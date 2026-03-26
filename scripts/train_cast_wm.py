#!/usr/bin/env python3
"""
Train a latent world model on the CAST navigation dataset.

Architecture:
    Encoder:   Frozen DINOv2-ViT-S/14  (or V-JEPA2-ViT-L)  →  (B, T, 1, H, W, D)
    Predictor: AdaLN ViT (action-conditioned)                →  next-frame features
    Loss:      L2 (+ optional cosine / L1 / smooth-L1) on predicted vs. GT features

Data:
    8 consecutive observations per sample  (window_size=8)
    Single-step action per timestep        (action_horizon=1)
    Action = (local_x, local_y, sin_yaw, cos_yaw)

Usage:
    # Single GPU
    python scripts/train_cast_wm.py --encoder dino --debug

    # Multi-GPU (DDP via torchrun)
    torchrun --nproc_per_node 4 scripts/train_cast_wm.py --encoder dino

    # With V-JEPA2 encoder
    torchrun --nproc_per_node 4 scripts/train_cast_wm.py --encoder vjepa
"""

import datetime
import importlib.util
import io
import os
import socket
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from einops import rearrange

from absl import app, flags, logging
import tqdm
import wandb
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def _find_free_port():
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _setup_distributed():
    """
    Initialise distributed training.
    Works when launched via torchrun **or** plain `python` (single-GPU fallback).
    """
    if "RANK" not in os.environ:
        # Not launched via torchrun → single-GPU mode
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        torch.distributed.init_process_group(backend="nccl")

    from accelerate import PartialState
    return PartialState()

# ─── CAST dataset loading (same pipeline as finetune scripts) ────────────────
from octo.data.dataset import make_interleaved_dataset
from octo.data.utils.data_utils import NormalizationType
from octo.data.utils.cast_transforms import gnm_action_angle_dataset_transform
from octo.data.utils.text_processing import HFTokenizer
from octo.utils.spec import ModuleSpec
from octo.utils.torch_rlds_dataset import TorchRLDSDataset
from octo.utils.train_utils_pt import _np2pt

# ─── Encoders ────────────────────────────────────────────────────────────────
from octo.model.components.dino_encoder import DinoV2EncoderPt
from octo.model.components.vjepa_encoder import VJEPA2EncoderPt

# ─── AdaLN predictor ─────────────────────────────────────────────────────────
# NOTE: The AdaLN import is deferred to build_predictor() to avoid a
# `src.*` namespace collision with the VJEPA2 torch.hub cached code.
# Both jepa_wms/src and the hub's src define `src.models.utils.modules.Block`
# with different forward() signatures (H/W vs H_patches/W_patches).
_JEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jepa_wms"))

# ─── Flags ────────────────────────────────────────────────────────────────────
FLAGS = flags.FLAGS
flags.DEFINE_string("config", "scripts/configs/train_cast_wm_config.py", "Config path")
flags.DEFINE_string("encoder", "dino", "Encoder type: dino | vjepa")
flags.DEFINE_string("name", "cast_wm", "W&B run name")
flags.DEFINE_bool("debug", False, "Quick sanity-check mode")
flags.DEFINE_string("save_dir", None, "Override config save_dir")
flags.DEFINE_float("grad_clip", None, "Override config grad_clip")
flags.DEFINE_string("resume", None, "Path to checkpoint to resume from")
flags.DEFINE_integer("rollout_steps", None, "Override config rollout_steps")
flags.DEFINE_integer("batch_size", None, "Override config batch_size (reduce for rollout training)")
flags.DEFINE_integer("num_steps", None, "Override config num_steps (extend when resuming)")
flags.DEFINE_float("lr", None, "Override config learning rate (lower for rollout training)")
flags.DEFINE_bool("skip_optimizer_state", False, "Skip loading optimizer/scheduler state (reset momentum)")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def custom_collate_fn(batch):
    if isinstance(batch[0], dict):
        return {key: custom_collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, dim=0)
    elif isinstance(batch[0], (bool, np.bool_)):
        return torch.tensor(batch, dtype=torch.bool)
    elif isinstance(batch[0], (int, np.integer)):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(batch[0], (float, np.floating)):
        return torch.tensor(batch, dtype=torch.float32)
    else:
        return batch


def _to_device(data, device):
    if isinstance(data, dict):
        return {key: _to_device(val, device) for key, val in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    return data


def _make_sample_viz(images, actions, sample_idx=0):
    """
    Create a wandb.Image showing 4 evenly-spaced frames from the 8-frame
    window plus the action trajectory (cumulative XY waypoints).

    Args:
        images:  (B, T, C, H, W) tensor (uint8 or float)
        actions: (B, T, A) tensor — A=(local_x, local_y, sin_yaw, cos_yaw)
        sample_idx: which sample in the batch to visualise
    Returns:
        wandb.Image
    """
    T = images.shape[1]
    # Pick 4 evenly-spaced frame indices from [0, T)
    frame_idxs = np.linspace(0, T - 1, 4, dtype=int)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # ── Plot 4 frames ─────────────────────────────────────────────────
    for ax_i, fi in enumerate(frame_idxs):
        img = images[sample_idx, fi].cpu().numpy()  # (C, H, W)
        if img.shape[0] in (1, 3):  # CHW → HWC
            img = img.transpose(1, 2, 0)
        if img.max() > 1.0:
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0.0, 1.0)
        axes[ax_i].imshow(img)
        axes[ax_i].set_title(f"t={fi}", fontsize=10)
        axes[ax_i].axis("off")

    # ── Plot action trajectory in frame-0 coordinate ────────────────
    # Each a_t = (local_x, local_y, sin_yaw, cos_yaw) is in frame t's ego
    # coord.  To plot in frame 0: rotate each (dx, dy) by accumulated heading.
    act = actions[sample_idx].cpu().numpy()  # (T, 4)
    heading = 0.0  # accumulated heading relative to frame 0
    xy = [np.array([0.0, 0.0])]
    for t in range(act.shape[0]):
        dx_local, dy_local = act[t, 0], act[t, 1]
        # Rotate local displacement into frame-0 coordinate
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        dx_global = cos_h * dx_local - sin_h * dy_local
        dy_global = sin_h * dx_local + cos_h * dy_local
        xy.append(xy[-1] + np.array([dx_global, dy_global]))
        # Update heading with this step's yaw change
        heading += np.arctan2(act[t, 2], act[t, 3])  # sin_yaw, cos_yaw
    xy = np.array(xy)  # (T+1, 2)

    ax_traj = axes[4]
    ax_traj.plot(xy[:, 0], xy[:, 1], "b-o", markersize=4, linewidth=1.5)
    ax_traj.plot(0, 0, "k*", markersize=10, label="start")
    ax_traj.plot(xy[-1, 0], xy[-1, 1], "r*", markersize=10, label="end")
    # Annotate each timestep
    for ti in range(len(xy)):
        ax_traj.annotate(str(ti), (xy[ti, 0], xy[ti, 1]), fontsize=6,
                         textcoords="offset points", xytext=(3, 3))
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_title("Action trajectory (xy)")
    ax_traj.legend(fontsize=8)
    ax_traj.set_aspect("equal", "datalim")
    ax_traj.grid(True, alpha=0.4)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    from PIL import Image as PILImage
    pil_img = PILImage.open(buf).copy()
    plt.close(fig)
    buf.close()
    return wandb.Image(pil_img, caption=f"sample {sample_idx}")


# ═══════════════════════════════════════════════════════════════════════════════
# World Model wrapper (lightweight version of VideoWM)
# ═══════════════════════════════════════════════════════════════════════════════

class CastWorldModel(nn.Module):
    """
    Latent world model for CAST:
        encoder   → frozen DINOv2 / V-JEPA2
        predictor → AdaLN ViT (trainable)

    The predictor takes (image_features, actions) and outputs predicted
    next-frame features.  Loss is computed by shifting ground-truth features
    by 1 timestep and comparing against predictions.
    """

    def __init__(self, encoder, predictor, cfg):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

        self.enc_type = cfg.encoder_type  # "dino" | "vjepa"
        self.grid_size = cfg.grid_size
        self.batchify_video = cfg.batchify_video
        self.normalize_reps = cfg.normalize_reps

        # Loss weights
        self.cfgs_loss = {
            "l2_loss_weight": cfg.loss.l2_loss_weight,
            "l1_loss_weight": cfg.loss.l1_loss_weight,
            "cos_loss_weight": cfg.loss.cos_loss_weight,
            "smooth_l1_loss_weight": cfg.loss.smooth_l1_loss_weight,
        }
        self.proprio_loss = cfg.loss.proprio_loss

    # ── Encode observations ──────────────────────────────────────────────
    @torch.no_grad()
    def encode_obs(self, images):
        """
        Args:
            images: (B, T, C, H, W) uint8 in [0, 255]
        Returns:
            visual_embs: (B, T, 1, grid_h, grid_w, embed_dim)
        """
        B, T, C, H, W = images.shape
        self.encoder.eval()

        if self.enc_type == "dino":
            # DINO processes each frame independently
            flat = rearrange(images, "b t c h w -> (b t) c h w")
            embs = self.encoder(flat)                    # (BT, C_out, gh, gw)
            embs = rearrange(
                embs, "(b t) d h w -> b t 1 h w d",
                b=B, t=T,
            )
        elif self.enc_type == "vjepa":
            flat = rearrange(images, "b t c h w -> (b t) c h w")
            embs = self.encoder(flat)                    # (BT, C_out, gh, gw)
            embs = rearrange(
                embs, "(b t) d h w -> b t 1 h w d",
                b=B, t=T,
            )
        else:
            raise ValueError(f"Unknown enc_type: {self.enc_type}")

        if self.normalize_reps:
            embs = F.layer_norm(embs, (embs.size(-1),))
        return embs

    # ── Forward predictor ────────────────────────────────────────────────
    def forward_pred(self, video_features, actions):
        """
        Args:
            video_features: (B, T, 1, H, W, D)
            actions:        (B, T, action_dim)
        Returns:
            pred_features:  (B, T, 1, H, W, D)
        """
        pred_video, _, _ = self.predictor(
            video_features,
            actions,
            proprio=None,
        )
        # predictor returns (B, T, H*W, D), reshape to spatial
        pred_video = rearrange(
            pred_video,
            "b t (v h w) d -> b t v h w d",
            v=1, h=self.grid_size, w=self.grid_size,
        )
        if self.normalize_reps:
            pred_video = F.layer_norm(pred_video, (pred_video.size(-1),))
        return pred_video

    # ── Full forward ─────────────────────────────────────────────────────
    def forward(self, images, actions):
        """
        Args:
            images:  (B, T, C, H, W)
            actions: (B, T, action_dim)
        Returns:
            pred_video_features: (B, T, 1, H, W, D)
            video_features:      (B, T, 1, H, W, D)  (detached encoder output)
        """
        video_features = self.encode_obs(images)
        pred_video_features = self.forward_pred(video_features, actions)
        return pred_video_features, video_features

    # ── Loss (mirrors VideoWM.compute_loss) ──────────────────────────────
    def compute_loss(self, pred_video_features, video_features, shift=1):
        """
        Shift-by-1 prediction loss: predict[t] should match GT[t+shift].

        Args:
            pred_video_features: (B, T, V, H, W, D)
            video_features:      (B, T, V, H, W, D)
            shift: how many steps ahead we predict (default 1)
        Returns:
            dict with "loss" and per-component losses
        """
        B, T, V, H, W, C = pred_video_features.shape
        pred_flat = pred_video_features.reshape(B, T, V * H * W, C)
        gt_flat = video_features.reshape(B, T, V * H * W, C)

        # Shift: pred[:-shift] should match gt[shift:]
        # Special case: shift=0 means no temporal shift (already aligned)
        if shift == 0:
            pred_ = pred_flat
            gt_ = gt_flat
        else:
            pred_ = pred_flat[:, :-shift]
            gt_ = gt_flat[:, shift:]

        # L2 loss
        l2_loss = F.mse_loss(pred_, gt_, reduction="none").mean(dim=-1)

        # L1 loss
        l1_loss = F.l1_loss(pred_, gt_, reduction="none").mean(dim=-1)

        # Smooth L1 loss
        smooth_l1_loss = F.smooth_l1_loss(pred_, gt_, reduction="none").mean(dim=-1)

        # Cosine loss (negative cosine similarity)
        cos_loss = -(
            pred_ * gt_
            / (pred_.norm(dim=-1, keepdim=True) * gt_.norm(dim=-1, keepdim=True) + 1e-8)
        ).sum(dim=-1)

        # Combine
        loss = (
            self.cfgs_loss["l2_loss_weight"] * l2_loss
            + self.cfgs_loss["l1_loss_weight"] * l1_loss
            + self.cfgs_loss["smooth_l1_loss_weight"] * smooth_l1_loss
            + self.cfgs_loss["cos_loss_weight"] * cos_loss
        )

        return {
            "loss": loss.mean(),
            "l2_loss": l2_loss.mean(),
            "l1_loss": l1_loss.mean(),
            "smooth_l1_loss": smooth_l1_loss.mean(),
            "cos_loss": cos_loss.mean(),
        }

    # ── Multi-step rollout loss (autoregressive) ─────────────────────────
    def rollout_loss(self, video_features, actions, rollout_steps=1, ctxt_window=8):
        """
        Autoregressive rollout in latent space, accumulating prediction loss.

        At each step t (starting from ctxt_window-1):
            1. Take GT features up to t as context.
            2. Predict feature at t+1 using predictor.
            3. Replace GT feature at t with prediction for next iteration.
            4. Compute loss against GT at t+1.

        Args:
            video_features: (B, T, 1, H, W, D) — encoder output (detached)
            actions:        (B, T, A)
            rollout_steps:  how many autoregressive steps
            ctxt_window:    number of context frames the predictor sees
        Returns:
            dict of losses
        """
        B, T, V, H, W, D = video_features.shape
        total_loss = torch.tensor(0.0, device=video_features.device)
        n_steps = 0

        # Start from a random prefix to expose the predictor to varied contexts
        max_start = T - rollout_steps - 1
        if max_start < 1:
            # Not enough frames for rollout; fall back to 1-step
            return self.compute_loss(
                self.forward_pred(video_features, actions),
                video_features,
                shift=1,
            )

        t_start = torch.randint(0, max(max_start, 1), (1,)).item()
        pred_features = video_features.clone()

        for step in range(rollout_steps):
            t = t_start + step
            # Context window
            ctx_start = max(0, t + 1 - ctxt_window)
            ctx_end = t + 1  # up to and including t

            ctx_feats = pred_features[:, ctx_start:ctx_end].detach()
            ctx_acts = actions[:, ctx_start:ctx_end]

            # Predict next frame
            pred_next = self.forward_pred(ctx_feats, ctx_acts)

            # Loss: last prediction vs GT
            gt_next = video_features[:, t + 1 : t + 2]
            pred_last = pred_next[:, -1:]

            step_loss_dict = self.compute_loss(
                pred_last,
                gt_next,
                shift=0,  # already aligned
            )

            total_loss = total_loss + step_loss_dict["loss"]
            n_steps += 1

            # Feed prediction into context for next step (stop-gradient optional)
            pred_features = pred_features.clone()
            pred_features[:, t + 1 : t + 2] = pred_next[:, -1:].detach()

        if n_steps > 0:
            total_loss = total_loss / n_steps

        return {"loss": total_loss}


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(checkpoint_path, world_model, optimizer, lr_scheduler, device, skip_optimizer_state=False):
    """
    Load checkpoint and restore training state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        world_model: DDP-wrapped model
        optimizer: Optimizer instance
        lr_scheduler: Learning rate scheduler
        device: Device to load checkpoint to
        skip_optimizer_state: If True, skip loading optimizer/scheduler (reset momentum)
    
    Returns:
        resume_step: Step number to resume from
    """
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model weights (handle DDP wrapping)
    world_model.module.load_state_dict(checkpoint["model_state_dict"])
    
    if skip_optimizer_state:
        logging.info("  ⚠ Skipping optimizer/scheduler state (resetting momentum)")
    else:
        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logging.info("  ✓ Loaded optimizer state")
        
        # Load scheduler state
        if "lr_scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            logging.info("  ✓ Loaded lr_scheduler state")
    
    resume_step = checkpoint.get("step", 0)
    logging.info(f"  ✓ Resuming from step {resume_step}")
    
    return resume_step


# ═══════════════════════════════════════════════════════════════════════════════
# Build components
# ═══════════════════════════════════════════════════════════════════════════════

def build_encoder(cfg):
    """Create a frozen image encoder."""
    if cfg.encoder_type == "dino":
        encoder = DinoV2EncoderPt(
            model_name=cfg.enc_model_name,
            freeze=True,
            img_norm_type="imagenet",
        )
    elif cfg.encoder_type == "vjepa":
        encoder = VJEPA2EncoderPt(
            model_name=cfg.enc_model_name,
            freeze=True,
            img_norm_type="imagenet",
            force_resolution=cfg.img_size,
        )
    else:
        raise ValueError(f"Unknown encoder_type: {cfg.encoder_type}")
    return encoder


def build_predictor(cfg):
    """Create the AdaLN predictor."""
    # Lazy import: clear any hub `src.*` modules so jepa_wms's version is used.
    src_keys = [k for k in sys.modules if k == "src" or k.startswith("src.")]
    for k in src_keys:
        del sys.modules[k]
    if _JEPA_ROOT not in sys.path:
        sys.path.insert(0, _JEPA_ROOT)
    from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN

    predictor = vit_predictor_AdaLN(
        img_size=(cfg.img_size, cfg.img_size),
        patch_size=cfg.patch_size,
        num_frames=cfg.window_size,
        tubelet_size=1,               # no temporal convolution; frame-level
        embed_dim=cfg.embed_dim,
        predictor_embed_dim=cfg.pred_embed_dim,
        depth=cfg.pred_depth,
        num_heads=cfg.pred_num_heads,
        mlp_ratio=4.0,
        use_rope=cfg.use_rope,
        is_causal=cfg.is_causal,
        use_silu=cfg.use_silu,
        action_dim=cfg.action_dim,
        action_encoder_inpred=cfg.action_encoder_inpred,
        proprio_dim=cfg.proprio_dim,
        use_proprio=cfg.use_proprio,
        proprio_encoding=cfg.proprio_encoding,
        proprio_emb_dim=cfg.proprio_emb_dim,
        proprio_tokens=cfg.proprio_tokens,
        proprio_encoder_inpred=cfg.proprio_encoder_inpred,
        init_scale_factor_adaln=cfg.init_scale_factor_adaln,
    )
    return predictor


# ═══════════════════════════════════════════════════════════════════════════════
# CAST dataset helpers (mirrors finetune_cast_dino.py)
# ═══════════════════════════════════════════════════════════════════════════════

def build_cast_datasets(cfg):
    """
    Returns (train_dataset, val_dataset) as tf.data.Dataset objects
    using the octo RLDS pipeline.
    """
    transform = ModuleSpec.create(
        gnm_action_angle_dataset_transform,
        action_horizon=cfg.action_horizon,
    )

    dataset_kwargs_list = []
    for name in cfg.dataset_names:
        if name.startswith("atomic_"):
            subdir = "atomic_datasets"
        else:
            subdir = name
        dataset_kwargs_list.append({
            "name": name,
            "data_dir": f"{cfg.data_dir}/{subdir}",
            "image_obs_keys": {"primary": "image"},
            "proprio_obs_key": "state",
            "language_key": "language_instruction",
            "action_proprio_normalization_type": NormalizationType.BOUNDS,
            "standardize_fn": transform,
            "force_recompute_dataset_statistics": False,
            "skip_norm": False,
        })

    shared_kwargs = dict(
        sample_weights=cfg.sample_weights,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        frame_transform_kwargs={
            "resize_size": {"primary": (cfg.img_size, cfg.img_size)},
            "image_augment_kwargs": {"primary": cfg.img_augment_kwargs},
        },
        traj_transform_kwargs={
            "window_size": cfg.window_size,
            "action_horizon": cfg.action_horizon,
            "goal_relabeling_strategy": None,
            "task_augment_strategy": None,
            "subsample_length": 100,
        },
        batch_size=None,
        balance_weights=False,
        traj_transform_threads=cfg.traj_transform_threads,
        traj_read_threads=cfg.traj_read_threads,
    )

    train_dataset = make_interleaved_dataset(dataset_kwargs_list, train=True, **shared_kwargs)
    val_dataset = make_interleaved_dataset(dataset_kwargs_list, train=False, **shared_kwargs)

    # Capture statistics before filter/prefetch strips them
    dataset_statistics = train_dataset.dataset_statistics
    sample_weights = train_dataset.sample_weights

    # Filter out blank images
    train_dataset = train_dataset.filter(
        lambda x: tf.reduce_any(x["observation"]["image_primary"] != 255)
    )
    val_dataset = val_dataset.filter(
        lambda x: tf.reduce_any(x["observation"]["image_primary"] != 255)
    )

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # Re-attach statistics so TorchRLDSDataset.__len__ works
    train_dataset.dataset_statistics = dataset_statistics
    train_dataset.sample_weights = sample_weights
    val_dataset.dataset_statistics = dataset_statistics
    val_dataset.sample_weights = sample_weights

    return train_dataset, val_dataset, dataset_statistics


def make_text_processor():
    """A minimal HFTokenizer for language instructions (needed by TorchRLDSDataset)."""
    return HFTokenizer(
        tokenizer_name="t5-base",
        encode_with_model=False,
        tokenizer_kwargs={
            "max_length": 16,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "np",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def main(_):
    # ── Load config ───────────────────────────────────────────────────────
    spec = importlib.util.spec_from_file_location("cfg_module", FLAGS.config)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config(encoder_type=FLAGS.encoder)

    # Override from flags
    if FLAGS.save_dir is not None:
        cfg.save_dir = FLAGS.save_dir
    if FLAGS.grad_clip is not None:
        cfg.grad_clip = FLAGS.grad_clip
    if FLAGS.rollout_steps is not None:
        logging.info(f"Overriding rollout_steps: {cfg.rollout_steps} → {FLAGS.rollout_steps}")
        cfg.rollout_steps = FLAGS.rollout_steps
    if FLAGS.batch_size is not None:
        logging.info(f"Overriding batch_size: {cfg.batch_size} → {FLAGS.batch_size}")
        cfg.batch_size = FLAGS.batch_size
    if FLAGS.num_steps is not None:
        logging.info(f"Overriding num_steps: {cfg.num_steps} → {FLAGS.num_steps}")
        cfg.num_steps = FLAGS.num_steps
    if FLAGS.lr is not None:
        logging.info(f"Overriding learning rate: {cfg.lr} → {FLAGS.lr}")
        cfg.lr = FLAGS.lr

    # ── Distributed setup ─────────────────────────────────────────────────
    assert torch.cuda.is_available(), "CUDA required"
    distributed_state = _setup_distributed()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{device_id}")
    is_main = distributed_state.is_main_process

    if is_main:
        logging.set_verbosity(logging.INFO)
    else:
        logging.set_verbosity(logging.ERROR)

    if FLAGS.debug:
        logging.info("DEBUG mode: small run")
        cfg.num_steps = 20
        cfg.batch_size = 2
        cfg.shuffle_buffer_size = 100
        cfg.eval_interval = 5
        cfg.viz = True
        cfg.viz_interval = 2
        cfg.save_interval = 999999
        cfg.log_interval = 1

    # ── W&B ───────────────────────────────────────────────────────────────
    if is_main:
        wandb.init(
            project=cfg.wandb_project,
            name=FLAGS.encoder + "_" + FLAGS.name,
            config=cfg.to_dict(),
        )

    logging.info("=" * 80)
    logging.info(f"TRAINING WORLD MODEL ON CAST  (encoder={cfg.encoder_type})")
    logging.info("=" * 80)

    # ── Build dataset ─────────────────────────────────────────────────────
    logging.info("Loading CAST dataset ...")
    text_processor = make_text_processor()
    train_tf_ds, val_tf_ds, dataset_statistics = build_cast_datasets(cfg)

    train_pt = TorchRLDSDataset(train_tf_ds, text_processor, train=True)
    val_pt = TorchRLDSDataset(val_tf_ds, text_processor, train=False)

    train_loader = DataLoader(
        train_pt,
        batch_size=cfg.batch_size,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_pt,
        batch_size=cfg.batch_size,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )
    logging.info("Dataset ready.")

    # ── Build model ───────────────────────────────────────────────────────
    logging.info("Building encoder + predictor ...")
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)
    world_model = CastWorldModel(encoder, predictor, cfg).to(device)

    # Freeze encoder
    for p in world_model.encoder.parameters():
        p.requires_grad = False
    world_model.encoder.eval()

    # DDP
    world_model = DDP(
        world_model,
        device_ids=[device_id],
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )

    trainable_params = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in world_model.parameters())
    logging.info(f"Parameters: trainable {trainable_params:,} / total {total_params:,} "
                 f"({100 * trainable_params / total_params:.1f}%)")

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    trainable_list = [p for p in world_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_list, lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.num_steps
    logging.info(f"Total training steps: {total_steps}")

    def cosine_lr_lambda(current_step):
        if current_step < cfg.warmup_steps:
            return float(current_step) / float(max(1, cfg.warmup_steps))
        progress = float(current_step - cfg.warmup_steps) / float(
            max(1, total_steps - cfg.warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_lambda)

    # Mixed precision
    use_amp = cfg.mixed_precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.mixed_precision == "fp16"))

    # ── Resume from checkpoint ────────────────────────────────────────────
    resume_step = 0
    if FLAGS.resume:
        resume_step = load_checkpoint(
            FLAGS.resume, world_model, optimizer, lr_scheduler, device,
            skip_optimizer_state=FLAGS.skip_optimizer_state
        )
        # Step scheduler to the resumed position (unless we're resetting it)
        if not FLAGS.skip_optimizer_state:
            for _ in range(resume_step):
                lr_scheduler.step()
        logging.info(f"Resumed training from step {resume_step}")

    # ── Save directory ────────────────────────────────────────────────────
    save_dir = None
    if is_main and cfg.save_dir:
        if FLAGS.resume:
            # Extract original run_id from checkpoint path if resuming
            resume_path = Path(FLAGS.resume)
            if resume_path.parent.name.startswith(FLAGS.encoder):
                run_id = resume_path.parent.name
                save_dir = resume_path.parent
                logging.info(f"Resuming into existing checkpoint dir: {save_dir}")
            else:
                # Create new directory
                run_id = f"{FLAGS.encoder}_{FLAGS.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_resumed"
                save_dir = Path(cfg.save_dir) / run_id
                save_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"New checkpoint dir for resumed training: {save_dir}")
        else:
            run_id = f"{FLAGS.encoder}_{FLAGS.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            save_dir = Path(cfg.save_dir) / run_id
            save_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Checkpoints → {save_dir}")

    # ═══════════════════════════════════════════════════════════════════════
    # Validation
    # ═══════════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def run_validation(global_step, max_batches=20):
        world_model.eval()
        val_losses = []
        for vi, vbatch in enumerate(val_loader):
            if vi >= max_batches:
                break
            vbatch = _to_device(vbatch, device)
            images = vbatch["observation"]["image_primary"]  # (B, T, C, H, W)
            actions = vbatch["action"]                       # (B, T, H, A) or (B, T, A)
            if actions.dim() == 4:
                actions = actions[:, :, 0, :]  # (B, T, A) — per-frame ego action

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred_feats, gt_feats = world_model(images, actions)
                loss_dict = world_model.module.compute_loss(pred_feats, gt_feats, shift=1)
            val_losses.append(loss_dict["loss"].item())

        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        if is_main:
            wandb.log({"val/loss": mean_val}, step=global_step)
        logging.info(f"  [step {global_step}] val_loss = {mean_val:.6f}")
        world_model.train()
        world_model.module.encoder.eval()  # keep encoder in eval
        return mean_val

    # ═══════════════════════════════════════════════════════════════════════
    # Training loop
    # ═══════════════════════════════════════════════════════════════════════
    logging.info("\n" + "=" * 80)
    logging.info("STARTING TRAINING" if resume_step == 0 else "RESUMING TRAINING")
    logging.info("=" * 80)
    if resume_step > 0:
        logging.info(f"  Resume step:    {resume_step}")
    logging.info(f"  Total steps:    {cfg.num_steps}")
    logging.info(f"  Remaining steps: {max(0, cfg.num_steps - resume_step)}")
    logging.info(f"  Batch size:     {cfg.batch_size}")
    logging.info(f"  Learning rate:  {cfg.lr}")
    logging.info(f"  Warmup steps:   {cfg.warmup_steps}")
    logging.info(f"  Grad clip:      {cfg.grad_clip}")
    logging.info(f"  Window size:    {cfg.window_size}")
    logging.info(f"  Rollout steps:  {cfg.rollout_steps}")
    logging.info(f"  Mixed precision: {cfg.mixed_precision}")

    world_model.train()
    world_model.module.encoder.eval()

    best_val_loss = float("inf")

    # Infinite data iterator — restarts DataLoader when exhausted
    def infinite_loader(loader):
        while True:
            yield from loader

    train_iter = infinite_loader(train_loader)

    for global_step in tqdm.tqdm(
        range(resume_step, total_steps),
        initial=resume_step,
        total=total_steps,
        dynamic_ncols=True,
        desc="Training",
        disable=not is_main,
    ):
        batch = next(train_iter)
        batch = _to_device(batch, device)
        images = batch["observation"]["image_primary"]   # (B, T, C, H, W)
        actions = batch["action"]                        # (B, T, H, A) or (B, T, A)

        # action_horizon=1: actions is (B, T, 1, A). Squeeze horizon dim.
        if actions.dim() == 4:
            actions = actions[:, :, 0, :]  # (B, T, A) — each a_t in frame t's ego coord

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            # ── 1-step teacher-forced prediction ──────────────────────
            pred_feats, gt_feats = world_model(images, actions)
            loss_dict = world_model.module.compute_loss(
                pred_feats, gt_feats, shift=1,
            )
            loss = loss_dict["loss"]

            # ── Optional multi-step rollout loss ──────────────────────
            if cfg.rollout_steps > 1:
                rollout_dict = world_model.module.rollout_loss(
                    gt_feats,
                    actions,
                    rollout_steps=cfg.rollout_steps,
                    ctxt_window=cfg.ctxt_window,
                )
                # Average teacher-forced + rollout loss
                loss = (loss + rollout_dict["loss"]) / 2.0

        # Skip NaN
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"Step {global_step}: NaN/Inf loss — skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_list, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        # ── Logging ───────────────────────────────────────────────────
        if is_main and (global_step + 1) % cfg.log_interval == 0:
            log_dict = {
                "train/loss": loss.item(),
                "train/l2_loss": loss_dict["l2_loss"].item(),
                "train/cos_loss": loss_dict["cos_loss"].item(),
                "train/lr": lr_scheduler.get_last_lr()[0],
                "train/grad_norm": (
                    grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                ),
                "train/step": global_step + 1,
            }
            wandb.log(log_dict, step=global_step)

        # ── Image + action visualisation ──────────────────────────────
        if is_main and cfg.viz and (global_step + 1) % cfg.viz_interval == 0:
            viz_images = []
            for s_idx in range(min(4, images.shape[0])):
                viz_images.append(_make_sample_viz(images, actions, s_idx))
            wandb.log({"train/samples": viz_images}, step=global_step)

        # ── Validation ────────────────────────────────────────────────
        if (global_step + 1) % cfg.eval_interval == 0:
            val_loss = run_validation(global_step + 1)
            if is_main and val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_dir:
                    torch.save(
                        {
                            "step": global_step + 1,
                            "model_state_dict": world_model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": val_loss,
                            "config": cfg.to_dict(),
                        },
                        save_dir / "best_model.pt",
                    )
                    logging.info(f"  Saved best model (val_loss={val_loss:.6f})")

        # ── Checkpoint ────────────────────────────────────────────────
        if is_main and save_dir and (global_step + 1) % cfg.save_interval == 0:
            torch.save(
                {
                    "step": global_step + 1,
                    "model_state_dict": world_model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "config": cfg.to_dict(),
                },
                save_dir / f"checkpoint_step{global_step + 1}.pt",
            )

    # ── Final validation + checkpoint ─────────────────────────────────────
    logging.info("\nFinal validation ...")
    run_validation(total_steps)

    if is_main and save_dir:
        torch.save(
            {
                "step": total_steps,
                "model_state_dict": world_model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "config": cfg.to_dict(),
            },
            save_dir / "final_model.pt",
        )
        logging.info(f"Saved final checkpoint to {save_dir / 'final_model.pt'}")

    logging.info("Training complete.")
    if is_main:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
