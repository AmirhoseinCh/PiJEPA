#!/usr/bin/env python3
"""
Evaluate Octo + MPPI planner on the CAST validation dataset.

Four planners compared:
  1. Default MPPI   — random Gaussian initialization
  2. Octo-seeded MPPI — initialize from Octo action samples
  3. Octo WM         — best Octo sample selected via WM rollout (no MPPI)
  4. Octo mean       — simple mean of Octo action samples (no WM, no MPPI)

Reports aggregate:
  - ATE  (Absolute Trajectory Error) for XY and heading
  - RPE  (Relative Pose Error)       for XY and heading
  - Total wall-clock times per method

Usage:
    python scripts/eval_octo_mppi_cast.py [--num_examples 50] [--encoder_type dino]
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ── octo pipeline ────────────────────────────────────────────────────────────
from octo.data.dataset import make_interleaved_dataset
from octo.data.utils.data_utils import NormalizationType
from octo.data.utils.cast_transforms import gnm_action_angle_dataset_transform
from octo.data.utils.text_processing import HFTokenizer
from octo.model.components.dino_encoder import DinoV2EncoderPt
from octo.model.components.vjepa_encoder import VJEPA2EncoderPt
from octo.model.octo_model_pt import OctoModelPt
from octo.utils.spec import ModuleSpec
from octo.utils.torch_rlds_dataset import TorchRLDSDataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JEPA_ROOT = os.path.join(REPO_ROOT, "jepa_wms")


import logging
import warnings

# Suppress Python warnings
warnings.filterwarnings("ignore")

# Set logging level to ERROR for common noisy loggers
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)


# For TensorFlow (you already have TF_CPP_MIN_LOG_LEVEL)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# For absl logging (often used with TF)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory metrics: ATE and RPE  (scalar summaries only)
# ═══════════════════════════════════════════════════════════════════════════════

def wrap_angle(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def compute_ate_xy(xy_pred: np.ndarray, xy_gt: np.ndarray) -> Dict[str, float]:
    n = min(len(xy_pred), len(xy_gt))
    err = np.sqrt(((xy_pred[:n] - xy_gt[:n]) ** 2).sum(axis=-1))
    return {
        "ate_xy_rmse": float(np.sqrt((err ** 2).mean())),
        "ate_xy_mean": float(err.mean()),
        "ate_xy_final": float(err[-1]),
    }


def compute_ate_heading(h_pred: np.ndarray, h_gt: np.ndarray) -> Dict[str, float]:
    n = min(len(h_pred), len(h_gt))
    err = np.abs(wrap_angle(h_pred[:n] - h_gt[:n]))
    return {
        "ate_heading_rmse_deg": float(np.degrees(np.sqrt((err ** 2).mean()))),
        "ate_heading_mean_deg": float(np.degrees(err.mean())),
        "ate_heading_final_deg": float(np.degrees(err[-1])),
    }


def compute_rpe_xy(xy_pred: np.ndarray, xy_gt: np.ndarray) -> Dict[str, float]:
    n = min(len(xy_pred), len(xy_gt))
    dp = np.diff(xy_pred[:n], axis=0)
    dg = np.diff(xy_gt[:n], axis=0)
    err = np.sqrt(((dp - dg) ** 2).sum(axis=-1))
    if len(err) == 0:
        return {"rpe_xy_rmse": 0.0, "rpe_xy_mean": 0.0}
    return {
        "rpe_xy_rmse": float(np.sqrt((err ** 2).mean())),
        "rpe_xy_mean": float(err.mean()),
    }


def compute_rpe_heading(h_pred: np.ndarray, h_gt: np.ndarray) -> Dict[str, float]:
    n = min(len(h_pred), len(h_gt))
    dh_pred = np.diff(h_pred[:n])
    dh_gt = np.diff(h_gt[:n])
    err = np.abs(wrap_angle(dh_pred - dh_gt))
    if len(err) == 0:
        return {"rpe_heading_rmse_deg": 0.0, "rpe_heading_mean_deg": 0.0}
    return {
        "rpe_heading_rmse_deg": float(np.degrees(np.sqrt((err ** 2).mean()))),
        "rpe_heading_mean_deg": float(np.degrees(err.mean())),
    }


def compute_all_metrics(xy_pred, h_pred, xy_gt, h_gt) -> Dict[str, float]:
    m = {}
    m.update(compute_ate_xy(xy_pred, xy_gt))
    m.update(compute_ate_heading(h_pred, h_gt))
    m.update(compute_rpe_xy(xy_pred, xy_gt))
    m.update(compute_rpe_heading(h_pred, h_gt))
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Action / trajectory utilities
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_bound_action(action: np.ndarray, unnorm_stats: dict) -> np.ndarray:
    h = action.shape[0]
    p01 = unnorm_stats["p01"].cpu().numpy()[:h]
    p99 = unnorm_stats["p99"].cpu().numpy()[:h]
    mask = unnorm_stats.get("mask", np.ones(p01.shape[-1], dtype=bool))
    action = action.copy()
    action[..., : len(mask)] = np.where(
        mask,
        2.0 * (action[..., : len(mask)] - p01) / (p99 - p01 + 1e-8) - 1.0,
        action[..., : len(mask)],
    )
    return action


def unnormalize_bound_action(action: np.ndarray, unnorm_stats: dict) -> np.ndarray:
    h = action.shape[0]
    p01 = unnorm_stats["p01"].cpu().numpy()[:h]
    p99 = unnorm_stats["p99"].cpu().numpy()[:h]
    mask = unnorm_stats.get("mask", np.ones(p01.shape[-1], dtype=bool))
    action = action.copy()
    action[..., : len(mask)] = np.where(
        mask,
        (action[..., : len(mask)] + 1) * (p99 - p01) / 2 + p01,
        action[..., : len(mask)],
    )
    return action


def octo_to_wm_actions(actions_unnorm: np.ndarray, unnorm_stats: dict) -> np.ndarray:
    H = actions_unnorm.shape[0]
    delta_yaw = np.arctan2(actions_unnorm[:, 2], actions_unnorm[:, 3])
    heading = np.concatenate([np.zeros(1), np.cumsum(delta_yaw[:-1])])
    wm = actions_unnorm.copy()
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    wm[:, 0] = actions_unnorm[:, 0] * cos_h + actions_unnorm[:, 1] * sin_h
    wm[:, 1] = -actions_unnorm[:, 0] * sin_h + actions_unnorm[:, 1] * cos_h
    return normalize_bound_action(wm, unnorm_stats)


def wm_actions_to_trajectory(
    wm_actions_norm: np.ndarray, unnorm_stats: dict
) -> Tuple[np.ndarray, np.ndarray]:
    wm_unnorm = unnormalize_bound_action(wm_actions_norm, unnorm_stats)
    heading = 0.0
    xy = [np.array([0.0, 0.0])]
    headings = [0.0]
    for t in range(wm_unnorm.shape[0]):
        dx_l, dy_l = wm_unnorm[t, 0], wm_unnorm[t, 1]
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        xy.append(xy[-1] + np.array([cos_h * dx_l - sin_h * dy_l,
                                     sin_h * dx_l + cos_h * dy_l]))
        heading += np.arctan2(wm_unnorm[t, 2], wm_unnorm[t, 3])
        headings.append(heading)
    return np.array(xy), np.array(headings)


def gt_actions_to_trajectory(
    gt_actions_local: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    heading = 0.0
    xy = [np.array([0.0, 0.0])]
    headings = [0.0]
    for t in range(gt_actions_local.shape[0]):
        dx_l, dy_l = gt_actions_local[t, 0], gt_actions_local[t, 1]
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        xy.append(xy[-1] + np.array([cos_h * dx_l - sin_h * dy_l,
                                     sin_h * dx_l + cos_h * dy_l]))
        heading += np.arctan2(gt_actions_local[t, 2], gt_actions_local[t, 3])
        headings.append(heading)
    return np.array(xy), np.array(headings)


def octo_actions_to_trajectory(
    actions_unnorm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    xy = np.concatenate([np.zeros((1, 2)), np.cumsum(actions_unnorm[:, :2], axis=0)], axis=0)
    delta_yaw = np.arctan2(actions_unnorm[:, 2], actions_unnorm[:, 3])
    headings = np.concatenate([np.zeros(1), np.cumsum(delta_yaw)], axis=0)
    return xy, headings


# ═══════════════════════════════════════════════════════════════════════════════
# CastWorldModel
# ═══════════════════════════════════════════════════════════════════════════════

class CastWorldModel(nn.Module):
    def __init__(self, encoder, predictor, cfg):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.grid_size = cfg.grid_size
        self.normalize_reps = cfg.normalize_reps

    @torch.no_grad()
    def encode_obs(self, images):
        B, T, C, H, W = images.shape
        self.encoder.eval()
        flat = rearrange(images, "b t c h w -> (b t) c h w")
        embs = self.encoder(flat)
        embs = rearrange(embs, "(b t) d h w -> b t 1 h w d", b=B, t=T)
        if self.normalize_reps:
            embs = F.layer_norm(embs, (embs.size(-1),))
        return embs

    def forward_pred(self, video_features, actions):
        pred_video, _, _ = self.predictor(video_features, actions, proprio=None)
        pred_video = rearrange(
            pred_video, "b t (v h w) d -> b t v h w d",
            v=1, h=self.grid_size, w=self.grid_size,
        )
        if self.normalize_reps:
            pred_video = F.layer_norm(pred_video, (pred_video.size(-1),))
        return pred_video


# ═══════════════════════════════════════════════════════════════════════════════
# MPPI Planner
# ═══════════════════════════════════════════════════════════════════════════════

class L2FinalFrameObjective:
    def __init__(self, target_enc: torch.Tensor, sum_all_diffs: bool = False):
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs

    def __call__(self, encodings, actions, keepdims=False):
        diff = (self.target_enc - encodings).pow(2).mean(
            dim=tuple(range(2, encodings.ndim))
        )
        if not keepdims:
            return diff.sum(0) if self.sum_all_diffs else diff[-1]
        return diff

class L1FinalFrameObjective:
    def __init__(self, target_enc: torch.Tensor, sum_all_diffs: bool = False):
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs

    def __call__(self, encodings, actions, keepdims=False):
        diff = (self.target_enc - encodings).abs().mean(
            dim=tuple(range(2, encodings.ndim))
        )
        if not keepdims:
            return diff.sum(0) if self.sum_all_diffs else diff[-1]
        return diff

class MPPIPlannerLocal:
    def __init__(
        self, unroll_fn, objective, horizon, action_dim,
        iterations=6, num_samples=512, num_elites=64,
        temperature=0.5, max_std=2.0, min_std=0.05,
        init_mean=None, init_std_scale=None, device="cuda",
    ):
        self.unroll = unroll_fn
        self.objective = objective
        self.horizon = horizon
        self.action_dim = action_dim
        self.iterations = iterations
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.temperature = temperature
        self.max_std = max_std
        self.min_std = min_std
        self.device = torch.device(device)
        self.init_mean = init_mean
        self.init_std_scale = init_std_scale

    @torch.no_grad()
    def plan(self, z_init):
        H, A = self.horizon, self.action_dim
        if self.init_mean is not None:
            mean = self.init_mean.clone().to(self.device)
            if self.init_std_scale is not None:
                std = (self.init_std_scale.clone().to(self.device)
                       if isinstance(self.init_std_scale, torch.Tensor)
                       else self.init_std_scale * torch.ones(H, A, device=self.device))
            else:
                std = self.max_std * 0.5 * torch.ones(H, A, device=self.device)
        else:
            mean = torch.zeros(H, A, device=self.device)
            std = self.max_std * torch.ones(H, A, device=self.device)

        actions = torch.empty(H, self.num_samples, A, device=self.device)
        best_cost = float("inf")

        for _ in range(self.iterations):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                H, self.num_samples, A, device=self.device
            )
            actions.clamp_(-1.0, 1.0)

            predicted_encs = self.unroll(z_init, actions)
            cost = self.objective(predicted_encs, actions)

            cur_best = cost.min().item()
            if cur_best < best_cost:
                best_cost = cur_best

            elite_idxs = torch.topk(-cost, self.num_elites, dim=0).indices
            elite_loss = cost[elite_idxs]
            elite_actions = actions[:, elite_idxs]

            min_cost = cost.min()
            score = torch.exp(self.temperature * (min_cost - elite_loss))
            score /= score.sum() + 1e-9

            mean = (score.unsqueeze(0).unsqueeze(2) * elite_actions).sum(dim=1) / (score.sum() + 1e-9)
            std = torch.sqrt(
                (score.unsqueeze(0).unsqueeze(2) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1)
                / (score.sum() + 1e-9)
            ).clamp(self.min_std, self.max_std)

        score_np = score.cpu().numpy()
        chosen_idx = np.random.choice(np.arange(score_np.shape[0]), p=score_np)
        best_actions = elite_actions[:, chosen_idx]
        return best_actions, mean, best_cost


# ═══════════════════════════════════════════════════════════════════════════════
# Build functions
# ═══════════════════════════════════════════════════════════════════════════════

def _scrub_jepa_src():
    """Remove jepa_wms' ``src.*`` modules and JEPA_ROOT from sys.path so they
    don't shadow the torch-hub vjepa2 ``src.*`` namespace."""
    # 1. remove JEPA_ROOT from sys.path
    while JEPA_ROOT in sys.path:
        sys.path.remove(JEPA_ROOT)
    # 2. remove src.* modules that came from jepa_wms
    for k in [k for k in sys.modules
              if k == "src" or k.startswith("src.")]:
        del sys.modules[k]


def build_encoder(cfg):
    """Build the frozen vision encoder.
    For vjepa we first scrub any jepa_wms ``src.*`` residue so the torch-hub
    vjepa2 package gets a clean namespace."""
    if cfg.encoder_type == "dino":
        return DinoV2EncoderPt(
            model_name=cfg.enc_model_name, freeze=True, img_norm_type="imagenet"
        )
    elif cfg.encoder_type == "vjepa":
        _scrub_jepa_src()  # ensure clean namespace before hub import
        encoder = VJEPA2EncoderPt(
            model_name=cfg.enc_model_name, freeze=True,
            img_norm_type="imagenet", force_resolution=cfg.img_size,
        )
        return encoder
    raise ValueError(f"Unknown encoder_type: {cfg.encoder_type}")


def build_predictor(cfg):
    """Build the JEPA-WM predictor from ``jepa_wms``.
    Adds JEPA_ROOT to sys.path and imports from the local ``src`` package."""
    # Clear any prior src.* (e.g. from torch-hub vjepa2) to avoid confusion
    for k in [k for k in sys.modules
              if k == "src" or k.startswith("src.")]:
        del sys.modules[k]
    if JEPA_ROOT not in sys.path:
        sys.path.insert(0, JEPA_ROOT)

    from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
    predictor = vit_predictor_AdaLN(
        img_size=(cfg.img_size, cfg.img_size),
        patch_size=cfg.patch_size, num_frames=cfg.window_size,
        tubelet_size=1, embed_dim=cfg.embed_dim,
        predictor_embed_dim=cfg.pred_embed_dim,
        depth=cfg.pred_depth, num_heads=cfg.pred_num_heads,
        mlp_ratio=4.0, use_rope=cfg.use_rope,
        is_causal=cfg.is_causal, use_silu=cfg.use_silu,
        action_dim=cfg.action_dim,
        action_encoder_inpred=cfg.action_encoder_inpred,
        proprio_dim=cfg.proprio_dim, use_proprio=cfg.use_proprio,
        proprio_encoding=cfg.proprio_encoding,
        proprio_emb_dim=cfg.proprio_emb_dim,
        proprio_tokens=cfg.proprio_tokens,
        proprio_encoder_inpred=cfg.proprio_encoder_inpred,
        init_scale_factor_adaln=cfg.init_scale_factor_adaln,
    )

    # ── Clean up: remove jepa_wms src.* so it won't shadow the hub vjepa2
    #    namespace at runtime (critical for the vjepa encoder).
    _scrub_jepa_src()
    return predictor


def build_cast_val_dataset(cfg):
    transform = ModuleSpec.create(
        gnm_action_angle_dataset_transform, action_horizon=cfg.action_horizon
    )
    dataset_kwargs_list = []
    for name in cfg.dataset_names:
        subdir = "atomic_datasets" if name.startswith("atomic_") else name
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
        batch_size=None, balance_weights=False,
        traj_transform_threads=cfg.traj_transform_threads,
        traj_read_threads=cfg.traj_read_threads,
    )
    # For validation, use shuffle_buffer_size=1 to ensure deterministic ordering
    val_kwargs = shared_kwargs.copy()
    val_kwargs['shuffle_buffer_size'] = 1
    val_tf_ds = make_interleaved_dataset(dataset_kwargs_list, train=False, **val_kwargs)
    dataset_statistics = val_tf_ds.dataset_statistics
    sample_weights = val_tf_ds.sample_weights
    val_tf_ds = val_tf_ds.filter(
        lambda x: tf.reduce_any(x["observation"]["image_primary"] != 255)
    )
    val_tf_ds = val_tf_ds.prefetch(tf.data.AUTOTUNE)
    val_tf_ds.dataset_statistics = dataset_statistics
    val_tf_ds.sample_weights = sample_weights
    return val_tf_ds


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


# ═══════════════════════════════════════════════════════════════════════════════
# WM unroll wrapper for MPPI
# ═══════════════════════════════════════════════════════════════════════════════

def make_cast_wm_unroll(model, cfg):
    def cast_wm_unroll(z_init, act_suffix):
        T, B, A = act_suffix.shape
        V, gH, gW = 1, cfg.grid_size, cfg.grid_size
        D = z_init.shape[-1]
        current_feat = z_init.reshape(1, 1, V, gH, gW, D).expand(B, -1, -1, -1, -1, -1)
        preds = []
        for t in range(T):
            act_t = act_suffix[t : t + 1].permute(1, 0, 2)
            pred_t = model.forward_pred(current_feat, act_t)
            preds.append(pred_t.reshape(B, 1, V * gH * gW, D))
            current_feat = pred_t
        return torch.cat(preds, dim=1).permute(1, 0, 2, 3)
    return cast_wm_unroll


# ═══════════════════════════════════════════════════════════════════════════════
# Per-example evaluation  (lightweight result dataclass)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExampleResult:
    idx: int = 0
    text: str = ""
    dataset_name: str = ""
    metrics_default_mppi: Dict = field(default_factory=dict)
    metrics_octo_mppi: Dict = field(default_factory=dict)
    metrics_octo_wm: Dict = field(default_factory=dict)
    metrics_octo_mean: Dict = field(default_factory=dict)
    # Timing per planner (seconds)
    time_default_mppi: float = 0.0
    time_octo_mppi: float = 0.0
    time_octo_wm: float = 0.0
    time_octo_mean: float = 0.0
    # Trajectories for JSON export
    traj_gt_xy: Optional[np.ndarray] = None
    traj_gt_heading: Optional[np.ndarray] = None
    traj_default_mppi_xy: Optional[np.ndarray] = None
    traj_default_mppi_heading: Optional[np.ndarray] = None
    traj_octo_mppi_xy: Optional[np.ndarray] = None
    traj_octo_mppi_heading: Optional[np.ndarray] = None
    traj_octo_wm_xy: Optional[np.ndarray] = None
    traj_octo_wm_heading: Optional[np.ndarray] = None
    traj_octo_mean_xy: Optional[np.ndarray] = None
    traj_octo_mean_heading: Optional[np.ndarray] = None


def evaluate_one_example(
    batch, wm_model, octo_model, octo_ds_stats, cfg, device,
    cast_wm_unroll, mppi_cfg, planner_types,
):
    """Run selected planners on one validation example."""
    H_octo = mppi_cfg["horizon"]  # planning horizon for Octo and MPPI (in steps)
    images_raw = batch["observation"]["image_primary"][:, :H_octo + 1]  # (1, H+1, C, H, W)
    actions_raw = batch["action"]
    if actions_raw.dim() == 4:
        actions_raw = actions_raw[:, :H_octo, 0, :]

    # print(f" action shape after possible squeezing: {actions_raw.shape}")
    text_instruction = batch["raw_language"][0]
    dataset_name = batch["dataset_name"][0]
    images = images_raw.to(device)

    # ── Encode all frames ────────────────────────────────────────────────
    with torch.no_grad():
        video_features = wm_model.encode_obs(images)
    feat_0 = video_features[:, :1, :, :, :, :]

    unnorm_stats = octo_ds_stats[dataset_name]["action"]
    # print(f"Unnormalization stats shape: p01 {unnorm_stats['p01'].shape}, p99 {unnorm_stats['p99'].shape}")
    # ── GT trajectory ────────────────────────────────────────────────────
    gt_actions_unnorm = unnormalize_bound_action(actions_raw[0].numpy(), unnorm_stats)
    
    # print(f"Horizon for Octo and MPPI: {H_octo} steps")
    xy_gt, h_gt = gt_actions_to_trajectory(gt_actions_unnorm[:H_octo])
    # print(f"GT traj shape: {xy_gt.shape}, {h_gt.shape}")

    # Initialize results
    result = ExampleResult(
        text=text_instruction, dataset_name=dataset_name,
        traj_gt_xy=xy_gt, traj_gt_heading=h_gt,
    )

    # Check if we need Octo inference for any planner
    needs_octo = any(p in planner_types for p in ["octo_mppi", "octo_wm", "octo_mean"])
    if not needs_octo:
        # Skip Octo inference if not needed
        pred_actions_np = None
    else:
        # ── Octo inference ───────────────────────────────────────────────────
        obs_image = images_raw[:, :1].to(device)
        timestep_pad_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
        tasks = octo_model.create_tasks(texts=[text_instruction], device=device)
        observations = {"image_primary": obs_image, "timestep_pad_mask": timestep_pad_mask}

        N_octo_samples = mppi_cfg["n_octo_samples"]
        with torch.no_grad():
            pred_actions = octo_model.sample_actions(
                observations=observations, tasks=tasks,
                unnormalization_statistics=unnorm_stats,
                normalization_type=NormalizationType.BOUNDS,
                timestep_pad_mask=timestep_pad_mask,
                train=False, argmax=False, save_attention_mask=False,
                sample_shape=(N_octo_samples,),
                generator=torch.Generator(device).manual_seed(0),
            )
        pred_actions_np = pred_actions.squeeze(1).cpu().numpy()  # (N, H_full, 4)

    # ── Octo mean ────────────────────────────────────────────────────────
    if "octo_mean" in planner_types:
        t_octo_mean_start = time.time()
        octo_mean_actions = pred_actions_np[:, :H_octo].mean(axis=0)
        # choose action randomly from the N samples instead of mean, since mean can be out-of-distribution for the WM
        # random_idx = random.randint(0, N_octo_samples - 1)
        # octo_mean_actions = pred_actions_np[random_idx, :H_octo]
        xy_octo_mean, h_octo_mean = octo_actions_to_trajectory(octo_mean_actions)
        # print(f"Octo mean traj shape: {xy_octo_mean.shape}, {h_octo_mean.shape}")
        result.time_octo_mean = time.time() - t_octo_mean_start
        result.metrics_octo_mean = compute_all_metrics(xy_octo_mean, h_octo_mean, xy_gt, h_gt)
        result.traj_octo_mean_xy = xy_octo_mean
        result.traj_octo_mean_heading = h_octo_mean

    # ── Best Octo sample via WM rollout ─────────────────────────────────
    if "octo_wm" in planner_types:
        t_octo_wm_start = time.time()
        all_l2_final = []
        with torch.no_grad():
            for s in range(N_octo_samples):
                wm_act_np = octo_to_wm_actions(pred_actions_np[s, :H_octo], unnorm_stats)
                wm_act_t = torch.tensor(wm_act_np, dtype=torch.float32, device=device).unsqueeze(0)
                cur = feat_0
                pred_last = None
                for t in range(H_octo):
                    pred_last = wm_model.forward_pred(cur, wm_act_t[:, t : t + 1])
                    cur = pred_last
                gt_final = video_features[:, H_octo : H_octo + 1]
                all_l2_final.append(F.mse_loss(pred_last, gt_final).item())

        best_idx = int(np.argmin(all_l2_final))
        xy_octo_wm, h_octo_wm = octo_actions_to_trajectory(pred_actions_np[best_idx, :H_octo])
        result.time_octo_wm = time.time() - t_octo_wm_start
        result.metrics_octo_wm = compute_all_metrics(xy_octo_wm, h_octo_wm, xy_gt, h_gt)
        result.traj_octo_wm_xy = xy_octo_wm
        result.traj_octo_wm_heading = h_octo_wm

    # ── Prepare Octo-seeded MPPI init (if needed) ───────────────────────
    needs_mppi = any(p in planner_types for p in ["default_mppi", "octo_mppi"])
    if needs_mppi:
        # ── z_init and objective for MPPI ────────────────────────────────────
        z_init_flat = feat_0.reshape(1, 1, -1, feat_0.shape[-1])
        gt_target = video_features[:, H_octo : H_octo + 1, 0]
        gt_target_flat = gt_target.reshape(1, -1, gt_target.shape[-1])
        objective = L2FinalFrameObjective(target_enc=gt_target_flat, sum_all_diffs=False)
        # objective = L1FinalFrameObjective(target_enc=gt_target_flat, sum_all_diffs=False)

    if "octo_mppi" in planner_types:
        all_octo_wm = np.stack(
            [octo_to_wm_actions(pred_actions_np[s, :H_octo], unnorm_stats)
             for s in range(N_octo_samples)], axis=0,
        )
        octo_mean_wm = torch.tensor(all_octo_wm.mean(axis=0), dtype=torch.float32)
        # octo_std_wm_t = torch.tensor(
        #     np.clip(np.std(all_octo_wm, axis=0), mppi_cfg["min_std"], mppi_cfg["max_std"]),
        #     dtype=torch.float32,
        # )
        octo_std_wm_t = torch.tensor(
            np.std(all_octo_wm, axis=0), dtype=torch.float32,
        )
        # print(f"Octo-seeded MPPI init std (clipped to [{mppi_cfg['min_std']}, {mppi_cfg['max_std']}]):")
        # print(octo_std_wm_t.cpu().numpy())
    # ── Default MPPI ─────────────────────────────────────────────────────
    if "default_mppi" in planner_types:
        t_default_mppi_start = time.time()
        planner_default = MPPIPlannerLocal(
            unroll_fn=cast_wm_unroll, objective=objective,
            horizon=H_octo, action_dim=cfg.action_dim,
            iterations=mppi_cfg["iterations"], num_samples=mppi_cfg["num_samples"],
            num_elites=mppi_cfg["num_elites"], temperature=mppi_cfg["temperature"],
            max_std=mppi_cfg["max_std"], min_std=mppi_cfg["min_std"],
            init_mean=None, device=device,
        )
        with torch.no_grad():
            default_actions, _, _ = planner_default.plan(z_init_flat)
        xy_default, h_default = wm_actions_to_trajectory(default_actions.cpu().numpy(), unnorm_stats)
        # print(f"Default MPPI traj shape: {xy_default.shape}, {h_default.shape}")
        result.time_default_mppi = time.time() - t_default_mppi_start
        result.metrics_default_mppi = compute_all_metrics(xy_default, h_default, xy_gt, h_gt)
        result.traj_default_mppi_xy = xy_default
        result.traj_default_mppi_heading = h_default

    # ── Octo-seeded MPPI ─────────────────────────────────────────────────
    if "octo_mppi" in planner_types:
        t_octo_mppi_start = time.time()
        planner_octo = MPPIPlannerLocal(
            unroll_fn=cast_wm_unroll, objective=objective,
            horizon=H_octo, action_dim=cfg.action_dim,
            iterations=mppi_cfg["iterations"], num_samples=mppi_cfg["num_samples"],
            num_elites=mppi_cfg["num_elites"], temperature=mppi_cfg["temperature"],
            max_std=mppi_cfg["max_std"], min_std=mppi_cfg["min_std"],
            init_mean=octo_mean_wm, init_std_scale=octo_std_wm_t, device=device,
        )
        with torch.no_grad():
            octo_mppi_actions, _, _ = planner_octo.plan(z_init_flat)
        xy_octo_mppi, h_octo_mppi = wm_actions_to_trajectory(octo_mppi_actions.cpu().numpy(), unnorm_stats)
        # print(f"Octo-seeded MPPI traj shape: {xy_octo_mppi.shape}, {h_octo_mppi.shape}")
        result.time_octo_mppi = time.time() - t_octo_mppi_start
        result.metrics_octo_mppi = compute_all_metrics(xy_octo_mppi, h_octo_mppi, xy_gt, h_gt)
        result.traj_octo_mppi_xy = xy_octo_mppi
        result.traj_octo_mppi_heading = h_octo_mppi

    # ── Return results ───────────────────────────────────────────────────
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation & printing
# ═══════════════════════════════════════════════════════════════════════════════

SUMMARY_KEYS = [
    "ate_xy_rmse", "ate_xy_mean", "ate_xy_final",
    "ate_heading_rmse_deg", "ate_heading_mean_deg", "ate_heading_final_deg",
    "rpe_xy_rmse", "rpe_xy_mean",
    "rpe_heading_rmse_deg", "rpe_heading_mean_deg",
]

METHOD_ATTRS = ["metrics_default_mppi", "metrics_octo_mppi", "metrics_octo_wm", "metrics_octo_mean"]


def aggregate_metrics(results: List[ExampleResult], planner_types: List[str]) -> Dict:
    agg = {}
    for method in METHOD_ATTRS:
        method_name = method.replace("metrics_", "")
        if method_name not in planner_types:
            continue
        agg[method_name] = {}
        for k in SUMMARY_KEYS:
            vals = [getattr(r, method)[k] for r in results 
                    if getattr(r, method) is not None and k in getattr(r, method)]
            if vals:
                arr = np.array(vals)
                agg[method_name][k] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "median": float(np.median(arr)),
                }
    return agg


def print_summary_table(agg: Dict, n_examples: int, time_per_method: Dict[str, float], planner_types: List[str]):
    # Map planner names to display names
    display_names = {
        "default_mppi": "Default MPPI",
        "octo_mppi": "Octo-seed MPPI",
        "octo_wm": "Octo WM",
        "octo_mean": "Octo mean",
    }
    
    # Calculate dynamic widths
    n_planners = len(planner_types)
    col_width = 15
    metric_width = 35
    total_width = metric_width + 2 + n_planners * (col_width + 2)
    
    print(f"\n{'=' * total_width}")
    print(f"  OCTO + MPPI EVALUATION ON CAST VALIDATION  ({n_examples} examples)")
    print(f"{'=' * total_width}")

    header = f"  {'Metric':<{metric_width}s}"
    for ptype in planner_types:
        header += f"  {display_names[ptype]:>{col_width}s}"
    print(header)
    print("  " + "─" * (total_width - 2))

    display_keys = [
        ("ATE XY RMSE (m)", "ate_xy_rmse"),
        ("ATE XY Mean (m)", "ate_xy_mean"),
        ("ATE XY Final (m)", "ate_xy_final"),
        ("ATE Heading RMSE (deg)", "ate_heading_rmse_deg"),
        ("ATE Heading Mean (deg)", "ate_heading_mean_deg"),
        ("ATE Heading Final (deg)", "ate_heading_final_deg"),
        ("RPE XY RMSE (m)", "rpe_xy_rmse"),
        ("RPE XY Mean (m)", "rpe_xy_mean"),
        ("RPE Heading RMSE (deg)", "rpe_heading_rmse_deg"),
        ("RPE Heading Mean (deg)", "rpe_heading_mean_deg"),
    ]

    for label, key in display_keys:
        line = f"  {label:<{metric_width}s}"
        for method in planner_types:
            d = agg.get(method, {}).get(key, {})
            val_str = f"{d['mean']:.4f}±{d['std']:.4f}" if d else "N/A"
            line += f"  {val_str:>{col_width}s}"
        print(line)
    print("  " + "─" * (total_width - 2))
    # Print timing per method
    time_line = f"  {'Total Time':<{metric_width}s}"
    for m in planner_types:
        time_val = f"{time_per_method.get(m, 0):.2f}s"
        time_line += f"  {time_val:>{col_width}s}"
    print(time_line)
    avg_line = f"  {'Avg Time/Example':<{metric_width}s}"
    for m in planner_types:
        avg_val = f"{time_per_method.get(m, 0)/n_examples*1000:.1f}ms"
        avg_line += f"  {avg_val:>{col_width}s}"
    print(avg_line)
    print(f"{'=' * total_width}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate Octo+MPPI on CAST val")
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--encoder_type", type=str, default="dino", choices=["dino", "vjepa"])
    parser.add_argument("--planner_types", type=str, nargs="+",
                        default=["default_mppi", "octo_mppi", "octo_wm", "octo_mean"],
                        choices=["default_mppi", "octo_mppi", "octo_wm", "octo_mean"],
                        help="List of planner types to evaluate")
    parser.add_argument("--wm_checkpoint", type=str,
                        default="/mnt/weka/zhougrp/octo_wm_cast/dino_cast_wm_20260228_193050/best_model.pt")
    parser.add_argument("--octo_ckpt_dir", type=str,
                        default="/mnt/weka/zhougrp/octo_pt_cast/cast_dino_finetune_20260219_202345")
    parser.add_argument("--octo_ckpt_step", type=int, default=2000)
    parser.add_argument("--mppi_horizon", type=int, default=7)
    parser.add_argument("--mppi_iterations", type=int, default=3)
    parser.add_argument("--mppi_samples", type=int, default=100)
    parser.add_argument("--mppi_elites", type=int, default=10)
    parser.add_argument("--mppi_temperature", type=float, default=0.5)
    parser.add_argument("--mppi_max_std", type=float, default=2.0)
    parser.add_argument("--mppi_min_std", type=float, default=0.05)
    parser.add_argument("--n_octo_samples", type=int, default=3)
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set all random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tf.random.set_seed(args.seed)
    
    # Ensure deterministic behavior on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable TensorFlow deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Seed: {args.seed}")

    # ── Load WM config ───────────────────────────────────────────────────
    config_path = os.path.join(REPO_ROOT, "scripts/configs/train_cast_wm_config.py")
    spec = importlib.util.spec_from_file_location("cfg_module", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config(encoder_type=args.encoder_type)
    print(f"Config: encoder={cfg.encoder_type}, img_size={cfg.img_size}, "
          f"grid={cfg.grid_size}, embed_dim={cfg.embed_dim}")

    # ── Build and load WM ────────────────────────────────────────────────
    # Build predictor FIRST so that jepa_wms' src.* can be loaded and then
    # scrubbed from sys.path/sys.modules BEFORE the torch-hub vjepa2
    # encoder loads its own src.* (they collide on the ``src`` namespace).
    print("Building world model...")
    predictor = build_predictor(cfg)
    encoder = build_encoder(cfg)
    wm_model = CastWorldModel(encoder, predictor, cfg).to(device)
    for p in wm_model.encoder.parameters():
        p.requires_grad = False
    wm_model.encoder.eval()

    print(f"Loading WM checkpoint: {args.wm_checkpoint}")
    ckpt = torch.load(args.wm_checkpoint, map_location=device, weights_only=False)
    wm_model.load_state_dict(ckpt["model_state_dict"])
    wm_model.eval()
    print(f"  WM loaded (step={ckpt.get('step', 'N/A')}, val_loss={ckpt.get('val_loss', 'N/A')})")

    # ── Load Octo ────────────────────────────────────────────────────────
    print(f"Loading Octo from: {args.octo_ckpt_dir} (step={args.octo_ckpt_step})")
    octo_load = OctoModelPt.load_pretrained(args.octo_ckpt_dir, step=args.octo_ckpt_step)
    octo_model = octo_load["octo_model"].to(device)
    octo_model.eval()
    octo_ds_stats = octo_model.dataset_statistics

    # ── Build validation dataset ─────────────────────────────────────────
    print("Building CAST validation dataset...")
    text_processor = HFTokenizer(
        tokenizer_name="t5-base", encode_with_model=False,
        tokenizer_kwargs={
            "max_length": 16, "padding": "max_length",
            "truncation": True, "return_tensors": "np",
        },
    )
    val_tf_ds = build_cast_val_dataset(cfg)
    val_pt = TorchRLDSDataset(val_tf_ds, text_processor, train=False)
    val_loader = DataLoader(
        val_pt, batch_size=1, num_workers=0,
        collate_fn=custom_collate_fn, pin_memory=True,
    )
    print("Validation DataLoader ready.")

    cast_wm_unroll = make_cast_wm_unroll(wm_model, cfg)
    mppi_cfg = {
        "horizon": args.mppi_horizon,
        "iterations": args.mppi_iterations,
        "num_samples": args.mppi_samples,
        "num_elites": args.mppi_elites,
        "temperature": args.mppi_temperature,
        "max_std": args.mppi_max_std,
        "min_std": args.mppi_min_std,
        "n_octo_samples": args.n_octo_samples,
    }

    # ── Evaluate ─────────────────────────────────────────────────────────
    print(f"\nMPPI: iters={mppi_cfg['iterations']}, samples={mppi_cfg['num_samples']}, "
          f"elites={mppi_cfg['num_elites']}, temp={mppi_cfg['temperature']}")

    results: List[ExampleResult] = []
    val_iter = iter(val_loader)
    t_eval_start = time.time()
    exhausted = False
    logging.getLogger().setLevel(logging.ERROR)
    pbar = tqdm(range(args.num_examples), desc="Evaluating", unit="ex")
    for i in pbar:
        batch = None
        while batch is None:
            try:
                candidate = next(val_iter)
            except StopIteration:
                exhausted = True
                break
            if (candidate["observation"]["image_primary"][:, -1] == 255).all():
                continue
            batch = candidate

        if batch is None:
            pbar.write(f"Dataset exhausted after {i} examples.")
            break

        result = evaluate_one_example(
            batch=batch, wm_model=wm_model, octo_model=octo_model,
            octo_ds_stats=octo_ds_stats, cfg=cfg, device=device,
            cast_wm_unroll=cast_wm_unroll, mppi_cfg=mppi_cfg,
            planner_types=args.planner_types,
        )
        result.idx = i
        results.append(result)

        # Running averages for the progress bar
        postfix_parts = []
        if "default_mppi" in args.planner_types:
            avg_ate_d = np.mean([r.metrics_default_mppi["ate_xy_final"] 
                                for r in results if r.metrics_default_mppi])
            postfix_parts.append(f"D:{avg_ate_d:.3f}")
        if "octo_mppi" in args.planner_types:
            avg_ate_o = np.mean([r.metrics_octo_mppi["ate_xy_final"] 
                                for r in results if r.metrics_octo_mppi])
            postfix_parts.append(f"O:{avg_ate_o:.3f}")
        if postfix_parts:
            pbar.set_postfix_str("ATE_xy " + " ".join(postfix_parts) + "m")

    if not results:
        print("No examples evaluated!")
        return

    # ── Aggregate timing per method ──────────────────────────────────────
    time_per_method = {}
    for ptype in args.planner_types:
        attr_name = f"time_{ptype}"
        time_per_method[ptype] = sum(getattr(r, attr_name) for r in results)

    # ── Aggregate and print ──────────────────────────────────────────────
    agg = aggregate_metrics(results, args.planner_types)
    print_summary_table(agg, len(results), time_per_method, args.planner_types)

    # ── Save JSON ────────────────────────────────────────────────────────
    if args.output_json:
        out = {
            "config": {
                "encoder_type": args.encoder_type,
                "wm_checkpoint": args.wm_checkpoint,
                "octo_ckpt_dir": args.octo_ckpt_dir,
                "octo_ckpt_step": args.octo_ckpt_step,
                "mppi": mppi_cfg,
                "num_examples": len(results),
                "seed": args.seed,
                "planner_types": args.planner_types,
            },
            "time_per_method_s": time_per_method,
            "aggregate": agg,
            "trajectories": [
                {
                    "idx": r.idx,
                    "text": r.text,
                    "dataset_name": r.dataset_name,
                    "gt": {"xy": r.traj_gt_xy.tolist(), "heading": r.traj_gt_heading.tolist()},
                    **{
                        ptype: {"xy": getattr(r, f"traj_{ptype}_xy").tolist(), 
                               "heading": getattr(r, f"traj_{ptype}_heading").tolist()}
                        for ptype in args.planner_types
                        if getattr(r, f"traj_{ptype}_xy") is not None
                    },
                }
                for r in results
            ],
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()