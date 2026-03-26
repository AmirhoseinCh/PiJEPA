#!/usr/bin/env python3
"""
Evaluate Octo + MPPI planner on the CAST validation dataset.

Four planners compared:
  1. Default MPPI   — random Gaussian initialization
  2. Octo-seeded MPPI — initialize from Octo action samples
  3. Octo WM         — best Octo sample selected via WM rollout (no MPPI)
  4. Octo mean       — simple mean of Octo action samples (no WM, no MPPI)

Reports per-example and aggregate:
  - ATE  (Absolute Trajectory Error) for XY and heading
  - RPE  (Relative Pose Error)       for XY and heading

Usage:
    python scripts/eval_octo_mppi_cast.py [--num_examples 50] [--encoder_type dino]
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory metrics: ATE and RPE
# ═══════════════════════════════════════════════════════════════════════════════

def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def compute_ate_xy(xy_pred: np.ndarray, xy_gt: np.ndarray) -> Dict[str, float]:
    """
    Absolute Trajectory Error (position).
    xy_pred, xy_gt: (N, 2)
    Returns dict with RMSE, mean, and per-step errors.
    """
    n = min(len(xy_pred), len(xy_gt))
    err = np.sqrt(((xy_pred[:n] - xy_gt[:n]) ** 2).sum(axis=-1))  # (N,)
    return {
        "ate_xy_rmse": float(np.sqrt((err ** 2).mean())),
        "ate_xy_mean": float(err.mean()),
        "ate_xy_max": float(err.max()),
        "ate_xy_final": float(err[-1]),
        "ate_xy_per_step": err.tolist(),
    }


def compute_ate_heading(h_pred: np.ndarray, h_gt: np.ndarray) -> Dict[str, float]:
    """
    Absolute Trajectory Error (heading).
    h_pred, h_gt: (N,)
    Returns dict with RMSE, mean (in radians and degrees).
    """
    n = min(len(h_pred), len(h_gt))
    err = np.abs(wrap_angle(h_pred[:n] - h_gt[:n]))  # (N,)
    return {
        "ate_heading_rmse_rad": float(np.sqrt((err ** 2).mean())),
        "ate_heading_mean_rad": float(err.mean()),
        "ate_heading_max_rad": float(err.max()),
        "ate_heading_final_rad": float(err[-1]),
        "ate_heading_rmse_deg": float(np.degrees(np.sqrt((err ** 2).mean()))),
        "ate_heading_mean_deg": float(np.degrees(err.mean())),
        "ate_heading_final_deg": float(np.degrees(err[-1])),
        "ate_heading_per_step_rad": err.tolist(),
    }


def compute_rpe_xy(xy_pred: np.ndarray, xy_gt: np.ndarray) -> Dict[str, float]:
    """
    Relative Pose Error (translational).
    Measures error on relative displacements between consecutive timesteps.
    xy_pred, xy_gt: (N, 2)
    """
    n = min(len(xy_pred), len(xy_gt))
    dp = np.diff(xy_pred[:n], axis=0)  # (N-1, 2)
    dg = np.diff(xy_gt[:n], axis=0)    # (N-1, 2)
    err = np.sqrt(((dp - dg) ** 2).sum(axis=-1))  # (N-1,)
    if len(err) == 0:
        return {"rpe_xy_rmse": 0.0, "rpe_xy_mean": 0.0, "rpe_xy_max": 0.0}
    return {
        "rpe_xy_rmse": float(np.sqrt((err ** 2).mean())),
        "rpe_xy_mean": float(err.mean()),
        "rpe_xy_max": float(err.max()),
        "rpe_xy_per_step": err.tolist(),
    }


def compute_rpe_heading(h_pred: np.ndarray, h_gt: np.ndarray) -> Dict[str, float]:
    """
    Relative Pose Error (rotational).
    Measures error on relative heading changes between consecutive timesteps.
    h_pred, h_gt: (N,)
    """
    n = min(len(h_pred), len(h_gt))
    dh_pred = np.diff(h_pred[:n])  # (N-1,)
    dh_gt = np.diff(h_gt[:n])      # (N-1,)
    err = np.abs(wrap_angle(dh_pred - dh_gt))  # (N-1,)
    if len(err) == 0:
        return {"rpe_heading_rmse_rad": 0.0, "rpe_heading_mean_rad": 0.0, "rpe_heading_mean_deg": 0.0}
    return {
        "rpe_heading_rmse_rad": float(np.sqrt((err ** 2).mean())),
        "rpe_heading_mean_rad": float(err.mean()),
        "rpe_heading_max_rad": float(err.max()),
        "rpe_heading_rmse_deg": float(np.degrees(np.sqrt((err ** 2).mean()))),
        "rpe_heading_mean_deg": float(np.degrees(err.mean())),
        "rpe_heading_per_step_rad": err.tolist(),
    }


def compute_all_metrics(xy_pred, h_pred, xy_gt, h_gt) -> Dict[str, float]:
    """Compute ATE + RPE for both XY and heading."""
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
    """Map real actions to [-1, 1] using p01/p99 stats."""
    h = action.shape[0]
    p01 = unnorm_stats["p01"].cpu().numpy()[:h]
    p99 = unnorm_stats["p99"].cpu().numpy()[:h]
    mask = unnorm_stats.get("mask", np.ones_like(p01, dtype=bool))
    action = action.copy()
    action[..., : len(mask)] = np.where(
        mask,
        2.0 * (action[..., : len(mask)] - p01) / (p99 - p01 + 1e-8) - 1.0,
        action[..., : len(mask)],
    )
    return action


def unnormalize_bound_action(action: np.ndarray, unnorm_stats: dict) -> np.ndarray:
    """Map normalized [-1, 1] actions back to real scale."""
    h = action.shape[0]
    p01 = unnorm_stats["p01"].cpu().numpy()[:h]
    p99 = unnorm_stats["p99"].cpu().numpy()[:h]
    mask = unnorm_stats.get("mask", np.ones_like(p01, dtype=bool))
    action = action.copy()
    action[..., : len(mask)] = np.where(
        mask,
        (action[..., : len(mask)] + 1) * (p99 - p01) / 2 + p01,
        action[..., : len(mask)],
    )
    return action


def octo_to_wm_actions(actions_unnorm: np.ndarray, unnorm_stats: dict) -> np.ndarray:
    """
    Convert Octo unnormalized actions (frame-0 coords) to WM normalized local actions.
    actions_unnorm: (H, 4) — dx, dy in frame-0 + sin_yaw, cos_yaw
    Returns:        (H, 4) — normalized [-1,1] in frame-t local coords
    """
    H = actions_unnorm.shape[0]
    delta_yaw = np.arctan2(actions_unnorm[:, 2], actions_unnorm[:, 3])
    heading = np.concatenate([np.zeros(1), np.cumsum(delta_yaw[:-1])])

    wm = actions_unnorm.copy()
    for t in range(H):
        dx0, dy0 = actions_unnorm[t, 0], actions_unnorm[t, 1]
        th = heading[t]
        wm[t, 0] = dx0 * np.cos(th) + dy0 * np.sin(th)
        wm[t, 1] = -dx0 * np.sin(th) + dy0 * np.cos(th)
    return normalize_bound_action(wm, unnorm_stats)


def wm_actions_to_trajectory(
    wm_actions_norm: np.ndarray, unnorm_stats: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert WM-normalized local actions → cumulative XY + heading in frame-0.
    Returns: xy (H+1, 2), headings (H+1,)
    """
    wm_unnorm = unnormalize_bound_action(wm_actions_norm, unnorm_stats)
    heading = 0.0
    xy = [np.array([0.0, 0.0])]
    headings = [0.0]
    for t in range(wm_unnorm.shape[0]):
        dx_l, dy_l = wm_unnorm[t, 0], wm_unnorm[t, 1]
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        dx_0 = cos_h * dx_l - sin_h * dy_l
        dy_0 = sin_h * dx_l + cos_h * dy_l
        xy.append(xy[-1] + np.array([dx_0, dy_0]))
        heading += np.arctan2(wm_unnorm[t, 2], wm_unnorm[t, 3])
        headings.append(heading)
    return np.array(xy), np.array(headings)


def gt_actions_to_trajectory(
    gt_actions_local: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GT actions are in local frame. Accumulate into frame-0 coords.
    gt_actions_local: (H, 4) — unnormalized local-frame actions
    Returns: xy (H+1, 2), headings (H+1,)
    """
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
    """
    Octo unnormalized actions are in frame-0 coords (dx, dy already global).
    Returns: xy (H+1, 2), headings (H+1,)
    """
    xy = np.concatenate([np.zeros((1, 2)), np.cumsum(actions_unnorm[:, :2], axis=0)], axis=0)
    delta_yaw = np.arctan2(actions_unnorm[:, 2], actions_unnorm[:, 3])
    headings = np.concatenate([np.zeros(1), np.cumsum(delta_yaw)], axis=0)
    return xy, headings


# ═══════════════════════════════════════════════════════════════════════════════
# CastWorldModel (same as notebook)
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
    """Minimize L2 distance between predicted features and GT target."""

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


class MPPIPlannerLocal:
    """MPPI planner with optional initial mean/std seeding."""

    def __init__(
        self,
        unroll_fn,
        objective,
        horizon,
        action_dim,
        iterations=6,
        num_samples=512,
        num_elites=64,
        temperature=0.5,
        max_std=2.0,
        min_std=0.05,
        init_mean=None,
        init_std_scale=None,
        device="cuda",
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
        self.init_mean = init_mean            # (H, A) tensor or None
        self.init_std_scale = init_std_scale  # (H, A) tensor, scalar, or None

    @torch.no_grad()
    def plan(self, z_init):
        H, A = self.horizon, self.action_dim

        if self.init_mean is not None:
            mean = self.init_mean.clone().to(self.device)
            if self.init_std_scale is not None:
                if isinstance(self.init_std_scale, torch.Tensor):
                    std = self.init_std_scale.clone().to(self.device)
                else:
                    std = self.init_std_scale * torch.ones(H, A, device=self.device)
            else:
                std = self.max_std * 0.5 * torch.ones(H, A, device=self.device)
        else:
            mean = torch.zeros(H, A, device=self.device)
            std = self.max_std * torch.ones(H, A, device=self.device)

        actions = torch.empty(H, self.num_samples, A, device=self.device)
        losses_per_iter = []

        for _ in range(self.iterations):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                H, self.num_samples, A, device=self.device
            )
            actions.clamp_(-1.0, 1.0)

            predicted_encs = self.unroll(z_init, actions)
            cost = self.objective(predicted_encs, actions)

            losses_per_iter.append(cost.min().item())

            elite_idxs = torch.topk(-cost, self.num_elites, dim=0).indices
            elite_loss = cost[elite_idxs]
            elite_actions = actions[:, elite_idxs]

            min_cost = cost.min()
            score = torch.exp(self.temperature * (min_cost - elite_loss))
            score /= score.sum() + 1e-9

            mean = (
                score.unsqueeze(0).unsqueeze(2) * elite_actions
            ).sum(dim=1) / (score.sum() + 1e-9)
            std = torch.sqrt(
                (
                    score.unsqueeze(0).unsqueeze(2)
                    * (elite_actions - mean.unsqueeze(1)) ** 2
                ).sum(dim=1)
                / (score.sum() + 1e-9)
            )
            std = std.clamp(self.min_std, self.max_std)

        score_np = score.cpu().numpy()
        chosen_idx = np.random.choice(np.arange(score_np.shape[0]), p=score_np)
        best_actions = elite_actions[:, chosen_idx]
        return best_actions, mean, losses_per_iter


# ═══════════════════════════════════════════════════════════════════════════════
# Build functions
# ═══════════════════════════════════════════════════════════════════════════════

def build_encoder(cfg):
    if cfg.encoder_type == "dino":
        return DinoV2EncoderPt(
            model_name=cfg.enc_model_name, freeze=True, img_norm_type="imagenet"
        )
    elif cfg.encoder_type == "vjepa":
        return VJEPA2EncoderPt(
            model_name=cfg.enc_model_name,
            freeze=True,
            img_norm_type="imagenet",
            force_resolution=cfg.img_size,
        )
    raise ValueError(f"Unknown encoder_type: {cfg.encoder_type}")


def build_predictor(cfg):
    src_keys = [k for k in sys.modules if k == "src" or k.startswith("src.")]
    for k in src_keys:
        del sys.modules[k]
    if JEPA_ROOT not in sys.path:
        sys.path.insert(0, JEPA_ROOT)
    from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN

    return vit_predictor_AdaLN(
        img_size=(cfg.img_size, cfg.img_size),
        patch_size=cfg.patch_size,
        num_frames=cfg.window_size,
        tubelet_size=1,
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


def build_cast_val_dataset(cfg):
    transform = ModuleSpec.create(
        gnm_action_angle_dataset_transform, action_horizon=cfg.action_horizon
    )
    dataset_kwargs_list = []
    for name in cfg.dataset_names:
        subdir = "atomic_datasets" if name.startswith("atomic_") else name
        dataset_kwargs_list.append(
            {
                "name": name,
                "data_dir": f"{cfg.data_dir}/{subdir}",
                "image_obs_keys": {"primary": "image"},
                "proprio_obs_key": "state",
                "language_key": "language_instruction",
                "action_proprio_normalization_type": NormalizationType.BOUNDS,
                "standardize_fn": transform,
                "force_recompute_dataset_statistics": False,
                "skip_norm": False,
            }
        )
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
    _ = make_interleaved_dataset(dataset_kwargs_list, train=True, **shared_kwargs)
    val_tf_ds = make_interleaved_dataset(dataset_kwargs_list, train=False, **shared_kwargs)
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
    """Create the unroll function that MPPI expects."""

    def cast_wm_unroll(z_init, act_suffix):
        """
        z_init     : (1, 1, V*gh*gw, D)
        act_suffix : (T, B, A)
        Returns    : (T, B, V*gh*gw, D)
        """
        T, B, A = act_suffix.shape
        V, gH, gW = 1, cfg.grid_size, cfg.grid_size
        D = z_init.shape[-1]
        current_feat = z_init.reshape(1, 1, V, gH, gW, D).expand(
            B, -1, -1, -1, -1, -1
        )
        preds = []
        for t in range(T):
            act_t = act_suffix[t : t + 1].permute(1, 0, 2)
            pred_t = model.forward_pred(current_feat, act_t)
            preds.append(pred_t.reshape(B, 1, V * gH * gW, D))
            current_feat = pred_t
        return torch.cat(preds, dim=1).permute(1, 0, 2, 3)

    return cast_wm_unroll


# ═══════════════════════════════════════════════════════════════════════════════
# Per-example evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExampleResult:
    idx: int
    text: str
    dataset_name: str
    gt_endpoint_xy: List[float] = field(default_factory=list)
    metrics_default_mppi: Dict = field(default_factory=dict)
    metrics_octo_mppi: Dict = field(default_factory=dict)
    metrics_octo_wm: Dict = field(default_factory=dict)
    metrics_octo_mean: Dict = field(default_factory=dict)
    mppi_default_cost: float = 0.0
    mppi_octo_cost: float = 0.0
    time_default_s: float = 0.0
    time_octo_s: float = 0.0
    time_octo_wm_s: float = 0.0
    time_octo_mean_s: float = 0.0
    # Trajectories: xy (H+1, 2), heading (H+1,)
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
    batch,
    wm_model,
    octo_model,
    octo_ds_stats,
    cfg,
    device,
    cast_wm_unroll,
    mppi_cfg,
):
    """Run all three planners on one validation example and return metrics."""

    images_raw = batch["observation"]["image_primary"]  # (1, T, C, H, W)
    actions_raw = batch["action"]                       # (1, T, 1, A) or (1, T, A)
    if actions_raw.dim() == 4:
        actions_raw = actions_raw[:, :, 0, :]

    text_instruction = batch["raw_language"][0]
    dataset_name = batch["dataset_name"][0]

    images = images_raw.to(device)
    actions = actions_raw.to(device)

    # ── Encode all frames ────────────────────────────────────────────────
    with torch.no_grad():
        video_features = wm_model.encode_obs(images)  # (1, T, 1, gh, gw, D)

    feat_0 = video_features[:, :1, :, :, :, :]

    # ── Get unnorm stats for this dataset ────────────────────────────────
    unnorm_stats = octo_ds_stats[dataset_name]["action"]

    # ── GT trajectory ────────────────────────────────────────────────────
    gt_actions_unnorm = unnormalize_bound_action(
        actions_raw[0].numpy(), unnorm_stats
    )
    H_octo = actions_raw.shape[1] - 1
    xy_gt, h_gt = gt_actions_to_trajectory(gt_actions_unnorm[:H_octo])

    # ── Octo inference ───────────────────────────────────────────────────
    t0 = time.time()
    obs_image = images_raw[:, :1].to(device)
    timestep_pad_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
    tasks = octo_model.create_tasks(texts=[text_instruction], device=device)
    observations = {"image_primary": obs_image, "timestep_pad_mask": timestep_pad_mask}

    N_octo_samples = mppi_cfg["n_octo_samples"]
    with torch.no_grad():
        pred_actions = octo_model.sample_actions(
            observations=observations,
            tasks=tasks,
            unnormalization_statistics=unnorm_stats,
            normalization_type=NormalizationType.BOUNDS,
            timestep_pad_mask=timestep_pad_mask,
            train=False,
            argmax=False,
            save_attention_mask=False,
            sample_shape=(N_octo_samples,),
            generator=torch.Generator(device).manual_seed(0),
        )
    pred_actions_np = pred_actions.squeeze(1).cpu().numpy()  # (N, H_full, 4)
    # ── Octo mean trajectory (simple mean of all samples) ────────────────
    octo_mean_actions = pred_actions_np[:, :H_octo].mean(axis=0)  # (H_octo, 4)
    xy_octo_mean, h_octo_mean = octo_actions_to_trajectory(octo_mean_actions)
    dt_octo_mean = time.time() - t0
    # ── Select best Octo sample via WM rollout ──────────────────────────
    all_l2_final = []
    with torch.no_grad():
        for s in range(N_octo_samples):
            wm_act_np = octo_to_wm_actions(
                pred_actions_np[s, :H_octo], unnorm_stats
            )
            wm_act_t = (
                torch.tensor(wm_act_np, dtype=torch.float32, device=device)
                .unsqueeze(0)
            )
            # Autoregressive rollout
            preds, cur = [], feat_0
            for t in range(H_octo):
                p = wm_model.forward_pred(cur, wm_act_t[:, t : t + 1])
                preds.append(p)
                cur = p
            pred_s = torch.cat(preds, dim=1)

            gt_final = video_features[:, H_octo : H_octo + 1]
            l2_final = F.mse_loss(pred_s[:, -1:], gt_final).item()
            all_l2_final.append(l2_final)

    all_l2_final = np.array(all_l2_final)
    best_idx = int(np.argmin(all_l2_final))

    # ── Octo WM best trajectory (best sample selected by WM) ────────────
    xy_octo_wm, h_octo_wm = octo_actions_to_trajectory(
        pred_actions_np[best_idx, :H_octo]
    )
    dt_octo_wm = time.time() - t0

    # ── Prepare Octo-seeded MPPI init ────────────────────────────────────
    all_octo_wm = np.stack(
        [octo_to_wm_actions(pred_actions_np[s, :H_octo], unnorm_stats)
         for s in range(N_octo_samples)],
        axis=0,
    )
    octo_mean_wm = torch.tensor(
        all_octo_wm.mean(axis=0), dtype=torch.float32
    )
    octo_std_wm = np.std(all_octo_wm, axis=0)  # (H_octo, 4)
    octo_std_wm_t = torch.tensor(
        np.clip(octo_std_wm * 2, 0.3, mppi_cfg["max_std"]),
        dtype=torch.float32,
    )

    # ── z_init and target for MPPI ───────────────────────────────────────
    z_init_flat = feat_0.reshape(1, 1, -1, feat_0.shape[-1])
    gt_target = video_features[:, H_octo : H_octo + 1, 0]
    gt_target_flat = gt_target.reshape(1, -1, gt_target.shape[-1])
    objective = L2FinalFrameObjective(
        target_enc=gt_target_flat, sum_all_diffs=False
    )

    # ── Default MPPI ─────────────────────────────────────────────────────
    t0 = time.time()
    planner_default = MPPIPlannerLocal(
        unroll_fn=cast_wm_unroll,
        objective=objective,
        horizon=H_octo,
        action_dim=cfg.action_dim,
        iterations=mppi_cfg["iterations"],
        num_samples=mppi_cfg["num_samples"],
        num_elites=mppi_cfg["num_elites"],
        temperature=mppi_cfg["temperature"],
        max_std=mppi_cfg["max_std"],
        min_std=mppi_cfg["min_std"],
        init_mean=None,
        device=device,
    )
    with torch.no_grad():
        default_actions, _, default_losses = planner_default.plan(z_init_flat)
    default_np = default_actions.cpu().numpy()
    xy_default, h_default = wm_actions_to_trajectory(default_np, unnorm_stats)
    dt_default = time.time() - t0
    # ── Octo-seeded MPPI ─────────────────────────────────────────────────
    t0 = time.time()
    planner_octo = MPPIPlannerLocal(
        unroll_fn=cast_wm_unroll,
        objective=objective,
        horizon=H_octo,
        action_dim=cfg.action_dim,
        iterations=mppi_cfg["iterations"],
        num_samples=mppi_cfg["num_samples"],
        num_elites=mppi_cfg["num_elites"],
        temperature=mppi_cfg["temperature"],
        max_std=mppi_cfg["max_std"],
        min_std=mppi_cfg["min_std"],
        init_mean=octo_mean_wm,
        init_std_scale=octo_std_wm_t,
        device=device,
    )
    with torch.no_grad():
        octo_mppi_actions, _, octo_losses = planner_octo.plan(z_init_flat)
    octo_mppi_np = octo_mppi_actions.cpu().numpy()
    xy_octo_mppi, h_octo_mppi = wm_actions_to_trajectory(
        octo_mppi_np, unnorm_stats
    )
    dt_octo = time.time() - t0 + dt_octo_mean  # include Octo mean time since it's part of the seeding

    # ── Compute metrics ──────────────────────────────────────────────────
    result = ExampleResult(
        idx=0,
        text=text_instruction,
        dataset_name=dataset_name,
        gt_endpoint_xy=xy_gt[-1].tolist(),
        metrics_default_mppi=compute_all_metrics(
            xy_default, h_default, xy_gt, h_gt
        ),
        metrics_octo_mppi=compute_all_metrics(
            xy_octo_mppi, h_octo_mppi, xy_gt, h_gt
        ),
        metrics_octo_wm=compute_all_metrics(
            xy_octo_wm, h_octo_wm, xy_gt, h_gt
        ),
        metrics_octo_mean=compute_all_metrics(
            xy_octo_mean, h_octo_mean, xy_gt, h_gt
        ),
        mppi_default_cost=default_losses[-1],
        mppi_octo_cost=octo_losses[-1],
        time_octo_wm_s=dt_octo_wm,
        time_octo_mean_s=dt_octo_mean,
        time_default_s=dt_default,
        time_octo_s=dt_octo,
        traj_gt_xy=xy_gt,
        traj_gt_heading=h_gt,
        traj_default_mppi_xy=xy_default,
        traj_default_mppi_heading=h_default,
        traj_octo_mppi_xy=xy_octo_mppi,
        traj_octo_mppi_heading=h_octo_mppi,
        traj_octo_wm_xy=xy_octo_wm,
        traj_octo_wm_heading=h_octo_wm,
        traj_octo_mean_xy=xy_octo_mean,
        traj_octo_mean_heading=h_octo_mean,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_metrics(results: List[ExampleResult]) -> Dict:
    """Aggregate per-example metrics into dataset-level summary."""

    def _agg(key, method_attr):
        vals = [getattr(r, method_attr)[key] for r in results if key in getattr(r, method_attr)]
        if not vals:
            return {}
        arr = np.array(vals)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "median": float(np.median(arr))}

    summary_keys = [
        "ate_xy_rmse", "ate_xy_mean", "ate_xy_final",
        "ate_heading_rmse_deg", "ate_heading_mean_deg", "ate_heading_final_deg",
        "rpe_xy_rmse", "rpe_xy_mean",
        "rpe_heading_rmse_deg", "rpe_heading_mean_deg",
    ]

    agg = {}
    for method in ["metrics_default_mppi", "metrics_octo_mppi", "metrics_octo_wm", "metrics_octo_mean"]:
        method_name = method.replace("metrics_", "")
        agg[method_name] = {}
        for k in summary_keys:
            agg[method_name][k] = _agg(k, method)

    # Timing
    agg["timing"] = {
        "default_mppi_mean_s": float(np.mean([r.time_default_s for r in results])),
        "octo_mppi_mean_s": float(np.mean([r.time_octo_s for r in results])),
        "octo_wm_mean_s": float(np.mean([r.time_octo_wm_s for r in results])),
        "octo_mean_mean_s": float(np.mean([r.time_octo_mean_s for r in results])),
    }

    return agg


def print_summary_table(agg: Dict, n_examples: int):
    """Pretty-print the aggregated metrics."""

    print(f"\n{'=' * 90}")
    print(f"  OCTO + MPPI EVALUATION ON CAST VALIDATION  ({n_examples} examples)")
    print(f"{'=' * 90}")

    header = f"  {'Metric':<35s}  {'Default MPPI':>15s}  {'Octo-seed MPPI':>15s}  {'Octo WM':>15s}  {'Octo mean':>15s}"
    print(header)
    print("  " + "─" * 104)

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
        vals = []
        for method in ["default_mppi", "octo_mppi", "octo_wm", "octo_mean"]:
            d = agg[method].get(key, {})
            if d:
                vals.append(f"{d['mean']:.4f}±{d['std']:.4f}")
            else:
                vals.append("N/A")
        print(f"  {label:<35s}  {vals[0]:>15s}  {vals[1]:>15s}  {vals[2]:>15s}  {vals[3]:>15s}")

    print()
    print(f"  {'Mean planning time (s)':<35s}  "
          f"{agg['timing']['default_mppi_mean_s']:>15.2f}  "
          f"{agg['timing']['octo_mppi_mean_s']:>15.2f}  "
          f"{agg['timing']['octo_wm_mean_s']:>15.2f}  "
          f"{agg['timing']['octo_mean_mean_s']:>15.2f}")
    print(f"{'=' * 108}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Octo+MPPI on CAST val")
    parser.add_argument("--num_examples", type=int, default=50,
                        help="Number of validation examples to evaluate")
    parser.add_argument("--encoder_type", type=str, default="dino",
                        choices=["dino", "vjepa"])
    parser.add_argument("--wm_checkpoint", type=str,
                        default="/mnt/weka/zhougrp/octo_wm_cast/dino_cast_wm_20260228_193050/best_model.pt")
    parser.add_argument("--octo_ckpt_dir", type=str,
                        default="/mnt/weka/zhougrp/octo_pt_cast/cast_dino_finetune_20260219_202345")
    parser.add_argument("--octo_ckpt_step", type=int, default=2000)
    parser.add_argument("--mppi_iterations", type=int, default=4)
    parser.add_argument("--mppi_samples", type=int, default=256)
    parser.add_argument("--mppi_elites", type=int, default=16)
    parser.add_argument("--mppi_temperature", type=float, default=0.5)
    parser.add_argument("--mppi_max_std", type=float, default=2.0)
    parser.add_argument("--mppi_min_std", type=float, default=0.05)
    parser.add_argument("--n_octo_samples", type=int, default=10)
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save detailed results JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load WM config ───────────────────────────────────────────────────
    config_path = os.path.join(REPO_ROOT, "scripts/configs/train_cast_wm_config.py")
    spec = importlib.util.spec_from_file_location("cfg_module", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config(encoder_type=args.encoder_type)
    print(f"Config: encoder={cfg.encoder_type}, img_size={cfg.img_size}, "
          f"grid={cfg.grid_size}, embed_dim={cfg.embed_dim}")

    # ── Build and load WM ────────────────────────────────────────────────
    print("Building world model...")
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)
    wm_model = CastWorldModel(encoder, predictor, cfg).to(device)
    for p in wm_model.encoder.parameters():
        p.requires_grad = False
    wm_model.encoder.eval()

    print(f"Loading WM checkpoint: {args.wm_checkpoint}")
    ckpt = torch.load(args.wm_checkpoint, map_location=device, weights_only=False)
    wm_model.load_state_dict(ckpt["model_state_dict"])
    wm_model.eval()
    print(f"  WM loaded (step={ckpt.get('step', 'N/A')}, "
          f"val_loss={ckpt.get('val_loss', 'N/A')})")

    # ── Load Octo ────────────────────────────────────────────────────────
    print(f"Loading Octo from: {args.octo_ckpt_dir} (step={args.octo_ckpt_step})")
    octo_load = OctoModelPt.load_pretrained(
        args.octo_ckpt_dir, step=args.octo_ckpt_step
    )
    octo_model = octo_load["octo_model"].to(device)
    octo_model.eval()
    octo_ds_stats = octo_model.dataset_statistics

    # ── Build validation dataset ─────────────────────────────────────────
    print("Building CAST validation dataset...")
    text_processor = HFTokenizer(
        tokenizer_name="t5-base",
        encode_with_model=False,
        tokenizer_kwargs={
            "max_length": 16,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "np",
        },
    )
    val_tf_ds = build_cast_val_dataset(cfg)
    val_pt = TorchRLDSDataset(val_tf_ds, text_processor, train=False)
    val_loader = DataLoader(
        val_pt,
        batch_size=1,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )
    print("Validation DataLoader ready.")

    # ── Create WM unroll function ────────────────────────────────────────
    cast_wm_unroll = make_cast_wm_unroll(wm_model, cfg)

    mppi_cfg = {
        "iterations": args.mppi_iterations,
        "num_samples": args.mppi_samples,
        "num_elites": args.mppi_elites,
        "temperature": args.mppi_temperature,
        "max_std": args.mppi_max_std,
        "min_std": args.mppi_min_std,
        "n_octo_samples": args.n_octo_samples,
    }

    # ── Evaluate ─────────────────────────────────────────────────────────
    print(f"\nEvaluating {args.num_examples} examples...")
    print(f"  MPPI: iters={mppi_cfg['iterations']}, samples={mppi_cfg['num_samples']}, "
          f"elites={mppi_cfg['num_elites']}, temp={mppi_cfg['temperature']}")

    results: List[ExampleResult] = []
    val_iter = iter(val_loader)

    for i in range(args.num_examples):
        batch = None
        while batch is None:
            try:
                candidate = next(val_iter)
            except StopIteration:
                print(f"  Dataset exhausted after {i} examples.")
                break
            # Skip if the last frame image is all 255 (padding)
            last_frame = candidate["observation"]["image_primary"][:, -1]
            if (last_frame == 255).all():
                continue
            batch = candidate

        if batch is None:
            break

        t0_ex = time.time()
        result = evaluate_one_example(
            batch=batch,
            wm_model=wm_model,
            octo_model=octo_model,
            octo_ds_stats=octo_ds_stats,
            cfg=cfg,
            device=device,
            cast_wm_unroll=cast_wm_unroll,
            mppi_cfg=mppi_cfg,
        )
        result.idx = i
        results.append(result)
        dt_ex = time.time() - t0_ex

        # Per-example summary
        m_d = result.metrics_default_mppi
        m_o = result.metrics_octo_mppi
        m_w = result.metrics_octo_wm
        m_m = result.metrics_octo_mean
        print(
            f"  [{i+1:3d}/{args.num_examples}] "
            f"ATE_xy(D/O/W/M): {m_d['ate_xy_final']:.3f}/{m_o['ate_xy_final']:.3f}/{m_w['ate_xy_final']:.3f}/{m_m['ate_xy_final']:.3f}m  "
            f"ATE_h(D/O/W/M): {m_d['ate_heading_final_deg']:.1f}/{m_o['ate_heading_final_deg']:.1f}/{m_w['ate_heading_final_deg']:.1f}/{m_m['ate_heading_final_deg']:.1f}°  "
            f"[{dt_ex:.1f}s]"
        )

    if not results:
        print("No examples evaluated!")
        return

    # ── Aggregate and print ──────────────────────────────────────────────
    agg = aggregate_metrics(results)
    print_summary_table(agg, len(results))

    # ── Save detailed JSON ───────────────────────────────────────────────
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
            },
            "aggregate": agg,
            "per_example": [
                {
                    "idx": r.idx,
                    "text": r.text,
                    "dataset_name": r.dataset_name,
                    "mppi_default_cost": r.mppi_default_cost,
                    "mppi_octo_cost": r.mppi_octo_cost,
                    "time_default_mppi_s": r.time_default_s,
                    "time_octo_mppi_s": r.time_octo_s,
                    "time_octo_wm_s": r.time_octo_wm_s,
                    "time_octo_mean_s": r.time_octo_mean_s,
                    "gt": {
                        "xy": r.traj_gt_xy.tolist(),
                        "heading": r.traj_gt_heading.tolist(),
                    },
                    "default_mppi": {
                        "xy": r.traj_default_mppi_xy.tolist(),
                        "heading": r.traj_default_mppi_heading.tolist(),
                    },
                    "octo_mppi": {
                        "xy": r.traj_octo_mppi_xy.tolist(),
                        "heading": r.traj_octo_mppi_heading.tolist(),
                    },
                    "octo_wm": {
                        "xy": r.traj_octo_wm_xy.tolist(),
                        "heading": r.traj_octo_wm_heading.tolist(),
                    },
                    "octo_mean": {
                        "xy": r.traj_octo_mean_xy.tolist(),
                        "heading": r.traj_octo_mean_heading.tolist(),
                    },
                }
                for r in results
            ],
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nDetailed results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
