#!/usr/bin/env python3
"""
Run eval_octo_wm_plan.py and save per-example data (images, trajectories,
metrics, text) to a directory for later offline plotting.

Saves one .npz per example plus a summary JSON.

Usage:
    python save_eval_samples.py \
        --num_examples 20 \
        --encoder_type dino \
        --save_dir ./eval_samples/dino_run1 \
        [--wm_checkpoint ...] [--octo_ckpt_dir ...] [other eval args]

Each example directory contains:
    sample_{idx}.npz   with keys:
        images         (T, C, H, W) uint8  – all observation frames
        goal_image     (C, H, W) uint8     – last frame (goal)
        text           str                  – language instruction
        dataset_name   str
        gt_xy, gt_heading
        default_mppi_xy, default_mppi_heading
        octo_mppi_xy,    octo_mppi_heading
        octo_wm_xy,      octo_wm_heading
        octo_mean_xy,    octo_mean_heading
        metrics_*        dicts (via JSON string)
    summary.json – aggregate metrics + config
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import logging, warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# ── Import everything from the eval script ──────────────────────────────────
# Adjust this path to point to your eval script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from eval_octo_wm_plan import (
    build_encoder, build_predictor, build_cast_val_dataset,
    CastWorldModel, MPPIPlannerLocal, L2FinalFrameObjective, L1FinalFrameObjective,
    make_cast_wm_unroll, custom_collate_fn,
    normalize_bound_action, unnormalize_bound_action,
    octo_to_wm_actions, wm_actions_to_trajectory,
    gt_actions_to_trajectory, octo_actions_to_trajectory,
    compute_all_metrics, aggregate_metrics, print_summary_table,
    ExampleResult, SUMMARY_KEYS, METHOD_ATTRS,
)
from octo.data.dataset import make_interleaved_dataset
from octo.data.utils.data_utils import NormalizationType
from octo.data.utils.text_processing import HFTokenizer
from octo.model.octo_model_pt import OctoModelPt
from octo.utils.torch_rlds_dataset import TorchRLDSDataset

# Insert jepa_wms to path for importing WM decoder head
JEPA_ROOT = os.path.join(REPO_ROOT, "jepa_wms")
if JEPA_ROOT not in sys.path:
    sys.path.insert(0, JEPA_ROOT)
from app.plan_common.models.wm_heads import WorldModelViTImageHead
from app.plan_common.datasets.transforms import make_inverse_transforms

def images_to_numpy(images_tensor):
    """Convert (B, T, C, H, W) float tensor [0,1] or [0,255] to (T, H, W, C) uint8 numpy."""
    imgs = images_tensor[0].cpu()  # (T, C, H, W)
    if imgs.dtype == torch.float32:
        if imgs.max() <= 1.0:
            imgs = (imgs * 255).clamp(0, 255)
        imgs = imgs.to(torch.uint8)
    # (T, C, H, W) -> (T, H, W, C)
    imgs = imgs.permute(0, 2, 3, 1).numpy()
    return imgs


def evaluate_and_save_one(
    batch, wm_model, wm_decoder, octo_model, octo_ds_stats, cfg, device,
    cast_wm_unroll, mppi_cfg,
):
    """Same as evaluate_one_example but also returns raw images."""
    H_octo = 7
    images_raw = batch["observation"]["image_primary"][:, :H_octo + 1]
    actions_raw = batch["action"]
    if actions_raw.dim() == 4:
        actions_raw = actions_raw[:, :H_octo, 0, :]

    text_instruction = batch["raw_language"][0]
    dataset_name = batch["dataset_name"][0]
    images = images_raw.to(device)

    # ── Encode all frames
    with torch.no_grad():
        video_features = wm_model.encode_obs(images)
    feat_0 = video_features[:, :1, :, :, :, :]
    unnorm_stats = octo_ds_stats[dataset_name]["action"]

    # ── GT trajectory
    gt_actions_unnorm = unnormalize_bound_action(actions_raw[0].numpy(), unnorm_stats)
    xy_gt, h_gt = gt_actions_to_trajectory(gt_actions_unnorm[:H_octo])

    # ── Octo inference
    obs_image = images_raw[:, :1].to(device)
    timestep_pad_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
    tasks = octo_model.create_tasks(texts=[text_instruction], device=device)
    observations = {"image_primary": obs_image, "timestep_pad_mask": timestep_pad_mask}

    N_octo_samples = mppi_cfg["n_octo_samples"]
    t_octo_mean_start = time.time()
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
    pred_actions_np = pred_actions.squeeze(1).cpu().numpy()

    # ── Octo mean (random sample as in updated code)
    random_idx = random.randint(0, N_octo_samples - 1)
    octo_mean_actions = pred_actions_np[random_idx, :H_octo]
    xy_octo_mean, h_octo_mean = octo_actions_to_trajectory(octo_mean_actions)
    time_octo_mean = time.time() - t_octo_mean_start

    # ── Octo-WM (best sample via WM rollout)
    t_octo_wm_start = time.time()
    all_l2_final = []
    with torch.no_grad():
        for s in range(N_octo_samples):
            wm_act_np = octo_to_wm_actions(pred_actions_np[s, :H_octo], unnorm_stats)
            wm_act_t = torch.tensor(wm_act_np, dtype=torch.float32, device=device).unsqueeze(0)
            cur = feat_0
            for t in range(H_octo):
                cur = wm_model.forward_pred(cur, wm_act_t[:, t:t+1])
            gt_final = video_features[:, H_octo:H_octo+1]
            all_l2_final.append(F.mse_loss(cur, gt_final).item())
    best_idx = int(np.argmin(all_l2_final))
    xy_octo_wm, h_octo_wm = octo_actions_to_trajectory(pred_actions_np[best_idx, :H_octo])
    time_octo_wm = time.time() - t_octo_wm_start
    
    # Decode images if decoder is available
    if wm_decoder is not None:
        all_decoded_image = []
        with torch.no_grad():
            wm_act_np = octo_to_wm_actions(pred_actions_np[best_idx, :H_octo], unnorm_stats)
            wm_act_t = torch.tensor(wm_act_np, dtype=torch.float32, device=device).unsqueeze(0)
            cur = feat_0
            for t in range(H_octo):
                cur = wm_model.forward_pred(cur, wm_act_t[:, t:t+1])
                decoded_image = wm_decoder.decode(cur).cpu().numpy()
                all_decoded_image.append(decoded_image)
        decoded_images_np = np.concatenate(all_decoded_image, axis=0)  # (H_octo, C, H, W)
    else:
        decoded_images_np = None


    # ── MPPI init from Octo
    all_octo_wm = np.stack(
        [octo_to_wm_actions(pred_actions_np[s, :H_octo], unnorm_stats)
         for s in range(N_octo_samples)], axis=0,
    )
    octo_mean_wm = torch.tensor(all_octo_wm.mean(axis=0), dtype=torch.float32)
    octo_std_wm_t = torch.tensor(np.std(all_octo_wm, axis=0), dtype=torch.float32)

    z_init_flat = feat_0.reshape(1, 1, -1, feat_0.shape[-1])
    gt_target = video_features[:, H_octo:H_octo+1, 0]
    gt_target_flat = gt_target.reshape(1, -1, gt_target.shape[-1])
    objective = L2FinalFrameObjective(target_enc=gt_target_flat, sum_all_diffs=False)

    # ── Default MPPI
    t_default_start = time.time()
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
    time_default = time.time() - t_default_start

    # ── Octo-seeded MPPI
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
    time_octo_mppi = time.time() - t_octo_mppi_start

    # ── Build result
    result = ExampleResult(
        text=text_instruction, dataset_name=dataset_name,
        metrics_default_mppi=compute_all_metrics(xy_default, h_default, xy_gt, h_gt),
        metrics_octo_mppi=compute_all_metrics(xy_octo_mppi, h_octo_mppi, xy_gt, h_gt),
        metrics_octo_wm=compute_all_metrics(xy_octo_wm, h_octo_wm, xy_gt, h_gt),
        metrics_octo_mean=compute_all_metrics(xy_octo_mean, h_octo_mean, xy_gt, h_gt),
        time_default_mppi=time_default, time_octo_mppi=time_octo_mppi,
        time_octo_wm=time_octo_wm, time_octo_mean=time_octo_mean,
        traj_gt_xy=xy_gt, traj_gt_heading=h_gt,
        traj_default_mppi_xy=xy_default, traj_default_mppi_heading=h_default,
        traj_octo_mppi_xy=xy_octo_mppi, traj_octo_mppi_heading=h_octo_mppi,
        traj_octo_wm_xy=xy_octo_wm, traj_octo_wm_heading=h_octo_wm,
        traj_octo_mean_xy=xy_octo_mean, traj_octo_mean_heading=h_octo_mean,
    )

    # ── Extract images as numpy
    images_np = images_to_numpy(images_raw)  # (T, H, W, C) uint8

    return result, images_np, decoded_images_np  # (H_octo, C, H, W) float32


def save_sample(save_dir, idx, result, images_np, decoded_images_np):
    """Save one example's data to an .npz file."""
    save_dict = {
        # Images
        "images": images_np,                          # (T, H, W, C) uint8
        "start_image": images_np[0],                  # (H, W, C)
        "goal_image": images_np[-1],                  # (H, W, C)
        # Metadata
        "text": str(result.text),
        "dataset_name": str(result.dataset_name),
        # Trajectories – GT
        "gt_xy": result.traj_gt_xy,
        "gt_heading": result.traj_gt_heading,
        # Trajectories – Default MPPI
        "default_mppi_xy": result.traj_default_mppi_xy,
        "default_mppi_heading": result.traj_default_mppi_heading,
        # Trajectories – Octo-MPPI
        "octo_mppi_xy": result.traj_octo_mppi_xy,
        "octo_mppi_heading": result.traj_octo_mppi_heading,
        # Trajectories – Octo-WM
        "octo_wm_xy": result.traj_octo_wm_xy,
        "octo_wm_heading": result.traj_octo_wm_heading,
        # Trajectories – Octo-Mean
        "octo_mean_xy": result.traj_octo_mean_xy,
        "octo_mean_heading": result.traj_octo_mean_heading,
        # Metrics as JSON strings
        "metrics_default_mppi": json.dumps(result.metrics_default_mppi),
        "metrics_octo_mppi": json.dumps(result.metrics_octo_mppi),
        "metrics_octo_wm": json.dumps(result.metrics_octo_wm),
        "metrics_octo_mean": json.dumps(result.metrics_octo_mean),
    }
    if decoded_images_np is not None:
        save_dict["decoded_images"] = decoded_images_np  # (H_octo, C, H, W) float32
    
    np.savez_compressed(
        os.path.join(save_dir, f"sample_{idx:04d}.npz"),
        **save_dict
    )


def main():
    parser = argparse.ArgumentParser(description="Eval + save samples for plotting")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for saved samples (default: 0)")
    parser.add_argument("--encoder_type", type=str, default="dino", choices=["dino", "vjepa"])
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save .npz samples and summary.json")
    parser.add_argument("--wm_checkpoint", type=str,
                        default="/mnt/weka/zhougrp/octo_wm_cast/dino_cast_wm_20260228_193050/best_model.pt")
    parser.add_argument("--octo_ckpt_dir", type=str,
                        default="/mnt/weka/zhougrp/octo_pt_cast/cast_dino_finetune_20260219_202345")
    parser.add_argument("--octo_ckpt_step", type=int, default=2000)
    parser.add_argument("--mppi_iterations", type=int, default=4)
    parser.add_argument("--mppi_samples", type=int, default=32)
    parser.add_argument("--mppi_elites", type=int, default=4)
    parser.add_argument("--mppi_temperature", type=float, default=0.8)
    parser.add_argument("--mppi_max_std", type=float, default=0.05)
    parser.add_argument("--mppi_min_std", type=float, default=0.01)
    parser.add_argument("--n_octo_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_decoder", action="store_true", default=False,
                        help="Load WM decoder for visualization (may be incompatible for some encoders)")
    parser.add_argument("--planner_types", type=str, nargs="+",
                        default=["default_mppi", "octo_mppi", "octo_wm", "octo_mean"],
                        choices=["default_mppi", "octo_mppi", "octo_wm", "octo_mean"],
                        help="List of planner types to evaluate")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

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
    print(f"Device: {device} | Seed: {args.seed} | Saving to: {args.save_dir}")

    # ── Load config
    config_path = os.path.join(REPO_ROOT, "scripts/configs/train_cast_wm_config.py")
    spec = importlib.util.spec_from_file_location("cfg_module", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config(encoder_type=args.encoder_type)

    # ── Build models
    print("Building world model...")
    predictor = build_predictor(cfg)
    encoder = build_encoder(cfg)
    wm_model = CastWorldModel(encoder, predictor, cfg).to(device)
    for p in wm_model.encoder.parameters():
        p.requires_grad = False
    wm_model.encoder.eval()
    ckpt = torch.load(args.wm_checkpoint, map_location=device, weights_only=False)
    wm_model.load_state_dict(ckpt["model_state_dict"])
    wm_model.eval()
    print(f"WM loaded (step={ckpt.get('step', 'N/A')})")

    heads_architectures = {
        "vjepa" :{
        "image_head": {
            "config": {
                "patch_size": 8,
                "in_chans": 3,
                "img_size": [256, 256],
                "embed_dim": 1408,
                "decoder_embed_dim": 1024,
                "depth": 12,
                "num_heads": 16,
                "mlp_ratio": 4.0,
                "num_views": 1,
                "use_activation_checkpointing": False,
                "use_lpips": True,
                "pixelloss_weight": 10,
                "perceptual_weight": 1
            },
            "kind": "vit"
        },
        "head_source": "https://dl.fbaipublicfiles.com/jepa-wms/vm2m_lpips_vj2vitgnorm_vitldec_dup_256_INet.pth.tar"
        },
        "dino": {
            "image_head": {
            "config": {
                "patch_size": 8,
                "in_chans": 3,
                "img_size": [224, 224],
                "embed_dim": 384,
                "decoder_embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
                "mlp_ratio": 4.0,
                "num_views": 1,
                "use_activation_checkpointing": False
            },
            "kind": "vit"
        },
        "head_source": "https://dl.fbaipublicfiles.com/jepa-wms/vm2m_lpips_dv2vits_vitldec_224_05norm.pth.tar"
        }
        }

    wm_decoder = None
    if args.load_decoder:
        print("Loading WM decoder...")
        inverse_transform = make_inverse_transforms(
                img_size=cfg.img_size
            )
        wm_decoder = WorldModelViTImageHead(
                                head_config=dict(heads_architectures[args.encoder_type]["image_head"]["config"]),
                                inverse_transform=inverse_transform,
                                device=device,
                            )
        wm_decoder.load_checkpoint(heads_architectures[args.encoder_type]["head_source"])
        wm_decoder.eval()
    else:
        print("Skipping WM decoder (--load_decoder not set)")
    print("Loading Octo...")
    octo_load = OctoModelPt.load_pretrained(args.octo_ckpt_dir, step=args.octo_ckpt_step)
    octo_model = octo_load["octo_model"].to(device)
    octo_model.eval()
    octo_ds_stats = octo_model.dataset_statistics

    # ── Dataset
    print("Building CAST val dataset...")
    text_processor = HFTokenizer(
        tokenizer_name="t5-base", encode_with_model=False,
        tokenizer_kwargs={"max_length": 16, "padding": "max_length",
                          "truncation": True, "return_tensors": "np"},
    )
    val_tf_ds = build_cast_val_dataset(cfg)
    val_pt = TorchRLDSDataset(val_tf_ds, text_processor, train=False)
    val_loader = DataLoader(val_pt, batch_size=1, num_workers=0,
                            collate_fn=custom_collate_fn, pin_memory=True)

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

    # ── Evaluate and save
    results: List[ExampleResult] = []
    val_iter = iter(val_loader)
    
    # Skip first start_idx items
    print(f"Skipping first {args.start_idx} items...")
    for _ in range(args.start_idx):
        try:
            next(val_iter)
        except StopIteration:
            print(f"Dataset exhausted before reaching start_idx={args.start_idx}")
            return

    pbar = tqdm(range(args.num_examples), desc="Eval+Save")
    for i in pbar:
        batch = None
        while batch is None:
            try:
                candidate = next(val_iter)
            except StopIteration:
                pbar.write(f"Dataset exhausted after {i} examples.")
                batch = None
                break
            if (candidate["observation"]["image_primary"][:, -1] == 255).all():
                continue
            batch = candidate
        if batch is None:
            break

        result, images_np, decoded_images_np = evaluate_and_save_one(
            batch, wm_model, wm_decoder, octo_model, octo_ds_stats, cfg, device,
            cast_wm_unroll, mppi_cfg,
        )
        save_idx = args.start_idx + i
        result.idx = save_idx
        results.append(result)
        save_sample(args.save_dir, save_idx, result, images_np, decoded_images_np)

        avg_ate_o = np.mean([r.metrics_octo_mppi["ate_xy_final"] for r in results])
        pbar.set_postfix_str(f"ATE_xy Octo-MPPI:{avg_ate_o:.3f}m")

    # ── Summary JSON
    if results:
        time_per_method = {
            "default_mppi": sum(r.time_default_mppi for r in results),
            "octo_mppi": sum(r.time_octo_mppi for r in results),
            "octo_wm": sum(r.time_octo_wm for r in results),
            "octo_mean": sum(r.time_octo_mean for r in results),
        }
        agg = aggregate_metrics(results, planner_types=args.planner_types)
        print_summary_table(agg, len(results), time_per_method, planner_types=args.planner_types)

        summary = {
            "config": {
                "encoder_type": args.encoder_type,
                "wm_checkpoint": args.wm_checkpoint,
                "octo_ckpt_dir": args.octo_ckpt_dir,
                "octo_ckpt_step": args.octo_ckpt_step,
                "mppi": mppi_cfg,
                "num_examples": len(results),
                "seed": args.seed,
            },
            "time_per_method_s": time_per_method,
            "aggregate": agg,
            "sample_files": [f"sample_{args.start_idx + i:04d}.npz" for i in range(len(results))],
        }
        with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved {len(results)} samples to {args.save_dir}/")


if __name__ == "__main__":
    main()
