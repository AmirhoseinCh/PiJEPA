#!/usr/bin/env python3
"""
Script to finetune Octo-small with DinoV2 encoder on CAST dataset.

This script:
1. Loads Octo-small pretrained weights
2. Replaces vision encoder with DinoV2
3. Skips loading projection layer (will be trained from random init)
4. Finetunes on CAST navigation dataset

Usage:
    # Single GPU debug mode (small dataset, fewer steps)
    torchrun --nproc_per_node 1 scripts/finetune_cast_dino.py  --debug

    # Multi-GPU (recommended)
    torchrun --nproc_per_node 4 scripts/finetune_cast_dino.py --config scripts/configs/finetune_cast_dino_config.py
"""

import datetime
import io
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from absl import app, flags, logging
import tqdm
import wandb
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hide TF warnings

from accelerate import PartialState

from octo.data.utils.data_utils import NormalizationType
from octo.model.octo_model_pt import OctoModelPt
from octo.utils.spec import ModuleSpec
from octo.data.dataset import make_interleaved_dataset
from octo.utils.torch_rlds_dataset import TorchRLDSDataset
from octo.utils.train_utils_pt import (
    _np2pt,
    freeze_weights_pt,
    get_cosine_schedule_with_warmup,
    tree_map,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "scripts/configs/finetune_cast_dino_config.py", "Path to config file")
flags.DEFINE_string("pretrained", "hf://rail-berkeley/octo-small-1.5", "Pretrained checkpoint path")
flags.DEFINE_string("name", "cast_dino_finetune", "Experiment name for logging")
flags.DEFINE_bool("debug", False, "Debug mode with small dataset")
flags.DEFINE_string("save_dir", "/mnt/weka/zhougrp/octo_pt_cast", "Directory to save checkpoints")
flags.DEFINE_integer("log_interval", 10, "Log training loss every N steps")
flags.DEFINE_integer("val_steps", 20, "Number of batches to compute validation loss")
flags.DEFINE_float("grad_clip", 1.0, "Max gradient norm for clipping")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def custom_collate_fn(batch):
    """Custom collate function to handle nested dictionaries and tensors."""
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
    else:
        return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(_):
    # ── Load config ───────────────────────────────────────────────────────
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", FLAGS.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.get_config()

    # ── Distributed setup ─────────────────────────────────────────────────
    assert torch.cuda.is_available()
    distributed_state = PartialState()

    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{device_id}")
    num_devices = distributed_state.num_processes

    if distributed_state.is_main_process:
        logging.set_verbosity(logging.INFO)
    else:
        logging.set_verbosity(logging.ERROR)

    if FLAGS.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("🐛 Debug mode: using smaller settings")
        config.num_steps = 4
        config.batch_size = 4
        config.dataset_kwargs.batch_size = 4
        config.dataset_kwargs.shuffle_buffer_size = 100
        config.eval_interval = 3

    # ── Wandb ─────────────────────────────────────────────────────────────
    if distributed_state.is_main_process:
        wandb.init(
            project="octo-cast-finetune",
            name=FLAGS.name,
            config=config.to_dict(),
        )

    logging.info("=" * 80)
    logging.info("FINETUNING OCTO-SMALL WITH DINOV2 ON CAST DATASET")
    logging.info("=" * 80)

    # ── Step 1: Load pretrained config/meta ────────────────────────────────
    logging.info(f"\n📥 Loading config from {FLAGS.pretrained}")
    meta = OctoModelPt.load_config_and_meta_from_jax(FLAGS.pretrained)

    # ── Step 2: Override with DinoV2 / CAST configuration ─────────────────
    logging.info("\n🔧 Configuring DinoV2 encoder...")
    meta['config']['model']['observation_tokenizers'] = config.model.observation_tokenizers.to_dict()
    meta['config']['model']['task_tokenizers'] = config.model.task_tokenizers.to_dict()
    meta['config']['model']['heads'] = config.model.heads.to_dict()
    meta['config']['model']['readouts'] = config.model.readouts.to_dict()

    # num_tokens_dict — only include tokenizers that actually exist
    meta['config']['model']['num_tokens_dict'] = {
        'primary': 256,   # 16×16 patches from DinoV2
        'language': 16,   # From T5 tokenizer
        'action': 1,      # Readout token
    }

    action_horizon = config.model.heads.action.kwargs.action_horizon
    action_dim = config.model.heads.action.kwargs.action_dim

    # Log for debugging
    logging.debug(f"   Action horizon: {action_horizon}")
    logging.debug(f"   Action dim: {action_dim}")
    logging.debug(f"   Window size: {config.window_size}")

    # ── Step 3: Load CAST dataset ─────────────────────────────────────────
    logging.info("\n📊 Loading CAST dataset...")

    dataset_kwargs_list = [
        config.dataset_kwargs.dataset_kwargs_list[k]
        for k in config.dataset_kwargs.dataset_kwargs_list
    ]

    train_dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        config.dataset_kwargs.sample_weights,
        train=True,
        shuffle_buffer_size=config.dataset_kwargs.shuffle_buffer_size,
        frame_transform_kwargs=config.dataset_kwargs.frame_transform_kwargs,
        traj_transform_kwargs=config.dataset_kwargs.traj_transform_kwargs,
        batch_size=None,
        balance_weights=config.dataset_kwargs.balance_weights,
        traj_transform_threads=config.dataset_kwargs.traj_transform_threads,
        traj_read_threads=config.dataset_kwargs.traj_read_threads,
    )
    val_dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        config.dataset_kwargs.sample_weights,
        train=False,
        shuffle_buffer_size=config.dataset_kwargs.shuffle_buffer_size,
        frame_transform_kwargs=config.dataset_kwargs.frame_transform_kwargs,
        traj_transform_kwargs=config.dataset_kwargs.traj_transform_kwargs,
        batch_size=None,
        balance_weights=config.dataset_kwargs.balance_weights,
        traj_transform_threads=config.dataset_kwargs.traj_transform_threads,
        traj_read_threads=config.dataset_kwargs.traj_read_threads,
    )

    # Use TRAIN statistics everywhere (model normalizes with these)
    train_statistics = train_dataset.dataset_statistics
    meta['dataset_statistics'] = train_statistics

    # Apply data filters
    train_dataset = train_dataset.filter(
        lambda x: tf.reduce_any(x["observation"]["image_primary"] != 255)
    )
    train_dataset = train_dataset.filter(
        lambda x: tf.reduce_all(x["action"][:, 3:, :] != 0.0)
    )
    val_dataset = val_dataset.filter(
        lambda x: tf.reduce_any(x["observation"]["image_primary"] != 255)
    )
    val_dataset = val_dataset.filter(
        lambda x: tf.reduce_all(x["action"][:, 3:, :] != 0.0)
    )

    # Prefetch for throughput
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    logging.info("   ✓ Dataset loaded successfully")

    pytorch_dataset = TorchRLDSDataset(train_dataset, meta["text_processor"])
    dataloader = DataLoader(
        pytorch_dataset,
        batch_size=config.dataset_kwargs.batch_size,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    val_pytorch_dataset = TorchRLDSDataset(val_dataset, meta["text_processor"], train=False)
    val_dataloader = DataLoader(
        val_pytorch_dataset,
        batch_size=config.dataset_kwargs.batch_size,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    # Grab a sample for model construction WITHOUT consuming training data
    _peek_ds = TorchRLDSDataset(train_dataset, meta["text_processor"])
    _peek_loader = DataLoader(_peek_ds, batch_size=1,
                              num_workers=0, collate_fn=custom_collate_fn)
    sample_batch = next(iter(_peek_loader))
    del _peek_ds, _peek_loader

    logging.debug(f"   Sample batch keys: {list(sample_batch.keys())}")
    logging.debug(f"   Image shape: {sample_batch['observation']['image_primary'].shape}")
    logging.debug(f"   Action shape: {sample_batch['action'].shape}")
    logging.debug(f"   Language attention_mask shape: "
                 f"{sample_batch['task']['language_instruction']['attention_mask'].shape}")
    logging.debug(f"   Dataset name: {sample_batch['dataset_name']}")
    logging.debug(f"   Raw language instruction: {sample_batch['raw_language']}")

    meta['example_batch'] = sample_batch

    # ── Step 4: Create model ──────────────────────────────────────────────
    logging.info("\n🏗️  Creating model...")
    model = OctoModelPt.from_config(**meta)

    # ── Step 5: Load pretrained weights ───────────────────────────────────
    logging.debug(f"\n⚡ Loading pretrained weights from {FLAGS.pretrained}")
    logging.debug("   Skipping: DinoV2 encoder (has own pretrained weights)")
    logging.debug("   Skipping: obs_primary_projection (train from random init)")
    logging.debug("   Skipping: heads.action (different action dim)")

    missing_keys, skipped_keys = model.load_weights_from_jax(
        FLAGS.pretrained,
        skip_keys_regex=(
            r'.*observation_tokenizers\.primary\.encoder.*'
            r'|.*obs_primary_projection.*'
            r'|.*heads\.action.*'
            r'|.*observation_tokenizers\.proprio.*'
        ),
    )
    logging.debug(f"   ✓ Loaded weights  (missing={len(missing_keys)}, skipped={len(skipped_keys)})")

    # ── Step 6: Move to device + DDP ─────────────────────────────────────
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id],
                find_unused_parameters=True,
                gradient_as_bucket_view=True)

    # ── Step 7: Freeze encoders ──────────────────────────────────────────
    logging.debug("\n❄️  Freezing DinoV2 and T5 text encoder weights...")
    frozen_keys = list(config.optimizer.frozen_keys)
    freeze_weights_pt(model, frozen_keys)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.debug(f"   Trainable: {trainable_params:,} / {total_params:,} "
                 f"({100 * trainable_params / total_params:.1f}%)")

    # ── Step 8: Optimizer + scheduler ────────────────────────────────────
    logging.debug("\n⚙️  Setting up optimizer and LR scheduler...")
    lr_cfg = config.optimizer.learning_rate
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_param_list, lr=lr_cfg.peak_value, weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=lr_cfg.warmup_steps,
        num_training_steps=int(config.num_steps),
    )

    # ── Mixed-precision scaler ───────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # ── Step 9: Save directory ───────────────────────────────────────────
    save_dir = None  # safe default for all ranks
    if distributed_state.is_main_process:
        if FLAGS.save_dir is not None:
            run_id = "{name}_{time}".format(
                name=FLAGS.name,
                time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            save_dir = Path(FLAGS.save_dir) / run_id
            save_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"   Checkpoints → {save_dir}")
            wandb.config.update(dict(save_dir=str(save_dir)), allow_val_change=True)
        else:
            logging.warning("   --save_dir not set; checkpoints will NOT be saved.")

    # ─────────────────────────────────────────────────────────────────────
    # Unnormalization helper (uses TRAIN statistics everywhere)
    # ─────────────────────────────────────────────────────────────────────
    def unnormalize_action(action, unnorm_stats, dataset_name=None):
        ds_key = f'{dataset_name}_kwargs'
        norm_type = config.dataset_kwargs.dataset_kwargs_list[ds_key][
            "action_proprio_normalization_type"
        ]
        if norm_type == NormalizationType.NORMAL:
            mean = unnorm_stats["mean"]
            std = unnorm_stats["std"]
            mask = unnorm_stats.get("mask", np.ones_like(mean, dtype=bool))
            action = action[..., :len(mask)]
            action = np.where(mask, action * std + mean, action)
        elif norm_type == NormalizationType.BOUNDS:
            p01 = unnorm_stats["p01"]
            p99 = unnorm_stats["p99"]
            mask = unnorm_stats.get("mask", np.ones_like(p01, dtype=bool))
            action = action[..., :len(mask)]
            action = np.where(mask, (action + 1) * (p99 - p01) / 2 + p01, action)
        return action

    # ─────────────────────────────────────────────────────────────────────
    # Visualisation helper
    # ─────────────────────────────────────────────────────────────────────
    def _make_sample_viz(batch, pred_actions, sample_idx: int):
        """Return a wandb.Image with observation + xy trajectory overlay."""
        img_tensor = batch["observation"]["image_primary"][sample_idx]
        if img_tensor.ndim == 4:
            img_tensor = img_tensor[0]
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        if img_np.max() > 1.0:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0.0, 1.0)

        raw_lang = batch["raw_language"][sample_idx]
        lang_str = raw_lang if isinstance(raw_lang, str) else raw_lang.decode("utf-8")
        dataset_name = batch["dataset_name"][sample_idx]
        if isinstance(dataset_name, bytes):
            dataset_name = dataset_name.decode("utf-8")

        pred = pred_actions[sample_idx].squeeze(0).cpu().numpy()
        gt = batch["action"][sample_idx].squeeze(0).cpu().numpy()

        # Always unnormalize with TRAIN statistics
        pred = unnormalize_action(pred, train_statistics[dataset_name]['action'], dataset_name)
        gt = unnormalize_action(gt, train_statistics[dataset_name]['action'], dataset_name)

        pred_xy = np.cumsum(pred[:, :2], axis=0) - pred[0, :2]
        gt_xy = np.cumsum(gt[:, :2], axis=0) - gt[0, :2]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img_np)
        axes[0].set_title(f"obs  [{lang_str[:60]}]", fontsize=6)
        axes[0].axis("off")

        axes[1].plot(gt_xy[:, 0], gt_xy[:, 1], "g-o", label="GT", markersize=5, linewidth=1.5)
        axes[1].plot(pred_xy[:, 0], pred_xy[:, 1], "r--s", label="Pred", markersize=5, linewidth=1.5)
        axes[1].plot(0, 0, "k*", markersize=10, label="start")
        axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
        axes[1].set_title("Waypoints (xy)")
        axes[1].legend(fontsize=8)
        axes[1].set_aspect("equal", "datalim")
        axes[1].grid(True, alpha=0.4)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        pil_img = PILImage.open(buf).copy()
        plt.close(fig)
        buf.close()
        return wandb.Image(pil_img, caption=f"lang: {lang_str[:120]}")

    # ─────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def run_validation(step: int):
        model.eval()
        val_iter = iter(val_dataloader)
        val_losses = []

        # First batch: loss + visualisation
        first_batch = _to_device(next(val_iter), device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, val_head = model(
                observations=first_batch["observation"],
                tasks=first_batch["task"],
                timestep_pad_mask=first_batch["observation"]["timestep_pad_mask"],
                action_pad_mask=first_batch["action_pad_mask"],
                gt_actions=first_batch["action"],
                train=True,
                verbose=False,
                save_attention_mask=False,
            )
        first_loss, _ = val_head["action"]
        val_losses.append(first_loss.item())

        # Full reverse diffusion for visualisation (first batch only)
        pred_actions = model.module.sample_actions(
            observations=first_batch["observation"],
            tasks=first_batch["task"],
            timestep_pad_mask=first_batch["observation"]["timestep_pad_mask"],
            unnormalization_statistics=None,
            train=False,
        ).to(device)

        if distributed_state.is_main_process:
            sample_logs = {}
            for s_idx in range(min(2, pred_actions.shape[0])):
                sample_logs[f"val/sample_{s_idx}"] = _make_sample_viz(
                    first_batch, pred_actions, s_idx
                )
            wandb.log(sample_logs, step=step)

        # Remaining batches: loss only
        for val_i, val_batch in enumerate(val_iter):
            if val_i >= FLAGS.val_steps - 1:
                break
            val_batch = _to_device(val_batch, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, val_head = model(
                    observations=val_batch["observation"],
                    tasks=val_batch["task"],
                    timestep_pad_mask=val_batch["observation"]["timestep_pad_mask"],
                    action_pad_mask=val_batch["action_pad_mask"],
                    gt_actions=val_batch["action"],
                    train=True,
                    verbose=False,
                    save_attention_mask=False,
                )
            val_loss, _ = val_head["action"]
            val_losses.append(val_loss.item())

        mean_val = float(np.mean(val_losses))
        if distributed_state.is_main_process:
            wandb.log({"val/loss": mean_val}, step=step)
        logging.info(f"   [step {step}] val_loss = {mean_val:.4f}")
        model.train()
        return mean_val

    # ─────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────
    logging.info("\n" + "=" * 80)
    logging.info("STARTING TRAINING")
    logging.info("=" * 80)
    logging.info(f"  Total steps:    {config.num_steps}")
    logging.info(f"  Batch size:     {config.dataset_kwargs.batch_size}")
    logging.info(f"  Peak LR:        {lr_cfg.peak_value}")
    logging.info(f"  Warmup steps:   {lr_cfg.warmup_steps}")
    logging.info(f"  Save interval:  {config.save_interval}")
    logging.info(f"  Eval interval:  {config.eval_interval}")
    logging.info(f"  Grad clip:      {FLAGS.grad_clip}")
    logging.info(f"  Mixed precision: fp16")

    model.train()
    for i, batch in tqdm.tqdm(
        enumerate(dataloader), total=int(config.num_steps), dynamic_ncols=True
    ):
        if i >= config.num_steps:
            break

        optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()
        batch = _to_device(batch, device)

        # ── Forward (mixed precision) ─────────────────────────────────────
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, head_outputs = model(
                observations=batch["observation"],
                tasks=batch["task"],
                timestep_pad_mask=batch["observation"]["timestep_pad_mask"],
                action_pad_mask=batch["action_pad_mask"],
                gt_actions=batch["action"],
                train=True,
                verbose=False,
                save_attention_mask=False,
            )
            loss, metrics = head_outputs["action"]

        # Skip NaN batches instead of corrupting optimizer state
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"Step {i}: NaN/Inf loss detected, skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        # ── Backward (scaled) ─────────────────────────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_param_list, FLAGS.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────
        if distributed_state.is_main_process and (i + 1) % FLAGS.log_interval == 0:
            log_dict = {
                "train/loss": loss.item(),
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            }
            for k, v in metrics.items():
                if isinstance(v, (int, float)) or (
                    isinstance(v, torch.Tensor) and v.numel() == 1
                ):
                    log_dict[f"train/{k}"] = float(v)
            wandb.log(log_dict, step=i)

        # ── Validation ────────────────────────────────────────────────────
        if (i + 1) % config.eval_interval == 0:
            logging.info(f"\nRunning validation at step {i + 1}...")
            run_validation(step=i + 1)

        # ── Checkpoint ────────────────────────────────────────────────────
        if (distributed_state.is_main_process
                and save_dir is not None
                and (i + 1) % config.save_interval == 0):
            logging.info(f"Saving checkpoint at step {i + 1}...")
            model.module.save_pretrained(
                step=i + 1,
                checkpoint_path=save_dir,
                optimizer=optimizer,
            )

    # ── Final validation & checkpoint ────────────────────────────────────
    logging.info("\nRunning final validation...")
    run_validation(step=int(config.num_steps))

    if distributed_state.is_main_process and save_dir is not None:
        logging.info("Saving final checkpoint...")
        model.module.save_pretrained(
            step=int(config.num_steps),
            checkpoint_path=save_dir,
            optimizer=optimizer,
        )

    logging.info("\n✅ Training complete!")
    if distributed_state.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)