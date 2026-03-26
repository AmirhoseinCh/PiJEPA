#!/usr/bin/env python3
"""
Script to finetune Octo-small with DinoV2 encoder on CAST dataset.

This script:
1. Loads Octo-small pretrained weights
2. Replaces vision encoder with DinoV2
3. Skips loading projection layer (will be trained from random init)
4. Finetunes on CAST navigation dataset

Usage:
    # Single GPU
    python scripts/finetune_cast_dino.py --config scripts/configs/finetune_cast_dino_config.py
    
    # Multi-GPU (recommended)
    torchrun --nproc_per_node 4 scripts/finetune_cast_dino.py --config scripts/configs/finetune_cast_dino_config.py
"""

import datetime
import json
import os
import sys
from pathlib import Path

from octo.data.utils.data_utils import NormalizationType

# Add examples directory to path for DinoV2 encoder
# sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from absl import app, flags, logging
import tqdm
import wandb

from accelerate import PartialState

from octo.model.octo_model_pt import OctoModelPt
from octo.utils.spec import ModuleSpec
import tensorflow as tf
from octo.data.dataset import make_interleaved_dataset
from octo.utils.torch_rlds_dataset import TorchRLDSDataset
from octo.utils.train_utils_pt import (
    _np2pt,
    freeze_weights_pt,
    get_cosine_schedule_with_warmup,
    tree_map,
)
from torch.utils.data import DataLoader

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "scripts/configs/finetune_cast_dino_config.py", "Path to config file")
flags.DEFINE_string("pretrained", "hf://rail-berkeley/octo-small-1.5", "Pretrained checkpoint path")
flags.DEFINE_string("name", "cast_dino_finetune", "Experiment name for logging")
flags.DEFINE_bool("debug", False, "Debug mode with small dataset")
flags.DEFINE_string("save_dir", "/mnt/weka/zhougrp/octo_pt_cast", "Directory to save checkpoints (None = no saving)")
flags.DEFINE_integer("log_interval", 10, "Log training loss every N steps")
flags.DEFINE_integer("val_steps", 20, "Number of batches to compute validation loss")

def custom_collate_fn(batch):
    """Custom collate function to handle nested dictionaries and tensors."""
    if isinstance(batch[0], dict):
        return {key: custom_collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, dim=0)
    elif isinstance(batch[0], (bool, np.bool_)):
        # Handle boolean scalars (like pad masks) - convert to tensor
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
        return data.to(device)
    else:
        return data  # pass strings, lists, None, etc. through unchanged

def main(_):
    # Import config module
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", FLAGS.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.get_config()
    
    # Setup distributed environment
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
        logging.info("🐛 Debug mode: using smaller settings")
        config.num_steps = 4
        config.batch_size = 4
        config.dataset_kwargs.batch_size = 4
        config.dataset_kwargs.shuffle_buffer_size = 100
        config.eval_interval = 3
    
    # Initialize wandb
    if distributed_state.is_main_process:
        wandb.init(
            project="octo-cast-finetune",
            name=FLAGS.name,
            config=config.to_dict(),
        )
    
    logging.info("=" * 80)
    logging.info("FINETUNING OCTO-SMALL WITH DINOV2 ON CAST DATASET")
    logging.info("=" * 80)
    
    # Step 1: Load config and metadata from pretrained checkpoint
    logging.info(f"\n📥 Loading config from {FLAGS.pretrained}")
    meta = OctoModelPt.load_config_and_meta_from_jax(FLAGS.pretrained)
    
    # Step 2: Override with DinoV2 configuration
    logging.info("\n🔧 Configuring DinoV2 encoder...")
    # Convert ConfigDict to regular dict to avoid serialization issues
    meta['config']['model']['observation_tokenizers'] = config.model.observation_tokenizers.to_dict()
    meta['config']['model']['task_tokenizers'] = config.model.task_tokenizers.to_dict()
    meta['config']['model']['heads'] = config.model.heads.to_dict()
    meta['config']['model']['readouts'] = config.model.readouts.to_dict()
    
    # Update num_tokens_dict for new tokenizers
    # primary: DinoV2 produces 16x16 = 256 patch tokens
    # proprio: LowdimObsTokenizer produces 1 token per dim with  discretization (2D waypoints)
    meta['config']['model']['num_tokens_dict'] = {
        'primary': 256,  # 16x16 patches from DinoV2
        'proprio': 2,    # 2D waypoint (x, y)
        'language': 16,  # From T5 tokenizer
        'action': 1,     # Readout token
    }
    
    # Update action dimension in example batch
    action_horizon = config.model.heads.action.kwargs.action_horizon
    action_dim = config.model.heads.action.kwargs.action_dim
    
    logging.info(f"   Action horizon: {action_horizon}")
    logging.info(f"   Action dim: {action_dim}")
    logging.info(f"   Window size: {config.window_size}")

    # Step 3: Load CAST dataset
    logging.info("\n📊 Loading CAST dataset...")
    

    # Create dataset configuration
    dataset_kwargs_list = [
        config.dataset_kwargs.dataset_kwargs_list[k] 
        for k in config.dataset_kwargs.dataset_kwargs_list
    ]
    
    # Don't batch in TensorFlow - let PyTorch DataLoader handle batching
    train_dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        config.dataset_kwargs.sample_weights,
        train=True,
        shuffle_buffer_size=config.dataset_kwargs.shuffle_buffer_size,
        frame_transform_kwargs=config.dataset_kwargs.frame_transform_kwargs,
        traj_transform_kwargs=config.dataset_kwargs.traj_transform_kwargs,
        batch_size=None,  # No batching in TF dataset
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
        batch_size=None,  # No batching in TF dataset
        balance_weights=config.dataset_kwargs.balance_weights,
        traj_transform_threads=config.dataset_kwargs.traj_transform_threads,
        traj_read_threads=config.dataset_kwargs.traj_read_threads,
    )
    # Keep dataset_statistics in TensorFlow format for the data pipeline
    # They will be converted to PyTorch when needed by the model
    meta['dataset_statistics'] = train_dataset.dataset_statistics
    val_statistics = val_dataset.dataset_statistics

    train_dataset = train_dataset.filter(lambda x: tf.reduce_any(x["observation"]["image_primary"] != 255))
    train_dataset = train_dataset.filter(lambda x: tf.reduce_all(x["action"][:,3:,:] != 0.0))

    val_dataset = val_dataset.filter(lambda x: tf.reduce_any(x["observation"]["image_primary"] != 255))
    val_dataset = val_dataset.filter(lambda x: tf.reduce_all(x["action"][:,3:,:] != 0.0))
    
    logging.info("   ✓ Dataset loaded successfully")
    pytorch_dataset = TorchRLDSDataset(train_dataset, meta["text_processor"])
    
    # Batch in PyTorch DataLoader for native torch tensors
    dataloader = DataLoader(
            pytorch_dataset,
            batch_size=config.dataset_kwargs.batch_size,
            num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
            collate_fn=custom_collate_fn,  # Handle nested dicts
        )
    

    val_pytorch_dataset = TorchRLDSDataset(val_dataset, meta["text_processor"], train=False)
    val_dataloader = DataLoader(
            val_pytorch_dataset,
            batch_size=config.dataset_kwargs.batch_size,
            num_workers=0,
            collate_fn=custom_collate_fn,
        )
    sample_batch = next(iter(dataloader))
    # Get a sample batch to verify
    # sample_batch = next(iter(train_dataset.batch(1).iterator()))
    logging.info(f"   Sample batch keys: {list(sample_batch.keys())}")
    logging.info(f"   Image shape: {sample_batch['observation']['image_primary'].shape}")
    logging.info(f"   Action shape: {sample_batch['action'].shape}")
    logging.info(f"   Proprio shape: {sample_batch['observation']['proprio'].shape}")
    logging.info(f"   Language instruction keys: {list(sample_batch['task']['language_instruction'].keys())}")
    logging.info(f"   Language instruction attention_mask shape: {sample_batch['task']['language_instruction']['attention_mask'].shape}")
    logging.info(f"   Language instruction input_ids shape: {sample_batch['task']['language_instruction']['input_ids'].shape}")
    logging.info(f"   Dataset name: {sample_batch['dataset_name']}")
    logging.info(f"  Raw language instruction: {sample_batch['raw_language']}")
    
    # Debug: check the actual dtype and shape details
    input_ids = sample_batch['task']['language_instruction']['input_ids']
    logging.info(f"   input_ids ndim: {input_ids.ndim}, dtype: {input_ids.dtype}, shape details: {input_ids.shape}")
    logging.info(f"   task keys: {list(sample_batch['task'].keys())}")
    if 'pad_mask_dict' in sample_batch['task']:
        logging.info(f"   pad_mask_dict keys: {list(sample_batch['task']['pad_mask_dict'].keys())}")
        if 'language_instruction' in sample_batch['task']['pad_mask_dict']:
            pad_mask_val = sample_batch['task']['pad_mask_dict']['language_instruction']
            logging.info(f"   pad_mask_dict language_instruction type: {type(pad_mask_val)}, value: {pad_mask_val}")
    
    meta['example_batch'] = sample_batch
    
    # Step 3: Create model
    logging.info("\n🏗️  Creating model...")
    model = OctoModelPt.from_config(**meta)
    
    # Step 4: Load pretrained weights (skip DinoV2 encoder, projection, and action head)
    logging.info(f"\n⚡ Loading pretrained weights from {FLAGS.pretrained}")
    logging.info("   Skipping: DinoV2 encoder (has own pretrained weights)")
    logging.info("   Skipping: obs_primary_projection (will train from random init)")
    logging.info("   Skipping: heads.action (different action dimension: 2 instead of 7)")
    logging.info("   Skipping: observation_tokenizers.proprio (new tokenizer)")
    
    missing_keys, skipped_keys = model.load_weights_from_jax(
        FLAGS.pretrained,
        skip_keys_regex='.*observation_tokenizers.primary.encoder.*|.*obs_primary_projection.*|.*heads.action.*|.*observation_tokenizers.proprio.*'
    )
    
    logging.info(f"   ✓ Loaded weights")
    logging.info(f"   Missing keys: {len(missing_keys)}")
    logging.info(f"   Skipped keys: {len(skipped_keys)}")
    
    # Step 5: Move model to device and wrap with DDP
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Step 6: Freeze DinoV2 encoder and T5 text encoder weights
    logging.info("\n❄️  Freezing DinoV2 and T5 text encoder weights...")
    frozen_keys = list(config.optimizer.frozen_keys)
    freeze_weights_pt(model, frozen_keys)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"   Trainable parameters: {trainable_params:,} / {total_params:,} "
                 f"({100 * trainable_params / total_params:.1f}%)")

    # Step 7: Optimizer and LR scheduler (cosine with warmup)
    logging.info("\n⚙️  Setting up optimizer and LR scheduler...")
    lr_cfg = config.optimizer.learning_rate
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_param_list, lr=lr_cfg.peak_value)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=lr_cfg.warmup_steps,
        num_training_steps=int(config.num_steps),
    )

    # Step 8: Setup save directory
    if distributed_state.is_main_process:
        
        if FLAGS.save_dir is not None:
            save_dir = Path(FLAGS.save_dir)
            run_id = "{name}_{time}".format(
                name=FLAGS.name,
                time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            save_dir = save_dir / run_id
            save_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"   Checkpoints will be saved to: {save_dir}")
            wandb.config.update(dict(save_dir=str(save_dir)), allow_val_change=True)
        else:
            save_dir = None
            logging.warning("   --save_dir not set; checkpoints will NOT be saved.")

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: build a wandb visualisation panel for one sample
    # ─────────────────────────────────────────────────────────────────────────
    def unnormalize_action(action, unnormalization_statistics, dataset_name=None):
        if config.dataset_kwargs.dataset_kwargs_list[f'{dataset_name}_kwargs']["action_proprio_normalization_type"] == NormalizationType.NORMAL:
                mean = unnormalization_statistics["mean"]
                std = unnormalization_statistics["std"]
                
                mask = unnormalization_statistics.get(
                    "mask",
                    np.ones_like(mean, dtype=bool),
                )
                
                
                action = action[..., : len(mask)]
                print(action.shape, std.shape, mean.shape, mask.shape)
                action = np.where(
                    mask,
                    (action * std)
                    + mean,
                    action,
                )
        elif config.dataset_kwargs.dataset_kwargs_list[f'{dataset_name}_kwargs']["action_proprio_normalization_type"] == NormalizationType.BOUNDS:
            p01 = unnormalization_statistics["p01"]
            p99 = unnormalization_statistics["p99"]
            
            mask = unnormalization_statistics.get(
                "mask", np.ones_like(p01, dtype=bool)
            )
            action = action[..., : len(mask)]
            action = np.where(
                mask,
                (action + 1) * (p99 - p01) / 2 + p01, 
                action,
            )
        return action
    def _make_sample_viz(batch, pred_actions, sample_idx: int):
        """Return a wandb.Image combining the observation and an xy trajectory plot."""
        # ── Observation image ────────────────────────────────────────────────
        img_tensor = batch["observation"]["image_primary"][sample_idx]
        if img_tensor.ndim == 4:
            img_tensor = img_tensor[0]  # first timestep → (C, H, W)
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)  # → (H, W, C)
        # Clip to valid display range (images may already be uint8 or float [0,1])
        if img_np.max() > 1.0:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0.0, 1.0)

        # ── Language instruction ─────────────────────────────────────────────
        raw_lang = batch["raw_language"][sample_idx]
        lang_str = raw_lang if isinstance(raw_lang, str) else raw_lang.decode("utf-8")  # Handle both str and bytes
        dataset_name = batch["dataset_name"][sample_idx]
        # ── Waypoints ────────────────────────────────────────────────────────
        pred = pred_actions[sample_idx].squeeze(0).cpu().numpy()  # (horizon, dim)
        gt   = batch["action"][sample_idx].squeeze(0).cpu().numpy()    # (horizon, dim)
        pred = unnormalize_action(pred, val_statistics[dataset_name]['action'], dataset_name)
        gt = unnormalize_action(gt, val_statistics[dataset_name]['action'], dataset_name)
        pred_xy = pred[:, :2]
        gt_xy   = gt[:, :2]

        pred_xy = np.cumsum(pred_xy, axis=0) - pred_xy[0]  # Convert from deltas to absolute positions
        gt_xy = np.cumsum(gt_xy, axis=0) - gt_xy[0]  # Center on start point (0,0) for better visual comparison

        # ── Composite figure: observation | trajectory plot ──────────────────
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(img_np)
        axes[0].set_title(f"obs  [{lang_str[:60]}]", fontsize=6)
        axes[0].axis("off")

        axes[1].plot(gt_xy[:, 0],   gt_xy[:, 1],   "g-o", label="GT",   markersize=5, linewidth=1.5)
        axes[1].plot(pred_xy[:, 0], pred_xy[:, 1], "r--s", label="Pred", markersize=5, linewidth=1.5)
        # Mark start point
        axes[1].plot(0, 0, "k*", markersize=10, label="start")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
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

    # ─────────────────────────────────────────────────────────────────────────
    # Validation helper
    # ─────────────────────────────────────────────────────────────────────────
    def run_validation(step: int):
        """Compute val loss over FLAGS.val_steps batches; visualise 2 samples from
        the first batch only (full action denoising is expensive)."""
        model.eval()
        val_iter = iter(val_dataloader)
        val_losses = []

        with torch.no_grad():
            # ── First batch: loss + sample_actions visualisation ─────────────
            first_batch = _to_device(next(val_iter), device)

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

            # Full reverse diffusion for the first batch only
            pred_actions = model.module.sample_actions(
                observations=first_batch["observation"],
                tasks=first_batch["task"],
                timestep_pad_mask=first_batch["observation"]["timestep_pad_mask"],
                unnormalization_statistics=None,
                train=False,
            )  # → (B, horizon, dim) on CPU
            pred_actions = pred_actions.to(device)

            sample_logs = {}
            for s_idx in range(min(2, pred_actions.shape[0])):
                sample_logs[f"val/sample_{s_idx}"] = _make_sample_viz(
                    first_batch, pred_actions, s_idx
                )
            if distributed_state.is_main_process:
                wandb.log(sample_logs, step=step)

            # ── Remaining batches: loss only ──────────────────────────────────
            for val_i, val_batch in enumerate(val_iter):
                if val_i >= FLAGS.val_steps - 1:
                    break
                val_batch = _to_device(val_batch, device)
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

        mean_val_loss = float(np.mean(val_losses))
        if distributed_state.is_main_process:
            wandb.log({"val/loss": mean_val_loss}, step=step)
        logging.info(f"   [step {step}] val_loss = {mean_val_loss:.4f}")
        model.train()
        return mean_val_loss

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    logging.info("\n" + "=" * 80)
    logging.info("STARTING TRAINING")
    logging.info("=" * 80)
    logging.info(f"  Total steps:    {config.num_steps}")
    logging.info(f"  Batch size:     {config.dataset_kwargs.batch_size}")
    logging.info(f"  Peak LR:        {lr_cfg.peak_value}")
    logging.info(f"  Warmup steps:   {lr_cfg.warmup_steps}")
    logging.info(f"  Save interval:  {config.save_interval}")
    logging.info(f"  Eval interval:  {config.eval_interval}")
    logging.info(f"  Log interval:   {FLAGS.log_interval}")

    model.train()
    for i, batch in tqdm.tqdm(
        enumerate(dataloader), total=int(config.num_steps), dynamic_ncols=True
    ):
        if i >= config.num_steps:
            break

        # ── Forward + loss ────────────────────────────────────────────────
        optimizer.zero_grad()
        batch = _to_device(batch, device)

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
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────
        if distributed_state.is_main_process and (i + 1) % FLAGS.log_interval == 0:
            log_dict = {
                "train/loss":          loss.item(),
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
            }
            # Log any extra scalar metrics returned by the action head
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

        # ── Checkpoint ───────────────────────────────────────────────────
        if distributed_state.is_main_process and save_dir is not None and (i + 1) % config.save_interval == 0:
            logging.info(f"Saving checkpoint at step {i + 1}...")
            model.module.save_pretrained(
                step=i + 1,
                checkpoint_path=save_dir,
                optimizer=optimizer,
            )

    # ── Final validation & checkpoint ────────────────────────────────────────
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
