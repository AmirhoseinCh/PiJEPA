"""
Configuration for finetuning Octo-small with V-JEPA2 encoder on CAST navigation dataset.

This config:
- Replaces SmallStem16 encoder with V-JEPA2 (large model, 1024 dim output)
- Uses single camera view (primary) at 256x256 (V-JEPA2's default resolution)
- Configures for navigation task with waypoint actions
- Sets up proper observation space (image + proprio)

Hyperparameter rationale (~3M samples, H200 GPUs):
- 15k steps × 256 batch ≈ 3.8M samples → ~1.3 epochs (good for finetuning)
- Warmup 1000 steps (~6.5% of training) — enough to stabilize
- Peak LR 3e-4 with cosine decay to near-zero
- Gradient clipping at 1.0 for diffusion head stability
"""

from copy import deepcopy
import importlib.util
import os
from typing import Any, Dict

from ml_collections import ConfigDict, FieldReference

# Load base config using importlib
config_path = os.path.join(os.path.dirname(__file__), "config.py")
spec = importlib.util.spec_from_file_location("base_config", config_path)
base_config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_config_module)
get_base_config = base_config_module.get_config

from octo.data.utils.text_processing import HFTokenizer
from octo.data.utils.data_utils import NormalizationType
from octo.data.utils.cast_transforms import gnm_dataset_transform, gnm_action_angle_dataset_transform
from octo.model.components.action_heads_pt import DiffusionActionHeadPt
from octo.model.components.tokenizers_pt import ImageTokenizerPt, LanguageTokenizerPt, LowdimObsTokenizerPt
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import hf_weights_loader

from octo.model.components.vjepa_encoder import VJEPA2EncoderPt


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string=None):
    # Use vit_s transformer size (matching octo-small)
    config = get_base_config(transformer_size="vit_s")

    # Navigation actions: waypoints (x, y, sin_yaw, cos_yaw) over horizon
    action_horizon = 8
    action_dim = FieldReference(4)
    action_normalization_type = NormalizationType.BOUNDS

    # === MODEL CONFIGURATION ===

    # Replace default vision encoder with V-JEPA2
    config["model"]["observation_tokenizers"] = {
        "primary": ModuleSpec.create(
            ImageTokenizerPt,
            obs_stack_keys=["image_primary"],
            task_stack_keys=[],
            encoder=ModuleSpec.create(
                VJEPA2EncoderPt,
                model_name="vjepa2_vit_large",  # 1024-dim output, 256x256 resolution
                freeze=True,
                img_norm_type="imagenet",
                force_resolution=256,  # V-JEPA2 default resolution
            ),
            use_token_learner=False,
        ),
    }

    # Keep language tokenizer for instructions
    config["model"]["task_tokenizers"] = {
        "language": ModuleSpec.create(
            LanguageTokenizerPt,
            encoder="t5-base",
            finetune_encoder=False,
            proper_pad_mask=False,
        ),
    }

    config["model"]["repeat_task_tokens"] = True
    config["model"]["readouts"] = {"action": 1}
    config["model"]["max_horizon"] = action_horizon

    # Action head: Diffusion for navigation waypoints
    config["model"]["heads"]["action"] = ModuleSpec.create(
        DiffusionActionHeadPt,
        readout_key="readout_action",
        use_map=True,
        action_horizon=action_horizon,
        action_dim=action_dim,
        n_diffusion_samples=1,
        dropout_rate=0.1,
        max_action=1.0,
    )

    # === DATASET CONFIGURATION ===

    data_dir = '/mnt/weka/zhougrp/datasets/CAST_dataset'

    primary_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        augment_order=[
            "random_brightness",
            "random_contrast",
        ],
    )

    transform = ModuleSpec.create(gnm_action_angle_dataset_transform, action_horizon=action_horizon)

    cast_datasets = {}
    for dataset_name in [
        "cast_filtered_dataset",
        "cast_counterfactual_dataset",
        "atomic_forward_dataset",
        "atomic_turn_left_dataset",
        "atomic_turn_right_dataset",
        "atomic_adjust_left_dataset",
        "atomic_adjust_right_dataset",
        "atomic_stop_dataset",
    ]:
        if dataset_name.startswith("atomic_"):
            subdir = "atomic_datasets"
        else:
            subdir = dataset_name

        cast_datasets[f"{dataset_name}_kwargs"] = {
            "name": dataset_name,
            "data_dir": f"{data_dir}/{subdir}",
            "image_obs_keys": {"primary": "image"},
            "proprio_obs_key": "state",
            "language_key": "language_instruction",
            "action_proprio_normalization_type": action_normalization_type,
            "standardize_fn": transform,
            "force_recompute_dataset_statistics": False,
            "skip_norm": False,
        }

    # ML-collections requires field deletion before type change
    del config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    del config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]

    config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = {
        "primary": (256, 256),  # V-JEPA2 default resolution
    }
    config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = {
        "primary": primary_augment_kwargs,
    }

    del config["dataset_kwargs"]["oxe_kwargs"]

    # ──────────────────────────────────────────────────────────────────────
    # Hyperparameters (tuned for ~3M samples on H200 GPUs)
    # ──────────────────────────────────────────────────────────────────────
    #
    # Epoch math:
    #   3M samples ÷ 256 batch = 11,719 steps/epoch
    #   15k steps  ≈ 1.3 epochs   (good: enough to converge, low overfit risk)
    #   20k steps  ≈ 1.7 epochs   (use if val loss still dropping at 15k)
    #
    # Warmup:
    #   1000 steps ≈ 6.5% of training — standard for cosine schedules
    #
    # Sample weights normalised to sum=1.0:
    #   filtered & counterfactual dominate (0.3 each), atomics fill in (0.067 each)
    #
    config = update_config(
        config,
        num_steps=15000,
        window_size=1,
        save_interval=2000,
        eval_interval=250,
        optimizer=dict(
            frozen_keys=("*hf_model*", "*vjepa2_model*"),
            learning_rate=dict(
                name="rsqrt",
                init_value=0.0,
                peak_value=2e-4,
                warmup_steps=1000,
                timescale=10000,
            ),
        ),
        dataset_kwargs=dict(
            dataset_kwargs_list=cast_datasets,
            # Normalised so they sum to 1.0
            sample_weights=[0.3, 0.3, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067],
            traj_transform_kwargs=dict(
                window_size=1,
                action_horizon=action_horizon,
                goal_relabeling_strategy=None,
                task_augment_strategy=None,
                subsample_length=100,
            ),
            batch_size=256,
            shuffle_buffer_size=50000,
            balance_weights=False,
            traj_transform_threads=16,
            traj_read_threads=16,
        ),
        text_processor=ModuleSpec.create(
            HFTokenizer,
            tokenizer_name="t5-base",
            encode_with_model=False,
            tokenizer_kwargs={
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            },
        ),
        pretrained_loaders=(
            ModuleSpec.create(
                hf_weights_loader,
                hf_model="t5-base",
            ),
        ),
        eval_datasets=["cast_filtered_dataset"],
    )

    return config


if __name__ == "__main__":
    config = get_config()
    print("✓ Config created successfully")
    print(f"  Action dimension: {config.model.heads.action.kwargs.action_dim}")
    print(f"  Action horizon:   {config.model.heads.action.kwargs.action_horizon}")
    print(f"  Image size:       {config.dataset_kwargs.frame_transform_kwargs.resize_size}")
    print(f"  Window size:      {config.window_size}")
    print(f"  Num steps:        {config.num_steps}")
    print(f"  Warmup steps:     {config.optimizer.learning_rate.warmup_steps}")
    print(f"  Batch size:       {config.dataset_kwargs.batch_size}")
    print(f"  Encoder:          V-JEPA2")
