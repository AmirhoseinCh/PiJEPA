"""
Configuration for finetuning Octo-small with DinoV2 encoder on CAST navigation dataset.

This config:
- Replaces SmallStem16 encoder with DinoV2 (384 dim output)
- Uses single camera view (primary) at 224x224 (DinoV2's native resolution)
- Configures for navigation task with waypoint actions
- Sets up proper observation space (image + proprio)
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

# Import DinoV2 encoder (ensure dino_encoder.py is in examples/)
import sys
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../examples'))
from octo.model.components.dino_encoder import DinoV2EncoderPt


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string=None):
    # config_string is not used here, but kept for API compatibility
    # Use vit_s transformer size (matching octo-small)
    config = get_base_config(transformer_size="vit_s")

    # Navigation actions: waypoints in 3D (x, y, yaw_delta) over horizon
    action_horizon = 8
    action_dim = FieldReference(4)  # 2D waypoints + sin/cos delta yaw instead of 7-DOF manipulation
    action_normalization_type = NormalizationType.BOUNDS

    # === MODEL CONFIGURATION ===
    
    # Replace default vision encoder with DinoV2
    config["model"]["observation_tokenizers"] = {
        "primary": ModuleSpec.create(
            ImageTokenizerPt,
            obs_stack_keys=["image_primary"],
            task_stack_keys=[],  # No goal conditioning for CAST navigation
            encoder=ModuleSpec.create(
                DinoV2EncoderPt,
                model_name="dinov2_vits14",  # 384-dim output
                freeze=True,  # Keep DinoV2 frozen during finetuning
                img_norm_type="imagenet",
            ),
            use_token_learner=False,  # Direct patch tokens
        ),
        # Add proprio tokenizer for position/state 
        # No need for proprio tokenizer since we're encoding waypoints directly as actions and window size is 1 (no temporal stacking)
        # "proprio": ModuleSpec.create(
        #     LowdimObsTokenizerPt,
        #     obs_keys=["proprio"],
        #     discretize=True,
        #     n_bins=256,
        #     bin_type="uniform",
        #     low=-2.0,
        #     high=2.0,
        # ),
    }
    
    # Keep language tokenizer for instructions
    config["model"]["task_tokenizers"] = {
        "language": ModuleSpec.create(
            LanguageTokenizerPt,
            encoder="t5-base",
            finetune_encoder=False,
            proper_pad_mask=False,  # Language is per-trajectory, not per-timestep
        ),
    }
    
    # Model architecture settings
    config["model"]["repeat_task_tokens"] = True
    config["model"]["readouts"] = {"action": 1}
    config["model"]["max_horizon"] = action_horizon  # Must match action_horizon
    
    # Action head: Diffusion for navigation waypoints
    config["model"]["heads"]["action"] = ModuleSpec.create(
        DiffusionActionHeadPt,
        readout_key="readout_action",
        use_map=True,
        action_horizon=action_horizon,
        action_dim=action_dim,
        n_diffusion_samples=1,
        dropout_rate=0.1,
        max_action=1.0,  # actions normalized to [-1, 1]
    )

    # === DATASET CONFIGURATION ===
    
    # CAST dataset path (update this to your path)
    data_dir = '/mnt/weka/zhougrp/datasets/CAST_dataset'
    
    # Image augmentation for navigation (lighter augmentation)
    primary_augment_kwargs = dict(
        # random_resized_crop=dict(scale=[0.9, 1.0], ratio=[0.95, 1.05]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        augment_order=[
            # "random_resized_crop",
            "random_brightness",
            "random_contrast",
        ],
    )

    # Define CAST subdatasets
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
        # Determine subdirectory
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
        "primary": (224, 224),  # DinoV2 native resolution
    }
    config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = {
        "primary": primary_augment_kwargs,
    }

    # Remove oxe_kwargs from base config since we're using custom dataset
    del config["dataset_kwargs"]["oxe_kwargs"]
    
    # Update config with CAST-specific settings
    config = update_config(
        config,
        num_steps=5000,  # Finetune for 5k steps
        window_size=1,  # CAST uses single frame window
        save_interval=1000,
        eval_interval=100,
        optimizer=dict(
            frozen_keys=("*hf_model*", "*dino_model*"),  # Freeze text encoder and DinoV2
            learning_rate=dict(
                name="rsqrt",
                init_value=0.0,
                peak_value=3e-4,  # Lower LR for finetuning
                warmup_steps=2000,
                timescale=10000,
            ),
        ),
        dataset_kwargs=dict(
            dataset_kwargs_list=cast_datasets,  # Use CAST dataset instead of OXE
            sample_weights=[0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Weight filtered/counterfactual more
            traj_transform_kwargs=dict(
                window_size=1,
                action_horizon=action_horizon,
                goal_relabeling_strategy=None,  # Disable goal relabeling for CAST
                task_augment_strategy=None,  # No task augmentation for navigation
                subsample_length=100,  # Subsample long trajectories
            ),
            batch_size=256,  # Adjust based on your GPU memory
            shuffle_buffer_size=50000,
            balance_weights=False,  # Use explicit sample weights instead
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
        eval_datasets=["cast_filtered_dataset"],  # Evaluate on main CAST dataset
    )

    return config


if __name__ == "__main__":
    # Test config creation
    config = get_config()
    print("✓ Config created successfully")
    print(f"  Action dimension: {config.model.heads.action.kwargs.action_dim}")
    print(f"  Action horizon: {config.model.heads.action.kwargs.action_horizon}")
    print(f"  Image size: {config.dataset_kwargs.frame_transform_kwargs.resize_size}")
    print(f"  Window size: {config.window_size}")
    print(f"  Encoder: DinoV2")
