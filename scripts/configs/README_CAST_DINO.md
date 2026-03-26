# Finetuning Octo with DinoV2 on CAST Dataset

This guide explains how to finetune the Octo-small model with DinoV2 vision encoder on the CAST navigation dataset.

## Overview

**What's different from standard Octo:**
- **Vision Encoder**: DinoV2-ViT-S/14 (384 dim) instead of SmallStem16 (512 dim)
- **Image Resolution**: 224×224 (DinoV2's native) instead of 256×256
- **Task**: Navigation with waypoint actions (2D) instead of manipulation (7-DOF)
- **Observation**: Single camera + proprioception instead of dual cameras
- **Action Horizon**: 8 waypoints instead of 4

## Files Created

```
scripts/
├── configs/
│   └── finetune_cast_dino_config.py    # Configuration file
└── finetune_cast_dino.py                # Training script with DinoV2 setup

examples/
├── dino_encoder.py                      # DinoV2 encoder implementation
├── initialize_projection.py             # Helper for projection init
└── DINOV2_INTEGRATION.md                # DinoV2 integration guide
```

## Quick Start

### 1. Prerequisites

Ensure you have the CAST dataset available:
```bash
# Update the data_dir in finetune_cast_dino_config.py (line 158)
data_dir = '/mnt/weka/zhougrp/datasets/CAST_dataset'
```

The CAST dataset should have this structure:
```
CAST_dataset/
├── cast_filtered_dataset/
├── cast_counterfactual_dataset/
└── atomic_datasets/
    ├── atomic_forward_dataset/
    ├── atomic_turn_left_dataset/
    └── ...
```

### 2. Test Configuration

Verify the config loads correctly:
```bash
python scripts/configs/finetune_cast_dino_config.py
```

Expected output:
```
✓ Config created successfully
  Action dimension: 2
  Action horizon: 8
  Image size: {'primary': (224, 224)}
  Window size: 1
  Encoder: DinoV2
```

### 3. Validate Setup (Debug Mode)

Test the complete pipeline with a small subset:
```bash
python scripts/finetune_cast_dino.py --debug
```

This will:
- ✓ Load pretrained Octo-small weights
- ✓ Replace encoder with DinoV2
- ✓ Initialize projection as identity
- ✓ Load CAST dataset
- ✓ Verify batch shapes

### 4. Run Full Finetuning

**Single GPU:**
```bash
python scripts/finetune_cast_dino.py \
    --config scripts/configs/finetune_cast_dino_config.py \
    --name cast_dino_exp1
```

**Multi-GPU (recommended):**
```bash
torchrun --nproc_per_node 4 scripts/finetune_cast_dino.py \
    --config scripts/configs/finetune_cast_dino_config.py \
    --name cast_dino_exp1
```

## Configuration Details

### Model Architecture

```python
# Vision encoder: DinoV2 (frozen)
encoder = DinoV2EncoderPt(
    model_name="dinov2_vits14",  # 384-dim output
    freeze=True,                  # Keep frozen during finetuning
    img_norm_type="imagenet",
)

# Projection: 384 → 384 (initialized as identity)
# This allows DinoV2 features to pass through nearly unchanged

# Action head: Diffusion for waypoint prediction
action_head = DiffusionActionHeadPt(
    action_horizon=8,    # 8 waypoints
    action_dim=2,        # 2D (x, y) per waypoint
)
```

### Dataset Configuration

```python
# CAST subdatasets with sample weights
datasets = [
    "cast_filtered_dataset",        # 0.4 weight
    "cast_counterfactual_dataset",  # 0.4 weight
    "atomic_forward_dataset",       # 0.1 weight
    "atomic_turn_left_dataset",     # 0.1 weight
    "atomic_turn_right_dataset",    # 0.1 weight
    "atomic_adjust_left_dataset",   # 0.1 weight
    "atomic_adjust_right_dataset",  # 0.1 weight
    "atomic_stop_dataset",          # 0.1 weight
]
```

### Training Hyperparameters

```python
num_steps = 50000
batch_size = 256
learning_rate = 3e-4  # Peak LR with rsqrt schedule
window_size = 1        # Single frame (navigation)
action_horizon = 8     # 8 waypoints
```

## Key Implementation Details

### 1. DinoV2 Initialization

The `load_weights_from_jax` function converts CAST trajectories to relative waypoints:

```python
missing_keys, skipped_keys = model.load_weights_from_jax(
    "hf://rail-berkeley/octo-small-1.5",
    skip_keys_regex='.*observation_tokenizers.primary.encoder.*|.*obs_primary_projection.*'
)
```

### 2. Projection Layer

Since DinoV2 outputs 384 dimensions (matching `token_embedding_size=384`), the projection is 384→384. For finetuning, we simply skip loading the pretrained projection weights and let it train from random initialization. This is simpler and works well.

### 3. CAST Dataset Transform

The `gnm_dataset_transform` function converts CAST trajectories to relative waypoints:

```python
def gnm_dataset_transform(trajectory, action_horizon=8):
    # Compute future waypoints
    # Rotate to robot frame
    # Normalize by trajectory-specific factor
    trajectory["action"] = relative_waypoints
    return trajectory
```

### 4. Observation Space

CAST provides:
- **Image**: Single camera at 224×224
- **Proprio**: Robot position (x, y, yaw)
- **Language**: Navigation instruction

The model processes these through:
- Image → DinoV2 → Image tokens
- Proprio → Bin tokenizer → Proprio tokens
- Language → T5 → Language tokens

## Monitoring Training

Metrics logged to W&B:
- Training loss
- Action prediction accuracy
- Dataset statistics
- Learning rate schedule
- GPU utilization

## Troubleshooting

### Issue: Dataset not found
```
❌ Failed to load dataset: No such file or directory
```
**Solution**: Update `data_dir` in `finetune_cast_dino_config.py` (line 158)

### Issue: Shape mismatch error
```
AssertionError: observations does not match example batch
```
**Solution**: Ensure images are 224×224 and use `pad_mask_dict` structure (see examples in notebook)

### Issue: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Reduce `batch_size` in config (line 239)
- Use gradient checkpointing
- Use fewer GPUs per node

### Issue: Slow data loading
```
Dataset iterator is slow
```
**Solutions**:
- Increase `traj_transform_threads` and `traj_read_threads` (line 258)
- Reduce `shuffle_buffer_size` if memory is an issue

## Next Steps

1. **Monitor training**: Check W&B dashboard for loss curves
2. **Evaluate**: Run inference on held-out CAST trajectories
3. **Deploy**: Test on real robot navigation tasks
4. **Iterate**: Adjust hyperparameters based on performance

## Advanced Customization

### Change DinoV2 Model Size

```python
# In finetune_cast_dino_config.py, line 129
encoder=ModuleSpec.create(
    DinoV2EncoderPt,
    model_name="dinov2_vitb14",  # Options: vits14, vitb14, vitl14, vitg14
    ...
)
```

Larger models have more parameters but may perform better:
- `dinov2_vits14`: 22M params, 384 dim (default)
- `dinov2_vitb14`: 86M params, 768 dim
- `dinov2_vitl14`: 300M params, 1024 dim  
- `dinov2_vitg14`: 1.1B params, 1536 dim

**Note**: If using larger models, update `token_embedding_size` and projection dimensions accordingly.

### Unfreeze DinoV2 for Full Finetuning

```python
# In finetune_cast_dino_config.py, line 130
freeze=False,  # Allow DinoV2 to finetune

# In optimizer config, line 234
frozen_keys=("*hf_model*",),  # Remove "*dino_model*"
```

### Add Wrist Camera

```python
# In observation_tokenizers config, line 118
"wrist": ModuleSpec.create(
    ImageTokenizerPt,
    obs_stack_keys=["image_wrist"],
    encoder=ModuleSpec.create(DinoV2EncoderPt, ...),
),
```

## References

- [Octo Paper](https://arxiv.org/abs/2305.16171)
- [DinoV2 Paper](https://arxiv.org/abs/2304.07193)
- [Original Octo Repository](https://github.com/octo-models/octo)
- [DinoV2 Integration Guide](../examples/DINOV2_INTEGRATION.md)

## Citation

If you use this code, please cite:

```bibtex
@article{octo_2023,
  title={Octo: An Open-Source Generalist Robot Policy},
  author={Octo Model Team},
  journal={arXiv preprint arXiv:2305.16171},
  year={2023}
}

@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```
