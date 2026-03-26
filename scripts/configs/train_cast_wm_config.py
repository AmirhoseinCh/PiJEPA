"""
Configuration for training a world model on CAST dataset.

The world model encodes images with DINOv2 (or V-JEPA2), then uses an
AdaLN-conditioned ViT predictor to predict next-frame image features given
the current features + action.

Window size 8, action horizon 1 (single-step prediction per timestep).
Action dim = 4 (local_x, local_y, sin_yaw, cos_yaw).
"""

from ml_collections import ConfigDict


def get_config(encoder_type: str = "dino"):
    """
    Args:
        encoder_type: "dino" or "vjepa". Selects the frozen image encoder.
    Returns:
        ConfigDict with all hyperparameters.
    """
    cfg = ConfigDict()

    # ── Encoder ──────────────────────────────────────────────────────────
    cfg.encoder_type = encoder_type  # "dino" | "vjepa"

    if encoder_type == "dino":
        cfg.enc_model_name = "dinov2_vits14"
        cfg.embed_dim = 384          # DINOv2 ViT-S/14 output dim
        cfg.img_size = 224           # DINOv2 native resolution
        cfg.patch_size = 14
        cfg.grid_size = 16           # 224 / 14 = 16
        cfg.frozen_patterns = ("*dino_model*",)
    elif encoder_type == "vjepa":
        cfg.enc_model_name = "vjepa2_vit_large"
        cfg.embed_dim = 1024         # V-JEPA2 ViT-L output dim
        cfg.img_size = 256           # V-JEPA2 native resolution
        cfg.patch_size = 16
        cfg.grid_size = 16           # 256 / 16 = 16
        cfg.frozen_patterns = ("*vjepa2_model*",)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    # ── Predictor (AdaLN) ────────────────────────────────────────────────
    cfg.pred_type = "AdaLN"
    cfg.pred_embed_dim = 384         # predictor internal dimension
    cfg.pred_depth = 6               # number of transformer blocks
    cfg.pred_num_heads = 12
    cfg.use_rope = True
    cfg.is_causal = False
    cfg.use_silu = False
    cfg.init_scale_factor_adaln = 10

    # ── Action ──────────────────────────────────────────────────────────
    cfg.action_dim = 4               # (local_x, local_y, sin_yaw, cos_yaw)
    cfg.action_horizon = 1           # per-frame ego action; a_t in frame t's local coord
    cfg.action_encoder_inpred = True # action encoder lives inside AdaLN predictor
    cfg.action_conditioning = "token"

    # ── Proprio (disabled for navigation) ────────────────────────────────
    cfg.use_proprio = False
    cfg.proprio_dim = 0
    cfg.proprio_emb_dim = 0
    cfg.proprio_tokens = 0
    cfg.proprio_encoding = "feature"
    cfg.proprio_encoder_inpred = True

    # ── World model flags ────────────────────────────────────────────────
    cfg.batchify_video = True        # encode each frame independently
    cfg.dup_image = False
    cfg.normalize_reps = False

    # ── Data ─────────────────────────────────────────────────────────────
    cfg.window_size = 8              # 8 consecutive observations per sample
    cfg.data_dir = "/mnt/weka/zhougrp/datasets/CAST_dataset"

    cfg.dataset_names = [
        "cast_filtered_dataset",
        # "cast_counterfactual_dataset",
        # "atomic_forward_dataset",
        # "atomic_turn_left_dataset",
        # "atomic_turn_right_dataset",
        # "atomic_adjust_left_dataset",
        # "atomic_adjust_right_dataset",
        # "atomic_stop_dataset",
    ]
    cfg.sample_weights = [1.0]#[0.3, 0.3, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067]

    cfg.batch_size = 32
    cfg.shuffle_buffer_size = 50000
    cfg.traj_transform_threads = 16
    cfg.traj_read_threads = 16

    cfg.img_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        augment_order=["random_brightness", "random_contrast"],
    )

    # ── Loss ─────────────────────────────────────────────────────────────
    cfg.loss = ConfigDict()
    cfg.loss.l2_loss_weight = 1.0
    cfg.loss.l1_loss_weight = 0.0
    cfg.loss.cos_loss_weight = 0.0
    cfg.loss.smooth_l1_loss_weight = 0.0
    cfg.loss.proprio_loss = False

    # ── Rollout ──────────────────────────────────────────────────────────
    cfg.rollout_steps = 1            # multi-step unrolling during training
    cfg.ctxt_window = 8              # context frames for predictor

    # ── Optimisation ─────────────────────────────────────────────────────
    cfg.num_steps = 15000           # total training steps
    cfg.lr = 5e-4
    cfg.weight_decay = 0.01
    cfg.warmup_steps = 500
    cfg.grad_clip = 1.0
    cfg.mixed_precision = "bf16"     # "bf16", "fp16", or "none"

    # ── Logging / checkpointing ──────────────────────────────────────────
    cfg.log_interval = 10
    cfg.eval_interval = 500
    cfg.viz = False
    cfg.viz_interval = 500              # log image+action viz every N steps
    cfg.save_interval = 2000
    cfg.save_dir = "/mnt/weka/zhougrp/octo_wm_cast"
    cfg.wandb_project = "cast-world-model"

    return cfg


if __name__ == "__main__":
    cfg = get_config("dino")
    print(cfg)
