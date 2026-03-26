import torch
import torch.nn as nn
from octo.model.components.jax_pt import FromJaxModel
from typing import Optional, Dict, Any

class VJEPA2EncoderPt(nn.Module, FromJaxModel):
    """V-JEPA2 Vision Encoder for Octo (loaded from torch.hub)."""

    def __init__(
        self,
        model_name: str = "vjepa2_vit_large",   # "vjepa2_vit_huge", "vjepa2_vit_giant", "vjepa2_vit_giant_384"
        freeze: bool = True,
        img_norm_type: str = "imagenet",
        force_resolution: Optional[int] = None,  # set to 256 or 384 to override
        repo: str = "facebookresearch/vjepa2",
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.img_norm_type = img_norm_type
        self.repo = repo

        # Load model from torch.hub
        hub_result = torch.hub.load(repo, model_name)
        
        # Handle case where torch.hub.load returns a tuple (model, extras)
        if isinstance(hub_result, tuple):
            self.vjepa2_model = hub_result[0]
        else:
            self.vjepa2_model = hub_result

        # Try to infer embed dim (varies by checkpoint)
        self.num_features = getattr(self.vjepa2_model, "embed_dim", None)

        # Default resolution from checkpoint name/table
        if force_resolution is not None:
            self.resolution = int(force_resolution)
        else:
            self.resolution = 384 if model_name.endswith("_384") else 256

        # Freeze weights if requested
        if self.freeze:
            for p in self.vjepa2_model.parameters():
                p.requires_grad = False
            self.vjepa2_model.eval()

    def pt_to_jax_args_map(self):
        return {}

    def normalize_images(self, img: torch.Tensor) -> torch.Tensor:
        """
        Input:
          - (B, C, H, W) in [0,255] OR (B, H, W, C)
          - C can be 3 or 6 (if 6, we keep the first 3 channels by default)
        Output:
          - (B, 3, resolution, resolution), float32, ImageNet-normalized if enabled
        """
        # Ensure (B, C, H, W)
        if img.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got shape {tuple(img.shape)}")

        if img.shape[1] not in (3, 6):  # likely (B, H, W, C)
            img = img.permute(0, 3, 1, 2)

        # If stacked frames (6 channels), keep first RGB frame (simple + safe)
        if img.shape[1] == 6:
            img = img[:, :3, :, :]

        # Scale to [0,1]
        img = img.float() / 255.0

        # ImageNet normalization (common for ViTs)
        if self.img_norm_type == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
            img = (img - mean) / std

        # Resize to model resolution (256 or 384)
        if img.shape[2] != self.resolution or img.shape[3] != self.resolution:
            img = nn.functional.interpolate(img, size=(self.resolution, self.resolution),
                                            mode="bilinear", align_corners=False)
        return img

    def _extract_patch_tokens(self, out: Any) -> torch.Tensor:
        """
        Tries to pull patch tokens from common V-JEPA / ViT output formats.
        Returns tensor shaped (B, N, C).
        """
        # If dict-like output
        if isinstance(out, dict):
            # Common key candidates
            for k in ["x_norm_patchtokens", "patch_tokens", "x_patchtokens", "tokens", "x"]:
                if k in out and isinstance(out[k], torch.Tensor):
                    t = out[k]
                    # If includes CLS, remove it if needed (heuristic: N+1 tokens)
                    # We'll only remove CLS if it looks like square+1.
                    if t.ndim == 3:
                        return t
            raise KeyError(f"Could not find patch tokens in dict keys: {list(out.keys())}")

        # If tuple/list output
        if isinstance(out, (tuple, list)):
            # pick the first tensor that looks like (B, N, C)
            for item in out:
                if isinstance(item, torch.Tensor) and item.ndim == 3:
                    return item
            raise ValueError("Tuple/list output did not contain a (B,N,C) tensor.")

        # If raw tensor
        if isinstance(out, torch.Tensor):
            if out.ndim == 3:
                return out
            raise ValueError(f"Unexpected tensor output shape: {tuple(out.shape)}")

        raise TypeError(f"Unsupported model output type: {type(out)}")

    def forward(self, observations: torch.Tensor, train: bool = True, **kwargs) -> torch.Tensor:
        """
        Args:
            observations: (B, C, H, W) uint8-ish in [0,255] or float, or (B, H, W, C)

        Returns:
            patch_map: (B, embed_dim, H_patches, W_patches)
                - if resolution=256 and patch=16 => (B, C, 16, 16)
                - if resolution=384 and patch=16 => (B, C, 24, 24)
        """
        x = self.normalize_images(observations)
        
        # V-JEPA expects 5D input (B, C, T, H, W) for video
        # The model uses Conv3D with temporal kernel size 2, so we need at least 2 frames
        # Duplicate the frame: (B, C, H, W) -> (B, C, 2, H, W)
        x = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)

        if self.freeze:
            self.vjepa2_model.eval()

        with torch.set_grad_enabled((not self.freeze) and train):
            # Different repos expose different forward APIs; try common patterns.
            if hasattr(self.vjepa2_model, "forward_features"):
                out = self.vjepa2_model.forward_features(x)
            else:
                out = self.vjepa2_model(x)

            tokens = self._extract_patch_tokens(out)  # expect (B, N, C)

        B, N, C = tokens.shape

        # If tokens includes CLS token, remove it when it looks like (square + 1)
        grid = int((N - 1) ** 0.5)
        if grid * grid == (N - 1):
            patch_tokens = tokens[:, 1:, :]  # drop CLS
            N = patch_tokens.shape[1]
        else:
            patch_tokens = tokens

        grid_size = int(N ** 0.5)
        if grid_size * grid_size != N:
            raise ValueError(
                f"Num patches {N} is not a perfect square; "
                f"check resolution ({self.resolution}) vs patch size."
            )

        patch_map = patch_tokens.reshape(B, grid_size, grid_size, C).permute(0, 3, 1, 2)
        # patch_map: (B, C, grid, grid)

        # Save feature dim if not set
        if self.num_features is None:
            self.num_features = C

        return patch_map