"""Packed-latent mask P for FLUX.2 fill; UpdateAndLock edits only P."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .lib.models.flux_latent import (
    decode_packed_latents,
    resize_mask_to_image,
    resize_mask_to_latent_spatial,
)
from .lib.pipeline.flux_runtime import (
    MutableGuidanceScale,
    precompute_packed_mask,
    resolve_flux_hw,
)


def packed_edit_mask(
    pipe: Any,
    mask_image: Image.Image,
    height: int,
    width: int,
    *,
    batch_size: int = 1,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute packed editable mask P from M_flux (paper packing step)."""
    return precompute_packed_mask(
        pipe,
        mask_image,
        height,
        width,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )


def update_and_lock(
    z: torch.Tensor,
    z_native: torch.Tensor,
    packed_mask_p: torch.Tensor,
) -> torch.Tensor:
    """UpdateAndLock: keep JEPA update only on editable packed coords P.

    Non-edit coordinates are pinned back to the *current* native Fill state
    ``z_native`` (not a permanent lock to the original source latent).
    """
    p = packed_mask_p.to(device=z.device, dtype=z.dtype)
    if p.ndim < z.ndim:
        while p.ndim < z.ndim:
            p = p.unsqueeze(-1)
        # broadcast over channel dims if needed
        if p.shape != z.shape:
            p = p.expand_as(z)
    return z * p + z_native * (1.0 - p)


__all__ = [
    "MutableGuidanceScale",
    "decode_packed_latents",
    "packed_edit_mask",
    "precompute_packed_mask",
    "resolve_flux_hw",
    "resize_mask_to_image",
    "resize_mask_to_latent_spatial",
    "update_and_lock",
]
