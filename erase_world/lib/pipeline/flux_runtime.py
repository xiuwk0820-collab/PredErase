"""FLUX Klein inpaint: pure-noise mask init + staged CFG via mutable guidance_scale."""
from __future__ import annotations

import math
from typing import Any

import torch
from PIL import Image


class MutableGuidanceScale:
    """Float-like box so diffusers CFG loop reads updated scale from callback."""

    __slots__ = ("val",)

    def __init__(self, val: float) -> None:
        self.val = float(val)

    def __float__(self) -> float:
        return self.val

    def __gt__(self, other: Any) -> bool:
        return self.val > float(other)

    def __ge__(self, other: Any) -> bool:
        return self.val >= float(other)

    def __lt__(self, other: Any) -> bool:
        return self.val < float(other)

    def __le__(self, other: Any) -> bool:
        return self.val <= float(other)

    def __eq__(self, other: Any) -> bool:
        return self.val == float(other)

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, torch.Tensor):
            return other * self.val
        if isinstance(other, (int, float)):
            return self.val * other
        return NotImplemented

    def __rmul__(self, other: Any) -> Any:
        return self.__mul__(other)

    def __repr__(self) -> str:
        return f"MutableGuidanceScale({self.val})"


def resolve_flux_hw(pipe: Any, image: Image.Image) -> tuple[int, int]:
    multiple_of = pipe.vae_scale_factor * 2
    raw_w, raw_h = image.size
    if raw_h * raw_w > 1024 * 1024:
        scale = math.sqrt(1024 * 1024 / (raw_h * raw_w))
        raw_w, raw_h = int(raw_w * scale), int(raw_h * scale)
    w = (raw_w // multiple_of) * multiple_of
    h = (raw_h // multiple_of) * multiple_of
    return h, w


def precompute_packed_mask(
    pipe: Any,
    mask_image: Image.Image,
    height: int,
    width: int,
    *,
    batch_size: int = 1,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    mask_condition = pipe.mask_processor.preprocess(
        mask_image, height=height, width=width, resize_mode="default"
    )
    return pipe.prepare_mask_latents(
        mask_condition, batch_size, 1, height, width, dtype, device
    )


def patch_prepare_latents_pure_noise(
    pipe: Any,
    mask_packed: torch.Tensor,
) -> Any:
    """Return (original_prepare, wrapper) that replaces mask-region anchor with noise."""
    original = pipe.prepare_latents

    def wrapper(*args, **kwargs):
        latents, noise, packed_img, img_enc, ids = original(*args, **kwargs)
        m = mask_packed.to(device=packed_img.device, dtype=packed_img.dtype)
        packed_img = packed_img * (1.0 - m) + noise * m
        latents = latents * (1.0 - m) + noise * m
        return latents, noise, packed_img, img_enc, ids

    return original, wrapper
