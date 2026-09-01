"""SD-XL spatial latents -> RGB, differentiable w.r.t. latents."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch


@contextmanager
def sdxl_vae_float32(pipe: Any) -> Iterator[None]:
    """Hold VAE in float32 across a full forward+backward guidance step."""
    vae = pipe.vae
    vae_dtype = next(vae.parameters()).dtype
    was_training = vae.training
    if vae_dtype != torch.float32:
        vae.to(dtype=torch.float32)
    try:
        yield
    finally:
        if vae_dtype != torch.float32:
            vae.to(dtype=vae_dtype)
        vae.train(was_training)


def enable_sdxl_vae_force_upcast(pipe: Any) -> None:
    """Let diffusers encode/decode cast image↔VAE dtypes correctly."""
    if hasattr(pipe.vae, "config"):
        pipe.vae.config.force_upcast = True


def decode_sdxl_latents(pipe: Any, latents: torch.Tensor) -> torch.Tensor:
    """Decode [B,4,H,W] SD-XL latents to [B,3,H_img,W_img] in [0,1].

    Keeps gradient flow through the VAE. Call inside ``sdxl_vae_float32`` when
    ``latents.requires_grad`` so weights stay float32 until backward finishes.
    """
    vae = pipe.vae
    scaling = float(vae.config.scaling_factor)
    param_dtype = next(vae.parameters()).dtype
    z = (latents / scaling).to(dtype=param_dtype)
    image = vae.decode(z, return_dict=False)[0]
    return (image.float() / 2 + 0.5).clamp(0, 1)
