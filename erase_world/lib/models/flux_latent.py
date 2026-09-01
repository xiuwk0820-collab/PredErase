from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def decode_packed_latents(
    pipe: Any,
    latents_packed: torch.Tensor,
    latent_image_ids: torch.Tensor,
    *,
    park_transformer: bool = False,
) -> torch.Tensor:
    """FLUX.2 Klein packed latents -> image tensor [B,3,H,W] in [0,1], differentiable w.r.t. input."""
    flux_dev = latents_packed.device
    transformer = getattr(pipe, "transformer", None)
    text_encoder = getattr(pipe, "text_encoder", None)
    parked: list[tuple[Any, torch.device]] = []
    if park_transformer and torch.cuda.is_available() and flux_dev.type == "cuda":
        for mod in (transformer, text_encoder):
            if mod is not None and next(mod.parameters()).device.type == "cuda":
                parked.append((mod, next(mod.parameters()).device))
                mod.to("cpu")
        if parked:
            torch.cuda.empty_cache()

    try:
        unpacked = pipe._unpack_latents_with_ids(latents_packed, latent_image_ids)
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        bn_std = torch.sqrt(
            pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
        ).to(unpacked.device, unpacked.dtype)
        latents = unpacked * bn_std + bn_mean
        latents = pipe._unpatchify_latents(latents)
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
    finally:
        for mod, dev in parked:
            mod.to(dev)
        if parked and torch.cuda.is_available():
            torch.cuda.empty_cache()


def resize_mask_to_image(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """[1,1,H,W] or [H,W] mask -> [1,1,height,width], nearest."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.shape[1] > 1:
        mask = mask[:, :1]
    return F.interpolate(mask.float(), size=(height, width), mode="nearest")


def resize_mask_to_latent_spatial(mask: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    """[H,W] / [1,H,W] / [1,1,H,W] -> [1,1,latent_h,latent_w], nearest."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.shape[1] > 1:
        mask = mask[:, :1]
    return F.interpolate(mask.float(), size=(latent_h, latent_w), mode="nearest")
