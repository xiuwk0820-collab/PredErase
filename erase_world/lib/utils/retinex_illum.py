"""Retinex-style pixel illumination loss for cast-shadow removal (training-free).

Shadows alter illumination, not reflectance. JEPA targets are lighting-invariant;
this module matches decoded RGB statistics in the shadow band to a clean neighbor ring.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, gaussian_filter


def blur_mask_edges(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Soft shadow mask with blurred edges to avoid paste seams."""
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    soft = gaussian_filter(m, sigma=sigma)
    return np.clip(soft, 0.0, 1.0)


def build_neighbor_ring(
    edit_mask: np.ndarray,
    ring_width: int = 16,
) -> np.ndarray:
    """Visible ring just outside edit mask (object + shadow)."""
    m = edit_mask > 0.5 if edit_mask.dtype != np.bool_ else edit_mask
    struct = np.ones((ring_width * 2 + 1, ring_width * 2 + 1), dtype=bool)
    dilated = binary_dilation(m, structure=struct)
    ring = dilated & ~m
    return ring


def precompute_retinex_stats(
    image: torch.Tensor,
    edit_mask: torch.Tensor,
    shadow_mask: torch.Tensor,
    ring_width: int = 16,
    blur_sigma: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Cache neighbor RGB mean/std + blurred shadow mask for L_illum."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if edit_mask.ndim == 2:
        edit_mask = edit_mask.unsqueeze(0).unsqueeze(0)
    elif edit_mask.ndim == 3:
        edit_mask = edit_mask.unsqueeze(1)
    if shadow_mask.ndim == 2:
        shadow_mask = shadow_mask.unsqueeze(0).unsqueeze(0)
    elif shadow_mask.ndim == 3:
        shadow_mask = shadow_mask.unsqueeze(1)

    dev = image.device
    h, w = image.shape[-2:]
    edit_np = (edit_mask[0, 0].detach().cpu().numpy() > 0.5)
    sh_np = shadow_mask[0, 0].detach().cpu().numpy()
    if sh_np.max() <= 1.0:
        sh_hard = sh_np > 0.08
    else:
        sh_hard = sh_np > 20

    edit_union = edit_np | sh_hard
    ring = build_neighbor_ring(edit_union, ring_width=ring_width)

    rgb_np = (
        image[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    )
    if ring.sum() < 64:
        ring = ~edit_union

    neighbor = rgb_np[ring]
    n_mean = neighbor.mean(axis=0)
    n_std = neighbor.std(axis=0).clip(min=1e-3)

    shadow_blur = blur_mask_edges(sh_hard.astype(np.float32), sigma=blur_sigma)

    return {
        "neighbor_mean": torch.from_numpy(n_mean).view(1, 3, 1, 1).to(dev, dtype=image.dtype),
        "neighbor_std": torch.from_numpy(n_std).view(1, 3, 1, 1).to(dev, dtype=image.dtype),
        "shadow_blur": torch.from_numpy(shadow_blur[None, None]).to(dev, dtype=image.dtype),
        "shadow_hard": torch.from_numpy(sh_hard.astype(np.float32)[None, None]).to(
            dev, dtype=image.dtype
        ),
    }


def retinex_illumination_loss(
    image: torch.Tensor,
    shadow_mask: torch.Tensor,
    neighbor_mean: torch.Tensor,
    neighbor_std: torch.Tensor,
    std_weight: float = 0.35,
) -> torch.Tensor:
    """L_illum: shadow-band RGB mean/std match neighbor visible ring."""
    m = shadow_mask.clamp(0.0, 1.0)
    denom = m.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    region_mean = (image * m).sum(dim=(2, 3), keepdim=True) / denom
    l_mean = F.l1_loss(region_mean, neighbor_mean)

    region_var = ((image - region_mean).pow(2) * m).sum(dim=(2, 3), keepdim=True) / denom
    region_std = region_var.sqrt().clamp(min=1e-6)
    l_std = F.l1_loss(region_std, neighbor_std)
    return l_mean + std_weight * l_std


def lowfreq_illum_loss(
    image: torch.Tensor,
    shadow_mask: torch.Tensor,
    kernel: int = 15,
) -> torch.Tensor:
    """Optional: suppress low-frequency brightness step at shadow boundary."""
    pad = kernel // 2
    m = shadow_mask.clamp(0.0, 1.0)
    gray = image.mean(dim=1, keepdim=True)
    low = F.avg_pool2d(F.pad(gray, (pad, pad, pad, pad), mode="reflect"), kernel, stride=1)
    low = low[..., : gray.shape[2], : gray.shape[3]]
    denom = m.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    band_mean = (low * m).sum(dim=(2, 3), keepdim=True) / denom
    outside = (1.0 - m).clamp(0.0, 1.0)
    out_denom = outside.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    out_mean = (low * outside).sum(dim=(2, 3), keepdim=True) / out_denom
    return F.l1_loss(band_mean, out_mean)
