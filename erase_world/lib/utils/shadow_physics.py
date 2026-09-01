from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def _local_mean(gray: torch.Tensor, kernel: int = 31) -> torch.Tensor:
    k = max(3, kernel | 1)
    pad = k // 2
    return F.avg_pool2d(gray.unsqueeze(0), k, stride=1, padding=pad).squeeze(0)


def _infer_cast_direction(
    gray: torch.Tensor,
    m: torch.Tensor,
    obj_x: torch.Tensor,
    obj_y: torch.Tensor,
    h: int,
    w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick cast direction with the strongest nearby dark mass outside the object."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=gray.device, dtype=gray.dtype),
        torch.arange(w, device=gray.device, dtype=gray.dtype),
        indexing="ij",
    )
    outside = (1.0 - m[0, 0]).float()
    local = _local_mean(gray[0, 0], kernel=31).unsqueeze(0).unsqueeze(0)
    dark = ((local - gray).clamp(min=0.0) * outside.unsqueeze(0).unsqueeze(0))[0, 0]

    outside_np = outside.detach().cpu().numpy().astype(bool)
    dist = distance_transform_edt(outside_np)
    dist_t = torch.from_numpy(dist).to(gray.device, dtype=gray.dtype)
    near = (dist_t < min(h, w) * 0.45).float() * outside

    dx = xs - obj_x
    dy = ys - obj_y
    best_score = torch.tensor(-1.0, device=gray.device, dtype=gray.dtype)
    best_sx = torch.tensor(1.0, device=gray.device, dtype=gray.dtype)
    best_sy = torch.tensor(0.0, device=gray.device, dtype=gray.dtype)

    for k in range(16):
        angle = torch.tensor(2.0 * math.pi * k / 16.0, device=gray.device, dtype=gray.dtype)
        sx = torch.cos(angle)
        sy = torch.sin(angle)
        along = (dx * sx + dy * sy).clamp(min=0.0)
        dir_w = along * near * torch.exp(-along / max(min(h, w) * 0.28, 1.0))
        score = (dark * dir_w).sum()
        if score > best_score:
            best_score = score
            best_sx, best_sy = sx, sy

    if float(best_score) > 1e-4:
        norm = (best_sx * best_sx + best_sy * best_sy).sqrt().clamp(min=1e-6)
        return best_sx / norm, best_sy / norm

    quads = (
        (gray[0, 0, : h // 2, : w // 2].mean(), (-1.0, -1.0)),
        (gray[0, 0, : h // 2, w // 2 :].mean(), (1.0, -1.0)),
        (gray[0, 0, h // 2 :, : w // 2].mean(), (-1.0, 1.0)),
        (gray[0, 0, h // 2 :, w // 2 :].mean(), (1.0, 1.0)),
    )
    _, (lx, ly) = max(quads, key=lambda t: t[0])
    vx = lx - (obj_x - w * 0.5) / max(w, 1)
    vy = ly - (obj_y - h * 0.5) / max(h, 1)
    norm = (vx * vx + vy * vy).sqrt().clamp(min=1e-6)
    vx, vy = vx / norm, vy / norm
    return -vx, -vy


def estimate_cast_shadow_soft(
    image: torch.Tensor,
    mask: torch.Tensor,
    cast_ratio: float = 0.75,
    dark_delta: float = 0.03,
) -> torch.Tensor:
    """Soft shadow band outside object mask: physics prior (cast along light axis).

    Returns [1,1,H,W] in [0,1], zero inside the object mask.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] > 1:
        mask = mask[:, :1]

    b, _, h, w = image.shape
    m = (mask > 0.5).float()
    gray = image.mean(dim=1, keepdim=True)

    ys, xs = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    obj_y = (ys * m[0, 0]).sum() / m.sum().clamp(min=1.0)
    obj_x = (xs * m[0, 0]).sum() / m.sum().clamp(min=1.0)

    sx, sy = _infer_cast_direction(gray, m, obj_x, obj_y, h, w)

    dx = xs - obj_x
    dy = ys - obj_y
    downstream = (dx * sx + dy * sy) > 0.0
    perp = torch.abs(dx * (-sy) + dy * sx)
    spread = min(h, w) * 0.30
    lateral = torch.exp(-(perp ** 2) / (2 * spread ** 2))

    outside = (1.0 - m[0, 0]).detach().cpu().numpy().astype(bool)
    dist = distance_transform_edt(outside)
    max_len = min(h, w) * cast_ratio
    dist_t = torch.from_numpy(dist).to(image.device, dtype=image.dtype)
    falloff = torch.exp(-dist_t / max(max_len, 1.0))
    cast = (downstream.float() * lateral * falloff * (1.0 - m[0, 0])).unsqueeze(0).unsqueeze(0)

    local = _local_mean(gray[0, 0], kernel=31).unsqueeze(0).unsqueeze(0)
    darkness = ((local - gray).clamp(min=0.0) / max(dark_delta, 1e-6)).clamp(0.0, 1.0)
    # Geometry-first cast wedge; darkness only modulates confidence on textured floors.
    soft = (cast * (0.55 + 0.45 * darkness) * (1.0 - m)).clamp(0.0, 1.0)
    soft = F.max_pool2d(soft, kernel_size=11, stride=1, padding=5)
    return soft


def build_counterfactual_visible(
    image: torch.Tensor,
    mask: torch.Tensor,
    shadow_soft: torch.Tensor,
    strength: float = 0.85,
) -> torch.Tensor:
    """Lift cast shadows in visible region → counterfactual 'object never existed' context."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] > 1:
        mask = mask[:, :1]

    gray = image.mean(dim=1, keepdim=True)
    local = _local_mean(gray[0, 0], kernel=31).unsqueeze(0).unsqueeze(0)
    local_rgb = local.expand_as(image)
    w = (shadow_soft * (1.0 - mask) * strength).clamp(0.0, 1.0)
    return image * (1.0 - w) + local_rgb * w


def shadow_visible_patch_indices(
    shadow_soft: torch.Tensor,
    mask: torch.Tensor,
    visible_idx: torch.Tensor,
    grid: int,
    threshold: float = 0.20,
) -> torch.Tensor:
    """Visible JEPA patches overlapping the cast-shadow band."""
    if shadow_soft.ndim == 4:
        shadow_soft = shadow_soft[0, 0]
    if mask.ndim == 4:
        mask = mask[0, 0]
    elif mask.ndim == 3:
        mask = mask[0]

    patch_shadow = F.adaptive_max_pool2d(
        shadow_soft.unsqueeze(0).unsqueeze(0), (grid, grid)
    )[0, 0].reshape(-1)
    patch_mask = F.adaptive_max_pool2d(
        mask.float().unsqueeze(0).unsqueeze(0), (grid, grid)
    )[0, 0].reshape(-1)

    vis_mask = torch.zeros(grid * grid, dtype=torch.bool, device=shadow_soft.device)
    vis_mask[visible_idx] = True
    shadow_patches = (patch_shadow > threshold) & (patch_mask <= 0.5) & vis_mask
    return shadow_patches.nonzero(as_tuple=False).squeeze(-1)
