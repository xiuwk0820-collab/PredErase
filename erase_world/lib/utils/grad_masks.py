"""Soft gradient regions + structure-line awareness (pixel ops only, no extra models)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _mask_bchw(mask: torch.Tensor) -> torch.Tensor:
    """Normalize mask to [B, 1, H, W]."""
    if mask.ndim == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.ndim == 3:
        return mask.unsqueeze(1)
    if mask.ndim == 4 and mask.shape[1] == 1:
        return mask
    if mask.ndim == 4:
        return mask[:, :1]
    raise ValueError(f"unexpected mask shape {tuple(mask.shape)}")


def feather_mask(mask: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Gaussian feather on [B,1,H,W] or [B,H,W] mask in [0,1] (3–5 px typical)."""
    if sigma <= 0:
        return _mask_bchw(mask).clamp(0.0, 1.0)
    m = _mask_bchw(mask)
    k = max(3, int(sigma * 2) | 1)
    pad = k // 2
    return F.avg_pool2d(m.float(), kernel_size=k, stride=1, padding=pad).clamp(0.0, 1.0)


def erode_mask(mask: torch.Tensor, px: int = 3) -> torch.Tensor:
    """Morphological erosion via min-pool."""
    if px <= 0:
        return _mask_bchw(mask).clamp(0.0, 1.0)
    m = _mask_bchw(mask)
    k = px * 2 + 1
    eroded = -F.max_pool2d(-m.float(), kernel_size=k, stride=1, padding=px)
    return eroded.clamp(0.0, 1.0)


def dilate_mask(mask: torch.Tensor, px: int = 3) -> torch.Tensor:
    if px <= 0:
        return _mask_bchw(mask).clamp(0.0, 1.0)
    m = _mask_bchw(mask)
    k = px * 2 + 1
    return F.max_pool2d(m.float(), kernel_size=k, stride=1, padding=px).clamp(0.0, 1.0)


def object_interior_mask(object_mask: torch.Tensor, erosion_px: int = 4) -> torch.Tensor:
    """Struct loss only on object interior — skip hard mask rim."""
    return erode_mask(object_mask, erosion_px)


def canny_edge_map(image_bchw: torch.Tensor) -> torch.Tensor:
    """Sobel + local-max NMS approximation (Canny-like), normalized [B,1,H,W]."""
    g = image_bchw.float()
    if g.shape[1] > 1:
        g = g.mean(dim=1, keepdim=True)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=g.device,
        dtype=g.dtype,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    pad = F.pad(g, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(pad, kx)
    gy = F.conv2d(pad, ky)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    nms = F.max_pool2d(mag, kernel_size=3, stride=1, padding=1)
    edges = (mag >= nms * 0.98).float() * mag
    peak = edges.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    return (edges / peak).clamp(0.0, 1.0)


def jepa_struct_line_keep(
    image_bchw: torch.Tensor,
    edit_mask: torch.Tensor,
    *,
    band_px: int = 10,
    edge_thresh: float = 0.22,
    max_suppress: float = 0.95,
    fine_phase: bool = False,
    object_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multiplier in [0,1]: near mask-boundary strong lines → attenuate JEPA, let FLUX lead."""
    edges = canny_edge_map(image_bchw)
    strong = (edges > edge_thresh).float()
    m = _mask_bchw(edit_mask)
    boundary = (dilate_mask(m.squeeze(1), band_px) - erode_mask(m.squeeze(1), band_px)).clamp(
        0.0, 1.0
    )
    if boundary.ndim == 3:
        boundary = boundary.unsqueeze(1)
    zone = strong * boundary
    suppress = zone * (1.0 if fine_phase else max_suppress)
    keep = (1.0 - suppress).clamp(0.0, 1.0)
    if object_mask is not None and max_suppress > 0:
        om = _mask_bchw(object_mask)
        obj_zone = dilate_mask(om.squeeze(1), band_px + 4).clamp(0.0, 1.0)
        if obj_zone.ndim == 3:
            obj_zone = obj_zone.unsqueeze(1)
        keep = torch.maximum(keep, obj_zone)
    return keep


def edge_continuity_loss(
    image_bchw: torch.Tensor,
    edit_mask: torch.Tensor,
    band_px: int = 4,
) -> torch.Tensor:
    """Light pixel loss: smooth gradient magnitude across mask boundary ring."""
    g = image_bchw.float().mean(dim=1, keepdim=True)
    gx = F.pad(g[:, :, :, 1:] - g[:, :, :, :-1], (0, 1, 0, 0))
    gy = F.pad(g[:, :, 1:, :] - g[:, :, :-1, :], (0, 0, 0, 1))
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    m = _mask_bchw(edit_mask)
    inner = erode_mask(m.squeeze(1), band_px)
    outer = dilate_mask(m.squeeze(1), band_px)
    if inner.ndim == 3:
        inner = inner.unsqueeze(1)
    if outer.ndim == 3:
        outer = outer.unsqueeze(1)
    ring_out = (outer - inner).clamp(0.0, 1.0)
    ring_in = (inner - erode_mask(m.squeeze(1), band_px * 2)).clamp(0.0, 1.0)
    if ring_in.ndim == 3:
        ring_in = ring_in.unsqueeze(1)
    denom_o = ring_out.sum().clamp(min=1e-6)
    denom_i = ring_in.sum().clamp(min=1e-6)
    mean_o = (mag * ring_out).sum() / denom_o
    mean_i = (mag * ring_in).sum() / denom_i
    return (mean_o - mean_i).pow(2)


def struct_region_weights(
    object_mask: torch.Tensor,
    image_bchw: torch.Tensor,
    *,
    erosion_px: int = 4,
    edge_thresh: float = 0.35,
    edge_suppress: float = 0.85,
) -> torch.Tensor:
    """M_obj interior with Canny edge suppression at architectural seams."""
    interior = object_interior_mask(object_mask, erosion_px)
    if edge_suppress <= 0:
        return interior.squeeze(1) if interior.ndim == 4 else interior
    edges = canny_edge_map(image_bchw)
    suppress = (edges > edge_thresh).float() * edge_suppress
    w = interior * (1.0 - suppress)
    return w.squeeze(1) if w.ndim == 4 else w
