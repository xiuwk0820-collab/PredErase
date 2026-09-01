"""Unified causal weight field W_causal from ori vs counterfactual repr diff."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_i_vis(image: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
    """I_vis = I * (1 - M_obj): occlude object only."""
    if object_mask.ndim == 2:
        m = object_mask.unsqueeze(0).unsqueeze(0)
    elif object_mask.ndim == 3:
        m = object_mask.unsqueeze(1)
    else:
        m = object_mask
    m = (m > 0.5).float().to(image.device, dtype=image.dtype)
    if m.shape[-2:] != image.shape[-2:]:
        m = F.interpolate(m, size=image.shape[-2:], mode="nearest")
    return image * (1.0 - m)


def _gaussian_smooth_2d(w: torch.Tensor, sigma_px: float) -> torch.Tensor:
    if sigma_px <= 0:
        return w
    k = max(3, int(sigma_px * 2) | 1)
    pad = k // 2
    return F.avg_pool2d(w, kernel_size=k, stride=1, padding=pad)


def build_w_causal_from_repr_diff(
    diff_struct: torch.Tensor,
    diff_light: torch.Tensor,
    object_mask: torch.Tensor,
    grid: int,
    *,
    struct_weight: float = 0.55,
    light_weight: float = 0.45,
    smooth_sigma_px: float = 18.0,
) -> torch.Tensor:
    """W_causal in [0,1]: repr diff map + M_obj=1 + isotropic Gaussian smooth."""
    diff = struct_weight * diff_struct + light_weight * diff_light
    diff_grid = diff.reshape(grid, grid)
    d_min = diff_grid.min()
    d_max = diff_grid.max()
    w_patch = (diff_grid - d_min) / (d_max - d_min + 1e-6)

    if object_mask.ndim == 4:
        m = object_mask[:, :1]
    elif object_mask.ndim == 3:
        m = object_mask.unsqueeze(1)
    else:
        m = object_mask.unsqueeze(0).unsqueeze(0)
    m = (m > 0.5).float()
    h, w = m.shape[-2], m.shape[-1]

    w_up = F.interpolate(w_patch.view(1, 1, grid, grid), size=(h, w), mode="bilinear", align_corners=False)
    w_up = torch.maximum(w_up, m)
    w_smooth = _gaussian_smooth_2d(w_up, smooth_sigma_px)
    return w_smooth.clamp(0.0, 1.0)


def patch_weights_from_guide(
    w_guide: torch.Tensor,
    grid: int,
) -> torch.Tensor:
    """Downsample spatial weight field to I-JEPA patch grid."""
    if w_guide.ndim == 3:
        w_guide = w_guide.unsqueeze(0)
    if w_guide.ndim == 4 and w_guide.shape[1] != 1:
        w_guide = w_guide.mean(dim=1, keepdim=True)
    wp = F.adaptive_avg_pool2d(w_guide.float(), (grid, grid))
    return wp.reshape(-1)


def build_guide_weight_field(
    object_mask: torch.Tensor,
    *,
    decay_down: float = 0.6,
    decay_side: float = 0.2,
) -> torch.Tensor:
    """Legacy directional W_guide (deprecated — use build_w_causal_from_repr_diff)."""
    del decay_down, decay_side
    if object_mask.ndim == 4:
        m = object_mask[0, 0]
    elif object_mask.ndim == 3:
        m = object_mask[0]
    else:
        m = object_mask
    h, w = m.shape[-2], m.shape[-1]
    obj = (m > 0.5).float()
    return obj.unsqueeze(0).unsqueeze(0)
