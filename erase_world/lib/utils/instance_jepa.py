"""Instance-level JEPA patch indexing: separate object vs attached cast-shadow."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_single_channel(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 4:
        return mask[0, 0]
    if mask.ndim == 3:
        return mask[0]
    return mask


def instance_shadow_patch_indices(
    object_mask: torch.Tensor,
    shadow_soft: torch.Tensor,
    grid: int,
    threshold: float = 0.08,
) -> torch.Tensor:
    """Patches belonging to THIS instance's cast shadow (not the object body)."""
    obj = _to_single_channel(object_mask).float()
    sh = _to_single_channel(shadow_soft).float().to(obj.device)

    patch_sh = F.adaptive_max_pool2d(
        sh.float().unsqueeze(0).unsqueeze(0), (grid, grid)
    )[0, 0].reshape(-1)
    patch_obj = F.adaptive_max_pool2d(
        obj.float().unsqueeze(0).unsqueeze(0), (grid, grid)
    )[0, 0].reshape(-1)

    shadow_idx = (patch_sh > threshold) & (patch_obj <= 0.5)
    return shadow_idx.nonzero(as_tuple=False).squeeze(-1).to(obj.device)
