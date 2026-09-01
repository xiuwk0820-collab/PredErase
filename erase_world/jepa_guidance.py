"""I-JEPA guidance: E_target cache, L_align, projected latent update (paper Alg.1)."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .lib.core.types import GuidanceCache, GuidanceConfig, GuidanceState
from .lib.models.ijepa_official import OfficialIJEPAModel, build_visible_fill
from .lib.strategy.jepa import (
    JEPAGuidanceStrategy,
    NoOpStrategy,
    _alignment_loss,
    _scale_grad,
)


def alignment_loss(
    e_curr: torch.Tensor,
    e_target: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """L_align = mean_{i in I_mask} ||u_i - v_i||_2^2 (MSE default)."""
    return _alignment_loss(e_curr, e_target, loss_type)


def cache_e_target(
    ijepa: OfficialIJEPAModel,
    image: torch.Tensor,
    object_mask: torch.Tensor,
) -> GuidanceCache:
    """GrayFill → frozen I-JEPA → E_target on object patches only."""
    strategy = JEPAGuidanceStrategy(ijepa)
    return strategy.precompute(image, object_mask)


class JEPAGuidance:
    """Thin wrapper around the internal JEPA guidance strategy."""

    def __init__(self, ijepa: OfficialIJEPAModel) -> None:
        self.strategy = JEPAGuidanceStrategy(ijepa)

    def precompute(self, image: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> GuidanceCache:
        return self.strategy.precompute(image, mask, **kwargs)

    def should_guide(self, step_idx: int, num_steps: int, cfg: GuidanceConfig) -> bool:
        return self.strategy.should_guide(step_idx, num_steps, cfg)

    def guide_latents(
        self,
        latents: torch.Tensor,
        cache: GuidanceCache,
        image: torch.Tensor,
        mask: torch.Tensor,
        cfg: GuidanceConfig,
        state: GuidanceState,
        pipe: Any,
    ) -> tuple[torch.Tensor, Optional[float]]:
        """Decode preview → L_align → z_tilde = z_pin - eta * grad (masked to edit region)."""
        return self.strategy.guide_latents(latents, cache, image, mask, cfg, state, pipe)


__all__ = [
    "JEPAGuidance",
    "JEPAGuidanceStrategy",
    "NoOpStrategy",
    "OfficialIJEPAModel",
    "alignment_loss",
    "build_visible_fill",
    "cache_e_target",
    "_scale_grad",
]
