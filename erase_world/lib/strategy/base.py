from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from ..core.types import GuidanceCache, GuidanceConfig, GuidanceState, StrategyMetrics


class InpaintGuidanceStrategy(ABC):
    """Pluggable training-free guidance — models are backends, strategy is the product."""

    @abstractmethod
    def precompute(self, image: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> GuidanceCache:
        """Module 1: one-time target representation from original image + mask."""

    @abstractmethod
    def should_guide(self, step_idx: int, num_steps: int, cfg: GuidanceConfig) -> bool:
        ...

    @abstractmethod
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
        """Module 3: return corrected latents (same shape) and optional loss."""

    def reset_metrics(self) -> StrategyMetrics:
        return StrategyMetrics()
