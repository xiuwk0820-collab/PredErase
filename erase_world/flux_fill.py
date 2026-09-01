"""FLUX.2-klein-4B mask-conditioned Fill wrapper (frozen weights)."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .lib.pipeline.flux_runtime import MutableGuidanceScale, resolve_flux_hw
from .lib.pipeline.inpaint import build_strategy, run_inpaint
from .lib.core.config import build_guidance_config
from .lib.core.types import InpaintInputs


DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def load_flux_fill(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Any:
    """Load frozen FLUX.2-klein-4B inpaint pipeline from Hugging Face / local path."""
    from diffusers import Flux2KleinInpaintPipeline

    pipe = Flux2KleinInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    return pipe


def run_flux_fill(
    pipe: Any,
    *,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    strategy: Any,
    guidance_cfg: Any,
    num_inference_steps: int = 14,
    guidance_scale: float = 3.5,
    strength: float = 1.0,
    seed: int = 22,
    device: str | None = None,
    **kwargs: Any,
) -> Image.Image:
    """Run T flow-matching Fill steps with optional JEPA callback guidance."""
    if device is None:
        device = str(pipe.device) if hasattr(pipe, "device") else "cuda:0"
    images, _ = run_inpaint(
        pipe=pipe,
        inputs=InpaintInputs(image=torch.empty(0), mask=torch.empty(0), prompt=prompt),
        strategy=strategy,
        guidance_cfg=guidance_cfg,
        mask_image=mask,
        source_image=image,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        strength=float(strength),
        generator=torch.Generator(device=device).manual_seed(int(seed)),
        **kwargs,
    )
    return images[0]


__all__ = [
    "DEFAULT_MODEL_ID",
    "MutableGuidanceScale",
    "build_guidance_config",
    "build_strategy",
    "load_flux_fill",
    "resolve_flux_hw",
    "run_flux_fill",
    "run_inpaint",
]
