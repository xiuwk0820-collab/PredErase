"""Main PredErase inference pipeline (paper Alg.1 orchestration)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from PIL import Image

from .flux_fill import DEFAULT_MODEL_ID, build_strategy, load_flux_fill, run_flux_fill
from .lib.core.config import build_guidance_config
from .lib.core.methods import ERASE_WORLD, FLUX_FILL_NATIVE, normalize_method
from .masks import build_m_flux
from .prompts import get_prompts
from .utils import PAPER_SEEDS, average_images, parse_seeds, resize_longest, resolve_devices


class EraseWorldPipeline:
    """Training-free object-and-effect removal.

    Steps (Alg.1):
      1. GrayFill + JEPA E_target cache (inside strategy.precompute)
      2. M_flux = dilate_r(M_obj ∪ M_shadow)
      3. Source prefill Encode(I ⊙ (1 - M_flux)) when enabled
      4. T frozen FLUX.2-klein-4B Fill steps
      5. JEPA guidance on selected steps (eta = guidance_scale)
      6. Decode
    """

    def __init__(
        self,
        config: dict[str, Any] | str | Path,
        *,
        flux_device: str | None = None,
        ijepa_device: str | None = None,
        load_pipe: bool = True,
    ) -> None:
        if isinstance(config, (str, Path)):
            with open(config, encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = config

        flux_cfg = self.cfg.get("flux", {})
        dev_cfg = self.cfg.get("devices", {})
        self.flux_dev, self.ijepa_dev = resolve_devices(
            flux_device or dev_cfg.get("flux"),
            ijepa_device or dev_cfg.get("ijepa"),
        )
        self.method = normalize_method(self.cfg.get("method", ERASE_WORLD))
        self.pipe = None
        if load_pipe:
            model_id = flux_cfg.get("model_id", DEFAULT_MODEL_ID)
            self.pipe = load_flux_fill(model_id, device=self.flux_dev)

        self.strategy = build_strategy(
            self.method,
            self.cfg.get("ijepa", {}),
            ijepa_device=self.ijepa_dev,
        )
        self.guidance_cfg = build_guidance_config(self.cfg.get("guidance", {}))

    def prepare_mask(
        self,
        image: Image.Image,
        object_mask: Image.Image,
    ) -> Image.Image:
        flux_cfg = self.cfg.get("flux", {})
        r = int(self.cfg.get("guidance", {}).get("mask_run_dilate_px", flux_cfg.get("dilate_r", 4)))
        use_shadow = bool(flux_cfg.get("use_contact_shadow", True))
        if self.method == FLUX_FILL_NATIVE and not flux_cfg.get("force_m_flux", False):
            # Pure native can still use M_obj only unless force_m_flux
            if not flux_cfg.get("use_m_flux_for_native", True):
                return object_mask.convert("L")
        return build_m_flux(image, object_mask, dilate_r=r, use_shadow=use_shadow)

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
        *,
        seed: int | None = None,
        prompt: str | None = None,
        max_side: int | None = None,
    ) -> Image.Image:
        if self.pipe is None:
            raise RuntimeError("FLUX pipeline not loaded")
        flux_cfg = dict(self.cfg.get("flux", {}))
        max_side = int(max_side if max_side is not None else flux_cfg.get("max_side", 768))
        image = resize_longest(image.convert("RGB"), max_side)
        mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)

        edit_mask = self.prepare_mask(image, mask)
        if prompt is None:
            prompt_variant = flux_cfg.get("prompt_variant", "full")
            prompt, _neg = get_prompts(prompt_variant)
            if flux_cfg.get("prompt"):
                prompt = str(flux_cfg["prompt"])

        seed = int(flux_cfg.get("seed", PAPER_SEEDS[0]) if seed is None else seed)
        # Paper w_cfg=3.5; fall back to guidance_scale_flux if set
        w_cfg = float(flux_cfg.get("guidance_scale_flux", flux_cfg.get("w_cfg", 3.5)))

        # Source prefill is handled inside run_inpaint when source_prefill=True
        # by filling the FLUX source image before encode (see texture_harmonize).
        source = image
        if flux_cfg.get("source_prefill", True):
            from .lib.utils.texture_harmonize import fill_flux_source
            import numpy as np

            source = fill_flux_source(
                image,
                np.asarray(edit_mask),
                ring_px=int(flux_cfg.get("source_prefill_ring_px", 24)),
            )
        elif flux_cfg.get("gray_prefill", False):
            # w/o Prefill ablation: gray-fill M_flux then encode
            from .masks import gray_fill

            source = gray_fill(image, edit_mask)

        return run_flux_fill(
            self.pipe,
            image=source,
            mask=edit_mask,
            prompt=prompt,
            strategy=self.strategy,
            guidance_cfg=self.guidance_cfg,
            num_inference_steps=int(flux_cfg.get("num_inference_steps", 14)),
            guidance_scale=w_cfg,
            strength=float(flux_cfg.get("strength", 1.0)),
            seed=seed,
            device=self.flux_dev,
            padding_mask_crop=flux_cfg.get("padding_mask_crop"),
        )

    def run_seed_averaged(
        self,
        image: Image.Image,
        mask: Image.Image,
        seeds: list[int] | str | None = None,
        **kwargs: Any,
    ) -> Image.Image:
        seeds_list = parse_seeds(seeds)
        outs = [self(image, mask, seed=s, **kwargs) for s in seeds_list]
        return average_images(outs)


def run_erase_world(
    image: str | Path | Image.Image,
    mask: str | Path | Image.Image,
    *,
    config: str | Path | dict[str, Any] = "configs/default.yaml",
    output: str | Path | None = None,
    seed: int | None = None,
    seed_average: bool = False,
) -> Image.Image:
    """Convenience single-image entry."""
    pipe = EraseWorldPipeline(config)
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    if not isinstance(mask, Image.Image):
        mask = Image.open(mask).convert("L")
    if seed_average:
        out = pipe.run_seed_averaged(image, mask, seeds=None if seed is None else [seed])
    else:
        out = pipe(image, mask, seed=seed)
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        out.save(output)
    return out
