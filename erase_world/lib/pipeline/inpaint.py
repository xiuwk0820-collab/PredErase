from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from ..core.config import resolve_path
from ..core.types import GuidanceCache, GuidanceConfig, GuidanceState, InpaintInputs
from ..core.methods import ERASE_WORLD, LEGACY_ERASE_WORLD, is_erase_world, normalize_method
from ..strategy.base import InpaintGuidanceStrategy
from ..strategy.jepa import JEPAGuidanceStrategy, NoOpStrategy
from .flux_runtime import MutableGuidanceScale


def build_strategy(
    method: str,
    ijepa_cfg: dict,
    ijepa_device: str,
) -> InpaintGuidanceStrategy:
    method = normalize_method(method)
    if method in ("flux_fill_native", "noop"):
        return NoOpStrategy()
    if is_erase_world(method):
        from ..models.ijepa_official import OfficialIJEPAModel
        from ..models.ijepa_weights import ensure_ijepa_checkpoint

        ckpt = ijepa_cfg.get("checkpoint")
        if ckpt in (None, "auto", "official"):
            ckpt = ensure_ijepa_checkpoint(ijepa_cfg.get("cache_dir"))
        elif ckpt == "hf":
            ckpt = None
        elif ckpt:
            # erase_world/lib/pipeline -> repo root
            root = Path(__file__).resolve().parents[3]
            ckpt = resolve_path(root, str(ckpt))

        ijepa = OfficialIJEPAModel(
            model_name=ijepa_cfg.get("model_name", "facebook/ijepa_vith14_1k"),
            checkpoint_path=ckpt,
            img_size=int(ijepa_cfg.get("img_size", 224)),
            device=ijepa_device,
        )
        if not ijepa.has_predictor:
            raise RuntimeError(
                "Official I-JEPA predictor weights required for erase_world. "
                "Run: python scripts/download_ijepa.py "
                f"(place IN1K-vit.h.14-300e.pth.tar under ~/.cache/erase-world/)"
            )
        return JEPAGuidanceStrategy(ijepa)
    raise NotImplementedError(f"method={method!r} not implemented yet")


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    import numpy as np

    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    import numpy as np

    arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy((arr > 0.5).astype(np.float32))


def run_inpaint(
    pipe: Any,
    inputs: InpaintInputs,
    strategy: InpaintGuidanceStrategy,
    guidance_cfg: GuidanceConfig,
    mask_image: Image.Image,
    source_image: Image.Image,
    guidance_source_image: Image.Image | None = None,
    object_mask_image: Image.Image | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 1.0,
    strength: float = 1.0,
    generator: Optional[torch.Generator] = None,
    *,
    pure_noise_mask_init: bool = False,
    guidance_scale_early: float | None = None,
    guidance_scale_late: float | None = None,
    guidance_switch_step: int = 7,
    **pipe_kwargs: Any,
) -> tuple[list[Image.Image], GuidanceCache]:
    """Run Flux2KleinInpaintPipeline with optional JEPA strategy hook."""

    guide_image = guidance_source_image or source_image
    if guide_image.size != source_image.size:
        guide_image = guide_image.resize(source_image.size, Image.Resampling.LANCZOS)
    image_t = _pil_to_tensor(guide_image)
    mask_t = _mask_to_tensor(mask_image)
    object_mask = object_mask_image or mask_image
    if object_mask.size != mask_image.size:
        object_mask = object_mask.resize(mask_image.size, Image.Resampling.NEAREST)
    object_t = _mask_to_tensor(object_mask)

    cache = strategy.precompute(
        image_t,
        mask_t,
        object_mask=object_t,
        cfg=guidance_cfg,
    )

    runtime: dict[str, Any] = {
        "latent_image_ids": None,
        "mask_packed": None,
        "height": int(image_t.shape[-2]),
        "width": int(image_t.shape[-1]),
        "gs_mutable": None,
    }

    use_staged_cfg = guidance_scale_early is not None and guidance_scale_late is not None
    gs_mutable: MutableGuidanceScale | None = None
    if use_staged_cfg:
        gs_mutable = MutableGuidanceScale(float(guidance_scale_early))
        runtime["gs_mutable"] = gs_mutable
        guidance_scale = gs_mutable  # type: ignore[assignment]

    def callback_on_step_end(pipe_obj, step_idx, timestep, callback_kwargs):
        if gs_mutable is not None and step_idx >= guidance_switch_step - 1:
            gs_mutable.val = float(guidance_scale_late)

        latents = callback_kwargs["latents"]
        num_steps = pipe_obj._num_timesteps

        if not strategy.should_guide(step_idx, num_steps, guidance_cfg):
            return {"latents": latents}

        state = GuidanceState(
            step_idx=step_idx,
            num_steps=num_steps,
            timestep=timestep,
            latent_image_ids=runtime["latent_image_ids"],
            mask_packed=runtime["mask_packed"],
            height=runtime["height"] or 0,
            width=runtime["width"] or 0,
            backend="flux",
        )
        latents_new, _ = strategy.guide_latents(
            latents, cache, image_t, mask_t, guidance_cfg, state, pipe_obj
        )
        return {"latents": latents_new}

    original_prepare = pipe.prepare_latents
    original_mask_prepare = pipe.prepare_mask_latents

    def prepare_latents_wrapper(*args, **kwargs):
        out = original_prepare(*args, **kwargs)
        runtime["latent_image_ids"] = out[-1]
        height = kwargs.get("height")
        width = kwargs.get("width")
        if height is None and len(args) > 4:
            height = args[4]
        if width is None and len(args) > 5:
            width = args[5]
        if height is not None and width is not None:
            runtime["height"] = int(height)
            runtime["width"] = int(width)
        if pure_noise_mask_init and height is not None and width is not None:
            batch_size = kwargs.get("batch_size", args[2] if len(args) > 2 else 1)
            dtype = kwargs.get("dtype", out[0].dtype)
            device = kwargs.get("device", out[0].device)
            mask_condition = pipe.mask_processor.preprocess(
                mask_image, height=int(height), width=int(width), resize_mode="default"
            )
            m = original_mask_prepare(
                mask_condition,
                batch_size,
                1,
                int(height),
                int(width),
                dtype,
                device,
            )
            runtime["mask_packed"] = m
            latents, noise, packed_img, img_enc, ids = out
            latents = latents * (1.0 - m) + noise * m
            packed_img = packed_img * (1.0 - m) + noise * m
            out = (latents, noise, packed_img, img_enc, ids)
        return out

    def prepare_mask_wrapper(*args, **kwargs):
        mask_packed = original_mask_prepare(*args, **kwargs)
        runtime["mask_packed"] = mask_packed
        return mask_packed

    pipe.prepare_latents = prepare_latents_wrapper
    pipe.prepare_mask_latents = prepare_mask_wrapper
    pipe._ew_prepare_mask_latents = original_mask_prepare

    try:
        pipe_call_kwargs = dict(
            prompt=inputs.prompt,
            image=source_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
            **pipe_kwargs,
        )
        if pipe_call_kwargs.get("padding_mask_crop") is None:
            pipe_call_kwargs.pop("padding_mask_crop", None)
        result = pipe(**pipe_call_kwargs)
    finally:
        pipe.prepare_latents = original_prepare
        pipe.prepare_mask_latents = original_mask_prepare
        if hasattr(pipe, "_ew_prepare_mask_latents"):
            del pipe._ew_prepare_mask_latents

    return result.images, cache
