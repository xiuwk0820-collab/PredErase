"""Unified cast-shadow search for object+effect removal.

Combines three existing priors into one edit mask:
  1. Contact shadow band (dark ring / below-object)
  2. Causal physics wedge (light-axis cast + optional I-JEPA shallow dark)
  3. Instance lighting / Retinex cast-shadow understanding

Used by SD-XL and FLUX demos so shadow search is not left behind JEPA guidance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from .causal_shadow import (
    counterfactual_to_pil,
    derive_causal_shadow_masks,
    pack_to_effect_mask_pil,
)
from .shadow_simple import (
    build_contact_shadow_band_u8,
    build_stable_effect_mask_u8,
    contact_shadow_soft_u8,
)
from ..modules.instance_lighting import InstanceLightingModule


@dataclass
class ShadowSearchResult:
    active: bool
    object_mask: Image.Image
    effect_mask: Image.Image
    shadow_soft: Image.Image
    shadow_hard: Image.Image
    counterfactual: Optional[Image.Image] = None
    cast_direction: tuple[float, float] = (0.0, 1.0)
    sources: dict[str, bool] = field(default_factory=dict)
    mask_ratio: float = 0.0
    shadow_ratio: float = 0.0


def _u8_mask(mask: Image.Image | np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    else:
        arr = mask.astype(np.uint8)
    if arr.shape[:2] != (size[1], size[0]):
        arr = np.asarray(
            Image.fromarray(arr).resize(size, Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return arr


def _overlay_shadow_debug(
    image: Image.Image,
    object_mask_u8: np.ndarray,
    shadow_u8: np.ndarray,
) -> Image.Image:
    """Red = object, cyan = searched shadow (for quick visual QA)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    obj = object_mask_u8 > 127
    sh = (shadow_u8 > 127) & ~obj
    out = rgb.copy()
    out[obj] = out[obj] * 0.45 + np.array([220, 40, 40], dtype=np.float32) * 0.55
    out[sh] = out[sh] * 0.45 + np.array([40, 200, 220], dtype=np.float32) * 0.55
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def search_object_cast_shadow(
    image: Image.Image,
    object_mask: Image.Image,
    *,
    ijepa: Any | None = None,
    cfg: dict | None = None,
) -> ShadowSearchResult:
    """Run shadow search and return object ∪ cast-shadow edit mask."""
    cfg = dict(cfg or {})
    enable_contact = bool(cfg.get("enable_contact", True))
    enable_causal = bool(cfg.get("enable_causal", True))
    enable_lighting = bool(cfg.get("enable_lighting", True))
    use_stable_fallback = bool(cfg.get("use_stable_fallback", True))
    use_counterfactual_source = bool(cfg.get("use_counterfactual_source", True))
    effect_thresh = float(cfg.get("effect_thresh", 0.12))
    max_effect_ratio = float(cfg.get("max_effect_ratio", 0.35))

    image = image.convert("RGB")
    w, h = image.size
    obj_u8 = _u8_mask(object_mask, (w, h))
    obj = obj_u8 > 127
    sources = {"contact": False, "causal": False, "lighting": False, "stable": False}

    shadow_acc = np.zeros((h, w), dtype=bool)
    soft_acc = np.zeros((h, w), dtype=np.float32)
    cast_dir = (0.0, 1.0)
    counterfactual: Image.Image | None = None

    # --- 1) Contact shadow band (lean demo path) ---
    if enable_contact:
        band = build_contact_shadow_band_u8(
            image,
            obj_u8,
            ring_px=int(cfg.get("ring_px", 18)),
            dark_delta=float(cfg.get("dark_delta", 0.015)),
            max_band_ratio=float(cfg.get("max_band_ratio", 0.028)),
        )
        soft = contact_shadow_soft_u8(
            band,
            obj_u8,
            sigma=float(cfg.get("soft_sigma", 3.0)),
        )
        band_b = band > 127
        if band_b.any():
            sources["contact"] = True
            shadow_acc |= band_b
            soft_acc = np.maximum(soft_acc, soft.astype(np.float32) / 255.0)

    # --- 2) Causal physics (+ optional I-JEPA shallow darkness) ---
    if enable_causal:
        from ..pipeline.inpaint import _mask_to_tensor, _pil_to_tensor

        image_t = _pil_to_tensor(image)
        mask_t = _mask_to_tensor(Image.fromarray(obj_u8, mode="L"))
        device = "cpu"
        if ijepa is not None and hasattr(ijepa, "device"):
            device = str(ijepa.device)
        elif torch.cuda.is_available():
            device = "cuda:0"
        image_t = image_t.to(device)
        mask_t = mask_t.to(device)
        if ijepa is not None and hasattr(ijepa, "to"):
            # OfficialIJEPAModel keeps its own device; tensors just need same device as encoder
            pass

        rule_only = bool(cfg.get("shadow_rule_only", ijepa is None))
        pack = derive_causal_shadow_masks(
            None if rule_only else ijepa,
            image_t,
            mask_t,
            shadow_thresh=float(cfg.get("shadow_thresh", 0.20)),
            min_shadow_mass=float(cfg.get("min_shadow_mass", 400.0)),
            lift_strength=float(cfg.get("lift_strength", 0.90)),
            shadow_rule_only=rule_only,
            shadow_expand_ratio=float(cfg.get("shadow_expand_ratio", 0.15)),
            shadow_feather_px=float(cfg.get("shadow_feather_px", 4.0)),
        )
        if pack.active:
            sources["causal"] = True
            cast_dir = pack.cast_direction
            eff = pack_to_effect_mask_pil(pack)
            eff_u8 = _u8_mask(eff, (w, h))
            sh = (eff_u8 > int(effect_thresh * 255)) & ~obj
            soft_c = eff_u8.astype(np.float32) / 255.0
            soft_c = soft_c * (~obj)
            shadow_acc |= sh
            soft_acc = np.maximum(soft_acc, soft_c)
            if use_counterfactual_source:
                counterfactual = counterfactual_to_pil(pack).resize((w, h), Image.Resampling.LANCZOS)

    # --- 3) Instance lighting / Retinex cast search ---
    if enable_lighting:
        lighting = InstanceLightingModule(
            retinex_sigma=float(cfg.get("retinex_sigma", 22.0)),
            dark_rel_thresh=float(cfg.get("dark_rel_thresh", 0.018)),
            min_shadow_pixels=int(cfg.get("min_shadow_pixels", 350)),
            lift_strength=float(cfg.get("lift_strength", 0.98)),
        )
        lit = lighting.analyze_pil(image, Image.fromarray(obj_u8, mode="L"))
        if lit.active:
            sources["lighting"] = True
            cast_dir = lit.cast_direction
            shadow_acc |= lit.shadow_hard
            soft_l = lit.shadow_soft.squeeze().cpu().numpy()
            if soft_l.shape != (h, w):
                soft_l = np.asarray(
                    Image.fromarray((np.clip(soft_l, 0, 1) * 255).astype(np.uint8)).resize(
                        (w, h), Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                ) / 255.0
            soft_acc = np.maximum(soft_acc, soft_l * (~obj))
            if use_counterfactual_source and counterfactual is None:
                counterfactual = Image.fromarray(
                    (np.clip(lit.counterfactual_rgb, 0, 1) * 255).astype(np.uint8)
                )

    # --- 4) Stable downward fallback if nothing fired ---
    if not shadow_acc.any() and use_stable_fallback:
        stable = build_stable_effect_mask_u8(
            image,
            obj_u8,
            down_ratio=float(cfg.get("shadow_down_ratio", 0.55)),
            ring_px=int(cfg.get("ring_px", 18)),
            dark_delta=float(cfg.get("dark_delta", 0.028)),
            dilate_px=int(cfg.get("dilate_px", 6)),
        )
        extra = (stable > 127) & ~obj
        if extra.any():
            sources["stable"] = True
            shadow_acc |= extra
            soft_acc = np.maximum(soft_acc, extra.astype(np.float32) * 0.65)

    effect = obj | shadow_acc
    effect_ratio = float(effect.mean())
    if effect_ratio > max_effect_ratio:
        # Cap runaway masks: keep object + strongest soft shadow mass.
        cap_px = int(max_effect_ratio * h * w)
        obj_px = int(obj.sum())
        budget = max(0, cap_px - obj_px)
        flat = soft_acc.copy()
        flat[obj] = 0.0
        if budget > 0 and flat.max() > 0:
            idx = np.argpartition(flat.ravel(), -budget)[-budget:]
            keep = np.zeros_like(effect)
            keep.ravel()[idx] = True
            keep &= flat > effect_thresh
            effect = obj | keep
            shadow_acc = effect & ~obj
            soft_acc = soft_acc * shadow_acc
        else:
            effect = obj
            shadow_acc = np.zeros_like(obj)
            soft_acc = np.zeros_like(soft_acc)

    active = bool(shadow_acc.any())
    shadow_hard_u8 = (shadow_acc.astype(np.uint8) * 255)
    soft_u8 = (np.clip(soft_acc, 0, 1) * 255).astype(np.uint8)
    effect_u8 = (effect.astype(np.uint8) * 255)

    return ShadowSearchResult(
        active=active,
        object_mask=Image.fromarray(obj_u8, mode="L"),
        effect_mask=Image.fromarray(effect_u8, mode="L"),
        shadow_soft=Image.fromarray(soft_u8, mode="L"),
        shadow_hard=Image.fromarray(shadow_hard_u8, mode="L"),
        counterfactual=counterfactual,
        cast_direction=cast_dir,
        sources=sources,
        mask_ratio=float(obj.mean()),
        shadow_ratio=float(shadow_acc.mean()),
    )


def save_shadow_debug(
    out_dir,
    image: Image.Image,
    result: ShadowSearchResult,
) -> None:
    """Write mask / overlay artifacts next to demo outputs."""
    out_dir = __import__("pathlib").Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.object_mask.save(out_dir / "object_mask.png")
    result.effect_mask.save(out_dir / "effect_mask.png")
    result.shadow_soft.save(out_dir / "shadow_soft.png")
    result.shadow_hard.save(out_dir / "shadow_hard.png")
    _overlay_shadow_debug(
        image,
        np.asarray(result.object_mask),
        np.asarray(result.shadow_hard),
    ).save(out_dir / "shadow_overlay.png")
    if result.counterfactual is not None:
        result.counterfactual.save(out_dir / "counterfactual.png")
