"""Causal shadow masks from frozen I-JEPA shallow features + physics priors.

No external detection models — one shallow encoder pass, spatial wedge rules,
dual-domain regularization handled in JEPAGuidanceStrategy (L_struct / L_light).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from ..models.ijepa_official import OfficialIJEPAModel
from .shadow_physics import build_counterfactual_visible, _local_mean
from .texture_harmonize import fill_object_absent_tensor


@dataclass
class CausalShadowPack:
    """Derived masks + counterfactual context (same forward as L_light precompute)."""

    active: bool
    shadow_soft: torch.Tensor  # [1, 1, H, W]
    object_mask: torch.Tensor  # [1, H, W] float 0/1
    flux_mask: torch.Tensor  # [1, H, W] legacy alias ≈ M_obj (FLUX edit = object only)
    counterfactual: torch.Tensor  # [1, 3, H, W]
    cast_direction: tuple[float, float]

    @property
    def flux_mask_u8(self) -> np.ndarray:
        m = self.flux_mask
        if m.ndim == 3:
            m = m[0]
        return (m.detach().cpu().numpy() * 255).astype(np.uint8)


def _floor_contact(obj: torch.Tensor) -> tuple[float, float, int, int]:
    """Bottom-center anchor of object mask [1,H,W]."""
    m = obj[0] > 0.5
    if not bool(m.any()):
        return 0.0, 0.0, 0, 0
    ys, xs = torch.where(m)
    y1 = int(ys.max().item())
    band = max(3, int((ys.max() - ys.min()).item() * 0.08))
    floor = ys >= y1 - band
    oy = float(ys[floor].float().mean().item())
    ox = float(xs[floor].float().mean().item())
    obj_h = int(ys.max() - ys.min() + 1)
    obj_w = int(xs.max() - xs.min() + 1)
    return oy, ox, obj_h, obj_w


def _infer_cast_direction_floor(
    gray: torch.Tensor,
    obj: torch.Tensor,
    oy: float,
    ox: float,
    obj_h: int,
    obj_w: int,
) -> tuple[float, float]:
    """Cast axis in down-right hemisphere (max dark mass in wedge)."""
    h, w = gray.shape[-2], gray.shape[-1]
    g = gray[0, 0]
    local = _local_mean(g, kernel=31)
    dark = (local - g).clamp(min=0.0)
    outside = (obj[0] < 0.5).float()

    yy, xx = torch.meshgrid(
        torch.arange(h, device=g.device, dtype=g.dtype),
        torch.arange(w, device=g.device, dtype=g.dtype),
        indexing="ij",
    )
    best_score, best_sx, best_sy = -1.0, 0.906, 0.423
    for k in range(20):
        ang = math.radians(15.0 + 32.0 * k / 19.0)
        sx, sy = math.cos(ang), math.sin(ang)
        if sy < 0.10:
            continue
        along = ((xx - ox) * sx + (yy - oy) * sy).clamp(min=0.0)
        perp = ((xx - ox) * (-sy) + (yy - oy) * sx).abs()
        spread = obj_w * 0.40 + along * 0.30
        wedge = (
            outside
            * (along > 0).float()
            * (along < max(obj_h * 2.4, obj_w * 1.4, min(h, w) * 0.38)).float()
            * (perp < spread).float()
        )
        score = float((dark * wedge).sum().item())
        if score > best_score:
            best_score, best_sx, best_sy = score, sx, sy
    n = max(math.hypot(best_sx, best_sy), 1e-6)
    return best_sx / n, best_sy / n


def _physics_wedge(
    obj: torch.Tensor,
    oy: float,
    ox: float,
    sx: float,
    sy: float,
    obj_h: int,
    obj_w: int,
    *,
    compact: bool = False,
) -> torch.Tensor:
    """Spatial causal constraint: shadow only +along from object floor contact."""
    h, w = obj.shape[-2], obj.shape[-1]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=obj.device, dtype=obj.dtype),
        torch.arange(w, device=obj.device, dtype=obj.dtype),
        indexing="ij",
    )
    along = (xx - ox) * sx + (yy - oy) * sy
    perp = ((xx - ox) * (-sy) + (yy - oy) * sx).abs()
    if compact:
        max_len = max(obj_h * 1.8, obj_w * 1.1, min(h, w) * 0.14)
        spread = obj_w * 0.30 + along.clamp(min=0.0) * 0.22
    else:
        max_len = max(obj_h * 2.4, obj_w * 1.35, min(h, w) * 0.36)
        spread = obj_w * 0.42 + along.clamp(min=0.0) * 0.32
    outside = (obj[0] < 0.5).float()
    wedge = (
        outside
        * (along > 0).float()
        * (along < max_len).float()
        * (perp < spread).float()
    )
    return wedge.unsqueeze(0).unsqueeze(0)


def _gaussian_feather(mask: torch.Tensor, sigma_px: float = 4.0) -> torch.Tensor:
    """3–5 px Gaussian feather on shadow / object masks."""
    if sigma_px <= 0:
        return mask.clamp(0.0, 1.0)
    if mask.ndim == 3:
        m = mask.unsqueeze(1)
    else:
        m = mask
    k = max(3, int(sigma_px * 2) | 1)
    pad = k // 2
    return F.avg_pool2d(m.float(), kernel_size=k, stride=1, padding=pad).clamp(0.0, 1.0)


def _rule_ground_shadow(
    obj: torch.Tensor,
    oy: float,
    ox: float,
    sx: float,
    sy: float,
    obj_h: int,
    obj_w: int,
    *,
    expand_ratio: float = 0.15,
) -> torch.Tensor:
    """Directional ground shadow candidate: gravity-down + coarse light axis, no detector."""
    h, w = obj.shape[-2], obj.shape[-1]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=obj.device, dtype=obj.dtype),
        torch.arange(w, device=obj.device, dtype=obj.dtype),
        indexing="ij",
    )
    along = (xx - ox) * sx + (yy - oy) * sy
    perp = ((xx - ox) * (-sy) + (yy - oy) * sx).abs()
    outside = (obj[0] < 0.5).float()
    below = (yy >= oy - 1).float()
    compact = float(obj.sum()) / (h * w) < 0.06
    if compact:
        depth = max(obj_h * (0.35 + expand_ratio * 3.5), obj_h * 0.10, 6.0)
        spread = obj_w * 0.42 + obj_h * expand_ratio * 0.8
    else:
        depth = max(min(h * 0.40, w * 0.40), obj_h * (1.0 + expand_ratio * 10.0), 20.0)
        spread = obj_w * 0.55 + obj_h * expand_ratio * 1.2
    wedge = (
        outside
        * below
        * (along > -obj_h * 0.04).float()
        * (along < depth).float()
        * (perp < spread).float()
    )
    return wedge.unsqueeze(0).unsqueeze(0)


def _dilate_binary(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    """Morphological dilation on [1,H,W] or [H,W] float mask."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    k = max(3, kernel | 1)
    return F.max_pool2d(mask.unsqueeze(1).float(), k, stride=1, padding=k // 2).squeeze(1)


def _extend_vertical_contact(flux: torch.Tensor, obj: torch.Tensor, h: int) -> torch.Tensor:
    """Wall/railing band above object — only when instance sits in upper scene."""
    m = obj[0] > 0.5
    if not bool(m.any()):
        return flux
    ys, xs = torch.where(m)
    y0 = int(ys.min().item())
    x0, x1 = int(xs.min().item()), int(xs.max().item())
    if y0 > h * 0.12:
        return flux
    obj_h = int(ys.max().item() - y0 + 1)
    band = max(4, int(obj_h * 0.22))
    y_top = max(0, y0 - band)
    out = flux.clone()
    out[0, y_top:y0, x0 : x1 + 1] = 1.0
    return out.clamp(0.0, 1.0)


def _flux_edit_mask(obj: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Slight M_obj dilation for seam only — shadow lives in effect_soft, not FLUX mask."""
    k = max(3, int(min(h, w) * 0.006) | 1)
    return _dilate_binary(obj, k)


def _merge_effect_soft(
    image: torch.Tensor,
    obj: torch.Tensor,
    shadow_soft: torch.Tensor,
    counterfactual: torch.Tensor,
) -> torch.Tensor:
    """Pixels outside M_obj that counterfactual / physics say deletion would change."""
    outside = (1.0 - obj.unsqueeze(1)).clamp(0.0, 1.0)
    sh = shadow_soft.clamp(0.0, 1.0) * outside
    delta = (image - counterfactual).abs().mean(dim=1, keepdim=True) * outside
    delta_n = (delta / 0.08).clamp(0.0, 1.0)
    return torch.maximum(sh, delta_n * 0.65).clamp(0.0, 1.0)


def dilate_mask_pil(mask: Image.Image, px: int) -> Image.Image:
    """Dilate edit mask after resize so thin structures (handles) survive 640px run."""
    if px <= 0:
        return mask
    k = px * 2 + 1
    arr = np.asarray(mask.convert("L"))
    return Image.fromarray(arr).filter(ImageFilter.MaxFilter(k))


@torch.no_grad()
def derive_causal_shadow_masks(
    ijepa: OfficialIJEPAModel | None,
    image: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    shadow_thresh: float = 0.20,
    flux_shadow_thresh: float = 0.18,
    min_shadow_mass: float = 400.0,
    lift_strength: float = 0.90,
    shadow_rule_only: bool = True,
    shadow_expand_ratio: float = 0.15,
    shadow_feather_px: float = 4.0,
) -> CausalShadowPack:
    """M_shadow from physics wedge (rule) or rule+I-JEPA shallow; M_flux = M_obj ∪ M_shadow."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if object_mask.ndim == 2:
        object_mask = object_mask.unsqueeze(0)
    obj = (object_mask > 0.5).float().to(image.device)
    h, w = image.shape[-2], image.shape[-1]

    empty = torch.zeros(1, 1, h, w, device=image.device, dtype=image.dtype)
    if obj.sum() < 10:
        return CausalShadowPack(
            active=False,
            shadow_soft=empty,
            object_mask=obj,
            flux_mask=obj.clone(),
            counterfactual=image.clone(),
            cast_direction=(1.0, 0.0),
        )

    oy, ox, obj_h, obj_w = _floor_contact(obj)
    gray = image.mean(dim=1, keepdim=True)
    sx, sy = _infer_cast_direction_floor(gray, obj, oy, ox, obj_h, obj_w)

    rule_shadow = _rule_ground_shadow(
        obj, oy, ox, sx, sy, obj_h, obj_w, expand_ratio=shadow_expand_ratio
    )
    compact = obj_h < h * 0.15
    wedge = _physics_wedge(
        obj, oy, ox, sx, sy, obj_h, obj_w, compact=compact
    )
    local = _local_mean(gray[0, 0], kernel=31).unsqueeze(0).unsqueeze(0)
    rel_dark = ((local - gray) / 0.05).clamp(0.0, 1.0)
    outside = (1.0 - obj.unsqueeze(1))

    if shadow_rule_only:
        wedge_causal = wedge * (0.20 + 0.80 * rel_dark) * outside
        rule_causal = rule_shadow * rel_dark * outside
        causal = torch.maximum(wedge_causal, rule_causal)
    else:
        patch_dark = ijepa.shallow_patch_darkness(image, reference_mask=obj)  # type: ignore[union-attr]
        grid = patch_dark.shape[0]
        dark_up = F.interpolate(
            patch_dark.view(1, 1, grid, grid),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        wedge = _physics_wedge(
            obj, oy, ox, sx, sy, obj_h, obj_w, compact=float(obj.sum()) / (h * w) < 0.08
        )
        shallow_w = dark_up.clamp(0.0, 1.0)
        causal = wedge * (0.55 * shallow_w + 0.45 * rel_dark) * outside
        causal = torch.maximum(causal, rule_shadow * rel_dark * outside)

    shadow_soft = _gaussian_feather(causal.clamp(0.0, 1.0), sigma_px=shadow_feather_px)

    shadow_mass = float((shadow_soft * (1.0 - obj.unsqueeze(1))).sum().item())
    active = shadow_mass > min_shadow_mass and float(shadow_soft.max()) > shadow_thresh * 0.5

    if not active:
        shadow_soft = torch.zeros_like(shadow_soft)
        flux = _flux_edit_mask(obj, h, w)
        cf = fill_object_absent_tensor(image.clone(), obj)
        effect = torch.zeros_like(shadow_soft)
    else:
        cf = build_counterfactual_visible(
            image, obj.unsqueeze(1), shadow_soft, strength=lift_strength
        )
        cf = fill_object_absent_tensor(cf, obj)
        effect = _merge_effect_soft(image, obj, shadow_soft, cf)
        flux = _flux_edit_mask(obj, h, w)

    return CausalShadowPack(
        active=active,
        shadow_soft=effect,
        object_mask=obj,
        flux_mask=flux,
        counterfactual=cf,
        cast_direction=(sx, sy),
    )


def pack_to_flux_mask_pil(pack: CausalShadowPack) -> Image.Image:
    """M_obj only — FLUX inpaint region (legacy name kept for callers)."""
    obj = pack.object_mask
    if obj.ndim == 3:
        obj = obj[0]
    return Image.fromarray((obj.detach().cpu().numpy() * 255).astype(np.uint8), mode="L")


def pack_to_paste_mask_pil(pack: CausalShadowPack, *, thresh: float = 0.12) -> Image.Image:
    """Keep model output inside M_obj ∪ effect_soft (paste original elsewhere)."""
    obj = pack.object_mask.squeeze().detach().cpu().numpy() > 0.5
    eff = pack.shadow_soft.squeeze().detach().cpu().numpy() > thresh
    union = (obj | eff).astype(np.uint8) * 255
    return Image.fromarray(union, mode="L")


def pack_to_effect_mask_pil(pack: CausalShadowPack) -> Image.Image:
    """Visualize world-model effect_soft (guidance / paste, not FLUX edit)."""
    sh = pack.shadow_soft.squeeze().detach().cpu().numpy()
    return Image.fromarray((np.clip(sh, 0, 1) * 255).astype(np.uint8), mode="L")


def counterfactual_to_pil(pack: CausalShadowPack) -> Image.Image:
    """Full counterfactual scene (object inpainted + shadow lifted) for FLUX encode."""
    cf = pack.counterfactual[0].permute(1, 2, 0).detach().cpu().numpy()
    if cf.max() > 1.01:
        cf = cf / 255.0
    return Image.fromarray((np.clip(cf, 0, 1) * 255).astype(np.uint8))


def apply_counterfactual_pil(image: Image.Image, pack: CausalShadowPack) -> Image.Image:
    if not pack.active:
        return image
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    cf = pack.counterfactual[0].permute(1, 2, 0).detach().cpu().numpy()
    if cf.shape[:2] != rgb.shape[:2]:
        cf = np.asarray(
            Image.fromarray((np.clip(cf, 0, 1) * 255).astype(np.uint8)).resize(
                (rgb.shape[1], rgb.shape[0]), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        ) / 255.0
    soft = pack.shadow_soft.squeeze().detach().cpu().numpy()
    if soft.shape != rgb.shape[:2]:
        soft = np.asarray(
            Image.fromarray((soft * 255).astype(np.uint8)).resize(
                (rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
    obj = pack.object_mask.squeeze().detach().cpu().numpy() > 0.5
    w = np.clip(soft * (~obj).astype(np.float32), 0.0, 1.0)[..., None]
    out = rgb * (1.0 - w) + cf * w
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def downscale_causal_pack(pack: CausalShadowPack, size: tuple[int, int]) -> CausalShadowPack:
    """Resize pack to FLUX run resolution."""
    w, h = size
    if pack.shadow_soft.shape[-2] == h and pack.shadow_soft.shape[-1] == w:
        return pack

    obj = F.interpolate(
        pack.object_mask.unsqueeze(1).float(), size=(h, w), mode="nearest"
    ).squeeze(1)
    flux = _flux_edit_mask(obj, h, w)
    soft = F.interpolate(pack.shadow_soft.float(), size=(h, w), mode="bilinear", align_corners=False)
    cf = F.interpolate(
        pack.counterfactual.float(), size=(h, w), mode="bilinear", align_corners=False
    ).to(pack.counterfactual.dtype)
    return CausalShadowPack(
        active=pack.active,
        shadow_soft=soft,
        object_mask=obj,
        flux_mask=flux,
        counterfactual=cf,
        cast_direction=pack.cast_direction,
    )


def debug_images_from_pack(pack: CausalShadowPack) -> dict[str, Image.Image]:
    obj = (pack.object_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    effect_vis = pack_to_effect_mask_pil(pack)
    paste = pack_to_paste_mask_pil(pack)
    prot = ((1.0 - np.asarray(paste, dtype=np.float32) / 255.0) * 255).astype(np.uint8)
    return {
        "mask_obj": Image.fromarray(obj, mode="L"),
        "mask_shadow": effect_vis,
        "mask_flux": paste,
        "mask_protect": Image.fromarray(prot, mode="L"),
    }
