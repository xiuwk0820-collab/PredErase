"""Minimal downward shadow soft weight — gravity only, no direction search."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, generate_binary_structure


def build_w_shadow_downward(
    object_mask: torch.Tensor,
    *,
    down_ratio: float = 0.5,
) -> torch.Tensor:
    """W_shadow: 1 inside M_obj, downward Gaussian decay below bbox bottom."""
    if object_mask.ndim == 4:
        m = object_mask[0, 0]
    elif object_mask.ndim == 3:
        m = object_mask[0]
    else:
        m = object_mask
    obj = (m.detach().cpu().numpy() > 0.5)
    h, w = obj.shape
    if not obj.any():
        return torch.zeros(1, 1, h, w, dtype=torch.float32)

    ys, _ = np.where(obj)
    y_bottom = int(ys.max())
    obj_h = float(ys.max() - ys.min() + 1)
    sigma = max(6.0, down_ratio * obj_h)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    below = np.maximum(0, yy.astype(np.float32) - float(y_bottom))
    decay = np.exp(-((below / sigma) ** 2))
    decay = np.where(below > 0, decay, 0.0) * (~obj)
    w_arr = np.maximum(obj.astype(np.float32), decay.astype(np.float32))
    t = torch.from_numpy(w_arr).float().unsqueeze(0).unsqueeze(0)
    return t.to(device=m.device, dtype=m.dtype)


def build_stable_effect_mask_u8(
    source: Image.Image,
    object_mask_u8: np.ndarray,
    *,
    down_ratio: float = 0.55,
    ring_px: int = 18,
    dark_delta: float = 0.028,
    dilate_px: int = 6,
) -> np.ndarray:
    """Object + contact-shadow band for stable FLUX/edit (no causal module)."""
    rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    obj = object_mask_u8 > 127
    if obj.shape != (h, w):
        obj = (
            np.asarray(Image.fromarray(object_mask_u8).resize((w, h), Image.Resampling.NEAREST))
            > 127
        )
    if not obj.any():
        return object_mask_u8

    obj_ratio = float(obj.sum()) / float(h * w)
    if obj_ratio >= 0.10:
        effect = binary_dilation(
            obj, structure=generate_binary_structure(2, 2), iterations=max(10, dilate_px + 4)
        )
        return (effect.astype(np.uint8) * 255)

    w_shadow = build_w_shadow_downward(
        torch.from_numpy(obj.astype(np.float32)), down_ratio=down_ratio
    )
    shadow_soft = w_shadow.squeeze().numpy()

    struct = generate_binary_structure(2, 2)
    outer = binary_dilation(obj, structure=struct, iterations=ring_px + 10)
    ring = binary_dilation(obj, structure=struct, iterations=ring_px) & ~obj
    bg = ~outer
    lum = rgb.mean(axis=-1)
    ref_lum = float(np.percentile(lum[bg], 56)) if bg.any() else float(np.median(lum[ring]))

    shadow_band = (shadow_soft > 0.12) & ~obj
    dark_ring = ring & (ref_lum - lum > dark_delta)

    ys, xs = np.where(obj)
    if ys.size:
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        obj_h = y1 - y0 + 1
        obj_w = x1 - x0 + 1
        below = np.zeros_like(obj)
        y_end = min(h, y1 + max(10, int(0.55 * obj_h)))
        x_pad = max(8, int(0.35 * obj_w))
        x_lo, x_hi = max(0, x0 - x_pad), min(w, x1 + x_pad + 1)
        below[y1:y_end, x_lo:x_hi] = True
        dark_below = below & (ref_lum - lum > dark_delta * 0.7)
        side = ring & (ref_lum - lum > dark_delta * 0.85)
        shadow_band = shadow_band | dark_below | dark_ring | side

    effect = obj | shadow_band
    if ys.size:
        margin_y = max(12, int(0.75 * obj_h))
        margin_x = max(16, int(0.45 * obj_w))
        roi = np.zeros_like(obj)
        roi[
            max(0, y0 - margin_y // 3) : min(h, y_end),
            max(0, x0 - margin_x) : min(w, x1 + margin_x + 1),
        ] = True
        effect = effect & roi
    if dilate_px > 0:
        effect = binary_dilation(effect, structure=struct, iterations=dilate_px)
    max_ratio = 0.22
    if effect.sum() > max_ratio * h * w:
        tight = obj | ((shadow_soft > 0.22) & ~obj)
        if ys.size:
            tight = tight & roi
        effect = binary_dilation(tight, structure=struct, iterations=max(2, dilate_px // 2))
    cap_px = int(min(h * w * 0.16, max(obj.sum() * 3.5, obj.sum() + 80000)))
    if effect.sum() > cap_px:
        effect = binary_dilation(obj, structure=struct, iterations=8)
        if ys.size:
            contact = (shadow_soft > 0.28) & roi & ~obj
            effect = effect | contact
        effect = binary_dilation(effect, structure=struct, iterations=3)
    return (effect.astype(np.uint8) * 255)


def build_contact_shadow_band_u8(
    source: Image.Image,
    object_mask_u8: np.ndarray,
    *,
    ring_px: int = 18,
    dark_delta: float = 0.015,
    max_band_ratio: float = 0.028,
    roi_margin_ratio: float = 0.55,
) -> np.ndarray:
    """Dark pixels adjacent to M_obj — tight, connected band for stable FLUX/paste."""
    from scipy.ndimage import label

    rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    obj = object_mask_u8 > 127
    if obj.shape != (h, w):
        obj = (
            np.asarray(Image.fromarray(object_mask_u8).resize((w, h), Image.Resampling.NEAREST))
            > 127
        )
    if not obj.any():
        return np.zeros((h, w), dtype=np.uint8)

    struct = generate_binary_structure(2, 2)
    ys, xs = np.where(obj)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    obj_h = y1 - y0 + 1
    obj_w = x1 - x0 + 1
    my = max(10, int(roi_margin_ratio * obj_h))
    mx = max(12, int(roi_margin_ratio * obj_w))
    roi = np.zeros_like(obj)
    roi[max(0, y0 - my // 3) : min(h, y1 + my + 1), max(0, x0 - mx) : min(w, x1 + mx + 1)] = True

    ring = binary_dilation(obj, structure=struct, iterations=ring_px) & ~obj & roi
    far = roi & ~binary_dilation(obj, structure=struct, iterations=ring_px + 20)
    lum = rgb.mean(axis=-1)
    ref_lum = float(np.percentile(lum[far], 58)) if far.any() else float(np.median(lum[ring]))
    band = ring & (ref_lum - lum > dark_delta)

    # Small contact pad below object only (not whole image bottom).
    below = np.zeros_like(obj)
    y_end = min(h, y1 + max(6, int(0.32 * obj_h)))
    x_pad = max(8, int(0.28 * obj_w))
    below[y1:y_end, max(0, x0 - x_pad) : min(w, x1 + x_pad + 1)] = True
    band = band | (below & ~obj & roi & (ref_lum - lum > dark_delta * 0.8))

    # Keep only shadow blobs touching the object.
    touch = binary_dilation(obj, structure=struct, iterations=2)
    labeled, n = label(band)
    keep = np.zeros_like(band)
    for lab in range(1, n + 1):
        comp = labeled == lab
        if (comp & touch).any():
            keep |= comp
    band = keep

    cap = int(min(h * w * max_band_ratio, max(obj.sum() * 1.6, obj.sum() + 6000)))
    if band.sum() > cap:
        gap = np.clip(ref_lum - lum, 0.0, 1.0) * band.astype(np.float32)
        idx = np.where(band)
        order = np.argsort(-gap[idx])[:cap]
        tight = np.zeros_like(band)
        tight[idx[0][order], idx[1][order]] = True
        band = binary_dilation(tight, structure=struct, iterations=1)

    return (band.astype(np.uint8) * 255)


def contact_shadow_soft_u8(
    contact_band_u8: np.ndarray,
    object_mask_u8: np.ndarray,
    *,
    sigma: float = 3.0,
) -> np.ndarray:
    """Soft weight on contact band only (excludes object interior)."""
    from scipy.ndimage import gaussian_filter

    obj = object_mask_u8 > 127
    band = (contact_band_u8 > 127) & ~obj
    if not band.any():
        return np.zeros_like(contact_band_u8, dtype=np.uint8)
    soft = gaussian_filter(band.astype(np.float32), sigma=sigma)
    return (np.clip(soft, 0.0, 1.0) * 255.0).astype(np.uint8)


def shadow_soft_u8_from_object(
    object_mask_u8: np.ndarray,
    *,
    down_ratio: float = 0.55,
) -> np.ndarray:
    obj = object_mask_u8 > 127
    w = build_w_shadow_downward(torch.from_numpy(obj.astype(np.float32)), down_ratio=down_ratio)
    soft = w.squeeze().numpy()
    soft = np.clip(soft - 0.08, 0.0, 1.0) / 0.92
    return (soft * 255.0).astype(np.uint8)
