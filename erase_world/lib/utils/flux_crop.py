"""Auto crop padding and FLUX mask helpers."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, generate_binary_structure


def object_mask_ratio(mask: Image.Image) -> float:
    m = np.asarray(mask.convert("L")) > 127
    if not m.any():
        return 0.0
    return float(m.sum()) / float(m.size)


def object_bbox_mask(
    object_mask: Image.Image,
    *,
    pad_px: int = 32,
) -> Image.Image:
    """Axis-aligned bbox mask — only for tiny objects where tight mask starves FLUX."""
    m = np.asarray(object_mask.convert("L")) > 127
    h, w = m.shape
    if not m.any():
        return object_mask
    ys, xs = np.where(m)
    x0 = max(0, int(xs.min()) - pad_px)
    x1 = min(w - 1, int(xs.max()) + pad_px)
    y0 = max(0, int(ys.min()) - pad_px)
    y1 = min(h - 1, int(ys.max()) + pad_px)
    bbox = Image.new("L", (w, h), 0)
    ImageDraw.Draw(bbox).rectangle([x0, y0, x1, y1], fill=255)
    return bbox


def dilate_object_mask(object_mask: Image.Image, *, px: int = 12) -> Image.Image:
    """Soft expand object silhouette — keeps context, avoids rectangular inpaint collapse."""
    m = np.asarray(object_mask.convert("L")) > 127
    if px <= 0 or not m.any():
        return object_mask
    struct = generate_binary_structure(2, 2)
    dil = binary_dilation(m, structure=struct, iterations=max(1, int(px) // 2))
    return Image.fromarray((dil.astype(np.uint8) * 255), mode="L")


def choose_flux_edit_mask(
    object_mask: Image.Image,
    *,
    tiny_ratio: float = 0.015,
    large_ratio: float = 0.06,
    tiny_pad: int = 28,
    large_dilate_px: int = 12,
    use_bbox_large: bool = False,
    use_bbox_tiny: bool = True,
) -> tuple[Image.Image, str]:
    """Pick FLUX inpaint mask.

    Large objects: keep object silhouette (+ light dilate), never full bbox — bbox inpaint
    removes all in-rectangle context and collapses to flat fill.
    Tiny objects: optional small bbox so FLUX sees enough edit region.
    """
    ratio = object_mask_ratio(object_mask)
    if ratio >= large_ratio:
        if use_bbox_large:
            return object_bbox_mask(object_mask, pad_px=48), "bbox_large"
        return dilate_object_mask(object_mask, px=large_dilate_px), "object_large"
    if ratio <= tiny_ratio and use_bbox_tiny:
        return object_bbox_mask(object_mask, pad_px=tiny_pad), "bbox_tiny"
    return object_mask, "object"


def auto_padding_mask_crop(
    object_mask: Image.Image,
    *,
    ratio_threshold: float = 0.06,
    padding: int = 64,
    max_padding: int = 128,
) -> int | None:
    """Return padding for FLUX crop-inpaint when the object is large in frame."""
    ratio = object_mask_ratio(object_mask)
    if ratio < ratio_threshold:
        return None
    scaled = int(padding * (1.0 + 0.5 * min(2.0, ratio / max(ratio_threshold, 1e-6))))
    return max(32, min(max_padding, scaled))
