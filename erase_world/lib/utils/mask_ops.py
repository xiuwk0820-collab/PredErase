from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, generate_binary_structure


def bridge_and_dilate_object_mask(
    mask: Image.Image,
    *,
    bridge_px: int = 16,
    dilate_px: int = 12,
    fill_holes: bool = True,
    extend_down_ratio: float = 0.20,
    extend_left_ratio: float = 0.32,
) -> Image.Image:
    """Bridge cup-handle gaps, thicken thin parts, dilate, extend for cast shadow."""
    m = np.asarray(mask.convert("L")) > 127
    if not m.any():
        return mask.convert("L")

    if fill_holes:
        m = binary_fill_holes(m)

    if bridge_px > 0:
        struct = generate_binary_structure(2, 2)
        struct = binary_dilation(struct, iterations=max(2, bridge_px // 2))
        m = binary_closing(m, structure=struct)
        # horizontal closing bridges cup body ↔ handle gap
        h_w = max(9, bridge_px + 3)
        h_struct = np.ones((3, h_w), dtype=bool)
        m = binary_closing(m, structure=h_struct)

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        m = np.asarray(
            Image.fromarray((m.astype(np.uint8) * 255)).filter(ImageFilter.MaxFilter(k)),
            dtype=bool,
        )

    if extend_down_ratio > 0 or extend_left_ratio > 0:
        ys, xs = np.where(m)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        obj_h = max(y1 - y0 + 1, 1)
        if extend_down_ratio > 0:
            band = max(6, int(obj_h * extend_down_ratio))
            y_end = min(m.shape[0], y1 + 1 + band)
            pad_left = max(8, int((x1 - x0 + 1) * extend_left_ratio)) if extend_left_ratio > 0 else 0
            pad_right = max(4, int((x1 - x0 + 1) * 0.10)) if extend_down_ratio > 0 else 0
            x0e = max(0, x0 - pad_left)
            x1e = min(m.shape[1], x1 + 1 + pad_right)
            m[y1 + 1 : y_end, x0e:x1e] = True
            if extend_left_ratio > 0:
                y_diag = min(m.shape[0], y1 + 1 + int(band * 0.65))
                x_diag = max(0, x0 - int(pad_left * 0.55))
                m[y1 + 1 : y_diag, x_diag:x0e] = True

    return Image.fromarray((m.astype(np.uint8) * 255), mode="L")


def _local_mean(gray: np.ndarray, kernel: int) -> np.ndarray:
    """Fast local mean via separable box blur."""
    k = max(3, kernel | 1)
    img = Image.fromarray((gray * 255).astype(np.uint8))
    blurred = img.filter(ImageFilter.BoxBlur(radius=k // 2))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def expand_removal_mask(
    image: Image.Image,
    mask: Image.Image,
    dilation_ratio: float = 0.04,
    shadow_delta: float = 0.12,
    local_kernel: int = 31,
) -> Image.Image:
    """Expand object mask to cover cast shadows (RemovalBench GT removes both).

    1. Morphological dilation scaled to image size.
    2. Heuristic: nearby pixels darker than local mean (typical shadow).
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    m = np.asarray(mask.convert("L")) > 127
    h, w = m.shape
    if not m.any():
        return mask.convert("L")

    radius = max(16, int(min(h, w) * dilation_ratio))
    k = 2 * radius + 1
    dilated = np.asarray(
        Image.fromarray((m.astype(np.uint8) * 255)).filter(ImageFilter.MaxFilter(k)),
        dtype=bool,
    )

    gray = rgb.mean(axis=2)
    local_mean = _local_mean(gray, local_kernel)
    shadow = (~m) & dilated & ((local_mean - gray) > shadow_delta)

    out = (dilated | shadow).astype(np.uint8) * 255
    return Image.fromarray(out, mode="L")
