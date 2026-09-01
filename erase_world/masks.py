"""Mask construction: GrayFill, contact-band M_shadow, M_flux = dilate_r(M_obj ∪ M_shadow)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .lib.models.ijepa_official import build_visible_fill


def gray_fill(
    image: Image.Image | np.ndarray | "torch.Tensor",
    mask: Image.Image | np.ndarray | "torch.Tensor",
    fill: float = 0.5,
):
    """GrayFill(I, M_obj) → I_vis used to cache JEPA E_target (Alg.1)."""
    import torch

    if isinstance(image, torch.Tensor):
        return build_visible_fill(image, mask, fill=fill)

    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        rgb = np.asarray(image, dtype=np.float32)
    if isinstance(mask, Image.Image):
        m = np.asarray(mask.convert("L")) > 127
    else:
        m = np.asarray(mask) > 127
    if m.ndim == 3:
        m = m[..., 0]
    out = rgb.copy()
    out[m] = fill * 255.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def dilate_mask_pil(mask: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return mask.convert("L")
    k = px * 2 + 1
    return mask.convert("L").filter(ImageFilter.MaxFilter(k))


def contact_band_shadow(
    image: Image.Image,
    object_mask: Image.Image,
    *,
    sigma_factor: float = 0.5,
    delta_x_factor: float = 0.35,
    **_unused,
) -> Image.Image:
    """Geometric contact band from the paper.

    C is the floor-contact contour (lowest object pixel in each column).
    M_shadow = {c + a n + b τ | c ∈ C, a ∈ [0, σ], |b| ≤ δ_x} ∩ Ω
    with n = (0, 1) (down) and τ = (1, 0) for upright images.
    σ = max(6, 0.5 h_obj), δ_x = max(8, 0.35 w_obj).
    """
    del image  # geometry uses the mask grid only
    m_u8 = np.asarray(object_mask.convert("L"))
    obj = m_u8 > 127
    h, w = obj.shape
    band = np.zeros((h, w), dtype=bool)
    if not obj.any():
        return Image.fromarray(np.zeros((h, w), dtype=np.uint8))

    ys, xs = np.where(obj)
    h_obj = int(ys.max() - ys.min() + 1)
    w_obj = int(xs.max() - xs.min() + 1)
    sigma = max(6, int(sigma_factor * h_obj))
    delta_x = max(8, int(delta_x_factor * w_obj))

    contact_y = np.full(w, -1, dtype=np.int32)
    for x in np.unique(xs):
        contact_y[int(x)] = int(np.where(obj[:, int(x)])[0].max())
    cx = np.where(contact_y >= 0)[0]
    cy = contact_y[cx]
    aa = np.arange(0, sigma + 1, dtype=np.int32)
    bb = np.arange(-delta_x, delta_x + 1, dtype=np.int32)
    yy = cy[:, None, None] + aa[None, :, None]
    xx = cx[:, None, None] + bb[None, None, :]
    yy, xx = np.broadcast_arrays(yy, xx)
    valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    band[yy[valid], xx[valid]] = True
    band &= ~obj
    return Image.fromarray((band.astype(np.uint8) * 255))


def build_m_flux(
    image: Image.Image,
    object_mask: Image.Image,
    *,
    dilate_r: int = 4,
    use_shadow: bool = True,
) -> Image.Image:
    """M_flux = dilate_r(M_obj ∪ M_shadow)."""
    m_obj = np.asarray(object_mask.convert("L")) > 127
    if use_shadow:
        m_sh = np.asarray(contact_band_shadow(image, object_mask).convert("L")) > 127
        union = (m_obj | m_sh).astype(np.uint8) * 255
    else:
        union = m_obj.astype(np.uint8) * 255
    union_pil = Image.fromarray(union, mode="L")
    if dilate_r > 0:
        return dilate_mask_pil(union_pil, dilate_r)
    return union_pil


def object_mask_u8(mask: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
    im = mask.convert("L")
    if size is not None and im.size != size:
        im = im.resize(size, Image.Resampling.NEAREST)
    return np.asarray(im)
