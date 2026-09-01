"""Post-inpaint texture harmonization — kill hard seams and cup-shaped ghosts."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation, gaussian_filter, generate_binary_structure


def _to_rgb_f32(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _align_object_mask(object_mask_u8: np.ndarray, h: int, w: int) -> np.ndarray:
    obj = object_mask_u8 > 127
    if obj.shape != (h, w):
        obj = (
            np.asarray(
                Image.fromarray(object_mask_u8).resize((w, h), Image.Resampling.NEAREST)
            )
            > 127
        )
    return obj


def fill_object_absent(
    image: Image.Image,
    object_mask_u8: np.ndarray,
    *,
    ring_px: int = 18,
    blur_sigma: float = 3.5,
    noise_scale: float = 0.45,
) -> Image.Image:
    """Spread ring background into object mask — for JEPA target only, not FLUX encode."""
    rgb = _to_rgb_f32(image)
    h, w = rgb.shape[:2]
    obj = _align_object_mask(object_mask_u8, h, w)
    if not obj.any():
        return image

    out = rgb.copy()
    try:
        import cv2

        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        mask_u8 = obj.astype(np.uint8) * 255
        radius = max(3, min(9, ring_px // 3))
        filled_u8 = cv2.inpaint(rgb_u8, mask_u8, radius, cv2.INPAINT_TELEA)
        out[obj] = filled_u8.astype(np.float32)[obj] / 255.0
    except Exception:
        from scipy.ndimage import distance_transform_edt

        # distance_transform_edt returns indices of the nearest zero.  For object
        # pixels, zeros must be the visible background outside the object.
        _, (iy, ix) = distance_transform_edt(obj, return_indices=True)
        filled = rgb[iy, ix].copy()
        out[obj] = filled[obj]

    struct = generate_binary_structure(2, 2)
    ring = binary_dilation(obj, structure=struct, iterations=ring_px) & ~obj
    if ring.any():
        ring_std = rgb[ring].std(axis=0) + 1e-4
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 1.0, size=(int(obj.sum()), 3)) * (ring_std * noise_scale)
        out[obj] = np.clip(out[obj] + noise, 0.0, 1.0)

    if blur_sigma > 0:
        soft = gaussian_filter(obj.astype(np.float32), sigma=blur_sigma)[..., None]
        blurred = gaussian_filter(out, sigma=blur_sigma, axes=(0, 1))
        out = np.clip(out * (1.0 - soft) + blurred * soft, 0.0, 1.0)

    return Image.fromarray((out * 255).astype(np.uint8))


def fill_flux_source(
    image: Image.Image,
    object_mask_u8: np.ndarray,
    *,
    ring_px: int = 24,
    row_margin: int = 48,
    noise_scale: float = 0.28,
    large_ratio: float = 0.06,
    elongation_ratio: float = 2.2,
    force_row: bool = False,
) -> Image.Image:
    """Background fill for FLUX source — row propagate for compact objs, inpaint for large/diagonal."""
    rgb = _to_rgb_f32(image)
    h, w = rgb.shape[:2]
    obj = _align_object_mask(object_mask_u8, h, w)
    if not obj.any():
        return image

    ys, xs = np.where(obj)
    obj_h = int(ys.max() - ys.min() + 1)
    obj_w = int(xs.max() - xs.min() + 1)
    obj_ratio = float(obj.sum()) / float(h * w)
    elongation = max(obj_h, obj_w) / max(1, min(obj_h, obj_w))
    # Very large masks: never row-propagate (causes horizontal stripe prefill).
    use_inpaint = force_row is False and (
        obj_ratio >= max(large_ratio, 0.14) or elongation >= elongation_ratio + 0.5
    )
    if obj_ratio >= 0.28:
        use_inpaint = True
        noise_scale = min(noise_scale, 0.08)

    out = rgb.copy()
    if use_inpaint:
        try:
            import cv2

            rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            mask_u8 = obj.astype(np.uint8) * 255
            radius = max(5, min(12, ring_px // 2))
            filled_u8 = cv2.inpaint(rgb_u8, mask_u8, radius, cv2.INPAINT_NS)
            out[obj] = filled_u8.astype(np.float32)[obj] / 255.0
        except Exception:
            use_inpaint = False

    if not use_inpaint:
        rows = np.where(obj.any(axis=1))[0]
        margin = max(row_margin, ring_px)
        for y in rows:
            xs_row = np.where(obj[y])[0]
            if xs_row.size == 0:
                continue
            x0, x1 = int(xs_row.min()), int(xs_row.max())
            chunks: list[np.ndarray] = []
            l0, l1 = max(0, x0 - margin), x0
            r0, r1 = min(w, x1 + 1), min(w, x1 + 1 + margin)
            if l1 > l0:
                vis = ~obj[y, l0:l1]
                if vis.any():
                    chunks.append(rgb[y, l0:l1][vis])
            if r1 > r0:
                vis = ~obj[y, r0:r1]
                if vis.any():
                    chunks.append(rgb[y, r0:r1][vis])
            if not chunks:
                continue
            target = np.median(np.concatenate(chunks, axis=0), axis=0)
            out[y, obj[y]] = target

    struct = generate_binary_structure(2, 2)
    ring = binary_dilation(obj, structure=struct, iterations=min(ring_px, 10)) & ~obj
    if ring.any() and noise_scale > 0:
        ring_std = rgb[ring].std(axis=0) + 1e-4
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 1.0, size=(int(obj.sum()), 3)) * (ring_std * noise_scale)
        out[obj] = np.clip(out[obj] + noise, 0.0, 1.0)

    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def fill_object_absent_tensor(
    image: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    ring_px: int = 18,
    blur_sigma: float = 3.5,
    noise_scale: float = 0.45,
) -> torch.Tensor:
    """Tensor wrapper for counterfactual / FLUX source prep."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if object_mask.ndim == 2:
        object_mask = object_mask.unsqueeze(0)
    obj_u8 = (object_mask[0] > 0.5).detach().cpu().numpy().astype(np.uint8) * 255
    arr = image[0].permute(1, 2, 0).detach().cpu().numpy()
    if arr.max() <= 1.01:
        pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    else:
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    filled = fill_object_absent(pil, obj_u8, ring_px=ring_px, blur_sigma=blur_sigma, noise_scale=noise_scale)
    out = np.asarray(filled, dtype=np.float32) / 255.0
    t = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0).to(
        device=image.device, dtype=image.dtype
    )
    return t


def neutralize_object_source(
    image: Image.Image,
    object_mask_u8: np.ndarray,
    *,
    ring_px: int = 18,
) -> Image.Image:
    """Legacy name — ring fill for object region (JEPA counterfactual, not FLUX source)."""
    return fill_object_absent(image, object_mask_u8, ring_px=ring_px)


def harmonize_edit_region(
    output: Image.Image,
    source: Image.Image,
    edit_mask: Image.Image,
    *,
    ring_px: int = 14,
    row_strength: float = 0.20,
    row_margin: int = 32,
    texture_strength: float = 0.12,
    texture_sigma: float = 2.8,
    shadow_lift: float = 0.18,
    object_mask_u8: np.ndarray | None = None,
    shadow_soft_u8: np.ndarray | None = None,
) -> Image.Image:
    """Row-wise color propagation from source neighbors + shadow tone + micro-texture."""
    out = _to_rgb_f32(output)
    src = _to_rgb_f32(source)
    h, w = out.shape[:2]
    edit = np.asarray(edit_mask.convert("L")) > 127
    if edit.shape != (h, w):
        edit = (
            np.asarray(edit_mask.convert("L").resize((w, h), Image.Resampling.NEAREST))
            > 127
        )
    if not edit.any():
        return output

    obj = edit.copy()
    if object_mask_u8 is not None:
        om = object_mask_u8 > 127
        if om.shape != (h, w):
            om = (
                np.asarray(Image.fromarray(object_mask_u8).resize((w, h), Image.Resampling.NEAREST))
                > 127
            )
        if om.any():
            obj = om

    low_o = gaussian_filter(out, sigma=3.0)
    low_adj = low_o.copy()
    rows = np.where(obj.any(axis=1))[0]
    if row_strength > 0:
        for y in rows:
            xs = np.where(obj[y])[0]
            if xs.size == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            chunks: list[np.ndarray] = []
            l0, l1 = max(0, x0 - row_margin), x0
            r0, r1 = min(w, x1 + 1), min(w, x1 + 1 + row_margin)
            if l1 > l0:
                vis = ~edit[y, l0:l1]
                if vis.any():
                    chunks.append(src[y, l0:l1][vis])
            if r1 > r0:
                vis = ~edit[y, r0:r1]
                if vis.any():
                    chunks.append(src[y, r0:r1][vis])
            if not chunks:
                continue
            target = np.median(np.concatenate(chunks, axis=0), axis=0)
            mrow = obj[y]
            low_adj[y, mrow] = low_adj[y, mrow] * (1.0 - row_strength) + target * row_strength

    harmonized = low_adj + (out - low_o)

    struct = generate_binary_structure(2, 2)
    ring = binary_dilation(edit, structure=struct, iterations=ring_px) & ~edit
    if texture_strength > 0 and ring.sum() >= 48:
        low_h = gaussian_filter(harmonized, sigma=texture_sigma)
        high_s = src - gaussian_filter(src, sigma=texture_sigma)
        tex = np.median(high_s[ring], axis=0)
        harmonized = harmonized + texture_strength * edit[..., None].astype(np.float32) * (
            tex.reshape(1, 1, 3) - (harmonized - low_h).mean(axis=(0, 1)).reshape(1, 1, 3)
        )

    if shadow_lift > 0 and shadow_soft_u8 is not None:
        soft = shadow_soft_u8.astype(np.float32) / 255.0
        if soft.shape != (h, w):
            soft = (
                np.asarray(Image.fromarray(shadow_soft_u8).resize((w, h), Image.Resampling.BILINEAR))
                / 255.0
            )
        obj = np.zeros((h, w), dtype=bool)
        if object_mask_u8 is not None:
            obj = object_mask_u8 > 127
            if obj.shape != (h, w):
                obj = (
                    np.asarray(Image.fromarray(object_mask_u8).resize((w, h), Image.Resampling.NEAREST))
                    > 127
                )
        band = (soft > 0.10) & (~obj)
        if band.any() and ring.sum() > 0:
            gray = harmonized.mean(axis=-1)
            tgt = float(np.percentile(gray[ring], 54))
            gap = np.clip(tgt - gray, 0.0, 0.045)
            lift_w = (gap / 0.028).clip(0.0, 1.0) * band.astype(np.float32) * shadow_lift
            harmonized = np.clip(harmonized + lift_w[..., None], 0.0, 1.0)

    return Image.fromarray((np.clip(harmonized, 0, 1) * 255).astype(np.uint8))


def decontaminate_object_rim(
    output: Image.Image,
    source: Image.Image,
    object_mask: Image.Image,
    *,
    ring_px: int = 10,
    stuck_thresh: float = 18.0,
    blend: float = 0.82,
) -> Image.Image:
    """Remove object-color fringe just outside M_obj (mask antialias / paste bleed)."""
    out = _to_rgb_f32(output)
    src = _to_rgb_f32(source)
    h, w = out.shape[:2]
    obj = np.asarray(object_mask.convert("L")) > 127
    if obj.shape != (h, w):
        obj = (
            np.asarray(object_mask.convert("L").resize((w, h), Image.Resampling.NEAREST))
            > 127
        )
    if not obj.any():
        return output

    struct = generate_binary_structure(2, 2)
    ring = binary_dilation(obj, structure=struct, iterations=ring_px) & ~obj
    if not ring.any():
        return output

    diff = np.linalg.norm(out - src, axis=2)
    stuck = ring & (diff < stuck_thresh)
    if not stuck.any():
        return output

    ref_band = binary_dilation(obj, structure=struct, iterations=ring_px + 8) & ~binary_dilation(
        obj, structure=struct, iterations=max(2, ring_px // 2)
    )
    ref_band &= ~stuck
    if ref_band.any():
        ref = np.median(out[ref_band], axis=0)
    else:
        ref = np.median(out[~obj], axis=0)

    fix = out.copy()
    stuck_f = stuck.astype(np.float32)
    for c in range(3):
        ch = out[:, :, c]
        num = gaussian_filter(ch * (1.0 - stuck_f), sigma=3.0)
        den = gaussian_filter(1.0 - stuck_f, sigma=3.0)
        local = num / (den + 1e-6)
        target = blend * local + (1.0 - blend) * ref[c]
        fix[:, :, c] = np.where(stuck, target, ch)
    return Image.fromarray((np.clip(fix, 0, 1) * 255).astype(np.uint8))


def decontaminate_bright_halo(
    output: Image.Image,
    source: Image.Image,
    object_mask: Image.Image,
    *,
    ring_px: int = 14,
    bright_margin: float = 0.055,
    blend: float = 0.72,
) -> Image.Image:
    """Pull over-bright paste halo outside M_obj back toward source floor."""
    out = _to_rgb_f32(output)
    src = _to_rgb_f32(source)
    h, w = out.shape[:2]
    obj = np.asarray(object_mask.convert("L")) > 127
    if obj.shape != (h, w):
        obj = (
            np.asarray(object_mask.convert("L").resize((w, h), Image.Resampling.NEAREST))
            > 127
        )
    if not obj.any():
        return output

    struct = generate_binary_structure(2, 2)
    ring = binary_dilation(obj, structure=struct, iterations=ring_px) & ~obj
    far = binary_dilation(obj, structure=struct, iterations=ring_px + 10) & ~binary_dilation(
        obj, structure=struct, iterations=max(3, ring_px // 2)
    )
    if not ring.any():
        return output

    out_l = out.mean(axis=-1)
    src_l = src.mean(axis=-1)
    ref = float(np.median(src_l[far])) if far.any() else float(np.median(src_l[~obj]))
    hot = ring & (out_l > ref + bright_margin)
    if not hot.any():
        return output

    fix = out.copy()
    hot_f = hot.astype(np.float32)
    for c in range(3):
        ch = out[:, :, c]
        num = gaussian_filter(ch * (1.0 - hot_f), sigma=2.5)
        den = gaussian_filter(1.0 - hot_f, sigma=2.5)
        local = num / (den + 1e-6)
        ref_c = float(np.median(src[~obj, c])) if (~obj).any() else ref
        target = blend * local + (1.0 - blend) * ref_c
        fix[:, :, c] = np.where(hot, target, ch)
    return Image.fromarray((np.clip(fix, 0, 1) * 255).astype(np.uint8))


def erase_silhouette_rim(
    output: Image.Image,
    source: Image.Image,
    object_mask_u8: np.ndarray,
    *,
    rim_px: int = 10,
    core_px: int = 16,
    strength: float = 0.75,
    dark_thresh: float = 0.035,
) -> Image.Image:
    """Pull dark object ghost edge inside M_obj toward row-neighbor floor colors."""
    out = _to_rgb_f32(output)
    src = _to_rgb_f32(source)
    h, w = out.shape[:2]
    obj = _align_object_mask(object_mask_u8, h, w)
    if not obj.any() or rim_px <= 0:
        return output

    from scipy.ndimage import distance_transform_edt

    dist_in = distance_transform_edt(obj)
    rim = obj & (dist_in <= float(rim_px))
    core = obj & (dist_in > float(core_px))
    if not core.any():
        core = obj & (dist_in > float(rim_px + 2))
    if not rim.any():
        return output

    gray = out.mean(axis=-1)
    core_med = float(np.median(gray[core])) if core.any() else float(np.median(gray[obj]))
    stuck = rim & (gray < core_med - dark_thresh)
    if not stuck.any():
        return output

    fix = out.copy()
    rows = np.where(stuck.any(axis=1))[0]
    for y in rows:
        xs = np.where(stuck[y])[0]
        if xs.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        margin = max(10, rim_px + 6)
        chunks: list[np.ndarray] = []
        for lo, hi in ((max(0, x0 - margin), x0), (min(w, x1 + 1), min(w, x1 + 1 + margin))):
            if hi > lo:
                vis = ~obj[y, lo:hi]
                if vis.any():
                    chunks.append(out[y, lo:hi][vis])
        if not chunks:
            vis = ~obj[y]
            if vis.any():
                row_tgt = np.median(out[y][vis], axis=0)
            else:
                continue
        else:
            row_tgt = np.median(np.concatenate(chunks, axis=0), axis=0)
        mrow = stuck[y]
        fix[y, mrow] = fix[y, mrow] * (1.0 - strength) + row_tgt * strength

    return Image.fromarray((np.clip(fix, 0.0, 1.0) * 255).astype(np.uint8))


def replace_object_interior(
    output: Image.Image,
    source: Image.Image,
    object_mask_u8: np.ndarray,
    *,
    ring_px: int = 24,
) -> Image.Image:
    """Discard FLUX interior; fill M_obj from source row-propagate only."""
    filled = fill_flux_source(source, object_mask_u8, ring_px=ring_px)
    out = _to_rgb_f32(output)
    fill = _to_rgb_f32(filled)
    h, w = out.shape[:2]
    obj = _align_object_mask(object_mask_u8, h, w)
    if not obj.any():
        return output
    out[obj] = fill[obj]
    return Image.fromarray((np.clip(out, 0.0, 1.0) * 255).astype(np.uint8))


def composite_feather_seam(
    output: Image.Image,
    source: Image.Image,
    paste_mask: Image.Image,
    *,
    seam_px: int = 8,
) -> Image.Image:
    """Final composite: hard-preserve far outside, feather alpha at paste edge."""
    out = _to_rgb_f32(output)
    src = _to_rgb_f32(source)
    raw = np.asarray(paste_mask.convert("L"), dtype=np.float32) / 255.0
    if raw.shape[:2] != out.shape[:2]:
        raw = (
            np.asarray(
                paste_mask.convert("L").resize((out.shape[1], out.shape[0]), Image.Resampling.BILINEAR)
            ).astype(np.float32)
            / 255.0
        )
    edit = raw > 0.5
    if not edit.any():
        return source
    from scipy.ndimage import distance_transform_edt

    dist_out = distance_transform_edt(~edit)
    alpha = gaussian_filter(raw, sigma=max(1.0, float(seam_px) / 2.2))
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = np.where(dist_out > float(seam_px) + 1.0, 0.0, alpha)
    alpha3 = alpha[..., None]
    blended = np.clip(out * alpha3 + src * (1.0 - alpha3), 0.0, 1.0)
    return Image.fromarray((blended * 255).astype(np.uint8))


def refine_edit_boundary(
    output: Image.Image,
    source: Image.Image,
    object_mask_u8: np.ndarray,
    *,
    edge_px: int = 5,
    outer_px: int = 6,
    strength: float = 0.5,
) -> Image.Image:
    """Soften hard inpaint seam in a thin band around M_obj only."""
    out = _to_rgb_f32(output)
    src = _to_rgb_f32(source)
    h, w = out.shape[:2]
    obj = _align_object_mask(object_mask_u8, h, w)
    if not obj.any() or edge_px <= 0:
        return output

    from scipy.ndimage import distance_transform_edt

    dist_in = distance_transform_edt(obj)
    dist_out = distance_transform_edt(~obj)
    inner = obj & (dist_in <= float(edge_px))
    outer = (~obj) & (dist_out <= float(outer_px))
    band = inner | outer
    if not band.any():
        return output

    w_in = np.clip(1.0 - (dist_in - 1.0) / max(edge_px, 1), 0.0, 1.0) * inner.astype(np.float32)
    w_out = np.clip(1.0 - dist_out / max(outer_px, 1), 0.0, 1.0) * outer.astype(np.float32)
    weight = np.clip(np.maximum(w_in, w_out) * strength, 0.0, 1.0)

    target = out.copy()
    rows = np.where(obj.any(axis=1))[0]
    for y in rows:
        xs = np.where(obj[y])[0]
        if xs.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        margin = max(8, edge_px + outer_px + 4)
        chunks: list[np.ndarray] = []
        l0, l1 = max(0, x0 - margin), x0
        r0, r1 = min(w, x1 + 1), min(w, x1 + 1 + margin)
        if l1 > l0:
            vis = ~obj[y, l0:l1]
            if vis.any():
                chunks.append(src[y, l0:l1][vis])
        if r1 > r0:
            vis = ~obj[y, r0:r1]
            if vis.any():
                chunks.append(src[y, r0:r1][vis])
        if not chunks:
            continue
        row_tgt = np.median(np.concatenate(chunks, axis=0), axis=0)
        mrow = band[y]
        target[y, mrow] = target[y, mrow] * (1.0 - strength) + row_tgt * strength

    weight3 = weight[..., None]
    blended = np.clip(out * (1.0 - weight3) + target * weight3, 0.0, 1.0)
    return Image.fromarray((blended * 255).astype(np.uint8))


def feather_paste_visible(
    output: Image.Image,
    source: Image.Image,
    mask: Image.Image,
    feather_px: int = 12,
) -> Image.Image:
    """Soft blend outside M_flux — removes hard texture cliff at edit boundary."""
    out = _to_rgb_f32(output)
    src = _to_rgb_f32(source)
    m = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    if m.shape[:2] != out.shape[:2]:
        m = (
            np.asarray(mask.convert("L").resize((out.shape[1], out.shape[0]), Image.Resampling.BILINEAR))
            / 255.0
        )
    # mask white = edit; preserve outside = 1 where mask black
    preserve = np.clip(1.0 - m, 0.0, 1.0)
    if feather_px > 0:
        preserve = gaussian_filter(preserve, sigma=max(1.0, feather_px / 2.5))
    preserve3 = preserve[..., None]
    blended = np.clip(out * (1.0 - preserve3) + src * preserve3, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))


def lift_shadow_residual(
    output: Image.Image,
    shadow_soft_u8: np.ndarray,
    object_mask_u8: np.ndarray,
    *,
    lift_thresh: float = 0.028,
    max_lift: float = 0.12,
    source: Image.Image | None = None,
) -> Image.Image:
    """Lift pixels still too dark inside M_shadow (base crescent / penumbra)."""
    out = _to_rgb_f32(output)
    src = _to_rgb_f32(source) if source is not None else out
    h, w = out.shape[:2]
    soft = shadow_soft_u8.astype(np.float32) / 255.0
    obj = object_mask_u8 > 127
    if soft.shape != (h, w):
        soft = (
            np.asarray(Image.fromarray(shadow_soft_u8).resize((w, h), Image.Resampling.BILINEAR))
            / 255.0
        )
    if obj.shape != (h, w):
        obj = np.asarray(Image.fromarray(object_mask_u8).resize((w, h), Image.Resampling.NEAREST)) > 127
    band = (soft > 0.12) & (~obj)
    if not band.any():
        return output
    struct = generate_binary_structure(2, 2)
    ring = binary_dilation(band, structure=struct, iterations=12) & ~band & ~obj
    if not ring.any():
        ring = binary_dilation(band, structure=struct, iterations=16) & ~obj
    gray = out.mean(axis=-1)
    ref_gray = src.mean(axis=-1)
    tgt = float(np.percentile(ref_gray[ring], 56)) if ring.any() else float(ref_gray[~obj].mean())
    gap = np.clip(tgt - gray, 0.0, max_lift)
    lift = (gap / lift_thresh).clip(0.0, 1.0) * band.astype(np.float32) * 0.32
    lifted = np.clip(out + lift[..., None], 0.0, 1.0)
    return Image.fromarray((lifted * 255).astype(np.uint8))


def flatten_dark_blobs(
    output: Image.Image,
    effect_mask_u8: np.ndarray,
    *,
    ring_px: int = 14,
    dark_margin: float = 0.018,
    lift_strength: float = 0.92,
) -> Image.Image:
    """Remove local dark contact-shadow blobs still inside effect."""
    out = _to_rgb_f32(output)
    h, w = out.shape[:2]
    effect = _align_object_mask(effect_mask_u8, h, w)
    if not effect.any():
        return output

    struct = generate_binary_structure(2, 2)
    far = ~binary_dilation(effect, structure=struct, iterations=ring_px + 24)
    ring = binary_dilation(effect, structure=struct, iterations=ring_px) & ~effect
    ref = ring
    if far.sum() > 256:
        ref = far
    if not np.any(ref):
        return output

    gray = out.mean(axis=-1)
    tgt = float(np.percentile(gray[ref], 58))
    gap = np.clip(tgt - gray - dark_margin, 0.0, 0.38)
    w = gap * effect.astype(np.float32) * lift_strength
    lifted = np.clip(out + w[..., None], 0.0, 1.0)
    # Second pass for stubborn contact shadows.
    gray2 = lifted.mean(axis=-1)
    gap2 = np.clip(tgt - gray2 - dark_margin * 0.5, 0.0, 0.22)
    w2 = gap2 * effect.astype(np.float32) * lift_strength
    lifted = np.clip(lifted + w2[..., None], 0.0, 1.0)
    return Image.fromarray((lifted * 255).astype(np.uint8))


def color_match_effect_to_ring(
    output: Image.Image,
    effect_mask_u8: np.ndarray,
    *,
    ring_px: int = 14,
) -> Image.Image:
    """Match effect region color statistics to clean neighbor ring."""
    out = _to_rgb_f32(output)
    h, w = out.shape[:2]
    effect = _align_object_mask(effect_mask_u8, h, w)
    if not effect.any():
        return output

    struct = generate_binary_structure(2, 2)
    ring = binary_dilation(effect, structure=struct, iterations=ring_px) & ~effect
    if ring.sum() < 32:
        return output

    ref = np.median(out[ring], axis=0)
    cur = np.median(out[effect], axis=0)
    delta = ref - cur
    matched = out.copy()
    matched[effect] = np.clip(matched[effect] + delta.reshape(1, 1, 3), 0.0, 1.0)
    return Image.fromarray((matched * 255).astype(np.uint8))


def lift_effect_to_ring(
    output: Image.Image,
    effect_mask_u8: np.ndarray,
    *,
    ring_px: int = 16,
    max_lift: float = 0.16,
) -> Image.Image:
    """Lift dark contact-shadow residue inside effect to neighbor ring luminance."""
    out = _to_rgb_f32(output)
    h, w = out.shape[:2]
    effect = _align_object_mask(effect_mask_u8, h, w)
    if not effect.any():
        return output

    struct = generate_binary_structure(2, 2)
    ring = binary_dilation(effect, structure=struct, iterations=ring_px) & ~effect
    if not ring.any():
        return output

    gray = out.mean(axis=-1)
    tgt = float(np.percentile(gray[ring], 57))
    gap = np.clip(tgt - gray, 0.0, max_lift)
    lift = gap * effect.astype(np.float32) * 0.92
    lifted = np.clip(out + lift[..., None], 0.0, 1.0)
    return Image.fromarray((lifted * 255).astype(np.uint8))


def fill_effect_inpaint(
    source: Image.Image,
    effect_mask_u8: np.ndarray,
    *,
    radius: int = 9,
    lift_max: float = 0.22,
) -> Image.Image:
    """Telea inpaint on lifted effect mask — good for small floor/contact-shadow patches."""
    prepared = lift_effect_to_ring(
        source, effect_mask_u8, ring_px=14, max_lift=lift_max
    )
    try:
        import cv2

        rgb_u8 = np.asarray(prepared.convert("RGB"))
        mask_u8 = (effect_mask_u8 > 127).astype(np.uint8) * 255
        if mask_u8.shape[:2] != rgb_u8.shape[:2]:
            mask_u8 = (
                np.asarray(
                    Image.fromarray(mask_u8).resize(
                        (rgb_u8.shape[1], rgb_u8.shape[0]), Image.Resampling.NEAREST
                    )
                )
            )
        filled = cv2.inpaint(rgb_u8, mask_u8, max(3, radius), cv2.INPAINT_TELEA)
        return Image.fromarray(filled)
    except Exception:
        return fill_flux_source(prepared, effect_mask_u8)


def restore_inpaint_core(
    output: Image.Image,
    source: Image.Image,
    object_mask_u8: np.ndarray,
    *,
    effect_mask_u8: np.ndarray | None = None,
    ring_px: int = 3,
    interior_strength: float = 1.0,
    fill_ring_px: int = 24,
    hard_replace: bool = True,
    force_row_fill: bool = False,
) -> Image.Image:
    """Replace FLUX interior with neighbor inpaint; default hard fill inside effect mask."""
    fill_mask = effect_mask_u8 if effect_mask_u8 is not None else object_mask_u8
    out = _to_rgb_f32(output)
    h, w = out.shape[:2]
    effect = _align_object_mask(fill_mask, h, w)
    obj = _align_object_mask(object_mask_u8, h, w)
    obj_ratio = float(obj.sum()) / float(h * w) if obj.any() else 0.0
    if force_row_fill or (obj_ratio >= 0.10 and obj_ratio < 0.28):
        filled_src = fill_flux_source(
            lift_effect_to_ring(source, fill_mask, ring_px=max(12, fill_ring_px // 2)),
            fill_mask,
            ring_px=fill_ring_px,
            force_row=force_row_fill,
        )
    else:
        filled_src = fill_effect_inpaint(source, fill_mask, radius=9, lift_max=0.22)
    filled = _to_rgb_f32(filled_src)
    if not effect.any():
        return output

    if hard_replace and interior_strength >= 0.99:
        effect3 = effect[..., None].astype(np.float32)
        mixed = np.clip(out * (1.0 - effect3) + filled * effect3, 0.0, 1.0)
        mixed_pil = Image.fromarray((mixed * 255).astype(np.uint8))
        mixed_pil = lift_effect_to_ring(mixed_pil, fill_mask, ring_px=max(10, fill_ring_px // 2))
        mixed_pil = flatten_dark_blobs(mixed_pil, fill_mask, ring_px=max(12, fill_ring_px // 2))
        return color_match_effect_to_ring(mixed_pil, fill_mask, ring_px=max(12, fill_ring_px // 2))

    from scipy.ndimage import distance_transform_edt

    dist = distance_transform_edt(effect)
    w = np.clip(dist / max(ring_px, 1), 0.0, 1.0) * interior_strength
    w3 = (w * effect)[..., None]
    mixed = np.clip(out * (1.0 - w3) + filled * w3, 0.0, 1.0)
    return Image.fromarray((mixed * 255).astype(np.uint8))


def graft_ring_texture(
    output: Image.Image,
    source: Image.Image,
    edit_mask: Image.Image,
    *,
    sigma: float = 3.0,
    strength: float = 0.38,
    ring_px: int = 14,
) -> Image.Image:
    """Inject high-frequency detail from clean ring around edit (stars / grain)."""
    out = _to_rgb_f32(output)
    src = _to_rgb_f32(source)
    edit = np.asarray(edit_mask.convert("L")) > 127
    if edit.shape != out.shape[:2]:
        edit = (
            np.asarray(edit_mask.convert("L").resize((out.shape[1], out.shape[0]), Image.Resampling.NEAREST))
            > 127
        )
    if not edit.any():
        return output

    struct = generate_binary_structure(2, 2)
    ring = binary_dilation(edit, structure=struct, iterations=ring_px) & ~edit
    if ring.sum() < 64:
        return output

    low_o = gaussian_filter(out, sigma=sigma)
    low_s = gaussian_filter(src, sigma=sigma)
    high_s = src - low_s
    ring_high = high_s[ring]
    if ring_high.shape[0] < 64:
        return output

    rng = np.random.default_rng(0)
    edit_idx = np.where(edit)
    n_edit = int(edit.sum())
    sampled = ring_high[rng.integers(0, ring_high.shape[0], size=n_edit)]
    mixed = out.copy()
    mixed[edit_idx] = low_o[edit_idx] + strength * sampled
    return Image.fromarray((np.clip(mixed, 0, 1) * 255).astype(np.uint8))
