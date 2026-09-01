from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_LPIPS_MODEL = None
_LPIPS_DEVICE: str | None = None


def set_lpips_device(device: str | None) -> None:
    global _LPIPS_DEVICE, _LPIPS_MODEL
    dev = (device or "cpu").strip().lower()
    if dev == "cuda" and __import__("torch").cuda.is_available():
        dev = "cuda:0"
    if dev != _LPIPS_DEVICE:
        _LPIPS_MODEL = None
    _LPIPS_DEVICE = dev


def _resolve_lpips_device() -> str:
    if _LPIPS_DEVICE is not None:
        return _LPIPS_DEVICE
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _to_rgb_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float64)


def _mask_bool(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask.convert("L")) > 127


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _get_lpips_model():
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips
        import torch

        _LPIPS_MODEL = lpips.LPIPS(net="alex")
        _LPIPS_MODEL.eval()
        dev = _resolve_lpips_device()
        _LPIPS_MODEL = _LPIPS_MODEL.to(dev if dev.startswith("cuda") else "cpu")
    return _LPIPS_MODEL


def compute_lpips(
    pred: Image.Image,
    gt: Image.Image,
    mask: Optional[Image.Image] = None,
) -> float:
    """Alex LPIPS; optional mask bbox crop (SmartEraser GT-consistency region)."""
    import torch

    p = pred.convert("RGB")
    g = gt.convert("RGB")
    if mask is not None:
        m = _mask_bool(mask)
        if m.any():
            y0, y1, x0, x1 = _mask_bbox(m)
            p = p.crop((x0, y0, x1, y1))
            g = g.crop((x0, y0, x1, y1))

    to_t = lambda im: torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
    t0, t1 = to_t(p), to_t(g)
    model = _get_lpips_model()
    dev = next(model.parameters()).device
    with torch.no_grad():
        return float(model(t0.to(dev), t1.to(dev)).item())


def _resize_to_match(pred: Image.Image, gt: Image.Image) -> Image.Image:
    if pred.size != gt.size:
        return pred.resize(gt.size, Image.Resampling.BILINEAR)
    return pred


def compute_image_metrics(
    pred: Image.Image,
    gt: Image.Image,
    mask: Image.Image,
    compute_lpips_metric: bool = True,
) -> dict[str, float]:
    """SmartEraser-style metrics: full-image PSNR, mask-bbox LPIPS/SSIM."""
    pred = _resize_to_match(pred.convert("RGB"), gt.convert("RGB"))
    p = _to_rgb_float(pred)
    g = _to_rgb_float(gt)
    m = _mask_bool(mask)

    out: dict[str, float] = {
        "psnr_full": float(peak_signal_noise_ratio(g, p, data_range=255)),
        "ssim_full": float(structural_similarity(g, p, channel_axis=2, data_range=255)),
    }
    if m.any():
        y0, y1, x0, x1 = _mask_bbox(m)
        pc = p[y0:y1, x0:x1]
        gc = g[y0:y1, x0:x1]
        mc = m[y0:y1, x0:x1]

        out["psnr_mask"] = float(peak_signal_noise_ratio(gc[mc], pc[mc], data_range=255))
        out["ssim_mask"] = float(structural_similarity(gc, pc, channel_axis=2, data_range=255))
        out["mask_ratio"] = float(m.mean())
        out["paper_psnr"] = out["psnr_full"]
        out["paper_ssim"] = out["ssim_mask"]
        if compute_lpips_metric:
            out["paper_lpips"] = compute_lpips(pred, gt, mask)
    else:
        out["psnr_mask"] = float("nan")
        out["ssim_mask"] = float("nan")
        out["mask_ratio"] = 0.0
        out["paper_psnr"] = float("nan")
        out["paper_ssim"] = float("nan")
        if compute_lpips_metric:
            out["paper_lpips"] = float("nan")
    return out


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
