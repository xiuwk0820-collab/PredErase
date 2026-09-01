"""Official ReMOVE metric (SAM ViT-H + square crop). SmartEraser Table-2/3 protocol.

Reference: Chandrasekar et al., ReMOVE, CVPRW 2024.
Official code: https://github.com/chandrasekaraditya/ReMOVE
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from erase_world.lib.bench.remove_sam_predictor import RemoveSamPredictor

_ROOT = Path(__file__).resolve().parents[3]  # repo root
_DEFAULT_SAM_CKPT = _ROOT / "checkpoints" / "sam_vit_h_4b8939.pth"

_SAM_PREDICTOR: RemoveSamPredictor | None = None
_REMOVE_DEVICE: str | None = None
_SAM_CKPT: Path | None = None


def set_remove_device(device: str | None) -> None:
    global _REMOVE_DEVICE, _SAM_PREDICTOR
    dev = (device or "cpu").strip().lower()
    if dev == "cuda" and __import__("torch").cuda.is_available():
        dev = "cuda:0"
    if dev != _REMOVE_DEVICE:
        _SAM_PREDICTOR = None
    _REMOVE_DEVICE = dev


def set_sam_checkpoint(path: str | Path | None) -> None:
    global _SAM_CKPT, _SAM_PREDICTOR
    p = Path(path) if path is not None else _DEFAULT_SAM_CKPT
    if _SAM_CKPT != p:
        _SAM_PREDICTOR = None
    _SAM_CKPT = p


def _resolve_remove_device() -> str:
    if _REMOVE_DEVICE is not None:
        return _REMOVE_DEVICE
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _get_sam_predictor() -> RemoveSamPredictor:
    global _SAM_PREDICTOR
    if _SAM_PREDICTOR is None:
        import torch
        from segment_anything import sam_model_registry

        ckpt = _SAM_CKPT or _DEFAULT_SAM_CKPT
        if not ckpt.is_file():
            raise FileNotFoundError(f"SAM checkpoint missing: {ckpt}")
        dev = _resolve_remove_device()
        sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
        if dev.startswith("cuda"):
            sam = sam.to(dev)
        else:
            sam = sam.to("cpu")
        sam.eval()
        _SAM_PREDICTOR = RemoveSamPredictor(sam)
    return _SAM_PREDICTOR


def find_smallest_bounding_square(mask_u8: np.ndarray) -> tuple[int, int, int] | None:
    """ReMOVE square crop (crop.py). Returns x, y, size."""
    if mask_u8.ndim == 3:
        mask_u8 = np.array(Image.fromarray(mask_u8).convert("L"))
    white = mask_u8 > 127
    if not white.any():
        return None
    ys, xs = np.where(white)
    min_row, max_row = int(ys.min()), int(ys.max())
    min_col, max_col = int(xs.min()), int(xs.max())
    width = max_col - min_col + 1
    height = max_row - min_row + 1
    size = max(width, height)
    h, w = mask_u8.shape
    sub = 16
    if not (min_col - sub >= 0 and min_row - sub >= 0 and max_row + sub <= h and max_col + sub <= w):
        sub = max(min(min_col, min_row, h - 1 - max_row, w - 1 - max_col), 0)
    return min_col - sub, min_row - sub, size + 2 * sub


def compute_remove(
    pred: Image.Image,
    mask: Image.Image,
    *,
    crop: bool = True,
) -> float:
    """ReMOVE ↑ : SAM encoder cosine similarity (fg vs bg), official crop variant."""
    import torch
    from torch.nn.functional import cosine_similarity

    predictor = _get_sam_predictor()
    img = np.asarray(pred.convert("RGB"), dtype=np.uint8)
    m = np.asarray(mask.convert("L"), dtype=np.uint8)

    if crop:
        bb = find_smallest_bounding_square(m)
        if bb is not None:
            x0, y0, size = bb
            img = img[y0 : y0 + size, x0 : x0 + size]
            m = m[y0 : y0 + size, x0 : x0 + size]

    if not (m > 127).any():
        return float("nan")

    mask_fg = (
        np.array(Image.fromarray(m).resize((64, 64), Image.Resampling.NEAREST))
        .reshape((1, 1, 64, 64))
        // 255
    )
    mask_bg = 1 - mask_fg

    embeddings = predictor.get_aggregate_features(img, [mask_fg, mask_bg])
    return float(cosine_similarity(embeddings[0], embeddings[1]).item())
