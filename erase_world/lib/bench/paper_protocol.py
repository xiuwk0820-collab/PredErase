"""OmniEraser RemovalBench metrics (arxiv:2501.07397, Table 1).

Per-image (pred resized to GT, then mean):
  PSNR  — full-image paired PSNR (shadows/reflections live outside object mask)
  LPIPS — SqueezeNet, full image
  DINO  — DINOv2-base patch L2 in mask region (optional structural check)

Dataset-level:
  FID, CMMD, AS — same as paper tables
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from erase_world.lib.bench.omnieraser_protocol import OmniEraserProtocol, OmniEraserProtocolConfig, _cmmd

_LPIPS_SQUEEZE = None
_DINO_PROC = None
_DINO_MODEL = None
_CLIP_FOR_CMMD = None
_CLIP_PROC_CMMD = None
_METRICS_DEVICE: str | None = None


def set_metrics_device(device: str | None) -> None:
    """Pin LPIPS/DINO/CLIP helpers (cpu | cuda:N). Use cpu for multi-GPU bench shards."""
    global _METRICS_DEVICE, _LPIPS_SQUEEZE, _DINO_PROC, _DINO_MODEL, _CLIP_FOR_CMMD, _CLIP_PROC_CMMD
    dev = (device or "cpu").strip().lower()
    if dev == "cuda" and __import__("torch").cuda.is_available():
        dev = "cuda:0"
    if dev != _METRICS_DEVICE:
        _LPIPS_SQUEEZE = None
        _DINO_PROC = None
        _DINO_MODEL = None
        _CLIP_FOR_CMMD = None
        _CLIP_PROC_CMMD = None
    _METRICS_DEVICE = dev


def _resolve_metrics_device() -> str:
    if _METRICS_DEVICE is not None:
        return _METRICS_DEVICE
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _to_metrics_device(module):
    dev = _resolve_metrics_device()
    if dev.startswith("cuda"):
        return module.to(dev)
    return module.to("cpu")


def _resize_pred_to_gt(pred: Image.Image, gt: Image.Image) -> Image.Image:
    pred = pred.convert("RGB")
    gt = gt.convert("RGB")
    if pred.size != gt.size:
        pred = pred.resize(gt.size, Image.Resampling.BILINEAR)
    return pred


def _inpaint_mask(mask: Image.Image, shape: tuple[int, int]) -> np.ndarray:
    """Binary mask (H,W,1) with 1 inside inpaint / removal region."""
    m = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    if m.shape != shape[:2]:
        m = (
            np.asarray(
                Image.fromarray((m * 255).astype(np.uint8)).resize(
                    (shape[1], shape[0]), Image.Resampling.NEAREST
                ),
                dtype=np.float32,
            )
            / 255.0
        )
    return m[..., None]


def compute_full_psnr(pred: Image.Image, gt: Image.Image) -> float:
    """Full-image PSNR (pred aligned to GT). Counts shadow/reflection outside object mask."""
    from skimage.metrics import peak_signal_noise_ratio

    gt = gt.convert("RGB")
    pred = _resize_pred_to_gt(pred, gt)
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    return float(peak_signal_noise_ratio(g, p, data_range=255))


def compute_mask_psnr(pred: Image.Image, gt: Image.Image, mask: Image.Image) -> float:
    """Optional: BrushNet-style PSNR on mask pixels only (not used in main table)."""
    gt = gt.convert("RGB")
    pred = _resize_pred_to_gt(pred, gt)
    p = np.asarray(pred, dtype=np.float32) / 255.0
    g = np.asarray(gt, dtype=np.float32) / 255.0
    m = _inpaint_mask(mask, g.shape)
    diff = (p - g) * m
    denom = max(float(m.sum()) * 3.0, 1.0)
    mse = float((diff ** 2).sum() / denom)
    if mse < 1.0e-10:
        return 1000.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def _get_lpips_squeeze():
    global _LPIPS_SQUEEZE
    if _LPIPS_SQUEEZE is None:
        import lpips
        import torch

        _LPIPS_SQUEEZE = lpips.LPIPS(net="squeeze")
        _LPIPS_SQUEEZE.eval()
        _LPIPS_SQUEEZE = _to_metrics_device(_LPIPS_SQUEEZE)
    return _LPIPS_SQUEEZE


def compute_paper_lpips(pred: Image.Image, gt: Image.Image) -> float:
    """Squeeze LPIPS on full image (pred aligned to GT resolution)."""
    import torch

    gt = gt.convert("RGB")
    pred = _resize_pred_to_gt(pred, gt)
    to_t = lambda im: (
        torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        * 2.0
        - 1.0
    )
    model = _get_lpips_squeeze()
    dev = next(model.parameters()).device
    with torch.no_grad():
        return float(model(to_t(pred).to(dev), to_t(gt).to(dev)).item())


def _get_dino():
    global _DINO_PROC, _DINO_MODEL
    if _DINO_MODEL is None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        _DINO_PROC = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        _DINO_MODEL = AutoModel.from_pretrained("facebook/dinov2-base")
        _DINO_MODEL.eval()
        _DINO_MODEL = _to_metrics_device(_DINO_MODEL)
    return _DINO_PROC, _DINO_MODEL


def _masked_pair(pred: Image.Image, gt: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    gt = gt.convert("RGB")
    pred = _resize_pred_to_gt(pred, gt)
    g = np.asarray(gt, dtype=np.float32) / 255.0
    p = np.asarray(pred, dtype=np.float32) / 255.0
    m = _inpaint_mask(mask, g.shape)
    return (
        Image.fromarray(np.clip(p * m * 255.0, 0, 255).astype(np.uint8)),
        Image.fromarray(np.clip(g * m * 255.0, 0, 255).astype(np.uint8)),
    )


def compute_paper_dino(pred: Image.Image, gt: Image.Image, mask: Image.Image) -> float:
    """DINOv2 patch L2 (L2-normalized tokens), averaged over mask patches."""
    import torch
    import torch.nn.functional as F

    proc, model = _get_dino()
    dev = next(model.parameters()).device
    mp, mg = _masked_pair(pred, gt, mask)
    m = _inpaint_mask(mask, np.asarray(gt.convert("RGB")).shape)
    with torch.no_grad():
        f0 = model(
            pixel_values=proc(images=mp, return_tensors="pt")["pixel_values"].to(dev)
        ).last_hidden_state[:, 1:, :]
        f1 = model(
            pixel_values=proc(images=mg, return_tensors="pt")["pixel_values"].to(dev)
        ).last_hidden_state[:, 1:, :]
    f0 = f0 / (f0.norm(dim=-1, keepdim=True) + 1e-8)
    f1 = f1 / (f1.norm(dim=-1, keepdim=True) + 1e-8)
    ph = int(f0.shape[1] ** 0.5)
    mt = (
        F.interpolate(
            torch.from_numpy(m[..., 0])[None, None].float().to(dev),
            size=(ph, ph),
            mode="nearest",
        )
        .reshape(1, -1)
    )
    if float(mt.sum()) <= 0:
        return float("nan")
    dist = torch.norm(f0 - f1, dim=-1)
    return float((dist * mt).sum() / mt.sum())


def _get_clip_cmmd():
    global _CLIP_FOR_CMMD, _CLIP_PROC_CMMD
    if _CLIP_FOR_CMMD is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        name = "openai/clip-vit-large-patch14-336"
        _CLIP_FOR_CMMD = CLIPModel.from_pretrained(name)
        _CLIP_PROC_CMMD = CLIPProcessor.from_pretrained(name)
        _CLIP_FOR_CMMD.eval()
        _CLIP_FOR_CMMD = _to_metrics_device(_CLIP_FOR_CMMD)
    return _CLIP_FOR_CMMD, _CLIP_PROC_CMMD


def _clip_embed_cmmd(img: Image.Image) -> np.ndarray:
    import torch

    model, proc = _get_clip_cmmd()
    dev = next(model.parameters()).device
    inputs = proc(images=img.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**{k: v.to(dev) for k, v in inputs.items()})
        if not torch.is_tensor(feats):
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[0]
        feats = feats.float()
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    return feats.detach().cpu().numpy()[0]


@dataclass
class PaperProtocolAccumulator:
    """Streaming accumulator for RemovalBench paper metrics."""

    cfg: OmniEraserProtocolConfig | None = None

    def __post_init__(self) -> None:
        self._dataset = OmniEraserProtocol(self.cfg or OmniEraserProtocolConfig())
        self._pred_clip: list[np.ndarray] = []
        self._gt_clip: list[np.ndarray] = []
        self._psnr: list[float] = []
        self._lpips: list[float] = []
        self._dino: list[float] = []

    def update(self, pred: Image.Image, gt: Image.Image, mask: Image.Image) -> None:
        pred = _resize_pred_to_gt(pred, gt)
        self._dataset.update(pred, gt)
        self._pred_clip.append(_clip_embed_cmmd(pred))
        self._gt_clip.append(_clip_embed_cmmd(gt))
        self._psnr.append(compute_full_psnr(pred, gt))
        self._lpips.append(compute_paper_lpips(pred, gt))
        self._dino.append(compute_paper_dino(pred, gt, mask))

    def compute(self) -> dict[str, float]:
        ds = self._dataset.compute()
        out = {
            "protocol": "omnieraser",
            "FID": ds.get("FID", float("nan")),
            "CMMD": float("nan"),
            "LPIPS": float(np.nanmean(self._lpips)) if self._lpips else float("nan"),
            "DINO": float(np.nanmean(self._dino)) if self._dino else float("nan"),
            "PSNR": float(np.nanmean(self._psnr)) if self._psnr else float("nan"),
            "AS": ds.get("AS", float("nan")),
            "n": len(self._psnr),
        }
        if self._pred_clip and self._gt_clip:
            out["CMMD"] = float(_cmmd(np.stack(self._pred_clip), np.stack(self._gt_clip)))
        return out


def compute_paper_metrics(
    pred: Image.Image,
    gt: Image.Image,
    mask: Image.Image,
) -> dict[str, float]:
    pred = _resize_pred_to_gt(pred, gt)
    return {
        "paper_psnr": compute_full_psnr(pred, gt),
        "paper_psnr_mask": compute_mask_psnr(pred, gt, mask),
        "paper_lpips": compute_paper_lpips(pred, gt),
        "paper_dino": compute_paper_dino(pred, gt, mask),
    }
