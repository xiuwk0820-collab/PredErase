"""SmartEraser bench metrics (DEFACTO-Val / RORD-Val / Syn4Removal-Val).

Dataset-level: FID↓, CMMD↓
Per-image mean: ReMOVE↑, LPIPS↓ (Alex, mask bbox), SSIM↑ (mask bbox), PSNR↑ (full image)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from erase_world.lib.bench.metrics import compute_image_metrics
from erase_world.lib.bench.omnieraser_protocol import OmniEraserProtocol, OmniEraserProtocolConfig, _cmmd
from erase_world.lib.bench.remove_metric import compute_remove


@dataclass
class SmartEraserProtocolConfig:
    remove_crop: bool = True


class SmartEraserProtocolAccumulator:
    def __init__(self, cfg: SmartEraserProtocolConfig | None = None):
        self.cfg = cfg or SmartEraserProtocolConfig()
        omni_cfg = OmniEraserProtocolConfig(compute_fid=True, compute_cmmd=True, compute_as=False)
        self._dataset = OmniEraserProtocol(omni_cfg)
        self._pred_clip: list[np.ndarray] = []
        self._gt_clip: list[np.ndarray] = []
        self._remove: list[float] = []
        self._lpips: list[float] = []
        self._ssim: list[float] = []
        self._psnr: list[float] = []

    def update(self, pred: Image.Image, gt: Image.Image, mask: Image.Image) -> dict[str, float]:
        pred = pred.convert("RGB")
        gt = gt.convert("RGB")
        self._dataset.update(pred, gt)
        from erase_world.lib.bench.paper_protocol import _clip_embed_cmmd

        self._pred_clip.append(_clip_embed_cmmd(pred))
        self._gt_clip.append(_clip_embed_cmmd(gt))

        per = compute_smarteraser_metrics(pred, gt, mask, remove_crop=self.cfg.remove_crop)
        self._remove.append(per["REMOVE"])
        self._lpips.append(per["LPIPS"])
        self._ssim.append(per["SSIM"])
        self._psnr.append(per["PSNR"])
        return per

    def compute(self) -> dict[str, float]:
        ds = self._dataset.compute()
        out = {
            "protocol": "smarteraser",
            "FID": ds.get("FID", float("nan")),
            "CMMD": float("nan"),
            "REMOVE": float(np.nanmean(self._remove)) if self._remove else float("nan"),
            "LPIPS": float(np.nanmean(self._lpips)) if self._lpips else float("nan"),
            "SSIM": float(np.nanmean(self._ssim)) if self._ssim else float("nan"),
            "PSNR": float(np.nanmean(self._psnr)) if self._psnr else float("nan"),
            "n": len(self._psnr),
        }
        if self._pred_clip and self._gt_clip:
            out["CMMD"] = float(_cmmd(np.stack(self._pred_clip), np.stack(self._gt_clip)))
        return out


def compute_smarteraser_metrics(
    pred: Image.Image,
    gt: Image.Image,
    mask: Image.Image,
    *,
    remove_crop: bool = True,
) -> dict[str, float]:
    m = compute_image_metrics(pred, gt, mask, compute_lpips_metric=True)
    return {
        "REMOVE": compute_remove(pred, mask, crop=remove_crop),
        "LPIPS": m.get("paper_lpips", float("nan")),
        "SSIM": m.get("paper_ssim", float("nan")),
        "PSNR": m.get("paper_psnr", float("nan")),
        "psnr_mask": m.get("psnr_mask", float("nan")),
        "ssim_full": m.get("ssim_full", float("nan")),
    }
