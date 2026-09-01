from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


def _pil_to_numpy_rgb01(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


@dataclass
class OmniEraserProtocolConfig:
    """OmniEraser RORD-Val table columns.

    Columns: FID, CMMD, LPIPS, PSNR, AS
    """

    compute_fid: bool = True
    compute_cmmd: bool = True
    compute_as: bool = True
    # CMMD uses Gaussian RBF kernel with fixed sigma=10 and 1000 scaling
    cmmd_sigma: float = 10.0
    cmmd_scale: float = 1000.0
    # CLIP backbone recommended by CMMD paper: ViT-L/14@336px
    clip_model_name: str = "openai/clip-vit-large-patch14-336"
    # Aesthetic predictor: LAION linear head on CLIP ViT-L/14
    aesthetic_head_url: str = (
        "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth"
    )
    # Inception backbone for FID (torchvision inception_v3, pool3 features)
    fid_resize: int = 299


class OmniEraserProtocol:
    """Dataset-level metrics used by OmniEraser tables.

    Dataset-level: FID/CMMD compare pred vs GT feature distributions (full image).
    AS is the mean LAION aesthetic score on predicted images only.
    Per-image LPIPS/PSNR are aggregated separately in metrics.py (full-image paired).
    """

    def __init__(self, cfg: Optional[OmniEraserProtocolConfig] = None):
        self.cfg = cfg or OmniEraserProtocolConfig()
        self._pred_incep: list[np.ndarray] = []
        self._gt_incep: list[np.ndarray] = []
        self._pred_clip: list[np.ndarray] = []
        self._gt_clip: list[np.ndarray] = []
        self._as_scores: list[float] = []

        self._inception = None
        self._clip = None
        self._clip_processor = None
        self._aesthetic_head = None

    # ----------------------------- backbones -----------------------------
    def _get_inception(self):
        if self._inception is not None:
            return self._inception
        import torch
        import torchvision

        # torchvision enforces aux_logits=True for pretrained weights; we can
        # ignore auxiliary head outputs and still use the avgpool features.
        m = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True,
            transform_input=False,
        )
        m.eval()
        if torch.cuda.is_available():
            m = m.cuda()
        # We'll take features from the final avgpool (2048)
        self._inception = m
        return m

    def _get_clip(self):
        if self._clip is not None:
            return self._clip, self._clip_processor
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(self.cfg.clip_model_name)
        proc = CLIPProcessor.from_pretrained(self.cfg.clip_model_name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self._clip = model
        self._clip_processor = proc
        return model, proc

    def _get_aesthetic_head(self):
        if self._aesthetic_head is not None:
            return self._aesthetic_head
        import torch
        import torch.nn as nn

        head = nn.Linear(768, 1)
        sd = torch.hub.load_state_dict_from_url(self.cfg.aesthetic_head_url, map_location="cpu")
        head.load_state_dict(sd)
        head.eval()
        if torch.cuda.is_available():
            head = head.cuda()
        self._aesthetic_head = head
        return head

    # ----------------------------- feature extraction -----------------------------
    def _inception_feat(self, img: Image.Image) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        m = self._get_inception()
        dev = next(m.parameters()).device
        x = _pil_to_numpy_rgb01(img)
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW, [0,1]
        t = F.interpolate(t, size=(self.cfg.fid_resize, self.cfg.fid_resize), mode="bilinear", align_corners=False)
        # torchvision Inception expects roughly [-1,1] when transform_input=False? Official weights are trained on [0,1] normalized by ImageNet stats.
        # We'll use the canonical ImageNet normalization here.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t = (t - mean) / std
        t = t.to(dev)
        with torch.no_grad():
            # forward to logits, but we want avgpool features
            # We'll call underlying layers explicitly for stability across torchvision versions.
            y = m.Conv2d_1a_3x3(t)
            y = m.Conv2d_2a_3x3(y)
            y = m.Conv2d_2b_3x3(y)
            y = m.maxpool1(y)
            y = m.Conv2d_3b_1x1(y)
            y = m.Conv2d_4a_3x3(y)
            y = m.maxpool2(y)
            y = m.Mixed_5b(y)
            y = m.Mixed_5c(y)
            y = m.Mixed_5d(y)
            y = m.Mixed_6a(y)
            y = m.Mixed_6b(y)
            y = m.Mixed_6c(y)
            y = m.Mixed_6d(y)
            y = m.Mixed_6e(y)
            y = m.Mixed_7a(y)
            y = m.Mixed_7b(y)
            y = m.Mixed_7c(y)
            y = m.avgpool(y)
            y = torch.flatten(y, 1)  # 1x2048
        return y.detach().float().cpu().numpy()[0]

    def _clip_embed(self, img: Image.Image) -> np.ndarray:
        import torch

        model, proc = self._get_clip()
        dev = next(model.parameters()).device
        inputs = proc(images=img.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            feats = model.get_image_features(**{k: v.to(dev) for k, v in inputs.items()})
            if not torch.is_tensor(feats):
                if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                    feats = feats.image_embeds
                elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
                else:
                    feats = feats.last_hidden_state[:, 0]
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.detach().float().cpu().numpy()[0]

    def _aesthetic_score_from_clip(self, clip_embed: np.ndarray) -> float:
        import torch

        head = self._get_aesthetic_head()
        dev = next(head.parameters()).device
        x = torch.from_numpy(clip_embed).to(dev).unsqueeze(0)
        with torch.no_grad():
            y = head(x).squeeze(0).squeeze(-1)
        return float(y.detach().cpu().item())

    # ----------------------------- public API -----------------------------
    def update(self, pred: Image.Image, gt: Image.Image) -> None:
        if self.cfg.compute_fid:
            try:
                self._pred_incep.append(self._inception_feat(pred))
                self._gt_incep.append(self._inception_feat(gt))
            except Exception:
                # Network/caching issues should not crash the whole benchmark.
                self.cfg.compute_fid = False
                self._pred_incep.clear()
                self._gt_incep.clear()

        if self.cfg.compute_cmmd or self.cfg.compute_as:
            try:
                pe = self._clip_embed(pred)
                ge = self._clip_embed(gt)
            except Exception:
                self.cfg.compute_cmmd = False
                self.cfg.compute_as = False
                self._pred_clip.clear()
                self._gt_clip.clear()
                self._as_scores.clear()
                return

            if self.cfg.compute_cmmd:
                self._pred_clip.append(pe)
                self._gt_clip.append(ge)

            if self.cfg.compute_as:
                try:
                    self._as_scores.append(self._aesthetic_score_from_clip(pe))
                except Exception:
                    self.cfg.compute_as = False
                    self._as_scores.clear()

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.cfg.compute_fid and self._pred_incep and self._gt_incep:
            out["FID"] = float(_fid(np.stack(self._pred_incep), np.stack(self._gt_incep)))
        if self.cfg.compute_cmmd and self._pred_clip and self._gt_clip:
            out["CMMD"] = float(
                _cmmd(
                    np.stack(self._pred_clip),
                    np.stack(self._gt_clip),
                    sigma=self.cfg.cmmd_sigma,
                    scale=self.cfg.cmmd_scale,
                )
            )
        if self.cfg.compute_as and self._as_scores:
            out["AS"] = float(np.mean(self._as_scores))
        return out


def _mu_cov(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0)
    xc = x - mu
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    return mu, cov


def _fid(pred_feats: np.ndarray, gt_feats: np.ndarray) -> float:
    """Fréchet distance between two Gaussians fitted to features."""
    import scipy.linalg

    mu1, cov1 = _mu_cov(pred_feats)
    mu2, cov2 = _mu_cov(gt_feats)
    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(cov1 @ cov2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * covmean))


def _rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    y_sq = np.sum(y * y, axis=1, keepdims=True).T
    dist2 = np.maximum(x_sq + y_sq - 2.0 * (x @ y.T), 0.0)
    gamma = 1.0 / (2.0 * float(sigma) ** 2)
    return np.exp(-gamma * dist2)


def _cmmd(pred_clip: np.ndarray, gt_clip: np.ndarray, sigma: float = 10.0, scale: float = 1000.0) -> float:
    """CMMD = 1000 * (E[k(xx)] + E[k(yy)] - 2E[k(xy)]) with RBF kernel.

    Matches the common CLIP-MMD definition used in CMMD literature.
    """
    k_xx = _rbf_kernel(pred_clip, pred_clip, sigma=sigma).mean()
    k_yy = _rbf_kernel(gt_clip, gt_clip, sigma=sigma).mean()
    k_xy = _rbf_kernel(pred_clip, gt_clip, sigma=sigma).mean()
    return float(scale * (k_xx + k_yy - 2.0 * k_xy))

