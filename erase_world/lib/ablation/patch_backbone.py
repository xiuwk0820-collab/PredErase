"""Patch-level guidance ablations: swap JEPA encoder for DINO / CLIP (RemovalBench)."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..core.types import GuidanceCache, GuidanceConfig, GuidanceState, StrategyMetrics
from ..models.flux_latent import decode_packed_latents, resize_mask_to_image
from ..models.ijepa_official import build_visible_fill, mask_to_patch_indices
from ..models.sdxl_latent import decode_sdxl_latents, sdxl_vae_float32
from ..strategy.base import InpaintGuidanceStrategy
from ..strategy.jepa import _alignment_loss, _scale_grad
from ..utils.texture_harmonize import fill_flux_source


def _pil_to_bchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        return image.unsqueeze(0)
    return image


def _mask_bchw(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    return mask


def _ring_fill_tensor(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Ring-fill masked region (counterfactual proxy for patch target)."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    b, _, h, w = image.shape
    out = []
    for i in range(b):
        pil = Image.fromarray((image[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        m = (mask[i].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
        filled = fill_flux_source(pil, m, ring_px=24, noise_scale=0.0)
        arr = np.asarray(filled, dtype=np.float32) / 255.0
        out.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(out, dim=0).to(device=image.device, dtype=image.dtype)


class _PatchEncoderBase:
    img_size: int = 224
    patch_size: int = 14

    @property
    def grid_size(self) -> int:
        return self.img_size // self.patch_size

    @property
    def device(self) -> torch.device:
        raise NotImplementedError

    def encode_patches(self, image_bchw: torch.Tensor) -> torch.Tensor:
        """Return [B, N_patches, D]."""
        raise NotImplementedError


class DinoV2PatchEncoder(_PatchEncoderBase):
    def __init__(self, model_id: str = "facebook/dinov2-base", img_size: int = 224, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModel

        self.img_size = img_size
        self._device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self._device).eval()
        self.patch_size = int(getattr(self.model.config, "patch_size", 14))
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_patches(self, image_bchw: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            image_bchw.float(),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(self.processor.image_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        out = self.model(pixel_values=x)
        tokens = out.last_hidden_state
        if tokens.shape[1] > self.grid_size * self.grid_size:
            tokens = tokens[:, 1:, :]
        return tokens


class CLIPPatchEncoder(_PatchEncoderBase):
    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        img_size: int = 224,
        device: str = "cuda",
    ):
        from transformers import CLIPModel, CLIPProcessor

        self.img_size = img_size
        self._device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self._device).eval()
        vis = self.model.vision_model
        self.patch_size = int(getattr(vis.config, "patch_size", 14))
        cfg_side = getattr(vis.config, "image_size", None)
        if cfg_side is not None:
            self.img_size = int(cfg_side)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_patches(self, image_bchw: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            image_bchw.float(),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(self.processor.image_processor.image_mean, device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1
        )
        std = torch.tensor(self.processor.image_processor.image_std, device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1
        )
        x = (x - mean) / std
        out = self.model.vision_model(pixel_values=x)
        tokens = out.last_hidden_state
        if tokens.shape[1] > self.grid_size * self.grid_size:
            tokens = tokens[:, 1:, :]
        return tokens


class CLIPTextEncoder(_PatchEncoderBase):
    """Global CLIP image-text alignment (different mechanism from patch-JEPA)."""

    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        img_size: int = 224,
        device: str = "cuda",
        prompt: str = "a clean background with the object removed, seamless photo",
    ):
        from transformers import CLIPModel, CLIPProcessor

        self.img_size = img_size
        self._device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self._device).eval()
        vis = self.model.vision_model
        cfg_side = getattr(vis.config, "image_size", None)
        if cfg_side is not None:
            self.img_size = int(cfg_side)
        self.prompt = prompt
        tok = self.processor(text=[prompt], return_tensors="pt", padding=True)
        with torch.no_grad():
            self.text_emb = F.normalize(
                self.model.get_text_features(input_ids=tok["input_ids"].to(self._device)), dim=-1
            )
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_patches(self, image_bchw: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("CLIP text mode uses global image embedding")

    def image_embedding(self, image_bchw: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            image_bchw.float(),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        px = self.processor.image_processor(images=x, return_tensors="pt")
        px = {k: v.to(self._device) for k, v in px.items()}
        emb = self.model.get_image_features(**px)
        return F.normalize(emb, dim=-1)


class PatchBackboneGuidanceStrategy(InpaintGuidanceStrategy):
    """Struct-only patch alignment with a swappable frozen ViT backbone."""

    def __init__(
        self,
        encoder: _PatchEncoderBase,
        *,
        mode: str = "patch",
        target_mode: str = "ring_fill",
        guidance_scale_mul: float = 1.0,
    ):
        self.encoder = encoder
        self.mode = mode  # patch | clip_text
        self.target_mode = target_mode
        self.guidance_scale_mul = float(guidance_scale_mul)
        self.struct_weight_override: float | None = None
        self.metrics = StrategyMetrics()

    def _build_target_image(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "ring_fill":
            return _ring_fill_tensor(image, mask)
        if self.target_mode == "gray_visible":
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            m = mask.unsqueeze(1).expand(-1, 3, -1, -1)
            return image * (1.0 - m) + 0.5 * m
        if self.target_mode == "object_patches":
            # Naive swap: supervise masked patches toward *original object* tokens.
            return image
        raise ValueError(f"unknown target_mode={self.target_mode!r}")

    def precompute(self, image: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> GuidanceCache:
        image = _pil_to_bchw(image)
        mask = _mask_bchw(mask)
        image = image.to(self.encoder.device)
        mask = mask.to(self.encoder.device)

        if self.mode == "clip_text":
            return GuidanceCache(
                e_target=torch.empty(0),
                visible_idx=torch.empty(0, dtype=torch.long),
                masked_idx=torch.empty(0, dtype=torch.long),
                patch_grid=(0, 0),
                patch_size=0,
                img_size=self.encoder.img_size,
            )

        cf = self._build_target_image(image, mask)
        grid = self.encoder.grid_size
        with torch.no_grad():
            tokens = self.encoder.encode_patches(cf)
        visible_idx, masked_idx = mask_to_patch_indices(mask, grid, grid, image.device)
        if masked_idx.numel() == 0:
            raise ValueError("empty edit mask")
        from ..models.ijepa_encoder import gather_tokens

        e_target = gather_tokens(tokens, masked_idx.unsqueeze(0)).squeeze(0)
        e_vis = gather_tokens(tokens, visible_idx.unsqueeze(0)).squeeze(0) if visible_idx.numel() else None
        return GuidanceCache(
            e_target=e_target,
            visible_idx=visible_idx,
            masked_idx=masked_idx,
            patch_grid=(grid, grid),
            patch_size=self.encoder.patch_size,
            img_size=self.encoder.img_size,
            e_vis_target=e_vis,
        )

    def should_guide(self, step_idx: int, num_steps: int, cfg: GuidanceConfig) -> bool:
        if not cfg.enabled:
            return False
        if cfg.guidance_schedule == "deferred":
            start = int(cfg.guidance_start_ratio * num_steps)
            return step_idx >= start and step_idx % max(1, cfg.guidance_every_n) == 0
        return step_idx < int(cfg.guidance_ratio * num_steps) and step_idx % max(1, cfg.guidance_every_n) == 0

    def _pin_view(
        self, decoded: torch.Tensor, original: torch.Tensor, mask_img: torch.Tensor, cfg: GuidanceConfig
    ) -> torch.Tensor:
        if cfg.pin_visible_before_jepa:
            return original * (1.0 - mask_img) + decoded * mask_img
        return decoded

    def guide_latents(
        self,
        latents: torch.Tensor,
        cache: GuidanceCache,
        image: torch.Tensor,
        mask: torch.Tensor,
        cfg: GuidanceConfig,
        state: GuidanceState,
        pipe: Any,
    ) -> tuple[torch.Tensor, Optional[float]]:
        backend = getattr(state, "backend", None) or (
            "flux" if state.latent_image_ids is not None else "sdxl"
        )
        if backend == "flux" and state.latent_image_ids is None:
            return latents, None

        image = _pil_to_bchw(image.to(latents.device))
        mask = _mask_bchw(mask.to(latents.device))

        vae_ctx = sdxl_vae_float32(pipe) if backend == "sdxl" else nullcontext()
        with vae_ctx, torch.enable_grad():
            if backend == "flux":
                z = latents.detach().requires_grad_(True)
                decoded = decode_packed_latents(
                    pipe, z, state.latent_image_ids, park_transformer=False
                )
            else:
                z = latents.detach().float().requires_grad_(True)
                decoded = decode_sdxl_latents(pipe, z)

            if not torch.isfinite(decoded).all():
                return latents, None

            h, w = decoded.shape[-2:]
            mask_img = resize_mask_to_image(mask, h, w).to(decoded.device, decoded.dtype)
            original = F.interpolate(image.float(), size=(h, w), mode="bilinear", align_corners=False)
            view = self._pin_view(decoded, original, mask_img, cfg)

            if self.mode == "clip_text":
                enc = self.encoder  # type: ignore[assignment]
                emb = enc.image_embedding(view.to(enc.device, dtype=torch.float32))
                loss = 1.0 - (emb * enc.text_emb).sum(dim=-1).mean()
            else:
                enc_dev = self.encoder.device
                x = view.to(enc_dev, dtype=torch.float32)
                tokens = self.encoder.encode_patches(x)
                from ..models.ijepa_encoder import gather_tokens

                e_curr = gather_tokens(
                    tokens, cache.masked_idx.to(enc_dev).unsqueeze(0)
                ).squeeze(0)
                loss = float(
                    self.struct_weight_override
                    if self.struct_weight_override is not None
                    else cfg.struct_weight
                ) * _alignment_loss(
                    e_curr, cache.e_target.to(enc_dev), cfg.loss_type
                )

            if not torch.isfinite(loss):
                return latents, None

            grad = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]
            if not torch.isfinite(grad).all():
                return latents, None

            if state.mask_packed is not None:
                region = state.mask_packed.to(device=grad.device, dtype=grad.dtype)
            else:
                region = torch.ones_like(grad)
            grad = grad * region
            grad = _scale_grad(grad, z, cfg)

            scale = float(cfg.guidance_scale) * self.guidance_scale_mul
            if cfg.time_decay and state.num_steps > 0:
                scale *= max(0.0, 1.0 - float(state.step_idx) / float(state.num_steps))

            z_new = z - scale * grad
            if not torch.isfinite(z_new).all():
                return latents, None

        loss_f = float(loss.detach().cpu())
        self.metrics.align_losses.append(loss_f)
        return z_new.detach().to(dtype=latents.dtype), loss_f

    def reset_metrics(self) -> StrategyMetrics:
        self.metrics = StrategyMetrics()
        return self.metrics


def build_patch_ablation_strategy(name: str, cfg: dict, device: str) -> InpaintGuidanceStrategy:
    """Factory for ablation backbones only."""
    name = name.lower()
    ab = dict(cfg.get("ablation", {}) or {})
    img_size = int(ab.get("img_size", 224))
    flux_cfg = cfg.get("flux", {})
    prompt = str(flux_cfg.get("prompt", ""))[:200]
    target_mode = str(ab.get("target_mode", "ring_fill"))
    g_mul = float(ab.get("guidance_scale_mul", 1.0))
    struct_w = float(cfg.get("guidance", {}).get("struct_weight", 1.0))

    ov = dict(ab.get("backbone_overrides", {}).get(name, {}) or {})
    target_mode = str(ov.get("target_mode", target_mode))
    g_mul *= float(ov.get("guidance_scale_mul", 1.0))
    struct_w_override = ov.get("struct_weight")
    struct_w = float(struct_w_override if struct_w_override is not None else struct_w)

    def _patch(enc: _PatchEncoderBase) -> PatchBackboneGuidanceStrategy:
        strat = PatchBackboneGuidanceStrategy(
            enc, mode="patch", target_mode=target_mode, guidance_scale_mul=g_mul
        )
        strat.struct_weight_override = struct_w
        return strat

    if name in ("dino_v2", "dino"):
        enc = DinoV2PatchEncoder(
            model_id=ab.get("dino_model", "facebook/dinov2-base"),
            img_size=img_size,
            device=device,
        )
        return _patch(enc)

    if name in ("dino_v3",):
        enc = DinoV2PatchEncoder(
            model_id=ab.get("dino_v3_model", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
            img_size=img_size,
            device=device,
        )
        return _patch(enc)

    if name in ("clip_patch", "clip"):
        enc = CLIPPatchEncoder(
            model_id=ab.get("clip_model", "openai/clip-vit-large-patch14"),
            img_size=img_size,
            device=device,
        )
        return _patch(enc)

    if name in ("clip_text",):
        enc = CLIPTextEncoder(
            model_id=ab.get("clip_model", "openai/clip-vit-large-patch14"),
            img_size=img_size,
            device=device,
            prompt=str(ab.get("clip_prompt", prompt)),
        )
        return PatchBackboneGuidanceStrategy(
            enc, mode="clip_text", target_mode=target_mode, guidance_scale_mul=g_mul
        )

    if name in ("jepa", "erase_world"):
        from ..pipeline.inpaint import build_strategy

        return build_strategy("erase_world", cfg.get("ijepa", {}), ijepa_device=device)

    raise ValueError(f"unknown ablation backbone {name!r}")
