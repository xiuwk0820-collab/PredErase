"""Full training-free erase_world guidance: JEPA + shadow / light / retinex stack."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..core.types import GuidanceCache, GuidanceConfig, GuidanceState, StrategyMetrics
from ..models.flux_latent import (
    decode_packed_latents,
    resize_mask_to_image,
    resize_mask_to_latent_spatial,
)
from ..models.ijepa_official import OfficialIJEPAModel
from ..models.sdxl_latent import decode_sdxl_latents, sdxl_vae_float32
from ..utils.grad_masks import edge_continuity_loss, feather_mask
from ..utils.guide_weight import build_w_causal_from_repr_diff
from ..utils.instance_jepa import instance_shadow_patch_indices
from ..utils.retinex_illum import precompute_retinex_stats, retinex_illumination_loss, lowfreq_illum_loss
from ..utils.shadow_physics import (
    build_counterfactual_visible,
    estimate_cast_shadow_soft,
    shadow_visible_patch_indices,
)
from .base import InpaintGuidanceStrategy


class NoOpStrategy(InpaintGuidanceStrategy):
    """Native inpaint baseline."""

    def precompute(self, image: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> GuidanceCache:
        return GuidanceCache(
            e_target=torch.empty(0),
            visible_idx=torch.empty(0, dtype=torch.long),
            masked_idx=torch.empty(0, dtype=torch.long),
            patch_grid=(0, 0),
            patch_size=0,
            img_size=0,
        )

    def should_guide(self, step_idx: int, num_steps: int, cfg: GuidanceConfig) -> bool:
        return False

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
        return latents, None


def _alignment_loss(e_curr: torch.Tensor, e_target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if e_curr.numel() == 0 or e_target.numel() == 0:
        return e_curr.new_zeros(())
    if e_curr.shape != e_target.shape:
        n = min(e_curr.shape[0], e_target.shape[0])
        e_curr, e_target = e_curr[:n], e_target[:n]
    if loss_type == "mse":
        return F.mse_loss(e_curr, e_target)
    e_curr_n = F.normalize(e_curr.float(), dim=-1)
    e_target_n = F.normalize(e_target.float(), dim=-1)
    return 1.0 - (e_curr_n * e_target_n).sum(dim=-1).mean()


def _scale_grad(grad: torch.Tensor, latents: torch.Tensor, cfg: GuidanceConfig) -> torch.Tensor:
    if cfg.grad_mode == "raw":
        scaled = grad
    elif cfg.grad_mode == "normalize" or cfg.normalize_grad:
        scaled = grad / grad.norm(p=2).clamp(min=1.0e-8)
    else:
        latent_rms = latents.detach().float().pow(2).mean().sqrt().clamp(min=1.0e-6)
        grad_rms = grad.float().pow(2).mean().sqrt().clamp(min=1.0e-8)
        scaled = grad * (latent_rms / grad_rms).to(grad.dtype)

    clip_norm = float(getattr(cfg, "grad_clip_norm", 0.0) or 0.0)
    if clip_norm > 0:
        norm = scaled.float().norm(p=2).clamp(min=1.0e-8)
        scaled = scaled * min(1.0, clip_norm / float(norm))
    return scaled


def _as_bchw_mask(mask: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
    if mask.ndim == 2:
        m = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        m = mask.unsqueeze(1)
    else:
        m = mask[:, :1]
    if size is not None and m.shape[-2:] != size:
        m = F.interpolate(m.float(), size=size, mode="nearest")
    return m.float()


class JEPAGuidanceStrategy(InpaintGuidanceStrategy):
    """Full erase_world guidance (struct + dual-instance + light + retinex + causal W)."""

    def __init__(self, ijepa: OfficialIJEPAModel):
        self.ijepa = ijepa
        self.metrics = StrategyMetrics()

    def precompute(self, image: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> GuidanceCache:
        """mask = edit/effect mask. Optional: object_mask, shadow_soft, cfg, counterfactual."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        cfg: GuidanceConfig | None = kwargs.get("cfg")
        object_mask = kwargs.get("object_mask")
        shadow_soft = kwargs.get("shadow_soft")
        counterfactual = kwargs.get("counterfactual")

        image = image.to(self.ijepa.device)
        mask = mask.to(self.ijepa.device)
        if object_mask is None:
            object_mask = mask
        if object_mask.ndim == 2:
            object_mask = object_mask.unsqueeze(0)
        object_mask = object_mask.to(self.ijepa.device)

        e_target, visible_idx, masked_idx, e_vis_target = self.ijepa.precompute_target(
            image, mask
        )
        grid = self.ijepa.grid_size
        h, w = image.shape[-2:]

        # Shadow soft: caller or physics estimate
        if shadow_soft is None and cfg is not None and (
            cfg.shadow_physics or cfg.cast_shadow_module or cfg.dual_instance_guidance
        ):
            shadow_soft = estimate_cast_shadow_soft(image, object_mask)
        if shadow_soft is not None:
            if shadow_soft.ndim == 2:
                shadow_soft = shadow_soft.unsqueeze(0).unsqueeze(0)
            elif shadow_soft.ndim == 3:
                shadow_soft = shadow_soft.unsqueeze(1)
            shadow_soft = shadow_soft.to(self.ijepa.device, dtype=torch.float32)
            if shadow_soft.shape[-2:] != (h, w):
                shadow_soft = F.interpolate(shadow_soft, size=(h, w), mode="bilinear", align_corners=False)

        object_soft = _as_bchw_mask(object_mask, (h, w))

        # Counterfactual visible (shadow lifted)
        if counterfactual is not None:
            if counterfactual.ndim == 3:
                counterfactual = counterfactual.unsqueeze(0)
            cf = counterfactual.to(self.ijepa.device, dtype=image.dtype)
            if cf.shape[-2:] != (h, w):
                cf = F.interpolate(cf, size=(h, w), mode="bilinear", align_corners=False)
        elif shadow_soft is not None and cfg is not None and (
            cfg.shadow_physics or cfg.counterfactual_guidance or cfg.cast_shadow_module
        ):
            strength = float(getattr(cfg, "shadow_harmonize_strength", 0.85))
            cf = build_counterfactual_visible(image, object_soft, shadow_soft, strength=strength)
        else:
            cf = None

        cache = GuidanceCache(
            e_target=e_target.detach(),
            visible_idx=visible_idx.detach(),
            masked_idx=masked_idx.detach(),
            patch_grid=(grid, grid),
            patch_size=self.ijepa.patch_size,
            img_size=self.ijepa.img_size,
            e_vis_target=e_vis_target.detach(),
            shadow_soft=shadow_soft.detach() if shadow_soft is not None else None,
            object_soft=object_soft.detach(),
            counterfactual_visible=cf.detach() if cf is not None else None,
        )

        # Dual-instance: separate object / cast-shadow JEPA targets
        if cfg is not None and cfg.dual_instance_guidance:
            e_obj, vis_o, mask_o, _ = self.ijepa.precompute_target(image, object_mask)
            cache.e_object_target = e_obj.detach()
            cache.object_visible_idx = vis_o.detach()
            cache.object_masked_idx = mask_o.detach()
            if shadow_soft is not None:
                sh_idx = instance_shadow_patch_indices(
                    object_mask,
                    shadow_soft,
                    grid,
                    threshold=float(cfg.shadow_patch_threshold),
                )
                cache.shadow_masked_idx = sh_idx.detach()
                if sh_idx.numel() > 0:
                    src_for_shadow = cf if cf is not None else image
                    sh_tok = self.ijepa.encode_shallow_light(src_for_shadow)
                    cache.e_shadow_target = sh_tok[sh_idx].detach()

        # Causal W from ori vs counterfactual repr diff (soft grad field)
        if cfg is not None:
            try:
                causal = self.ijepa.precompute_counterfactual_causal(image, object_mask)
                w_guide = build_w_causal_from_repr_diff(
                    causal["diff_struct"],
                    causal["diff_light"],
                    object_mask,
                    grid,
                    struct_weight=float(cfg.causal_diff_struct_weight),
                    light_weight=float(cfg.causal_diff_light_weight),
                    smooth_sigma_px=float(cfg.causal_smooth_sigma_px),
                )
                if shadow_soft is not None:
                    w_guide = torch.maximum(w_guide, shadow_soft.clamp(0, 1))
                w_guide = torch.maximum(w_guide, object_soft)
                cache.w_guide = w_guide.detach()
                if cfg.counterfactual_guidance or cfg.shadow_physics:
                    cache.e_cf_shallow = causal["e_cf_shallow_full"].detach()
            except Exception:
                cache.w_guide = torch.maximum(
                    object_soft,
                    shadow_soft if shadow_soft is not None else object_soft * 0,
                ).detach()

        # Shadow-visible patches for physics / light loss
        if shadow_soft is not None and cfg is not None:
            cache.shadow_vis_idx = shadow_visible_patch_indices(
                shadow_soft,
                object_mask,
                visible_idx,
                grid,
                threshold=float(cfg.shadow_patch_threshold),
            ).detach()

        # Layered global light target (protect = edit region)
        if cfg is not None and cfg.layered_light_guidance:
            protect = (mask > 0.5).float()
            if protect.ndim == 2:
                protect = protect.unsqueeze(0)
            e_light, light_vis = self.ijepa.precompute_global_light(image, protect)
            cache.e_light_target = e_light.detach()
            cache.light_visible_idx = light_vis.detach()
            if hasattr(self.ijepa, "_light_layer_ids") or cfg.shallow_light_layers:
                self.ijepa._light_layer_ids = list(cfg.shallow_light_layers)  # type: ignore[attr-defined]

        # Retinex neighbor stats on shadow band
        if cfg is not None and cfg.retinex_illum and shadow_soft is not None:
            stats = precompute_retinex_stats(
                image,
                mask,
                shadow_soft,
                ring_width=int(cfg.retinex_ring_width),
                blur_sigma=float(cfg.retinex_blur_sigma),
            )
            cache.neighbor_mean = stats["neighbor_mean"].detach()
            cache.neighbor_std = stats["neighbor_std"].detach()
            cache.shadow_blur = stats["shadow_blur"].detach()

        if cache.w_guide is None:
            cache.w_guide = object_soft.detach()
        return cache

    def should_guide(self, step_idx: int, num_steps: int, cfg: GuidanceConfig) -> bool:
        if not cfg.enabled:
            return False
        if getattr(cfg, "stable_guidance", False):
            end_step = max(0, min(int(cfg.struct_end_step), num_steps))
            every_n = max(1, int(cfg.struct_guidance_every_n))
            if step_idx < end_step and step_idx % every_n == 0:
                return True
            # Fine phase: illumination-only guidance
            if (
                step_idx >= end_step
                and (cfg.retinex_illum or cfg.layered_light_guidance)
                and step_idx % max(1, int(cfg.illum_guide_every_n)) == 0
            ):
                # Limit fine steps
                fine_budget = max(0, int(cfg.retinex_fine_steps))
                if fine_budget <= 0:
                    return False
                fine_i = (step_idx - end_step) // max(1, int(cfg.illum_guide_every_n))
                return fine_i < fine_budget
            return False
        if cfg.guidance_schedule == "deferred":
            start = int(cfg.guidance_start_ratio * num_steps)
            return step_idx >= start and step_idx % max(1, cfg.guidance_every_n) == 0
        return (
            step_idx < int(cfg.guidance_ratio * num_steps)
            and step_idx % max(1, cfg.guidance_every_n) == 0
        )

    def _jepa_view(
        self,
        decoded: torch.Tensor,
        original: torch.Tensor,
        mask_img: torch.Tensor,
        cfg: GuidanceConfig,
    ) -> torch.Tensor:
        if cfg.pin_visible_before_jepa:
            return original * (1.0 - mask_img) + decoded * mask_img
        return decoded

    def _is_fine_phase(self, state: GuidanceState, cfg: GuidanceConfig) -> bool:
        if not getattr(cfg, "stable_guidance", False):
            return False
        return int(state.step_idx) >= int(cfg.struct_end_step)

    def _compose_loss(
        self,
        jepa_input: torch.Tensor,
        decoded: torch.Tensor,
        cache: GuidanceCache,
        cfg: GuidanceConfig,
        fine: bool,
    ) -> torch.Tensor:
        ijepa_dev = self.ijepa.device
        x = jepa_input.to(ijepa_dev, dtype=torch.float32)
        loss = x.new_zeros(())
        struct_w = float(cfg.struct_weight_fine if fine else cfg.struct_weight)

        # One shared deep encode (autograd) — reuse for effect + object branches.
        need_struct = (not fine or not cfg.fine_light_only) and cache.masked_idx.numel() > 0
        need_dual_obj = (
            cfg.dual_instance_guidance
            and not fine
            and cache.e_object_target is not None
            and cache.object_masked_idx is not None
            and cache.object_visible_idx is not None
            and cache.object_masked_idx.numel() > 0
        )
        tokens = None
        if need_struct or need_dual_obj:
            tokens = self.ijepa.encode_patches(x)

        if need_struct and tokens is not None:
            if self.ijepa.has_predictor:
                e_curr = self.ijepa.predict_masked_tokens(
                    tokens,
                    cache.visible_idx.to(ijepa_dev),
                    cache.masked_idx.to(ijepa_dev),
                ).squeeze(0)
            else:
                from ..models.ijepa_encoder import gather_tokens

                e_curr = gather_tokens(
                    tokens, cache.masked_idx.to(ijepa_dev).unsqueeze(0)
                ).squeeze(0)
            loss = loss + struct_w * _alignment_loss(
                e_curr, cache.e_target.to(ijepa_dev), cfg.loss_type
            )

        if need_dual_obj and tokens is not None:
            if self.ijepa.has_predictor:
                e_obj = self.ijepa.predict_masked_tokens(
                    tokens,
                    cache.object_visible_idx.to(ijepa_dev),
                    cache.object_masked_idx.to(ijepa_dev),
                ).squeeze(0)
            else:
                from ..models.ijepa_encoder import gather_tokens

                e_obj = gather_tokens(
                    tokens, cache.object_masked_idx.to(ijepa_dev).unsqueeze(0)
                ).squeeze(0)
            loss = loss + struct_w * _alignment_loss(
                e_obj, cache.e_object_target.to(ijepa_dev), cfg.loss_type
            )

        # One shared shallow encode for light / shadow / physics
        need_shallow = (
            (
                cfg.dual_instance_guidance
                and cache.e_shadow_target is not None
                and cache.shadow_masked_idx is not None
                and cache.shadow_masked_idx.numel() > 0
            )
            or (cfg.layered_light_guidance and cache.e_light_target is not None)
            or (
                cfg.shadow_physics
                and cache.e_cf_shallow is not None
                and cache.shadow_vis_idx is not None
                and cache.shadow_vis_idx.numel() > 0
            )
        )
        sh_tok = None
        if need_shallow:
            sh_tok = self.ijepa.encode_shallow_light(x)

        if (
            cfg.dual_instance_guidance
            and sh_tok is not None
            and cache.e_shadow_target is not None
            and cache.shadow_masked_idx is not None
            and cache.shadow_masked_idx.numel() > 0
        ):
            from ..models.ijepa_encoder import gather_tokens

            e_sh = gather_tokens(
                sh_tok.unsqueeze(0), cache.shadow_masked_idx.to(ijepa_dev).unsqueeze(0)
            ).mean(dim=1).squeeze(0)
            tgt = cache.e_shadow_target.to(ijepa_dev)
            if tgt.ndim == 2:
                tgt = tgt.mean(0)
            loss = loss + float(cfg.illum_weight) * _alignment_loss(
                e_sh.unsqueeze(0), tgt.unsqueeze(0), cfg.loss_type
            )

        light_w = float(cfg.light_weight_fine if fine else cfg.light_weight_struct)
        if cfg.layered_light_guidance and cache.e_light_target is not None and sh_tok is not None:
            idx = cache.shadow_masked_idx if cache.shadow_masked_idx is not None else cache.masked_idx
            if idx is not None and idx.numel() > 0:
                from ..models.ijepa_encoder import gather_tokens

                e_l = gather_tokens(sh_tok.unsqueeze(0), idx.to(ijepa_dev).unsqueeze(0)).mean(dim=1).squeeze(0)
                loss = loss + light_w * F.mse_loss(
                    e_l, cache.e_light_target.to(ijepa_dev, dtype=e_l.dtype)
                )

        if (
            cfg.shadow_physics
            and sh_tok is not None
            and cache.e_cf_shallow is not None
            and cache.shadow_vis_idx is not None
            and cache.shadow_vis_idx.numel() > 0
        ):
            idx = cache.shadow_vis_idx.to(ijepa_dev)
            loss = loss + float(cfg.shadow_loss_weight) * F.mse_loss(
                sh_tok[idx], cache.e_cf_shallow.to(ijepa_dev)[idx]
            )

        # Retinex pixel illumination on decoded RGB
        if (
            cfg.retinex_illum
            and cache.shadow_blur is not None
            and cache.neighbor_mean is not None
            and cache.neighbor_std is not None
        ):
            sh = cache.shadow_blur.to(decoded.device, dtype=decoded.dtype)
            if sh.shape[-2:] != decoded.shape[-2:]:
                sh = F.interpolate(sh, size=decoded.shape[-2:], mode="bilinear", align_corners=False)
            rw = float(
                cfg.retinex_illum_weight_fine if fine else cfg.retinex_illum_weight_struct
            )
            loss = loss + rw * retinex_illumination_loss(
                decoded,
                sh,
                cache.neighbor_mean.to(decoded.device, dtype=decoded.dtype),
                cache.neighbor_std.to(decoded.device, dtype=decoded.dtype),
                std_weight=float(cfg.retinex_std_weight),
            )
            if cfg.retinex_lowfreq:
                loss = loss + float(cfg.retinex_lowfreq_weight) * lowfreq_illum_loss(decoded, sh)

        ecw = float(cfg.edge_continuity_weight_fine if fine else cfg.edge_continuity_weight)
        if ecw > 0 and cache.object_soft is not None:
            om = cache.object_soft.to(decoded.device, dtype=decoded.dtype)
            if om.shape[-2:] != decoded.shape[-2:]:
                om = F.interpolate(om, size=decoded.shape[-2:], mode="nearest")
            loss = loss + ecw * edge_continuity_loss(decoded, om)

        return loss

    def _grad_region_mask(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        cache: GuidanceCache,
        cfg: GuidanceConfig,
        backend: str,
        state: GuidanceState,
    ) -> torch.Tensor:
        """Soft causal / shadow-aware latent gradient gate."""
        if backend == "sdxl" and z.ndim == 4:
            hz, wz = z.shape[-2], z.shape[-1]
            hard = resize_mask_to_latent_spatial(mask, hz, wz).to(z.device, dtype=z.dtype)
            soft = hard
            if cache.w_guide is not None:
                wg = cache.w_guide.to(z.device, dtype=torch.float32)
                if wg.ndim == 3:
                    wg = wg.unsqueeze(0)
                wg = F.interpolate(wg, size=(hz, wz), mode="bilinear", align_corners=False)
                soft = wg.to(dtype=z.dtype)
            elif cache.shadow_soft is not None:
                ss = cache.shadow_soft.to(z.device, dtype=torch.float32)
                ss = F.interpolate(ss, size=(hz, wz), mode="bilinear", align_corners=False)
                soft = torch.maximum(hard, ss.to(dtype=z.dtype))
            feather = float(getattr(cfg, "grad_mask_feather", 0.0) or 0.0)
            if feather > 0:
                soft = feather_mask(soft, sigma=max(1.0, feather / float(max(mask.shape[-1], 1)) * wz))
            pin = float(getattr(cfg, "shadow_pin_release", 0.0) or 0.0)
            if pin > 0:
                return torch.maximum(hard, soft * pin + hard * (1.0 - pin))
            return torch.maximum(hard, soft * float(cfg.shadow_grad_weight if cfg.shadow_physics else 1.0))

        if state.mask_packed is not None:
            return state.mask_packed.to(device=z.device, dtype=z.dtype)
        return torch.ones_like(z)

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
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        device = latents.device
        image = image.to(device)
        mask = mask.to(device)
        fine = self._is_fine_phase(state, cfg)

        vae_ctx = sdxl_vae_float32(pipe) if backend == "sdxl" else nullcontext()
        with vae_ctx, torch.enable_grad():
            if backend == "flux":
                z = latents.detach().requires_grad_(True)
                decoded = decode_packed_latents(
                    pipe,
                    z,
                    state.latent_image_ids,
                    park_transformer=False,
                )
            else:
                z = latents.detach().float().requires_grad_(True)
                decoded = decode_sdxl_latents(pipe, z)

            if not torch.isfinite(decoded).all():
                return latents, None

            height, width = decoded.shape[-2:]
            mask_img = resize_mask_to_image(mask, height, width).to(decoded.device, decoded.dtype)
            original = F.interpolate(
                image.float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            # Prefer counterfactual as visible context when physics is on
            if cfg.shadow_physics and cache.counterfactual_visible is not None:
                cf = cache.counterfactual_visible.to(device=decoded.device, dtype=decoded.dtype)
                if cf.shape[-2:] != (height, width):
                    cf = F.interpolate(cf, size=(height, width), mode="bilinear", align_corners=False)
                context = cf
            else:
                context = original
            jepa_input = self._jepa_view(decoded, context, mask_img, cfg)

            loss = self._compose_loss(jepa_input, decoded, cache, cfg, fine=fine)
            if not torch.isfinite(loss):
                return latents, None
            if float(loss.detach()) == 0.0 and not fine:
                # Fall back to basic align if all optional branches skipped
                e_curr = self.ijepa.mask_region_repr(
                    jepa_input.to(self.ijepa.device, dtype=torch.float32),
                    cache.visible_idx.to(self.ijepa.device),
                    cache.masked_idx.to(self.ijepa.device),
                )
                loss = _alignment_loss(
                    e_curr, cache.e_target.to(self.ijepa.device), cfg.loss_type
                )

            grad = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]
            if not torch.isfinite(grad).all():
                return latents, None

            region = self._grad_region_mask(z, mask, cache, cfg, backend, state)
            grad = grad * region.to(device=grad.device, dtype=grad.dtype)

            grad = _scale_grad(grad, z, cfg)
            scale = float(cfg.guidance_scale)
            if fine:
                scale = float(cfg.retinex_guidance_scale) if cfg.retinex_illum else scale * 0.5
            if cfg.time_decay and state.num_steps > 0:
                scale *= max(0.0, 1.0 - float(state.step_idx) / float(state.num_steps))
            # Intra decay within struct window
            if getattr(cfg, "stable_guidance", False) and cfg.struct_end_step > 0:
                t = float(state.step_idx) / float(max(1, cfg.struct_end_step))
                scale *= max(0.15, 1.0 - float(cfg.guidance_intra_decay) * 0.15 * t)

            z_new = z - scale * grad

            # Latent pin visible (hard)
            if cfg.latent_pin_visible and backend == "sdxl" and z.ndim == 4:
                hard = resize_mask_to_latent_spatial(mask, z.shape[-2], z.shape[-1]).to(
                    z.device, dtype=z.dtype
                )
                z_new = latents.float() * (1.0 - hard) + z_new * hard

            if not torch.isfinite(z_new).all():
                return latents, None

        loss_f = float(loss.detach().cpu())
        self.metrics.align_losses.append(loss_f)
        return z_new.detach().to(dtype=latents.dtype), loss_f

    def reset_metrics(self) -> StrategyMetrics:
        self.metrics = StrategyMetrics()
        return self.metrics
