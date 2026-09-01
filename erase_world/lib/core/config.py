from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import GuidanceConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, maybe_rel: str | None) -> str | None:
    if maybe_rel is None:
        return None
    p = Path(maybe_rel)
    if p.is_absolute():
        return str(p)
    return str((base / p).resolve())


def build_guidance_config(guidance_raw: dict[str, Any], enabled: bool = True) -> GuidanceConfig:
    return GuidanceConfig(
        enabled=enabled and bool(guidance_raw.get("enabled", True)),
        guidance_scale=float(guidance_raw.get("guidance_scale", 0.5)),
        guidance_ratio=float(guidance_raw.get("guidance_ratio", 0.5)),
        guidance_every_n=int(guidance_raw.get("guidance_every_n", 1)),
        visible_preserve_weight=float(guidance_raw.get("visible_preserve_weight", 0.05)),
        time_decay=bool(guidance_raw.get("time_decay", False)),
        loss_type=str(guidance_raw.get("loss_type", "cosine")),
        pin_visible_before_jepa=bool(guidance_raw.get("pin_visible_before_jepa", True)),
        grad_mode=str(guidance_raw.get("grad_mode", "latent_std")),
        normalize_grad=bool(guidance_raw.get("normalize_grad", False)),
        guidance_schedule=str(guidance_raw.get("guidance_schedule", "early")),
        guidance_start_ratio=float(guidance_raw.get("guidance_start_ratio", 0.7)),
        pred_weight=float(guidance_raw.get("pred_weight", 1.0)),
        enc_weight=float(guidance_raw.get("enc_weight", 0.5)),
        dynamics_weight=float(guidance_raw.get("dynamics_weight", 0.8)),
        dynamics_noise_scale=float(guidance_raw.get("dynamics_noise_scale", 0.35)),
        dynamics_inner_steps=int(guidance_raw.get("dynamics_inner_steps", 2)),
        visible_enc_weight=float(guidance_raw.get("visible_enc_weight", 0.15)),
        dynamics_min_noise=float(guidance_raw.get("dynamics_min_noise", 0.2)),
        guidance_phase_ratio=float(guidance_raw.get("guidance_phase_ratio", 0.5)),
        guidance_intra_decay=float(guidance_raw.get("guidance_intra_decay", 1.5)),
        grad_clip_norm=float(guidance_raw.get("grad_clip_norm", 1.0)),
        shadow_physics=bool(guidance_raw.get("shadow_physics", True)),
        shadow_grad_weight=float(guidance_raw.get("shadow_grad_weight", 0.65)),
        shadow_loss_weight=float(guidance_raw.get("shadow_loss_weight", 0.40)),
        shadow_harmonize_strength=float(guidance_raw.get("shadow_harmonize_strength", 0.85)),
        shadow_patch_threshold=float(guidance_raw.get("shadow_patch_threshold", 0.08)),
        latent_pin_visible=bool(guidance_raw.get("latent_pin_visible", True)),
        illum_loss_weight=float(guidance_raw.get("illum_loss_weight", 0.0)),
        cast_shadow_module=bool(guidance_raw.get("cast_shadow_module", True)),
        dual_instance_guidance=bool(guidance_raw.get("dual_instance_guidance", True)),
        struct_weight=float(guidance_raw.get("struct_weight", guidance_raw.get("pred_weight", 1.0))),
        illum_weight=float(guidance_raw.get("illum_weight", guidance_raw.get("shadow_loss_weight", 0.55))),
        retinex_illum=bool(guidance_raw.get("retinex_illum", True)),
        retinex_illum_weight_struct=float(guidance_raw.get("retinex_illum_weight_struct", 0.4)),
        retinex_illum_weight_fine=float(guidance_raw.get("retinex_illum_weight_fine", 1.2)),
        retinex_guidance_scale=float(guidance_raw.get("retinex_guidance_scale", 0.38)),
        retinex_ring_width=int(guidance_raw.get("retinex_ring_width", 16)),
        retinex_blur_sigma=float(guidance_raw.get("retinex_blur_sigma", 3.0)),
        retinex_std_weight=float(guidance_raw.get("retinex_std_weight", 0.35)),
        retinex_lowfreq=bool(guidance_raw.get("retinex_lowfreq", False)),
        retinex_lowfreq_weight=float(guidance_raw.get("retinex_lowfreq_weight", 0.15)),
        struct_phase_ratio=float(guidance_raw.get("struct_phase_ratio", 0.667)),
        jepa_fine_scale=float(guidance_raw.get("jepa_fine_scale", 0.2)),
        illum_guide_every_n=int(guidance_raw.get("illum_guide_every_n", 1)),
        retinex_fine_steps=int(guidance_raw.get("retinex_fine_steps", 2)),
        fine_light_only=bool(guidance_raw.get("fine_light_only", True)),
        grad_mask_feather=float(guidance_raw.get("grad_mask_feather", 5.0)),
        struct_interior_erosion=int(guidance_raw.get("struct_interior_erosion", 4)),
        edge_suppress_struct=float(guidance_raw.get("edge_suppress_struct", 0.85)),
        shadow_rule_only=bool(guidance_raw.get("shadow_rule_only", True)),
        shadow_expand_ratio=float(guidance_raw.get("shadow_expand_ratio", 0.15)),
        shadow_feather_px=float(guidance_raw.get("shadow_feather_px", 4.0)),
        shallow_light_layers=tuple(guidance_raw.get("shallow_light_layers", [2, 3, 4])),
        struct_line_band_px=int(guidance_raw.get("struct_line_band_px", 10)),
        struct_line_decay=float(guidance_raw.get("struct_line_decay", 0.92)),
        edge_continuity_weight=float(guidance_raw.get("edge_continuity_weight", 0.06)),
        edge_continuity_weight_fine=float(guidance_raw.get("edge_continuity_weight_fine", 0.0)),
        mask_bridge_px=int(guidance_raw.get("mask_bridge_px", 16)),
        mask_dilate_px=int(guidance_raw.get("mask_dilate_px", 12)),
        mask_extend_down_ratio=float(guidance_raw.get("mask_extend_down_ratio", 0.26)),
        mask_extend_left_ratio=float(guidance_raw.get("mask_extend_left_ratio", 0.32)),
        mask_run_dilate_px=int(guidance_raw.get("mask_run_dilate_px", 8)),
        fast_vae_decode=bool(guidance_raw.get("fast_vae_decode", True)),
        layered_light_guidance=bool(guidance_raw.get("layered_light_guidance", True)),
        light_weight_struct_obj=float(guidance_raw.get("light_weight_struct_obj", 0.35)),
        light_weight_struct_shadow=float(guidance_raw.get("light_weight_struct_shadow", 0.85)),
        light_weight_fine_shadow=float(guidance_raw.get("light_weight_fine_shadow", 1.1)),
        light_weight_fine_obj=float(guidance_raw.get("light_weight_fine_obj", 0.25)),
        shallow_light_layer_ratio=float(guidance_raw.get("shallow_light_layer_ratio", 0.25)),
        counterfactual_guidance=bool(guidance_raw.get("counterfactual_guidance", False)),
        stable_guidance=bool(guidance_raw.get("stable_guidance", True)),
        struct_end_step=int(guidance_raw.get("struct_end_step", 11)),
        light_weight_struct=float(guidance_raw.get("light_weight_struct", 0.3)),
        shadow_down_ratio=float(guidance_raw.get("shadow_down_ratio", 0.5)),
        attentive_eraser=bool(guidance_raw.get("attentive_eraser", False)),
        guide_decay_down=float(guidance_raw.get("guide_decay_down", 0.6)),
        guide_decay_side=float(guidance_raw.get("guide_decay_side", 0.2)),
        struct_weight_fine=float(guidance_raw.get("struct_weight_fine", 0.3)),
        light_weight_fine=float(guidance_raw.get("light_weight_fine", 1.0)),
        dynamics_weight_fine=float(guidance_raw.get("dynamics_weight_fine", 0.2)),
        struct_guidance_every_n=int(guidance_raw.get("struct_guidance_every_n", 2)),
        causal_smooth_sigma_px=float(guidance_raw.get("causal_smooth_sigma_px", 18.0)),
        causal_diff_struct_weight=float(guidance_raw.get("causal_diff_struct_weight", 0.55)),
        causal_diff_light_weight=float(guidance_raw.get("causal_diff_light_weight", 0.45)),
        shadow_pin_release=float(guidance_raw.get("shadow_pin_release", 0.0)),
        jepa_pseudo_blur_sigma=float(guidance_raw.get("jepa_pseudo_blur_sigma", 3.5)),
        jepa_pseudo_noise_scale=float(guidance_raw.get("jepa_pseudo_noise_scale", 0.45)),
        jepa_pseudo_ring_px=int(guidance_raw.get("jepa_pseudo_ring_px", 18)),
    )
