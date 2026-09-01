from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class GuidanceCache:
    """Precomputed JEPA target representation (module 1 output)."""

    e_target: torch.Tensor
    visible_idx: torch.Tensor
    masked_idx: torch.Tensor
    patch_grid: tuple[int, int]
    patch_size: int
    img_size: int
    e_vis_target: Optional[torch.Tensor] = None
    # Instance dual guidance: this object vs its cast shadow
    object_masked_idx: Optional[torch.Tensor] = None
    object_visible_idx: Optional[torch.Tensor] = None
    e_object_target: Optional[torch.Tensor] = None
    shadow_masked_idx: Optional[torch.Tensor] = None
    e_shadow_target: Optional[torch.Tensor] = None
    # Legacy / shared
    shadow_soft: Optional[torch.Tensor] = None  # [1,1,H,W]
    counterfactual_visible: Optional[torch.Tensor] = None  # [1,3,H,W]
    shadow_vis_idx: Optional[torch.Tensor] = None
    # Retinex pixel illumination (cached once)
    neighbor_mean: Optional[torch.Tensor] = None
    neighbor_std: Optional[torch.Tensor] = None
    shadow_blur: Optional[torch.Tensor] = None
    # Shallow I-JEPA global lighting branch
    e_light_target: Optional[torch.Tensor] = None
    light_visible_idx: Optional[torch.Tensor] = None
    e_cf_shallow: Optional[torch.Tensor] = None  # shallow tokens on masked patches
    object_soft: Optional[torch.Tensor] = None  # [1,1,H,W] M_obj only
    w_guide: Optional[torch.Tensor] = None  # [1,1,H,W] spatial guidance weights


@dataclass
class GuidanceState:
    """Runtime context passed into per-step guidance."""

    step_idx: int
    num_steps: int
    timestep: torch.Tensor
    # FLUX packed ids; None for SD-XL / spatial-latent backends
    latent_image_ids: Optional[torch.Tensor] = None
    # FLUX packed mask, or SD-XL spatial latent mask [B,1,H_z,W_z]
    mask_packed: Optional[torch.Tensor] = None
    height: int = 0
    width: int = 0
    backend: str = "flux"  # flux | sdxl


@dataclass
class GuidanceConfig:
    enabled: bool = True
    guidance_scale: float = 0.5
    guidance_ratio: float = 0.5
    guidance_every_n: int = 1
    visible_preserve_weight: float = 0.05
    time_decay: bool = False
    loss_type: str = "cosine"  # cosine | mse
    pin_visible_before_jepa: bool = True
    grad_mode: str = "latent_std"  # latent_std | normalize | raw
    normalize_grad: bool = False
    # early | deferred | adaptive (JEPA first phase only, off later for texture)
    guidance_schedule: str = "early"
    guidance_start_ratio: float = 0.7
    guidance_phase_ratio: float = 0.5
    guidance_intra_decay: float = 1.5
    grad_clip_norm: float = 1.0
    # D-JEPA style dual objective: L_p (predictor) + L_d (embedding dynamics on noisy patches)
    pred_weight: float = 1.0
    enc_weight: float = 0.5
    dynamics_weight: float = 0.8
    dynamics_noise_scale: float = 0.35
    dynamics_inner_steps: int = 2
    visible_enc_weight: float = 0.15
    dynamics_min_noise: float = 0.2
    # Shadow physics: counterfactual JEPA + soft cast-shadow gradient (no mask dilation)
    shadow_physics: bool = True
    shadow_grad_weight: float = 0.65
    shadow_loss_weight: float = 0.40
    shadow_harmonize_strength: float = 0.85
    shadow_patch_threshold: float = 0.08
    latent_pin_visible: bool = True
    illum_loss_weight: float = 0.0
    cast_shadow_module: bool = True
    dual_instance_guidance: bool = True
    struct_weight: float = 1.0
    illum_weight: float = 0.55
    # Retinex pixel L_illum (FuLLaMa / AdaEraser style)
    retinex_illum: bool = True
    retinex_illum_weight_struct: float = 0.4
    retinex_illum_weight_fine: float = 1.2
    retinex_guidance_scale: float = 0.38
    retinex_ring_width: int = 16
    retinex_blur_sigma: float = 3.0
    retinex_std_weight: float = 0.35
    retinex_lowfreq: bool = False
    retinex_lowfreq_weight: float = 0.15
    # Staged schedule: struct phase ratio + illumination-only fine tail
    struct_phase_ratio: float = 0.667
    jepa_fine_scale: float = 0.2
    illum_guide_every_n: int = 1
    retinex_fine_steps: int = 2
    fine_light_only: bool = True
    grad_mask_feather: float = 5.0
    struct_interior_erosion: int = 4
    edge_suppress_struct: float = 0.85
    shadow_rule_only: bool = True
    shadow_expand_ratio: float = 0.15
    shadow_feather_px: float = 4.0
    shallow_light_layers: tuple[int, ...] = (2, 3, 4)
    struct_line_band_px: int = 10
    struct_line_decay: float = 0.92
    edge_continuity_weight: float = 0.06
    edge_continuity_weight_fine: float = 0.0
    mask_bridge_px: int = 16
    mask_dilate_px: int = 12
    mask_extend_down_ratio: float = 0.26
    mask_extend_left_ratio: float = 0.32
    mask_run_dilate_px: int = 8
    fast_vae_decode: bool = True
    # Layered global lighting (Doubao): split M_obj / M_shadow, shallow E_light
    layered_light_guidance: bool = True
    light_weight_struct_obj: float = 0.35
    light_weight_struct_shadow: float = 0.85
    light_weight_fine_shadow: float = 1.1
    light_weight_fine_obj: float = 0.25
    shallow_light_layer_ratio: float = 0.25
    counterfactual_guidance: bool = False
    stable_guidance: bool = True
    struct_end_step: int = 11
    light_weight_struct: float = 0.3
    shadow_down_ratio: float = 0.5
    attentive_eraser: bool = False
    guide_decay_down: float = 0.6
    guide_decay_side: float = 0.2
    struct_weight_fine: float = 0.3
    light_weight_fine: float = 1.0
    dynamics_weight_fine: float = 0.2
    struct_guidance_every_n: int = 2
    causal_smooth_sigma_px: float = 18.0
    causal_diff_struct_weight: float = 0.55
    causal_diff_light_weight: float = 0.45
    shadow_pin_release: float = 0.0  # stable: allow partial latent update in W_shadow outside edit mask
    jepa_pseudo_blur_sigma: float = 3.5
    jepa_pseudo_noise_scale: float = 0.45
    jepa_pseudo_ring_px: int = 18


@dataclass
class InpaintInputs:
    image: torch.Tensor
    mask: torch.Tensor
    prompt: str = ""
    negative_prompt: str = ""


@dataclass
class StrategyMetrics:
    align_losses: list[float] = field(default_factory=list)
    enc_losses: list[float] = field(default_factory=list)
    dynamics_losses: list[float] = field(default_factory=list)
