#!/usr/bin/env python3
"""Run paper ablations: Full / native / w/o JEPA / w/o Prefill / w/o Shadow Prompt / prior swap."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from PIL import Image

from erase_world.pipeline import EraseWorldPipeline
from erase_world.lib.ablation.patch_backbone import build_patch_ablation_strategy
from erase_world.lib.core.config import build_guidance_config
from erase_world.flux_fill import run_flux_fill
from erase_world.masks import build_m_flux
from erase_world.utils import resize_longest

VARIANT_CONFIGS = {
    "full": "configs/default.yaml",
    "native": "configs/ablation_native.yaml",
    "wo_jepa": "configs/ablation_wo_jepa.yaml",
    "wo_prefill": "configs/ablation_wo_prefill.yaml",
    "wo_shadow_prompt": "configs/ablation_wo_shadow_prompt.yaml",
    "prior_clip": "configs/prior_clip.yaml",
    "prior_dinov2": "configs/prior_dinov2.yaml",
    "prior_jepa": "configs/default.yaml",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Erase-World ablation runner")
    p.add_argument("--variant", choices=sorted(VARIANT_CONFIGS), default="full")
    p.add_argument("--config", default=None, help="override yaml path")
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=22)
    args = p.parse_args()

    cfg_rel = args.config or VARIANT_CONFIGS[args.variant]
    cfg_path = ROOT / cfg_rel if not Path(cfg_rel).is_absolute() else Path(cfg_rel)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    out_path = Path(args.output or f"outputs/ablation_{args.variant}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prior = cfg.get("ablation_prior")
    if prior in ("clip_patch", "dino_v2"):
        # Prior-swap path: same schedule / M_flux / eta, different critic
        pipe = EraseWorldPipeline(cfg_path, load_pipe=True)
        flux_cfg = cfg["flux"]
        max_side = int(flux_cfg.get("max_side", 768))
        image = resize_longest(image, max_side)
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
        edit = build_m_flux(
            image, mask,
            dilate_r=int(cfg.get("guidance", {}).get("mask_run_dilate_px", 4)),
            use_shadow=bool(flux_cfg.get("use_contact_shadow", True)),
        )
        strategy = build_patch_ablation_strategy(prior, cfg, pipe.ijepa_dev)
        guidance_cfg = build_guidance_config(cfg.get("guidance", {}))
        # optional source prefill
        source = image
        if flux_cfg.get("source_prefill", True):
            import numpy as np
            from erase_world.lib.utils.texture_harmonize import fill_flux_source
            source = fill_flux_source(image, np.asarray(edit), ring_px=24)
        result = run_flux_fill(
            pipe.pipe,
            image=source,
            mask=edit,
            prompt=str(flux_cfg.get("prompt", "")),
            strategy=strategy,
            guidance_cfg=guidance_cfg,
            num_inference_steps=int(flux_cfg.get("num_inference_steps", 14)),
            guidance_scale=float(flux_cfg.get("guidance_scale_flux", 3.5)),
            seed=args.seed,
            device=pipe.flux_dev,
        )
    else:
        pipe = EraseWorldPipeline(cfg_path)
        result = pipe(image, mask, seed=args.seed)

    result.save(out_path)
    print(f"[erase-world] variant={args.variant} -> {out_path}")


if __name__ == "__main__":
    main()
