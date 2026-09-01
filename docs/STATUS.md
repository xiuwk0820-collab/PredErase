# Implementation status

## Paper stack (this release)

- [x] FLUX.2-klein-4B Fill (`erase_world/flux_fill.py`, `lib/pipeline/`)
- [x] I-JEPA guidance / \(L_{\mathrm{align}}\) / UpdateAndLock (`jepa_guidance.py`, `lib/strategy/jepa.py`)
- [x] GrayFill + geometric contact-band \(M_{\mathrm{shadow}}\) + \(M_{\mathrm{flux}}\) (`masks.py`)
- [x] Packed-latent helpers (`packing.py`)
- [x] Paper prompts (`prompts.py`)
- [x] Configs: Full, native, w/o JEPA, w/o Prefill, w/o Shadow Prompt, CLIP / DINOv2
- [x] Scripts: inference, ablation, benchmark, OmniEraser / SmartEraser metrics
- [x] I-JEPA encoder loader (`lib/models/ijepa_encoder.py`)

## Needs local assets

- [ ] GPU smoke test (FLUX + I-JEPA weights)
- [ ] ReMOVE needs optional SAM ViT-H checkpoint
- [ ] Some distilled FLUX schedules ignore the negative prompt (positive prompt is applied)

## Not in this tree

- Model weights / datasets / `outputs/`
- Author-private paths
