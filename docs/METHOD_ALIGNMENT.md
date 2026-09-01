# Method ↔ Code Alignment (Erase-World)

This document maps paper Alg.1 / symbols to modules in this release.

## Alg.1 overview

| Step | Paper | Code |
|------|-------|------|
| 1 | Input \(I\), \(M_{obj}\) | `scripts/run_inference.py` / `EraseWorldPipeline.__call__` |
| 2 | \(I_{vis}=\mathrm{GrayFill}(I,M_{obj})\); cache \(E_{target}\) on object patches | `erase_world/masks.py::gray_fill` → `lib/models/ijepa_official.py::build_visible_fill` + `precompute_target`; called from `JEPAGuidanceStrategy.precompute` |
| 3 | \(M_{shadow}\) contact-band; \(M_{flux}=\mathrm{dilate}_r(M_{obj}\cup M_{shadow})\) | `masks.contact_band_shadow` (geometric upright band), `masks.build_m_flux` (`r=mask_run_dilate_px`) |
| 4 | Source prefill \(\mathrm{Encode}(I\odot(1-M_{flux}))\) | `lib/utils/texture_harmonize.py::fill_flux_source` when `flux.source_prefill: true` |
| 5 | \(T=14\) frozen FLUX.2-klein-4B Fill | `flux_fill.load_flux_fill` + `lib/pipeline/inpaint.py::run_inpaint` |
| 6 | JEPA guide on \(T_{guide}\): decode → \(L_{align}\) → \(z\leftarrow z-\eta\nabla\); UpdateAndLock on packed \(P\) | `jepa_guidance.py` / `lib/strategy/jepa.py::guide_latents`; edit region via packed mask + `latent_pin_visible` |
| 7 | Decode | Diffusers VAE decode in Fill pipeline |

## Symbols

| Paper | Config / code |
|-------|----------------|
| \(T\) | `flux.num_inference_steps` (14) |
| \(\eta\) | `guidance.guidance_scale` (0.45) |
| \(r\) | `guidance.mask_run_dilate_px` / `flux.dilate_r` (4) |
| \(w_{cfg}\) | `flux.guidance_scale_flux` (3.5) |
| \(\sigma,\delta_x\) | derived in `masks.contact_band_shadow` from object bbox |
| \(L_{align}\) | `jepa_guidance.alignment_loss` / `_alignment_loss` (MSE) |
| Packed \(P\) | `packing.packed_edit_mask` / `lib/pipeline/flux_runtime.precompute_packed_mask` |
| UpdateAndLock | `packing.update_and_lock` (+ runtime pin of non-edit coords to **current** Fill state) |

## \(T_{guide}=\{4,2\}\)

Paper lists explicit guidance steps. Runtime implements an equivalent schedule via:

- `guidance.struct_end_step: 4`
- `guidance.struct_guidance_every_n: 2`

i.e. guide while `step_idx <= struct_end_step` and `step_idx % every_n == 0` (see `JEPAGuidanceStrategy.should_guide`). This is the lean paper default in `configs/default.yaml`.

## Ablations

| Variant | Config | Mechanism |
|---------|--------|-----------|
| Full | `configs/default.yaml` | JEPA + M_flux + prefill + Full prompt |
| Pure FLUX.2 / native | `configs/ablation_native.yaml` | `method: flux_fill_native`, no prefill / shadow band |
| w/o JEPA | `configs/ablation_wo_jepa.yaml` | native Fill, keep M_flux + prefill + Full prompt |
| w/o Prefill | `configs/ablation_wo_prefill.yaml` | `source_prefill: false`, `gray_prefill: true` |
| w/o Shadow Prompt | `configs/ablation_wo_shadow_prompt.yaml` | short prompt; negative drops “shadow residual” |
| Prior CLIP / DINOv2 | `configs/prior_*.yaml` | `lib/ablation/patch_backbone.py` |

## Important naming notes

- Official backbone name: **FLUX.2-klein-4B** (`black-forest-labs/FLUX.2-klein-4B`).
- “Fill” denotes mask-conditioned inpainting usage, not a separate model name.
- \(L_{align}\) is computed on **object** hole patches; shadows/contact are cleaned via \(M_{flux}\) + prompt.
- Latent locking pins non-edit coords to the **current** Fill step state (not a permanent lock of the original source latent).

## Known deltas vs a full research tree

Internal experiments sometimes used stronger JEPA schedules (`struct_end_step=12`, `guidance_scale=1.8`, `max_side=640`). This release defaults to the **paper table** values. Do not mix the two when claiming paper numbers.
