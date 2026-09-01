<p align="center">
  <h1 align="center">Erase-World</h1>
  <p align="center">
    <b>Remove Objects and Their Effects in Images<br>with Training-Free World-Model Guidance</b>
  </p>
  <p align="center">
    <a href="https://github.com/xiuwk/Erase-World"><img src="https://img.shields.io/badge/GitHub-Erase--World-black?logo=github" alt="GitHub"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
    <a href="https://github.com/xiuwk/Erase-World"><img src="https://img.shields.io/badge/code-training--free-0A7B3E" alt="training-free"></a>
  </p>
  <p align="center">
    <a href="https://github.com/xiuwk">Waikit Xiu</a><sup>1</sup>&nbsp;&nbsp;
    Qiang Lu<sup>2</sup>&nbsp;&nbsp;
    Junbiao Chen<sup>2</sup>&nbsp;&nbsp;
    Ziyu Lin<sup>3</sup>&nbsp;&nbsp;
    Xiying Li<sup>2*</sup>
  </p>
  <p align="center">
    <sup>1</sup>The University of Hong Kong&nbsp;&nbsp;
    <sup>2</sup>Sun Yat-sen University&nbsp;&nbsp;
    <sup>3</sup>Tsinghua University<br>
    <a href="mailto:xiuwk0820@connect.hku.hk">xiuwk0820@connect.hku.hk</a> (Waikit Xiu)
    &nbsp;·&nbsp;
    <sup>*</sup>Corresponding author
  </p>
  <p align="center">
    <a href="https://github.com/xiuwk/Erase-World"><b>Code</b></a>
    &nbsp;·&nbsp;
    <a href="#citation"><b>BibTeX</b></a>
  </p>
</p>

<p align="center">
  <img src="assets/teaser_results.png" width="100%" alt="Erase-World teaser: object-and-effect removal">
</p>
<p align="center"><em>Instance-only masks (red). Erase-World removes the object together with cast shadows and contact shading — no paired training, no weight updates.</em></p>

---

Object-and-effect removal is not the same as filling a hole. A frozen inpainter constrained to \(M_{\mathrm{obj}}\) often leaves shadows and contact shading on the support. **Erase-World** keeps **FLUX.2-klein-4B** and **I-JEPA ViT-H/14** frozen, and only steers latents at test time:

1. **Where to edit** — a contact-aware geometric prior expands the user mask into \(M_{\mathrm{flux}}\).
2. **What to reconstruct** — a cached I-JEPA hole prediction guides sparse projected updates inside that support.
3. **What not to touch** — packed-latent locking pins coordinates outside the editable region.

Evaluated on **RemovalBench**, **RORD-Val**, and **DEFACTO-Val** under the OmniEraser / SmartEraser protocols.

<p align="center">
  <img src="assets/erase_world_pipeline.png" width="100%" alt="Erase-World pipeline">
</p>
<p align="center"><em>Pipeline: gray-fill + frozen I-JEPA target (top), contact-band \(M_{\mathrm{flux}}\) (bottom), sparse guidance on frozen FLUX.2-klein-4B (center).</em></p>

## Qualitative

<p align="center">
  <img src="assets/qualitative_comparison.png" width="100%" alt="Qualitative comparison on RemovalBench">
</p>
<p align="center"><em>RemovalBench, instance-only masks. Left to right: Input, mask, native FLUX.2, OmniEraser, <b>ours</b>, clean plate.</em></p>

## Setup

```bash
git clone https://github.com/xiuwk/Erase-World.git
cd Erase-World
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # accept FLUX.2-klein-4B
```

| Checkpoint | Source |
|------------|--------|
| FLUX.2-klein-4B | `black-forest-labs/FLUX.2-klein-4B` |
| I-JEPA ViT-H/14-1K | Meta `IN1K-vit.h.14-300e.pth.tar` → `./checkpoints/` or `~/.cache/erase-world/` |

Datasets (not shipped): see [`docs/DATA.md`](docs/DATA.md).

## Quick start

```bash
python scripts/run_inference.py \
  --config configs/default.yaml \
  --image examples/demo_image.jpg \
  --mask examples/demo_mask.png \
  --output outputs/demo.png
```

Seed-averaged inference (\(S=\{22,\ldots,26\}\), **not** best-of-5):

```bash
python scripts/run_inference.py ... --seed-average
```

### Benchmarks

```bash
python scripts/run_benchmark.py \
  --config configs/default.yaml \
  --dataset removalbench \
  --bench-root data/RemovalBench \
  --output-dir outputs/bench_rb_full \
  --seed-average
```

### Ablations & priors

```bash
python scripts/run_ablation.py --variant full --image ... --mask ... --output outputs/full.png
python scripts/run_ablation.py --variant native --image ... --mask ...
python scripts/run_ablation.py --variant wo_jepa --image ... --mask ...
python scripts/run_ablation.py --variant prior_clip --image ... --mask ...
python scripts/run_ablation.py --variant prior_dinov2 --image ... --mask ...
```

### Metrics

```bash
python scripts/compute_metrics_omnieraser.py \
  --dataset removalbench --bench-root data/RemovalBench \
  --pred-dir outputs/bench_rb_full/images

python scripts/compute_metrics_smarteraser.py \
  --dataset defacto --bench-root data/bench_extracted --bench-format folder \
  --pred-dir outputs/bench_defacto/images
```

Paper ↔ code map: [`docs/METHOD_ALIGNMENT.md`](docs/METHOD_ALIGNMENT.md) · reproduce: [`docs/REPRODUCE.md`](docs/REPRODUCE.md)

## Citation

```bibtex
@article{xiu2026eraseworld,
  title={Erase-World: Remove Objects and Their Effects in Images with Training-Free World-Model Guidance},
  author={Xiu, Waikit and Lu, Qiang and Chen, Junbiao and Lin, Ziyu and Li, Xiying},
  year={2026},
  url={https://github.com/xiuwk/Erase-World}
}
```

## License

MIT License. Fill and I-JEPA checkpoints remain under their upstream licenses.

## Acknowledgements

We thank the authors of FLUX.2, I-JEPA, OmniEraser, and SmartEraser for models, benchmarks, and evaluation protocols.
