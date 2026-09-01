# Data layout

Place datasets under `data/` (not shipped). Do **not** commit large archives.

## RemovalBench

Expected layout (Hugging Face `BaiLing/RemovalBench` style):

```
data/RemovalBench/
  images/   # input (object present)
  masks/    # instance masks
  gt/       # ground-truth removals
```

Download from the official RemovalBench release / HF dataset card, then point:

```bash
python scripts/run_benchmark.py \
  --dataset removalbench \
  --bench-root data/RemovalBench \
  --config configs/default.yaml \
  --output-dir outputs/bench_removalbench
```

## RORD-Val

SmartEraser-style extracted folders:

```
data/bench_extracted/
  val_rord/
    paste/
    mask/
    gt/
```

```bash
python scripts/run_benchmark.py \
  --dataset rord \
  --bench-root data/bench_extracted \
  --bench-format folder \
  --output-dir outputs/bench_rord
```

## DEFACTO-Val

```
data/bench_extracted/
  val_defacto/
    paste/
    mask/
    gt/
```

```bash
python scripts/run_benchmark.py \
  --dataset defacto \
  --bench-root data/bench_extracted \
  --bench-format folder \
  --output-dir outputs/bench_defacto
```

## Checkpoints (not included)

| Asset | Where to put |
|-------|----------------|
| FLUX.2-klein-4B | Hugging Face cache via `diffusers` (`black-forest-labs/FLUX.2-klein-4B`) |
| I-JEPA ViT-H/14 | `checkpoints/` or `~/.cache/erase-world/` (auto-download when `ijepa.checkpoint: auto`) |
| SAM ViT-H (ReMOVE only) | `checkpoints/sam_vit_h_4b8939.pth` |

Official I-JEPA URL (Meta):  
`https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar`

I-JEPA encoder loading lives in `erase_world/lib/models/ijepa_encoder.py`.
