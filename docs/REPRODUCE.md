# Reproduce paper tables (checklist)

Defaults match `configs/default.yaml` and `docs/METHOD_ALIGNMENT.md`.  
Use **seed-averaged** predictions over \(S=\{22,23,24,25,26\}\) (mean pixels, **not** best-of-5).

## 0. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Accept FLUX.2-klein-4B license on Hugging Face, then:
huggingface-cli login
```

Place datasets per `docs/DATA.md`. Ensure I-JEPA checkpoint is available (`ijepa.checkpoint: auto`).

## 1. Smoke test

```bash
python scripts/run_inference.py \
  --config configs/default.yaml \
  --image examples/demo_image.jpg \
  --mask examples/demo_mask.png \
  --output outputs/demo.png
```

## 2. Table — main results (RemovalBench / RORD)

```bash
# Full (seed-averaged)
python scripts/run_benchmark.py \
  --config configs/default.yaml \
  --dataset removalbench \
  --bench-root data/RemovalBench \
  --output-dir outputs/bench_rb_full \
  --seed-average

python scripts/compute_metrics_omnieraser.py \
  --dataset removalbench \
  --bench-root data/RemovalBench \
  --pred-dir outputs/bench_rb_full/images \
  --out-dir outputs/bench_rb_full
```

Repeat with `--dataset rord --bench-root data/bench_extracted --bench-format folder`.

For DEFACTO / SmartEraser columns:

```bash
python scripts/compute_metrics_smarteraser.py \
  --dataset defacto \
  --bench-root data/bench_extracted \
  --bench-format folder \
  --pred-dir outputs/bench_defacto/images
```

## 3. Table — module ablations

```bash
for v in full native wo_jepa wo_prefill wo_shadow_prompt; do
  python scripts/run_ablation.py \
    --variant $v \
    --image examples/demo_image.jpg \
    --mask examples/demo_mask.png \
    --output outputs/ablation_${v}.png
done
```

For full-bench ablations, swap `--config` in `run_benchmark.py`:

| Variant | Config |
|---------|--------|
| Full | `configs/default.yaml` |
| Pure FLUX.2 | `configs/ablation_native.yaml` |
| w/o JEPA | `configs/ablation_wo_jepa.yaml` |
| w/o Prefill | `configs/ablation_wo_prefill.yaml` |
| w/o Shadow Prompt | `configs/ablation_wo_shadow_prompt.yaml` |

## 4. Table — backbone / prior swap

```bash
python scripts/run_ablation.py --variant prior_clip ...
python scripts/run_ablation.py --variant prior_dinov2 ...
python scripts/run_ablation.py --variant prior_jepa ...   # same as full
```

Same \(\eta\), schedule, and \(M_{flux}\) as Full; only the patch prior changes.

## 5. Eval resolution

Predictions may be generated at longest-side 768; metrics scripts resize / compare against GT. Optional 1024² dump is saved as `output_1024.png` by `run_benchmark.py`.

## Notes

- Do not cherry-pick seeds; report the **mean** over \(S\).
- Do not claim paper numbers from a different hyperparameter set (see METHOD_ALIGNMENT.md).
- Code will be publicly released after publication; this package is the anonymized review / pre-release skeleton.
