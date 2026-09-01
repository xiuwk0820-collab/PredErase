#!/usr/bin/env python3
"""Benchmark runner for RemovalBench / RORD-Val / DEFACTO-Val."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
from tqdm import tqdm

from erase_world.lib.bench.dataset import build_bench_dataset
from erase_world.pipeline import EraseWorldPipeline
from erase_world.utils import PAPER_SEEDS, average_images, parse_seeds, resize_eval


def main() -> None:
    p = argparse.ArgumentParser(description="Erase-World benchmark")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", choices=["removalbench", "rord", "defacto"], default="removalbench")
    p.add_argument("--bench-root", required=True, help="dataset root (see docs/DATA.md)")
    p.add_argument("--bench-format", default="auto", choices=["auto", "folder", "zip"])
    p.add_argument("--output-dir", default="outputs/bench")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--seed-average", action="store_true", help="average seeds 22..26")
    p.add_argument("--seeds", default=None, help="comma-separated seeds (default paper set)")
    p.add_argument("--eval-size", type=int, default=1024)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sample-list", default=None, help="json with sample_ids")
    args = p.parse_args()

    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    img_root = out_root / "images"
    img_root.mkdir(parents=True, exist_ok=True)

    sample_ids = None
    if args.sample_list:
        payload = json.loads(Path(args.sample_list).read_text(encoding="utf-8"))
        sample_ids = list(payload["sample_ids"] if isinstance(payload, dict) else payload)

    ds = build_bench_dataset(
        args.bench_root,
        args.dataset,
        bench_format=args.bench_format,
        limit=args.limit,
        offset=args.offset,
        sample_ids=sample_ids,
    )
    pipe = EraseWorldPipeline(cfg_path)
    seeds = parse_seeds(args.seeds) if (args.seed_average or args.seeds) else None

    meta = []
    for i in tqdm(range(len(ds)), desc=f"{args.dataset}"):
        s = ds[i]
        stem = Path(s.sample_id).stem
        out_png = img_root / stem / "output.png"
        if args.resume and out_png.is_file():
            continue
        if seeds:
            preds = [pipe(s.image, s.mask, seed=seed) for seed in seeds]
            pred = average_images(preds)
        else:
            pred = pipe(s.image, s.mask)

        # save native-res + eval-res copy
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pred.save(out_png)
        resize_eval(pred, args.eval_size).save(out_png.parent / "output_1024.png")
        s.image.save(out_png.parent / "input.png")
        s.mask.save(out_png.parent / "mask.png")
        s.gt.save(out_png.parent / "gt.png")
        meta.append({"sample_id": s.sample_id, "path": str(out_png)})

    ds.close()
    (out_root / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[erase-world] wrote {len(meta)} preds -> {out_root}")
    print("[erase-world] next: python scripts/compute_metrics_omnieraser.py ...")


if __name__ == "__main__":
    main()
