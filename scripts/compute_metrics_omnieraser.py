#!/usr/bin/env python3
"""OmniEraser-protocol metrics on saved predictions (RemovalBench / RORD-style)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from erase_world.lib.bench.dataset import build_bench_dataset
from erase_world.lib.bench.paper_protocol import PaperProtocolAccumulator, compute_paper_metrics
from erase_world.lib.bench.omnieraser_protocol import OmniEraserProtocol


def _resolve_pred(pred_dir: Path, sample_id: str) -> Path | None:
    stem = Path(sample_id).stem
    for p in (
        pred_dir / sample_id,
        pred_dir / f"{stem}.png",
        pred_dir / stem / "output.png",
        pred_dir / "images" / stem / "output.png",
    ):
        if p.is_file():
            return p
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="OmniEraser / paper metrics")
    p.add_argument("--dataset", default="removalbench", choices=["removalbench", "rord", "defacto"])
    p.add_argument("--bench-root", required=True)
    p.add_argument("--bench-format", default="auto")
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--method", default="erase_world")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / pred_dir
    out_dir = Path(args.out_dir) if args.out_dir else pred_dir.parent
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = build_bench_dataset(args.bench_root, args.dataset, bench_format=args.bench_format, limit=args.limit)
    paper_acc = PaperProtocolAccumulator()
    omni_acc = OmniEraserProtocol()
    rows = []

    for i in tqdm(range(len(ds)), desc="omnieraser-metrics"):
        s = ds[i]
        pred_path = _resolve_pred(pred_dir, s.sample_id)
        if pred_path is None:
            continue
        pred = Image.open(pred_path).convert("RGB")
        m = compute_paper_metrics(pred, s.gt, s.mask)
        paper_acc.update(pred, s.gt, s.mask)
        omni_acc.update(pred, s.gt)
        rows.append({"sample_id": s.sample_id, **m})

    ds.close()
    summary = {
        "method": args.method,
        "n": len(rows),
        "paper": paper_acc.compute(),
        "omnieraser": omni_acc.compute(),
    }
    (out_dir / "metrics_omnieraser.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        with open(out_dir / "metrics_omnieraser.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
