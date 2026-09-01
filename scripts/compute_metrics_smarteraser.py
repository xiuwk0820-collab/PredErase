#!/usr/bin/env python3
"""SmartEraser-protocol metrics (CMMD / ReMOVE / LPIPS / SSIM / PSNR / FID)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from erase_world.lib.bench.dataset import build_bench_dataset
from erase_world.lib.bench.smarteraser_protocol import SmartEraserProtocolAccumulator


def _resolve_pred(pred_dir: Path, sample_id: str) -> Path | None:
    stem = Path(sample_id).stem
    for p in (
        pred_dir / f"{stem}.png",
        pred_dir / stem / "output.png",
        pred_dir / "images" / stem / "output.png",
    ):
        if p.is_file():
            return p
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="SmartEraser protocol metrics")
    p.add_argument("--dataset", default="defacto", choices=["removalbench", "rord", "defacto"])
    p.add_argument("--bench-root", required=True)
    p.add_argument("--bench-format", default="auto")
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--method", default="erase_world")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / pred_dir
    out_dir = Path(args.out_dir) if args.out_dir else pred_dir.parent
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = build_bench_dataset(args.bench_root, args.dataset, bench_format=args.bench_format, limit=args.limit)
    acc = SmartEraserProtocolAccumulator()

    n = 0
    for i in tqdm(range(len(ds)), desc="smarteraser-metrics"):
        s = ds[i]
        pred_path = _resolve_pred(pred_dir, s.sample_id)
        if pred_path is None:
            continue
        pred = Image.open(pred_path).convert("RGB")
        acc.update(pred, s.gt, s.mask)
        n += 1

    ds.close()
    summary = {"method": args.method, "n": n, **acc.compute()}
    (out_dir / "metrics_smarteraser.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
