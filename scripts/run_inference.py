#!/usr/bin/env python3
"""Single-image Erase-World inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from erase_world.pipeline import EraseWorldPipeline
from erase_world.utils import PAPER_SEEDS


def main() -> None:
    p = argparse.ArgumentParser(description="Erase-World single-image inference")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--output", default="outputs/result.png")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seed-average", action="store_true",
                   help=f"average over paper seeds {list(PAPER_SEEDS)}")
    p.add_argument("--flux-device", default=None)
    p.add_argument("--ijepa-device", default=None)
    args = p.parse_args()

    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    pipe = EraseWorldPipeline(
        cfg_path,
        flux_device=args.flux_device,
        ijepa_device=args.ijepa_device,
    )
    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    if args.seed_average:
        out = pipe.run_seed_averaged(image, mask)
    else:
        out = pipe(image, mask, seed=args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"[erase-world] saved {out_path}")


if __name__ == "__main__":
    main()
