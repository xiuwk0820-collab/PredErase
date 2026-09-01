"""Shared helpers: devices, resize, config IO, seed averaging."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
from PIL import Image

from .lib.utils.devices import resolve_devices

# Paper default seed set (seed-averaged, not best-of-N)
PAPER_SEEDS: tuple[int, ...] = (22, 23, 24, 25, 26)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resize_longest(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    m = max(w, h)
    if m <= max_side:
        return image
    scale = max_side / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return image.resize((nw, nh), Image.Resampling.LANCZOS)


def resize_eval(image: Image.Image, size: int = 1024) -> Image.Image:
    """Resize for evaluation (paper: 1024×1024)."""
    return image.resize((size, size), Image.Resampling.LANCZOS)


def average_images(images: Sequence[Image.Image]) -> Image.Image:
    """Pixel-wise mean over seed runs (seed-averaged)."""
    if not images:
        raise ValueError("empty image list")
    if len(images) == 1:
        return images[0]
    arrs = [np.asarray(im.convert("RGB"), dtype=np.float32) for im in images]
    mean = np.mean(np.stack(arrs, axis=0), axis=0)
    return Image.fromarray(np.clip(mean, 0, 255).astype(np.uint8))


def parse_seeds(spec: str | Iterable[int] | None) -> list[int]:
    if spec is None:
        return list(PAPER_SEEDS)
    if isinstance(spec, str):
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    return [int(x) for x in spec]


__all__ = [
    "PAPER_SEEDS",
    "average_images",
    "load_yaml",
    "parse_seeds",
    "resize_eval",
    "resize_longest",
    "resolve_devices",
]
