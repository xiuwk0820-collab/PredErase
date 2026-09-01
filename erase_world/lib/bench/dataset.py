from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class BenchSample:
    sample_id: str
    image: Image.Image
    mask: Image.Image
    gt: Image.Image


class BenchDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> BenchSample: ...
    def close(self) -> None: ...


def _collect_mask_ids(root: Path) -> list[str]:
    mask_dir = root / "mask"
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"missing mask/ under {root}")
    ids = sorted(
        p.name for p in mask_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not ids:
        raise FileNotFoundError(f"no mask images under {mask_dir}")
    return ids


class SmartEraserFolderDataset:
    """Extracted SmartEraser bench: val_*/paste, mask/, gt/."""

    DATASETS = {
        "rord": "val_rord",
        "defacto": "val_defacto",
        "paste": "val_paste",
    }

    def __init__(
        self,
        bench_root: str | Path,
        dataset: str,
        limit: int | None = None,
        offset: int = 0,
        sample_ids: list[str] | None = None,
    ):
        bench_root = Path(bench_root)
        if dataset not in self.DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}, choose from {list(self.DATASETS)}")
        self.root = bench_root / self.DATASETS[dataset]
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.dataset = dataset
        ids = _collect_mask_ids(self.root)
        if sample_ids is not None:
            missing = [sid for sid in sample_ids if sid not in set(ids)]
            if missing:
                raise FileNotFoundError(f"missing {len(missing)} sample ids under {self.root}, e.g. {missing[0]}")
            ids = list(sample_ids)
        self.ids = ids[offset : offset + limit if limit is not None else None]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> BenchSample:
        sample_id = self.ids[index]
        paste = Image.open(self.root / "paste" / sample_id).convert("RGB")
        mask = Image.open(self.root / "mask" / sample_id).convert("L")
        gt = Image.open(self.root / "gt" / sample_id).convert("RGB")
        return BenchSample(sample_id=sample_id, image=paste, mask=mask, gt=gt)

    def close(self) -> None:
        return


class SmartEraserZipDataset:
    """SmartEraser bench zip: paste/ (input), mask/, gt/."""

    DATASETS = {
        "rord": "val_rord.zip",
        "defacto": "val_defacto.zip",
        "paste": "val_paste.zip",
    }

    def __init__(
        self,
        bench_root: str | Path,
        dataset: str,
        limit: int | None = None,
        offset: int = 0,
    ):
        bench_root = Path(bench_root)
        if dataset not in self.DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}, choose from {list(self.DATASETS)}")
        self.zip_path = bench_root / self.DATASETS[dataset]
        if not self.zip_path.is_file():
            raise FileNotFoundError(self.zip_path)
        self.dataset = dataset
        self._zip = zipfile.ZipFile(self.zip_path)

        ids = sorted(
            Path(name).name
            for name in self._zip.namelist()
            if name.startswith("mask/") and name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        self.ids = ids[offset : offset + limit if limit is not None else None]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> BenchSample:
        sample_id = self.ids[index]
        paste = self._read_image(f"paste/{sample_id}")
        mask = self._read_image(f"mask/{sample_id}").convert("L")
        gt = self._read_image(f"gt/{sample_id}")
        return BenchSample(sample_id=sample_id, image=paste, mask=mask, gt=gt)

    def _read_image(self, name: str) -> Image.Image:
        if name not in self._zip.namelist():
            raise KeyError(f"missing {name} in {self.zip_path}")
        return Image.open(io.BytesIO(self._zip.read(name))).convert("RGB")

    def close(self) -> None:
        self._zip.close()


class RemovalBenchDataset:
    """RemovalBench (HF BaiLing/RemovalBench): images/, masks/, gt/."""

    def __init__(
        self,
        bench_root: str | Path,
        limit: int | None = None,
        offset: int = 0,
        sample_ids: list[str] | None = None,
    ):
        self.root = Path(bench_root)
        images_dir = self.root / "images"
        masks_dir = self.root / "masks"
        gt_dir = self.root / "gt"
        for path in (images_dir, masks_dir, gt_dir):
            if not path.is_dir():
                raise FileNotFoundError(path)

        image_ids = sorted(
            p.name for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        ids: list[str] = []
        for sample_id in image_ids:
            if not (masks_dir / sample_id).is_file() or not (gt_dir / sample_id).is_file():
                continue
            ids.append(sample_id)
        if not ids:
            raise FileNotFoundError(f"no complete triplets under {self.root}")
        if sample_ids is not None:
            available = set(ids)
            missing = [sid for sid in sample_ids if sid not in available]
            if missing:
                raise FileNotFoundError(
                    f"missing {len(missing)} sample ids under {self.root}, e.g. {missing[0]}"
                )
            ids = list(sample_ids)
        self.ids = ids[offset : offset + limit if limit is not None else None]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> BenchSample:
        sample_id = self.ids[index]
        image = Image.open(self.root / "images" / sample_id).convert("RGB")
        mask = Image.open(self.root / "masks" / sample_id).convert("L")
        gt = Image.open(self.root / "gt" / sample_id).convert("RGB")
        return BenchSample(sample_id=sample_id, image=image, mask=mask, gt=gt)

    def close(self) -> None:
        return


def build_bench_dataset(
    bench_root: str | Path,
    dataset: str,
    bench_format: str = "auto",
    limit: int | None = None,
    offset: int = 0,
    sample_ids: list[str] | None = None,
) -> BenchDataset:
    """auto: folder if val_{dataset} exists under bench_root, else zip."""
    if dataset == "removalbench":
        return RemovalBenchDataset(bench_root, limit=limit, offset=offset, sample_ids=sample_ids)

    bench_root = Path(bench_root)
    folder_name = SmartEraserFolderDataset.DATASETS.get(dataset)
    folder_path = bench_root / folder_name if folder_name else None

    if bench_format == "folder":
        return SmartEraserFolderDataset(
            bench_root, dataset, limit=limit, offset=offset, sample_ids=sample_ids
        )
    if bench_format == "zip":
        if sample_ids is not None:
            raise ValueError("sample_ids requires extracted folder bench_format=folder")
        return SmartEraserZipDataset(bench_root, dataset, limit=limit, offset=offset)
    if bench_format != "auto":
        raise ValueError(f"unknown bench_format={bench_format!r}")

    if folder_path is not None and folder_path.is_dir():
        return SmartEraserFolderDataset(
            bench_root, dataset, limit=limit, offset=offset, sample_ids=sample_ids
        )
    if sample_ids is not None:
        raise ValueError("sample_ids requires extracted folder under bench_root")
    return SmartEraserZipDataset(bench_root, dataset, limit=limit, offset=offset)
