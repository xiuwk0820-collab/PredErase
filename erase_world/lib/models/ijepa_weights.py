from __future__ import annotations

from pathlib import Path

# Meta official ViT-H/14 ImageNet-1K 300-epoch checkpoint (~10GB).
OFFICIAL_IJEPA_VITH14_URL = (
    "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar"
)
OFFICIAL_IJEPA_VITH14_NAME = "IN1K-vit.h.14-300e.pth.tar"
MIN_CHECKPOINT_BYTES = 9_000_000_000
LOG_PREFIX = "[erase-world]"


def _default_cache_dir() -> Path:
    new = Path.home() / ".cache" / "erase-world"
    legacy = Path.home() / ".cache" / "erase-world"
    if legacy.is_dir() and not new.exists():
        return legacy
    return new


def ensure_ijepa_checkpoint(cache_dir: str | Path | None = None) -> str | None:
    """Download official I-JEPA checkpoint if missing. Returns path or None."""
    if cache_dir is None:
        cache_dir = _default_cache_dir()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / OFFICIAL_IJEPA_VITH14_NAME

    if dest.is_file() and dest.stat().st_size >= MIN_CHECKPOINT_BYTES:
        return str(dest)

    partial = cache_dir / f"{OFFICIAL_IJEPA_VITH14_NAME}.partial"
    print(f"{LOG_PREFIX} downloading official I-JEPA checkpoint -> {dest}")

    # Prefer curl/wget for large files (more stable than requests on this host).
    import shutil
    import subprocess

    if shutil.which("curl"):
        cmd = [
            "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "3",
            "-C", "-", "-o", str(partial), OFFICIAL_IJEPA_VITH14_URL,
        ]
        print(f"{LOG_PREFIX} curl download (resume ok) ...")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0 and partial.is_file():
            partial.rename(dest)
            if dest.stat().st_size >= MIN_CHECKPOINT_BYTES:
                return str(dest)
        print(f"{LOG_PREFIX} curl failed (code={proc.returncode}), trying requests ...")

    try:
        import requests

        resume = partial.stat().st_size if partial.is_file() else 0
        headers = {"Range": f"bytes={resume}-"} if resume > 0 else {}
        with requests.get(OFFICIAL_IJEPA_VITH14_URL, stream=True, timeout=60, headers=headers) as r:
            r.raise_for_status()
            mode = "ab" if resume > 0 else "wb"
            with open(partial, mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        partial.rename(dest)
        if dest.stat().st_size < MIN_CHECKPOINT_BYTES:
            raise OSError(f"incomplete download ({dest.stat().st_size} bytes)")
        return str(dest)
    except Exception as exc:
        print(f"{LOG_PREFIX} warn: I-JEPA checkpoint download failed ({exc})")
        print(f"{LOG_PREFIX} retry: python scripts/download_ijepa.py")
        print(f"{LOG_PREFIX} or place file manually at:")
        print(f"  {dest}")
        return None
