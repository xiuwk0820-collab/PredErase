from __future__ import annotations

import torch


def _gpu_index(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    return 0


def pick_aux_gpu(primary: str) -> str:
    """Pick another GPU with the most free memory."""
    if not torch.cuda.is_available():
        return primary
    n = torch.cuda.device_count()
    if n <= 1:
        return primary
    primary_idx = _gpu_index(primary)
    best_idx, best_free = primary_idx, -1
    for i in range(n):
        if i == primary_idx:
            continue
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_idx = i
    return f"cuda:{best_idx}"


def resolve_devices(
    flux_device: str | None = None,
    ijepa_device: str | None = None,
) -> tuple[str, str]:
    flux = flux_device or "cuda:0"
    if ijepa_device in (None, "auto"):
        ijepa = pick_aux_gpu(flux)
    else:
        ijepa = ijepa_device
    return flux, ijepa
