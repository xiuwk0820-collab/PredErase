"""Method identifiers for PredErase (legacy key: erase_world)."""

ERASE_WORLD = "erase_world"
FLUX_FILL_NATIVE = "flux_fill_native"
LEGACY_ERASE_WORLD = "tf_jepa_fill"  # backward compat


def normalize_method(method: str) -> str:
    if method == LEGACY_ERASE_WORLD:
        return ERASE_WORLD
    return method


def is_erase_world(method: str) -> bool:
    return normalize_method(method) == ERASE_WORLD
