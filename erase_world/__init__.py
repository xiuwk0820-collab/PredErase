"""Erase-World: training-free object-and-effect removal (FLUX.2-klein-4B + I-JEPA)."""

__version__ = "0.1.0"

from .pipeline import EraseWorldPipeline, run_erase_world

__all__ = ["EraseWorldPipeline", "run_erase_world", "__version__"]
