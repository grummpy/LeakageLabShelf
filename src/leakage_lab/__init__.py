"""Leakage Lab Shelf: planted leaks beside clean controls."""

__version__ = "0.1.0"
DEFAULT_SEED = 42

from leakage_lab.catalog import run_shelf

__all__ = ["DEFAULT_SEED", "__version__", "run_shelf"]
