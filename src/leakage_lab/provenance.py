"""Small, versioned manifests for reproducible synthetic shelf runs."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from leakage_lab import __version__
from leakage_lab.cards import ExperimentCard

MANIFEST_VERSION = 1
SUMMARY_SEEDS = (0, 7, 42)


def shelf_manifest(cards: tuple[ExperimentCard, ...], seed: int) -> dict[str, Any]:
    """Describe the generated input, split, and runtime without raw records."""

    return {
        "manifest_version": MANIFEST_VERSION,
        "source": {"tool": "Leakage Lab Shelf", "version": __version__},
        "seed": int(seed),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "constraints": {
            "network": "not used",
            "input": "all tables are generated locally; no provider or user data is read",
            "html": "rendered deterministically from cards and seed",
        },
        "experiments": [
            {
                "slug": card.slug,
                "input": card.population,
                "features": {"leaky": list(card.leaky.features), "clean": list(card.clean.features)},
                "split": {
                    "leaky": {"train_rows": card.leaky.n_train, "test_rows": card.leaky.n_test},
                    "clean": {"train_rows": card.clean.n_train, "test_rows": card.clean.n_test},
                },
            }
            for card in cards
        ],
    }


def write_manifest(path: Path, cards: tuple[ExperimentCard, ...], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(shelf_manifest(cards, seed), indent=2, sort_keys=True) + "\n", encoding="utf-8")
