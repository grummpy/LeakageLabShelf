"""Small, versioned manifests for reproducible synthetic shelf runs."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from leakage_lab import __version__
from leakage_lab.cards import ExperimentCard

MANIFEST_VERSION = 2
SUMMARY_SEEDS = (0, 7, 42)
DEPENDENCY_PINS = {
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "scikit-learn": "1.7.2",
    "scipy": "1.14.1",
}


def _source_revision() -> str | None:
    """Return the checked-out revision when the source tree is available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _path_metrics(card: ExperimentCard) -> dict[str, float]:
    return {
        "leaky_roc_auc": card.leaky.scores.roc_auc,
        "clean_roc_auc": card.clean.scores.roc_auc,
        "paired_roc_auc_gap": card.roc_auc_gap,
    }


def multi_seed_records(runs: tuple[tuple[int, tuple[ExperimentCard, ...]], ...]) -> list[dict[str, Any]]:
    """Return every predeclared seed's paired results, without aggregation."""

    return [
        {
            "seed": seed,
            "experiments": [{"slug": card.slug, **_path_metrics(card)} for card in cards],
        }
        for seed, cards in runs
    ]


def shelf_manifest(
    cards: tuple[ExperimentCard, ...],
    seed: int,
    *,
    multi_seed_runs: tuple[tuple[int, tuple[ExperimentCard, ...]], ...] = (),
) -> dict[str, Any]:
    """Describe the generated input, split, and runtime without raw records."""

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "source": {
            "tool": "Leakage Lab Shelf",
            "version": __version__,
            "revision": _source_revision(),
        },
        "seed": int(seed),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "dependency_pins": DEPENDENCY_PINS,
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
    if multi_seed_runs:
        manifest["predeclared_multi_seed"] = {
            "seeds": [run_seed for run_seed, _cards in multi_seed_runs],
            "paired_roc_auc_results": multi_seed_records(multi_seed_runs),
            "interpretation": "descriptive synthetic results only; not a confidence interval or real-world benchmark",
        }
    return manifest


def write_manifest(
    path: Path,
    cards: tuple[ExperimentCard, ...],
    seed: int,
    *,
    multi_seed_runs: tuple[tuple[int, tuple[ExperimentCard, ...]], ...] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(shelf_manifest(cards, seed, multi_seed_runs=multi_seed_runs), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
