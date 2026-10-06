"""A copied-row comparison that keeps the entire holdout set fixed."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from leakage_lab.cards import ExperimentCard, PathResult
from leakage_lab.experiments.copied_rows import FEATURES, TARGET, TEST_SIZE, generate
from leakage_lab.metrics import fit_forest

COPY_FRACTION = 0.50

MECHANISM = (
    "This is a separate, matched holdout comparison. Both paths score the exact "
    "same untouched test table. Each training table has the same row count: the "
    "contaminated one receives copies of half the holdout claims, while the control "
    "receives an equal number of copies drawn from its own training rows. The model "
    "and holdout do not change. Only the presence of copied holdout rows in training "
    "changes, so the score difference is directly attributable to contamination."
)


def matched_split(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return equal-size clean/leaky training frames, one shared holdout, and copy mask."""

    frame = generate(seed)
    base_train, holdout = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=frame[TARGET],
    )
    copy_positions, _ = train_test_split(
        np.arange(len(holdout)),
        test_size=1.0 - COPY_FRACTION,
        random_state=seed + 1,
        stratify=holdout[TARGET],
    )
    control_positions, _ = train_test_split(
        np.arange(len(base_train)),
        test_size=1.0 - (len(copy_positions) / len(base_train)),
        random_state=seed + 2,
        stratify=base_train[TARGET],
    )
    leaky_train = pd.concat([base_train, holdout.iloc[copy_positions]], ignore_index=True)
    clean_train = pd.concat([base_train, base_train.iloc[control_positions]], ignore_index=True)
    holdout = holdout.reset_index(drop=True)
    copied_mask = np.zeros(len(holdout), dtype=bool)
    copied_mask[copy_positions] = True
    return leaky_train, clean_train, holdout, copied_mask


def _path(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> tuple[PathResult, np.ndarray]:
    scores, proba = fit_forest(
        train.loc[:, list(FEATURES)],
        train[TARGET].to_numpy(),
        test.loc[:, list(FEATURES)],
        test[TARGET].to_numpy(),
        seed,
    )
    return (
        PathResult(scores=scores, n_train=int(len(train)), n_test=int(len(test)), features=FEATURES),
        proba,
    )


def run(seed: int) -> ExperimentCard:
    leaky_train, clean_train, holdout, copied_mask = matched_split(seed)
    leaky, leaky_proba = _path(leaky_train, holdout, seed)
    clean, _clean_proba = _path(clean_train, holdout, seed)
    predicted = (leaky_proba >= 0.5).astype(int)
    copied_accuracy = float(accuracy_score(holdout[TARGET].to_numpy()[copied_mask], predicted[copied_mask]))
    unseen_accuracy = float(accuracy_score(holdout[TARGET].to_numpy()[~copied_mask], predicted[~copied_mask]))
    copied_count = int(copied_mask.sum())
    measured = (
        f"Both paths score the same {leaky.n_test} holdout rows. The contaminated training set contains "
        f"copies of {copied_count} of them, while the control has the same training row count with no holdout "
        f"copies. The leaky forest reaches ROC-AUC {leaky.scores.roc_auc:.4f}; the clean forest reaches "
        f"{clean.scores.roc_auc:.4f}. Accuracy is {copied_accuracy:.4f} on copied holdout rows and "
        f"{unseen_accuracy:.4f} on untouched holdout rows."
    )
    return ExperimentCard(
        slug="matched-holdout-copies",
        number=3,
        title="Matched holdout copies",
        kicker="Copied rows, same holdout on both paths",
        mechanism=MECHANISM,
        measured=measured,
        seed=seed,
        population="1,200 synthetic claims; one fixed holdout scored by both paths",
        positive_rate=float(holdout[TARGET].mean()),
        leaky=leaky,
        clean=clean,
        facts=(
            ("Shared holdout rows", str(leaky.n_test)),
            ("Copied holdout rows in contaminated training", f"{copied_count} of {leaky.n_test}"),
            ("Clean training rows matching holdout", "0"),
            ("Forest accuracy on copied holdout rows", f"{copied_accuracy:.4f}"),
            ("Forest accuracy on untouched holdout rows", f"{unseen_accuracy:.4f}"),
        ),
    )
