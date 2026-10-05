"""Train/test contamination from exact duplicate rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from leakage_lab.cards import ExperimentCard, PathResult, Scores
from leakage_lab.metrics import fit_forest, fit_logistic

FEATURES = ("x1", "x2", "x3", "x4", "x5", "x6")
TARGET = "claim_flagged"
TEST_SIZE = 0.30
N_ROWS = 1200

MECHANISM = (
    "The leaky pipeline appends a second copy of every claim, keeping the "
    "same row_id, and then takes a stratified row split. A test row can be "
    "the twin of a training row. The clean pipeline splits the original "
    "table, where each row_id appears once. Both paths train the same random "
    "forest (200 trees, min_samples_leaf=1, n_jobs=1). An exact copy has the "
    "same features as its twin, so trees that saw the twin vote its label. "
    "The card separates accuracy on those copied test rows from accuracy on "
    "the rows that were new. A logistic regression on the same two splits "
    "cannot memorize a twin; it shows what the features support when the "
    "model does not recall the row."
)


def generate(seed: int) -> pd.DataFrame:
    """Synthetic claims. x1–x3 carry a smooth signal; x4–x6 are noise."""

    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(N_ROWS, 3))
    noise = rng.normal(size=(N_ROWS, 3))
    logit = 0.15 + 0.85 * signal[:, 0] + 0.45 * signal[:, 1] - 0.35 * signal[:, 2]
    probability = 1.0 / (1.0 + np.exp(-logit))
    flagged = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "x1": signal[:, 0],
            "x2": signal[:, 1],
            "x3": signal[:, 2],
            "x4": noise[:, 0],
            "x5": noise[:, 1],
            "x6": noise[:, 2],
            TARGET: flagged,
            "row_id": np.arange(N_ROWS, dtype=int),
        }
    )


def contaminated_split(
    frame: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stack a second copy of every row, then split by row."""

    stacked = pd.concat([frame, frame], ignore_index=True)
    train, test = train_test_split(
        stacked,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=stacked[TARGET],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def clean_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=frame[TARGET],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def overlap_count(train: pd.DataFrame, test: pd.DataFrame) -> int:
    """Test rows whose row_id also appears in train."""

    return int(test["row_id"].isin(set(train["row_id"].tolist())).sum())


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return frame.loc[:, list(FEATURES)], frame[TARGET].to_numpy()


def forest_path(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> tuple[Scores, np.ndarray]:
    X_train, y_train = _xy(train)
    X_test, y_test = _xy(test)
    return fit_forest(X_train, y_train, X_test, y_test, seed)


def logistic_scores(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> Scores:
    X_train, y_train = _xy(train)
    X_test, y_test = _xy(test)
    return fit_logistic(X_train, y_train, X_test, y_test, seed)


def _path(scores: Scores, train: pd.DataFrame, test: pd.DataFrame) -> PathResult:
    return PathResult(
        scores=scores,
        n_train=int(len(train)),
        n_test=int(len(test)),
        features=FEATURES,
    )


def _subset_accuracy(y: np.ndarray, proba: np.ndarray, mask: np.ndarray) -> float:
    predicted = (proba >= 0.5).astype(int)
    return float(accuracy_score(y[mask], predicted[mask]))


def run(seed: int) -> ExperimentCard:
    frame = generate(seed)
    leaky_train, leaky_test = contaminated_split(frame, seed)
    clean_train, clean_test = clean_split(frame, seed)
    leaky_scores, leaky_proba = forest_path(leaky_train, leaky_test, seed)
    clean_scores, _clean_proba = forest_path(clean_train, clean_test, seed)
    leaky_logistic = logistic_scores(leaky_train, leaky_test, seed)
    clean_logistic = logistic_scores(clean_train, clean_test, seed)

    leaky_overlap = overlap_count(leaky_train, leaky_test)
    clean_overlap = overlap_count(clean_train, clean_test)
    twin = leaky_test["row_id"].isin(set(leaky_train["row_id"].tolist())).to_numpy()
    y_test = leaky_test[TARGET].to_numpy()
    twin_accuracy = _subset_accuracy(y_test, leaky_proba, twin)
    novel_accuracy = _subset_accuracy(y_test, leaky_proba, ~twin)

    leaky = _path(leaky_scores, leaky_train, leaky_test)
    clean = _path(clean_scores, clean_train, clean_test)
    if clean_overlap == 0:
        clean_overlap_text = "The clean split shares none"
    else:
        clean_overlap_text = f"The clean split shares {clean_overlap} row ids"
    measured = (
        f"{leaky_overlap} of {leaky.n_test} leaky test rows share a row_id with "
        f"training. The forest's accuracy on those copied rows is {twin_accuracy:.4f}. "
        f"On the leaky test rows that are not copies, accuracy is {novel_accuracy:.4f}. "
        f"{clean_overlap_text}; forest ROC-AUC there is {clean.scores.roc_auc:.4f}, "
        f"against {leaky.scores.roc_auc:.4f} on the contaminated split. "
        f"Logistic regression is {leaky_logistic.roc_auc:.4f} contaminated and "
        f"{clean_logistic.roc_auc:.4f} clean."
    )
    return ExperimentCard(
        slug="copied-rows",
        number=2,
        title="Copied rows",
        kicker="Train/test contamination from duplicate rows",
        mechanism=MECHANISM,
        measured=measured,
        seed=seed,
        population=(
            f"{N_ROWS:,} synthetic claims; the leaky path stacks a second copy "
            "before the split"
        ),
        positive_rate=float(frame[TARGET].mean()),
        leaky=leaky,
        clean=clean,
        facts=(
            ("Leaky test rows also in train", f"{leaky_overlap} of {leaky.n_test}"),
            ("Clean test rows also in train", f"{clean_overlap} of {clean.n_test}"),
            ("Forest accuracy on copied leaky test rows", f"{twin_accuracy:.4f}"),
            ("Forest accuracy on new leaky test rows", f"{novel_accuracy:.4f}"),
            ("Logistic ROC-AUC, contaminated split", f"{leaky_logistic.roc_auc:.4f}"),
            ("Logistic ROC-AUC, clean split", f"{clean_logistic.roc_auc:.4f}"),
            ("Positive rate", f"{float(frame[TARGET].mean()):.4f}"),
        ),
    )
