"""Target leakage from a cost that is only known after the outcome."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from leakage_lab.cards import ExperimentCard, PathResult
from leakage_lab.metrics import fit_logistic

DISCHARGE_FEATURES = (
    "age",
    "prior_admissions",
    "length_of_stay",
    "discharge_lab",
)
FUTURE_FEATURE = "cost_through_day_60"
LEAKY_FEATURES = DISCHARGE_FEATURES + (FUTURE_FEATURE,)
TARGET = "readmitted_30d"
TEST_SIZE = 0.25
N_ROWS = 2400

MECHANISM = (
    "Readmission is decided inside the 30 days after discharge. "
    "cost_through_day_60 keeps accumulating through that window. "
    "A readmission adds a charge of 3200 on top of a smaller length-of-stay "
    "term and Gaussian noise, so the column is a soft copy of the label. "
    "Length of stay is already available at discharge and is in both models; "
    "the future column is the only thing that changes. Both paths are the "
    "same logistic regression, and the scaler is fit on the training rows only."
)


def generate(seed: int) -> pd.DataFrame:
    """Synthetic discharges. The future cost is a planted function of the label."""

    rng = np.random.default_rng(seed)
    age = rng.normal(62, 13, N_ROWS)
    prior_admissions = rng.poisson(1.2, N_ROWS).astype(float)
    length_of_stay = rng.gamma(2.0, 1.8, N_ROWS)
    discharge_lab = rng.normal(0.0, 1.0, N_ROWS)
    logit = (
        -0.8
        + 0.03 * (age - 62)
        + 0.22 * prior_admissions
        + 0.05 * length_of_stay
        + 0.95 * discharge_lab
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    readmitted = rng.binomial(1, probability)
    cost_through_day_60 = (
        1800
        + 90 * length_of_stay
        + 3200 * readmitted
        + rng.normal(0.0, 1300, N_ROWS)
    )
    return pd.DataFrame(
        {
            "age": age,
            "prior_admissions": prior_admissions,
            "length_of_stay": length_of_stay,
            "discharge_lab": discharge_lab,
            FUTURE_FEATURE: cost_through_day_60,
            TARGET: readmitted,
        }
    )


def split_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        index,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )
    return train_idx, test_idx


def evaluate(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> PathResult:
    y = frame[TARGET].to_numpy()
    columns = list(features)
    scores = fit_logistic(
        frame.iloc[train_idx][columns],
        y[train_idx],
        frame.iloc[test_idx][columns],
        y[test_idx],
        seed,
    )
    return PathResult(
        scores=scores,
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        features=tuple(features),
    )


def run(seed: int) -> ExperimentCard:
    frame = generate(seed)
    y = frame[TARGET].to_numpy()
    train_idx, test_idx = split_indices(y, seed)
    leaky = evaluate(frame, LEAKY_FEATURES, train_idx, test_idx, seed)
    clean = evaluate(frame, DISCHARGE_FEATURES, train_idx, test_idx, seed)
    correlation = float(
        np.corrcoef(frame.iloc[test_idx][FUTURE_FEATURE].to_numpy(), y[test_idx])[0, 1]
    )
    gap = leaky.scores.roc_auc - clean.scores.roc_auc
    measured = (
        f"On seed {seed} the leaky ROC-AUC is {leaky.scores.roc_auc:.4f} "
        f"and the clean ROC-AUC is {clean.scores.roc_auc:.4f} "
        f"(gap {gap:.4f}) on the same {clean.n_test} test rows. "
        f"Accuracy at a probability cutoff of 0.5 is {leaky.scores.accuracy:.4f} "
        f"leaky and {clean.scores.accuracy:.4f} clean. "
        f"On the test rows, {FUTURE_FEATURE} correlates with the label at "
        f"{correlation:.4f}."
    )
    return ExperimentCard(
        slug="future-cost",
        number=1,
        title="Future cost",
        kicker="Target leakage via a feature from the future",
        mechanism=MECHANISM,
        measured=measured,
        seed=seed,
        population=f"{N_ROWS:,} synthetic discharges",
        positive_rate=float(y.mean()),
        leaky=leaky,
        clean=clean,
        facts=(
            ("Test rows, shared by both paths", str(clean.n_test)),
            ("Test-set correlation of future cost with the label", f"{correlation:.4f}"),
            ("Positive rate", f"{float(y.mean()):.4f}"),
        ),
    )
