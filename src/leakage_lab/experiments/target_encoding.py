"""Target encoding fit on the full table before the split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from leakage_lab.cards import ExperimentCard, PathResult
from leakage_lab.metrics import fit_logistic, positive_proba, scores_from_proba

CLINICAL_FEATURES = ("age", "systolic", "lab")
ID_FEATURE = "member_id"
ENCODED_FEATURE = "member_target_mean"
TARGET = "high_risk"
TEST_SIZE = 0.25
N_ROWS = 2000

MECHANISM = (
    "member_id is a different string on every row. It identifies that row and "
    "nothing else. The leaky pipeline fits sklearn's TargetEncoder "
    "(smooth='auto', target_type='binary') on the whole table and only then "
    "splits. For a category that appears once, the value stored by fit and "
    "read back by transform is that row's label, so the test matrix already "
    "contains the answer. The clean pipeline splits first. The encoder sits "
    "inside a Pipeline and is fit on the training rows only. Every test id is "
    "unseen, and TargetEncoder fills unseen categories with the training "
    "prevalence, a constant on the test set. fit_transform is a different "
    "call: it cross-fits, so a row does not receive an encoding built from "
    "its own label. Cross-fitting the full table is still the wrong order "
    "when a category appears on both sides of a later split. It is not the "
    "call the leaky path makes, and on these unique ids it does not paste "
    "the label back."
)


def _encoder(seed: int) -> TargetEncoder:
    return TargetEncoder(smooth="auto", target_type="binary", random_state=seed)


def generate(seed: int) -> pd.DataFrame:
    """Synthetic intake rows. Each member_id appears once."""

    rng = np.random.default_rng(seed)
    age = rng.normal(54, 15, N_ROWS)
    systolic = rng.normal(128, 18, N_ROWS)
    lab = rng.normal(size=N_ROWS)
    logit = -0.25 + 0.018 * (age - 54) + 0.01 * (systolic - 128) + 1.0 * lab
    probability = 1.0 / (1.0 + np.exp(-logit))
    high_risk = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "age": age,
            "systolic": systolic,
            "lab": lab,
            ID_FEATURE: [f"m{i:05d}" for i in range(N_ROWS)],
            TARGET: high_risk,
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


def leaky_encoding(frame: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    """Fit the encoder on every row, then transform those same rows."""

    encoder = _encoder(seed)
    encoder.fit(frame[[ID_FEATURE]], y)
    return np.asarray(encoder.transform(frame[[ID_FEATURE]])).ravel()


def cross_fit_encoding(frame: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    """fit_transform cross-fits. On unique ids this is not the label."""

    encoded = _encoder(seed).fit_transform(frame[[ID_FEATURE]], y)
    return np.asarray(encoded).ravel()


def _leaky_frame(frame: pd.DataFrame, encoded: np.ndarray) -> pd.DataFrame:
    design = frame.loc[:, list(CLINICAL_FEATURES)].copy()
    design[ENCODED_FEATURE] = encoded
    return design


def run(seed: int) -> ExperimentCard:
    frame = generate(seed)
    y = frame[TARGET].to_numpy()
    train_idx, test_idx = split_indices(y, seed)
    encoded = leaky_encoding(frame, y, seed)
    design = _leaky_frame(frame, encoded)
    leaky_features = CLINICAL_FEATURES + (ENCODED_FEATURE,)
    leaky_scores = fit_logistic(
        design.iloc[train_idx][list(leaky_features)],
        y[train_idx],
        design.iloc[test_idx][list(leaky_features)],
        y[test_idx],
        seed,
    )
    leaky = PathResult(
        scores=leaky_scores,
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        features=leaky_features,
    )

    clean_columns = list(CLINICAL_FEATURES + (ID_FEATURE,))
    preprocess = ColumnTransformer(
        transformers=[
            ("clinical", StandardScaler(), list(CLINICAL_FEATURES)),
            ("member", _encoder(seed), [ID_FEATURE]),
        ]
    )
    clean_model = Pipeline(
        [
            ("prep", preprocess),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    clean_model.fit(frame.iloc[train_idx][clean_columns], y[train_idx])
    clean_proba = positive_proba(clean_model, frame.iloc[test_idx][clean_columns])
    clean = PathResult(
        scores=scores_from_proba(y[test_idx], clean_proba),
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        features=CLINICAL_FEATURES + (ID_FEATURE,),
    )
    member_encoder = clean_model.named_steps["prep"].named_transformers_["member"]
    clean_test_encoded = np.asarray(
        member_encoder.transform(frame.iloc[test_idx][[ID_FEATURE]])
    ).ravel()
    clean_std = float(np.std(clean_test_encoded))
    crossed = cross_fit_encoding(frame, y, seed)
    crossed_correlation = float(np.corrcoef(crossed, y.astype(float))[0, 1])
    label_match = bool(np.array_equal(encoded, y.astype(float)))

    measured = (
        f"The encoding from fit on the full table matches the label on every row "
        f"({label_match}). Leaky ROC-AUC is {leaky.scores.roc_auc:.4f}. "
        f"The clean encoder is fit on {clean.n_train} training rows, and its "
        f"encoding of the {clean.n_test} test ids has standard deviation "
        f"{clean_std:.4f}. Clean ROC-AUC is {clean.scores.roc_auc:.4f}. "
        f"fit_transform on the same full table correlates with the label at "
        f"{crossed_correlation:.4f}."
    )
    return ExperimentCard(
        slug="target-encoding",
        number=3,
        title="Target encoding",
        kicker="Preprocessing fit on the full table before the split",
        mechanism=MECHANISM,
        measured=measured,
        seed=seed,
        population=f"{N_ROWS:,} synthetic intake rows, one row per member id",
        positive_rate=float(y.mean()),
        leaky=leaky,
        clean=clean,
        facts=(
            ("Leaky encoding equals the label on every row", str(label_match)),
            ("Clean test-encoding standard deviation", f"{clean_std:.4f}"),
            (
                "Correlation of fit_transform(full table) with the label",
                f"{crossed_correlation:.4f}",
            ),
            ("Test rows, shared by both paths", str(clean.n_test)),
            ("Positive rate", f"{float(y.mean()):.4f}"),
        ),
    )
