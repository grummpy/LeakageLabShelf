"""Repeated-entity target encoding: random rows versus a group holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from leakage_lab.cards import ExperimentCard, PathResult
from leakage_lab.metrics import positive_proba, scores_from_proba

CLINICAL_FEATURES = ("age", "systolic", "lab")
ENTITY_COLUMN = "member_id"
TARGET = "high_risk"
N_ENTITIES = 360
VISITS_PER_ENTITY = 3
TEST_SIZE = 0.25

MECHANISM = (
    "Each member has repeated visits and one stable outcome. The leaky path uses a random row split, "
    "but still fits the target encoder on training rows only. That is not enough: another visit from the "
    "same member is often in training, so the encoded member id carries the outcome into test. The clean "
    "path uses a group split by member_id and keeps the same encoder inside the training-only pipeline. "
    "Every clean-test member is unseen, so the encoder falls back to the training prevalence."
)


def generate(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for entity in range(N_ENTITIES):
        risk = float(rng.normal())
        target = int(risk > 0)
        age = 53.0 + 2.5 * risk + float(rng.normal(0, 10))
        systolic = 126.0 + 3.0 * risk + float(rng.normal(0, 17))
        lab = float(rng.normal() + 0.40 * risk)
        identifier = f"member-{entity:04d}"
        for visit in range(VISITS_PER_ENTITY):
            rows.append(
                {
                    "age": age + float(rng.normal(0, 1.5)),
                    "systolic": systolic + float(rng.normal(0, 4.0)),
                    "lab": lab + float(rng.normal(0, 0.4)),
                    ENTITY_COLUMN: identifier,
                    TARGET: target,
                    "visit": visit,
                }
            )
    return pd.DataFrame.from_records(rows)


def split_indices(frame: pd.DataFrame, seed: int, *, grouped: bool) -> tuple[np.ndarray, np.ndarray]:
    y = frame[TARGET].to_numpy()
    if not grouped:
        return tuple(
            np.asarray(part)
            for part in train_test_split(np.arange(len(frame)), test_size=TEST_SIZE, random_state=seed, stratify=y)
        )  # type: ignore[return-value]
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train, test = next(splitter.split(frame, y, groups=frame[ENTITY_COLUMN]))
    return np.asarray(train), np.asarray(test)


def _model(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("clinical", StandardScaler(), list(CLINICAL_FEATURES)),
                        ("member", TargetEncoder(target_type="binary", random_state=seed), [ENTITY_COLUMN]),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )


def evaluate(frame: pd.DataFrame, indices: tuple[np.ndarray, np.ndarray], seed: int) -> tuple[PathResult, float]:
    train_idx, test_idx = indices
    columns = list(CLINICAL_FEATURES + (ENTITY_COLUMN,))
    model = _model(seed)
    model.fit(frame.iloc[train_idx][columns], frame.iloc[train_idx][TARGET])
    proba = positive_proba(model, frame.iloc[test_idx][columns])
    encoder = model.named_steps["prep"].named_transformers_["member"]
    encoded_test = np.asarray(encoder.transform(frame.iloc[test_idx][[ENTITY_COLUMN]])).ravel()
    path = PathResult(
        scores=scores_from_proba(frame.iloc[test_idx][TARGET].to_numpy(), proba),
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        features=CLINICAL_FEATURES + (ENTITY_COLUMN,),
    )
    return path, float(np.std(encoded_test))


def run(seed: int) -> ExperimentCard:
    frame = generate(seed)
    random = split_indices(frame, seed, grouped=False)
    grouped = split_indices(frame, seed, grouped=True)
    leaky, leaky_std = evaluate(frame, random, seed)
    clean, clean_std = evaluate(frame, grouped, seed)
    random_shared = set(frame.iloc[random[0]][ENTITY_COLUMN]) & set(frame.iloc[random[1]][ENTITY_COLUMN])
    grouped_shared = set(frame.iloc[grouped[0]][ENTITY_COLUMN]) & set(frame.iloc[grouped[1]][ENTITY_COLUMN])
    measured = (
        f"The random split shares {len(random_shared)} entities across train and test; its training-only encoder "
        f"still reaches ROC-AUC {leaky.scores.roc_auc:.4f}. The group split shares {len(grouped_shared)} entities, "
        f"fits the same pipeline on training rows only, and reaches {clean.scores.roc_auc:.4f}. The clean test "
        f"encoding has standard deviation {clean_std:.4f}, because every member id is unseen."
    )
    return ExperimentCard(
        slug="group-aware-encoding",
        number=5,
        title="Group-aware repeated-entity encoding",
        kicker="Target encoding needs a group split, not just train-only fit",
        mechanism=MECHANISM,
        measured=measured,
        seed=seed,
        population=f"{N_ENTITIES:,} synthetic members, {VISITS_PER_ENTITY} visits each",
        positive_rate=float(frame[TARGET].mean()),
        leaky=leaky,
        clean=clean,
        facts=(
            ("Entities shared by random split", str(len(random_shared))),
            ("Entities shared by group split", str(len(grouped_shared))),
            ("Random test-encoding standard deviation", f"{leaky_std:.4f}"),
            ("Group-holdout test-encoding standard deviation", f"{clean_std:.4f}"),
            ("Fit boundary", "TargetEncoder remains inside each training-only pipeline"),
        ),
    )
