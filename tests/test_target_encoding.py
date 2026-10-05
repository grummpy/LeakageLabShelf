"""TargetEncoder fit on every row copies a unique id's label into the matrix."""

from __future__ import annotations

import numpy as np
import pytest

from leakage_lab.experiments import target_encoding
from leakage_lab.metrics import fit_logistic

SEEDS = (42, 0, 7)


def test_ids_are_unique():
    frame = target_encoding.generate(42)
    assert frame[target_encoding.ID_FEATURE].is_unique
    assert frame.equals(target_encoding.generate(42))


@pytest.mark.parametrize("seed", SEEDS)
def test_leaky_encoding_is_the_label(seed: int):
    frame = target_encoding.generate(seed)
    y = frame[target_encoding.TARGET].to_numpy()
    encoded = target_encoding.leaky_encoding(frame, y, seed)
    assert np.array_equal(encoded, y.astype(float))
    card = target_encoding.run(seed)
    assert card.leaky.scores.roc_auc == pytest.approx(1.0)
    assert card.leaky.scores.accuracy == pytest.approx(1.0)
    assert target_encoding.ENCODED_FEATURE in card.leaky.features
    assert target_encoding.ID_FEATURE not in card.leaky.features


@pytest.mark.parametrize("seed", SEEDS)
def test_clean_path_is_honest(seed: int):
    card = target_encoding.run(seed)
    facts = dict(card.facts)
    assert facts["Leaky encoding equals the label on every row"] == "True"
    assert facts["Clean test-encoding standard deviation"] == "0.0000"
    assert abs(float(facts["Correlation of fit_transform(full table) with the label"])) < 0.05
    # Intake features support a real model. The unseen id adds a constant.
    assert 0.70 <= card.clean.scores.roc_auc <= 0.85
    assert card.roc_auc_gap >= 0.20

    frame = target_encoding.generate(seed)
    y = frame[target_encoding.TARGET].to_numpy()
    train_idx, test_idx = target_encoding.split_indices(y, seed)
    clinical = fit_logistic(
        frame.iloc[train_idx][list(target_encoding.CLINICAL_FEATURES)],
        y[train_idx],
        frame.iloc[test_idx][list(target_encoding.CLINICAL_FEATURES)],
        y[test_idx],
        seed,
    )
    assert card.clean.scores.roc_auc == pytest.approx(clinical.roc_auc, abs=1e-3)


def test_fit_transform_does_not_paste_the_label():
    frame = target_encoding.generate(42)
    y = frame[target_encoding.TARGET].to_numpy()
    crossed = target_encoding.cross_fit_encoding(frame, y, 42)
    assert not np.array_equal(crossed, y.astype(float))
