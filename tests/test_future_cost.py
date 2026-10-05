"""The future-cost card: one column from after the outcome, otherwise the same fit."""

from __future__ import annotations

import pytest

from leakage_lab.experiments import future_cost

SEEDS = (42, 0, 7)


def test_generate_is_deterministic():
    first = future_cost.generate(42)
    second = future_cost.generate(42)
    assert first.equals(second)
    assert not first.equals(future_cost.generate(7))


@pytest.mark.parametrize("seed", SEEDS)
def test_leaky_path_is_suspiciously_good(seed: int):
    card = future_cost.run(seed)
    assert card.leaky.scores.roc_auc >= 0.96
    assert card.roc_auc_gap >= 0.20


@pytest.mark.parametrize("seed", SEEDS)
def test_clean_path_is_honest(seed: int):
    card = future_cost.run(seed)
    # Discharge-time fields support a real model, and they top out well below
    # the leaky score on every seed checked here.
    assert 0.70 <= card.clean.scores.roc_auc <= 0.80
    correlation = float(dict(card.facts)["Test-set correlation of future cost with the label"])
    assert correlation >= 0.70


def test_only_the_future_column_changes():
    frame = future_cost.generate(42)
    y = frame[future_cost.TARGET].to_numpy()
    train_idx, test_idx = future_cost.split_indices(y, 42)
    leaky = future_cost.evaluate(
        frame, future_cost.LEAKY_FEATURES, train_idx, test_idx, 42
    )
    clean = future_cost.evaluate(
        frame, future_cost.DISCHARGE_FEATURES, train_idx, test_idx, 42
    )
    card = future_cost.run(42)

    assert leaky.scores == card.leaky.scores
    assert clean.scores == card.clean.scores
    assert future_cost.FUTURE_FEATURE in card.leaky.features
    assert future_cost.FUTURE_FEATURE not in card.clean.features
    assert card.leaky.features[:-1] == card.clean.features
    assert future_cost.TARGET not in card.leaky.features
    assert card.leaky.n_test == card.clean.n_test
