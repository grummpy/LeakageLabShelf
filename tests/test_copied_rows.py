"""Copied claims: the forest memorizes twins, the linear model does not."""

from __future__ import annotations

import pandas as pd
import pytest

from leakage_lab.experiments import copied_rows


def test_overlap_count_matches_shared_ids():
    train = pd.DataFrame({"row_id": [1, 2, 3]})
    test = pd.DataFrame({"row_id": [3, 4, 3]})
    assert copied_rows.overlap_count(train, test) == 2


def test_generate_ids_are_unique_and_stable():
    frame = copied_rows.generate(42)
    assert frame["row_id"].is_unique
    assert frame.equals(copied_rows.generate(42))
    assert copied_rows.TARGET not in copied_rows.FEATURES
    assert "row_id" not in copied_rows.FEATURES


@pytest.mark.parametrize("seed", (42, 7))
def test_leaky_path_is_suspiciously_good(seed: int):
    card = copied_rows.run(seed)
    facts = dict(card.facts)
    overlap, _, n_test = facts["Leaky test rows also in train"].partition(" of ")
    assert int(overlap) / int(n_test) >= 0.60
    assert facts["Forest accuracy on copied leaky test rows"] == "1.0000"
    assert card.leaky.scores.roc_auc >= 0.94
    assert card.roc_auc_gap >= 0.20


@pytest.mark.parametrize("seed", (42, 7))
def test_clean_path_is_honest(seed: int):
    card = copied_rows.run(seed)
    facts = dict(card.facts)
    assert facts["Clean test rows also in train"].startswith("0 of ")
    # The same forest, with no twin to recall, lands in the ordinary range.
    assert 0.65 <= card.clean.scores.roc_auc <= 0.80
    leaky_linear = float(facts["Logistic ROC-AUC, contaminated split"])
    clean_linear = float(facts["Logistic ROC-AUC, clean split"])
    # A model that cannot memorize the twin barely moves when the copies appear.
    assert abs(leaky_linear - clean_linear) <= 0.05
    assert clean_linear >= 0.70


def test_clean_split_has_no_shared_ids():
    frame = copied_rows.generate(42)
    train, test = copied_rows.clean_split(frame, 42)
    assert copied_rows.overlap_count(train, test) == 0
    assert train["row_id"].is_unique
    assert test["row_id"].is_unique
