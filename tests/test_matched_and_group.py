"""The new lessons keep their comparisons and train-only fit boundaries explicit."""

import json
from pathlib import Path

from leakage_lab.__main__ import main
from leakage_lab.exports import write_split_audit_exports
from leakage_lab.experiments import group_encoding, matched_holdout


def test_matched_holdout_uses_one_test_population_and_only_one_path_has_copies():
    leaky_train, clean_train, holdout, copied = matched_holdout.matched_split(42)
    assert len(leaky_train) == len(clean_train)
    assert copied.sum() == len(holdout) // 2
    assert set(holdout["row_id"]) & set(clean_train["row_id"]) == set()
    assert set(holdout.loc[copied, "row_id"]).issubset(set(leaky_train["row_id"]))
    card = matched_holdout.run(42)
    assert card.leaky.n_test == card.clean.n_test == len(holdout)
    assert card.roc_auc_gap >= 0.15


def test_group_aware_lesson_splits_entities_before_the_clean_fit():
    frame = group_encoding.generate(42)
    random = group_encoding.split_indices(frame, 42, grouped=False)
    grouped = group_encoding.split_indices(frame, 42, grouped=True)
    random_shared = set(frame.iloc[random[0]]["member_id"]) & set(frame.iloc[random[1]]["member_id"])
    grouped_shared = set(frame.iloc[grouped[0]]["member_id"]) & set(frame.iloc[grouped[1]]["member_id"])
    assert random_shared
    assert not grouped_shared
    card = group_encoding.run(42)
    facts = dict(card.facts)
    assert card.roc_auc_gap >= 0.15
    assert facts["Group-holdout test-encoding standard deviation"] == "0.0000"


def test_manifest_summary_and_csv_exports_are_optional_and_standalone(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    export_dir = tmp_path / "exports"
    assert main(["shelf", "--seed", "42", "--manifest", str(manifest), "--splitcheck-export", str(export_dir)]) == 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["manifest_version"] == 2
    assert payload["seed"] == 42
    assert len(payload["source"]["revision"] or "") == 40
    assert payload["dependency_pins"]["scikit-learn"] == "1.7.2"
    assert payload["constraints"]["network"] == "not used"
    assert len(payload["experiments"]) == 5
    assert (export_dir / "copied_rows_train.csv").is_file()
    assert (export_dir / "repeated_entity_test.csv").is_file()
    note = json.loads((export_dir / "README.json").read_text(encoding="utf-8"))
    assert note["dependency"].startswith("none")
    source = Path(write_split_audit_exports.__code__.co_filename).read_text(encoding="utf-8").lower()
    assert "import splitcheck" not in source


def test_predeclared_multi_seed_summary_is_available_without_seed_selection(capsys):
    assert main(["shelf", "--seed", "42", "--multi-seed-summary"]) == 0
    out = capsys.readouterr().out
    assert "Predeclared multi-seed summary" in out
    assert "Seeds: 0, 7, 42" in out
    assert "Paired leaky-minus-clean gap" in out
    assert "not a confidence interval" in out


def test_multi_seed_manifest_retains_each_paired_result(tmp_path):
    manifest = tmp_path / "manifest.json"
    assert main(["shelf", "--manifest", str(manifest), "--multi-seed-summary"]) == 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    summary = payload["predeclared_multi_seed"]
    assert summary["seeds"] == [0, 7, 42]
    assert len(summary["paired_roc_auc_results"]) == 3
    first = summary["paired_roc_auc_results"][0]
    assert first["seed"] == 0
    assert len(first["experiments"]) == 5
    for experiment in first["experiments"]:
        assert experiment["paired_roc_auc_gap"] == experiment["leaky_roc_auc"] - experiment["clean_roc_auc"]
