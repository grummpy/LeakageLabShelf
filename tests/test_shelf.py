"""The shelf command, documentation, and same-runtime reproducibility."""

from __future__ import annotations

from pathlib import Path

from leakage_lab import DEFAULT_SEED
from leakage_lab.__main__ import main
from leakage_lab.catalog import run_shelf

ROOT = Path(__file__).resolve().parents[1]


def test_default_seed_and_order():
    assert DEFAULT_SEED == 42
    cards = run_shelf(42)
    assert tuple(card.slug for card in cards) == (
        "future-cost",
        "copied-rows",
        "matched-holdout-copies",
        "target-encoding",
        "group-aware-encoding",
    )
    assert tuple(card.number for card in cards) == (1, 2, 3, 4, 5)


def test_same_seed_repeats():
    assert run_shelf(42) == run_shelf(42)


def test_cli_prints_the_three_cards(capsys):
    assert main(["shelf", "--seed", "42"]) == 0
    out = capsys.readouterr().out
    assert "Future cost" in out
    assert "Copied rows" in out
    assert "Target encoding" in out
    assert "0.9712" in out
    assert "seed 42" in out


def test_html_shelf_lists_each_experiment(tmp_path):
    html_path = tmp_path / "shelf.html"
    assert main(["shelf", "--seed", "42", "--html", str(html_path)]) == 0
    text = html_path.read_text(encoding="utf-8")
    for slug in ("future-cost", "copied-rows", "matched-holdout-copies", "target-encoding", "group-aware-encoding"):
        assert f'id="{slug}"' in text
    assert "0.9712" in text
    assert "SplitCheck" in text


def test_committed_html_is_an_illustrative_seed_42_capture():
    committed = (ROOT / "docs" / "shelf.html").read_text(encoding="utf-8")
    assert "Synthetic tables, fixed seed 42." in committed
    assert "class=\"experiment\"" in committed


def test_readme_and_notes_describe_the_experiments_and_evidence_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expected_rows = {
        "future-cost.md": (
            "| [Future cost](docs/experiments/future-cost.md) "
            "| A day-60 cost that includes the readmission "
            "|"
        ),
        "copied-rows.md": (
            "| [Copied rows](docs/experiments/copied-rows.md) "
            "| Every claim pasted twice, then split by row "
            "|"
        ),
        "target-encoding.md": (
            "| [Target encoding](docs/experiments/target-encoding.md) "
            "| `TargetEncoder.fit` on the whole table, then the split "
            "|"
        ),
        "matched-holdout.md": (
            "| [Matched holdout copies](docs/experiments/matched-holdout.md) "
            "| Same holdout, with copies inserted into only one training path "
            "|"
        ),
        "group-aware-encoding.md": (
            "| [Group-aware encoding](docs/experiments/group-aware-encoding.md) "
            "| Repeated members: train-only encoder with random rows versus group holdout "
            "|"
        ),
    }
    for name, row in expected_rows.items():
        assert row in readme
        note = (ROOT / "docs" / "experiments" / name).read_text(encoding="utf-8")
        assert "ROC-AUC" in note

    assert "Logistic ROC-AUC, contaminated split" in readme
    assert "Logistic ROC-AUC, clean split" in readme
    assert "docs/cover.jpg" in readme
    assert "--manifest" in readme
    assert "--splitcheck-export" in readme
    assert "in the same runtime" in readme
    assert "cross-platform benchmark" in readme
    assert (ROOT / "docs" / "cover.jpg").is_file()
