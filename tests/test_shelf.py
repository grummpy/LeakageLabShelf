"""The shelf command, the written-up scores, and a second run with the same seed."""

from __future__ import annotations

from pathlib import Path

from leakage_lab import DEFAULT_SEED
from leakage_lab.__main__ import main
from leakage_lab.catalog import run_shelf
from leakage_lab.experiments import copied_rows, future_cost, group_encoding, matched_holdout, target_encoding
from leakage_lab.report import render_html

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


def test_committed_html_is_the_current_seed_42_rendering():
    cards = run_shelf(DEFAULT_SEED)
    committed = (ROOT / "docs" / "shelf.html").read_text(encoding="utf-8")
    assert committed == render_html(cards, DEFAULT_SEED, cover_src="cover.jpg")


def test_readme_and_notes_match_seed_42():
    future = future_cost.run(42)
    copied = copied_rows.run(42)
    matched = matched_holdout.run(42)
    encoded = target_encoding.run(42)
    grouped = group_encoding.run(42)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    notes = {
        "future-cost.md": future,
        "copied-rows.md": copied,
        "target-encoding.md": encoded,
    }
    expected_rows = {
        "future-cost.md": (
            "| [Future cost](docs/experiments/future-cost.md) "
            "| A day-60 cost that includes the readmission "
            f"| {future.leaky.scores.roc_auc:.4f} | {future.clean.scores.roc_auc:.4f} |"
        ),
        "copied-rows.md": (
            "| [Copied rows](docs/experiments/copied-rows.md) "
            "| Every claim pasted twice, then split by row "
            f"| {copied.leaky.scores.roc_auc:.4f} | {copied.clean.scores.roc_auc:.4f} |"
        ),
        "target-encoding.md": (
            "| [Target encoding](docs/experiments/target-encoding.md) "
            "| `TargetEncoder.fit` on the whole table, then the split "
            f"| {encoded.leaky.scores.roc_auc:.4f} | {encoded.clean.scores.roc_auc:.4f} |"
        ),
        "matched-holdout.md": (
            "| [Matched holdout copies](docs/experiments/matched-holdout.md) "
            "| Same holdout, with copies inserted into only one training path "
            f"| {matched.leaky.scores.roc_auc:.4f} | {matched.clean.scores.roc_auc:.4f} |"
        ),
        "group-aware-encoding.md": (
            "| [Group-aware encoding](docs/experiments/group-aware-encoding.md) "
            "| Repeated members: train-only encoder with random rows versus group holdout "
            f"| {grouped.leaky.scores.roc_auc:.4f} | {grouped.clean.scores.roc_auc:.4f} |"
        ),
    }
    for name, row in expected_rows.items():
        assert row in readme
        note = (ROOT / "docs" / "experiments" / name).read_text(encoding="utf-8")
        if name in notes:
            card = notes[name]
            assert f"leaky ROC-AUC {card.leaky.scores.roc_auc:.4f}" in note
            assert f"clean ROC-AUC {card.clean.scores.roc_auc:.4f}" in note

    copied_facts = dict(copied.facts)
    assert copied_facts["Logistic ROC-AUC, contaminated split"] in readme
    assert copied_facts["Logistic ROC-AUC, clean split"] in readme
    assert "docs/cover.jpg" in readme
    assert "--manifest" in readme
    assert "--splitcheck-export" in readme
    assert (ROOT / "docs" / "cover.jpg").is_file()
