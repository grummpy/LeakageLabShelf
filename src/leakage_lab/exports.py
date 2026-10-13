"""Optional, dependency-free CSV fixtures that another local audit tool can read."""

from __future__ import annotations

import json
from pathlib import Path

from leakage_lab.experiments import copied_rows, group_encoding


def write_split_audit_exports(destination: Path, seed: int) -> tuple[Path, ...]:
    """Write synthetic train/test CSVs only; never import or require SplitCheck.

    The copied-row pair intentionally leaks exact rows.  The group-encoding
    pair intentionally shares repeated entity ids across a random row split.
    Both are plain CSV evidence fixtures suitable for any split-audit tool.
    """

    destination.mkdir(parents=True, exist_ok=True)
    copied = copied_rows.generate(seed)
    copied_train, copied_test = copied_rows.contaminated_split(copied, seed)
    entity = group_encoding.generate(seed)
    train_idx, test_idx = group_encoding.split_indices(entity, seed, grouped=False)
    paths = (
        destination / "copied_rows_train.csv",
        destination / "copied_rows_test.csv",
        destination / "repeated_entity_train.csv",
        destination / "repeated_entity_test.csv",
    )
    copied_train.to_csv(paths[0], index=False)
    copied_test.to_csv(paths[1], index=False)
    entity.iloc[train_idx].to_csv(paths[2], index=False)
    entity.iloc[test_idx].to_csv(paths[3], index=False)
    (destination / "README.json").write_text(
        json.dumps(
            {
                "version": 1,
                "seed": seed,
                "synthetic_only": True,
                "dependency": "none; these are ordinary CSV files",
                "pairs": {
                    "copied_rows": {"train": paths[0].name, "test": paths[1].name, "group": "row_id"},
                    "repeated_entities": {"train": paths[2].name, "test": paths[3].name, "group": "member_id"},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
