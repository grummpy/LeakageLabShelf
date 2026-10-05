"""The shelf, in reading order."""

from __future__ import annotations

from leakage_lab.cards import ExperimentCard
from leakage_lab.experiments import copied_rows, future_cost, target_encoding

EXPERIMENTS = (future_cost, copied_rows, target_encoding)


def run_shelf(seed: int) -> tuple[ExperimentCard, ...]:
    return tuple(experiment.run(seed) for experiment in EXPERIMENTS)
