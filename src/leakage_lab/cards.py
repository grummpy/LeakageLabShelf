"""Shared result types for one shelf card."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scores:
    """Held-out scores for one fitted path.

    Accuracy uses a probability cutoff of 0.5. ROC-AUC and average precision
    use the predicted probability of the positive class.
    """

    roc_auc: float
    accuracy: float
    average_precision: float


@dataclass(frozen=True)
class PathResult:
    """One pipeline: the columns it saw, the rows it used, and its scores."""

    scores: Scores
    n_train: int
    n_test: int
    features: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentCard:
    """A leaky path, the clean control, and the text that says what moved."""

    slug: str
    number: int
    title: str
    kicker: str
    mechanism: str
    measured: str
    seed: int
    population: str
    positive_rate: float
    leaky: PathResult
    clean: PathResult
    facts: tuple[tuple[str, str], ...]

    @property
    def roc_auc_gap(self) -> float:
        return self.leaky.scores.roc_auc - self.clean.scores.roc_auc
