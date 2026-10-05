"""Fit the small models the shelf compares, and score the held-out rows."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from leakage_lab.cards import Scores


def scores_from_proba(y_true: np.ndarray, proba: np.ndarray) -> Scores:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    predicted = (proba >= 0.5).astype(int)
    return Scores(
        roc_auc=float(roc_auc_score(y_true, proba)),
        accuracy=float(accuracy_score(y_true, predicted)),
        average_precision=float(average_precision_score(y_true, proba)),
    )


def positive_proba(model, features) -> np.ndarray:
    proba = model.predict_proba(features)
    class_index = list(model.classes_).index(1)
    return proba[:, class_index]


def logistic_regression(seed: int) -> Pipeline:
    """Scaler fit on the rows the pipeline is fit on, then logistic regression."""

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )


def fit_logistic(X_train, y_train, X_test, y_test, seed: int) -> Scores:
    model = logistic_regression(seed)
    # A perfectly separating column drives coefficients outward. The ranking
    # is already decided; the warning is about the coefficient size.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)
    return scores_from_proba(y_test, positive_proba(model, X_test))


def random_forest(seed: int) -> RandomForestClassifier:
    """A forest deep enough to memorize a training row it sees again.

    ``n_jobs`` is 1 so the trees are built in a fixed order.
    """

    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        n_jobs=1,
    )


def fit_forest(X_train, y_train, X_test, y_test, seed: int) -> tuple[Scores, np.ndarray]:
    model = random_forest(seed)
    model.fit(X_train, y_train)
    proba = positive_proba(model, X_test)
    return scores_from_proba(y_test, proba), proba
