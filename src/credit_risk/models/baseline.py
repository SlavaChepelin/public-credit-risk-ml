"""Логистическая регрессия как baseline."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_risk.features.synthetic import build_preprocessor_for_frame


def fit_logistic(X, y) -> Pipeline:
    """Обучает логистическую регрессию с препроцессором.

    Args:
        X: Матрица признаков.
        y: Бинарный таргет.

    Returns:
        Обученный sklearn Pipeline.
    """
    pipe = Pipeline(
        steps=[
            ("pre", build_preprocessor_for_frame(X)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe
