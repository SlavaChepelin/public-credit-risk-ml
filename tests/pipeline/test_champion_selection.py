"""Регрессии выбора чемпиона в пайплайне."""

import numpy as np
import pandas as pd

from credit_risk.pipeline import _evaluate_models


def test_champion_selected_by_valid_gini() -> None:
    """Чемпион выбирается по valid Gini, не по test."""

    class _M:
        def __init__(self, by_key: dict[str, list[float]]):
            self._by_key = by_key

        def predict_proba(self, X):
            key = str(X.attrs.get("split"))
            p = np.asarray(self._by_key[key], dtype=float)
            return np.column_stack([1 - p, p])

    y_valid = pd.Series([0, 0, 1, 1, 0, 1])
    y_test = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    X_valid = pd.DataFrame({"a": range(6)})
    X_test = pd.DataFrame({"a": range(8)})
    X_valid.attrs["split"] = "valid"
    X_test.attrs["split"] = "test"

    a = _M(
        {
            "valid": [0.9, 0.8, 0.1, 0.2, 0.85, 0.15],
            "test": [0.05, 0.95, 0.05, 0.95, 0.05, 0.95, 0.05, 0.95],
        }
    )
    b = _M(
        {
            "valid": [0.05, 0.1, 0.9, 0.95, 0.15, 0.85],
            "test": [0.4, 0.45, 0.5, 0.55, 0.42, 0.48, 0.43, 0.52],
        }
    )
    _, best = _evaluate_models(
        {"a": a, "b": b},
        (X_valid, y_valid),
        (X_test, y_test),
    )
    assert best == "b"
