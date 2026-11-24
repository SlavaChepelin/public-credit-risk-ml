"""Обучение моделей-кандидатов."""

from __future__ import annotations

from typing import Any

import pandas as pd

from credit_risk.models import fit_boosting, fit_logistic, fit_torch_nn


def fit_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    torch_epochs: int,
) -> dict[str, Any]:
    """Обучает набор моделей-кандидатов.

    Args:
        X_train: Признаки обучения.
        y_train: Таргет обучения.
        torch_epochs: Число эпох для torch_nn.

    Returns:
        Словарь имя → обученная модель.
    """
    return {
        "logistic": fit_logistic(X_train, y_train),
        "boosting": fit_boosting(X_train, y_train),
        "torch_nn": fit_torch_nn(X_train, y_train, epochs=torch_epochs),
    }
