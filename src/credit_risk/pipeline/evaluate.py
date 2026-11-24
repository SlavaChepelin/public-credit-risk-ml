"""Оценка моделей, выбор чемпиона и temporal OOF."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credit_risk.evaluation.metrics import classification_report_scores
from credit_risk.models import fit_boosting
from credit_risk.pipeline.dataset import DatasetName, xy_from_split
from credit_risk.validation.leakage import check_split_leakage
from credit_risk.validation.split import temporal_folds


def predict_scores(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Считает метрики модели на заданной выборке.

    Args:
        model: Обученная модель с predict_proba или decision_function.
        X: Матрица признаков.
        y: Истинные метки.

    Returns:
        Словарь банковских метрик.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.decision_function(X)
    return classification_report_scores(y.to_numpy(), proba)


def evaluate_models(
    models: dict[str, Any],
    valid: tuple[pd.DataFrame, pd.Series],
    test: tuple[pd.DataFrame, pd.Series],
) -> tuple[dict[str, dict[str, dict[str, float]]], str | None]:
    """Оценивает модели на valid/test и выбирает чемпиона по valid Gini.

    Test метрики считаются для отчёта, но не участвуют в выборе модели
    (иначе holdout подглядывает в selection).

    Args:
        models: Словарь обученных моделей.
        valid: Валидационная выборка (X, y).
        test: Тестовая выборка (X, y).

    Returns:
        Кортеж (метрики по моделям, имя лучшей модели по valid Gini).
    """
    X_valid, y_valid = valid
    X_test, y_test = test
    report_models: dict[str, dict[str, dict[str, float]]] = {}
    best_name: str | None = None
    best_gini = -1.0

    for name, model in models.items():
        metrics = {
            "valid": predict_scores(model, X_valid, y_valid),
            "test": predict_scores(model, X_test, y_test),
        }
        report_models[name] = metrics
        gini = metrics["valid"]["gini"]
        if gini > best_gini:
            best_gini, best_name = gini, name

    return report_models, best_name


def run_oof(
    df: pd.DataFrame,
    *,
    dataset: DatasetName,
    target_col: str,
    time_col: str,
    n_folds: int,
    torch_epochs: int,
) -> dict[str, Any]:
    """Считает mean±std метрик бустинга по temporal folds.

    Args:
        df: Полный датафрейм.
        dataset: Тип датасета.
        target_col: Имя таргета.
        time_col: Колонка времени.
        n_folds: Число фолдов.
        torch_epochs: Не используется (OOF только boosting для скорости).

    Returns:
        Блок oof для metrics.json.
    """
    del torch_epochs
    folds = temporal_folds(df, time_col=time_col, n_folds=n_folds)
    fold_metrics: list[dict[str, float]] = []
    for train_df, valid_df in folds:
        check_split_leakage(train_df, valid_df, time_col=time_col)
        X_tr, y_tr = xy_from_split(train_df, dataset, target_col)
        X_va, y_va = xy_from_split(valid_df, dataset, target_col)
        model = fit_boosting(X_tr, y_tr)
        fold_metrics.append(predict_scores(model, X_va, y_va))

    keys = fold_metrics[0].keys()
    summary = {
        key: {
            "mean": float(np.mean([m[key] for m in fold_metrics])),
            "std": float(np.std([m[key] for m in fold_metrics])),
        }
        for key in keys
    }
    return {
        "n_folds": n_folds,
        "model": "boosting",
        "folds": fold_metrics,
        "summary": summary,
    }
