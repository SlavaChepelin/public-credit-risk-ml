"""Тесты метрик и time-based сплита."""

import numpy as np
import pytest

from credit_risk.evaluation.metrics import classification_report_scores, gini_from_auc, ks_statistic
from credit_risk.validation.split import time_based_split
from credit_risk.data.synthetic import generate_synthetic_credit


def test_gini_auc_relation() -> None:
    """Gini равен 2 * AUC - 1."""
    assert abs(gini_from_auc(0.7) - 0.4) < 1e-9


def test_ks_is_non_negative() -> None:
    """KS неотрицателен на отделимых скорах."""
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert ks_statistic(y, s) >= 0.0


def test_metrics_shape() -> None:
    """classification_report_scores возвращает ожидаемые ключи."""
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    metrics = classification_report_scores(y, s)
    assert {"roc_auc", "gini", "ks", "pr_auc", "brier", "ece"} <= set(metrics)
    assert 0.5 <= metrics["roc_auc"] <= 1.0


def test_time_split_is_chronological() -> None:
    """Сплит сохраняет хронологический порядок и полное покрытие."""
    df = generate_synthetic_credit(n_samples=1000, seed=0)
    train, valid, test = time_based_split(df)
    assert train["application_date"].max() <= valid["application_date"].min()
    assert valid["application_date"].max() <= test["application_date"].min()
    assert len(train) + len(valid) + len(test) == len(df)


def test_time_split_rejects_bad_ratios() -> None:
    """Некорректные доли сплита поднимают ValueError."""
    df = generate_synthetic_credit(n_samples=100, seed=0)
    with pytest.raises(ValueError):
        time_based_split(df, train_ratio=0.8, valid_ratio=0.3)
