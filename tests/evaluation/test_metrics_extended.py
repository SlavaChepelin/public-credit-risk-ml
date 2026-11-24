"""Тесты расширенных метрик: ECE, Brier, operating point."""

import numpy as np

from credit_risk.evaluation.metrics import (
    classification_report_scores,
    expected_calibration_error,
    operating_point_metrics,
)


def test_classification_report_includes_brier_ece() -> None:
    """Базовый отчёт содержит brier и ece."""
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    m = classification_report_scores(y, s)
    assert {"brier", "ece"} <= set(m)
    assert 0.0 <= m["brier"] <= 1.0
    assert 0.0 <= m["ece"] <= 1.0


def test_ece_perfect_calibration_near_zero() -> None:
    """Идеально калиброванные бины дают маленький ECE."""
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, n_bins=2) < 1e-9


def test_operating_point_metrics_shapes() -> None:
    """Operating point считает precision/recall/approval."""
    y = np.array([0, 0, 1, 1, 0, 1, 0])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.15])
    op = operating_point_metrics(y, s, threshold=0.5)
    assert 0.0 <= op["precision"] <= 1.0
    assert 0.0 <= op["recall"] <= 1.0
    assert 0.0 <= op["approval_rate"] <= 1.0
    assert op["tp"] + op["fp"] + op["tn"] + op["fn"] == len(y)
