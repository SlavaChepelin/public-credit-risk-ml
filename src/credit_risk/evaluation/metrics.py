"""Банковские метрики бинарной классификации."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def gini_from_auc(auc: float) -> float:
    """Считает Gini по ROC-AUC.

    Используется стандартное определение кредитного скоринга: Gini = 2 * AUC - 1.

    Args:
        auc: Значение ROC-AUC.

    Returns:
        Коэффициент Gini.
    """
    return 2.0 * auc - 1.0


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Считает статистику Колмогорова–Смирнова по скорам.

    Args:
        y_true: Бинарные метки классов.
        y_score: Предсказанные скоры / вероятности.

    Returns:
        Максимальная разница между CDF «хороших» и «плохих».
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Считает Expected Calibration Error (ECE).

    Args:
        y_true: Бинарные метки.
        y_proba: Калиброванные/сырые вероятности класса 1.
        n_bins: Число равномерных бинов по вероятности.

    Returns:
        Взвешенная по размеру бина абсолютная разница между
        средней вероятностью и долей положительных в бине.
    """
    y_true = np.asarray(y_true).astype(float)
    y_proba = np.asarray(y_proba).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)
        if not np.any(mask):
            continue
        conf = float(y_proba[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def classification_report_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Собирает набор банковских метрик классификации.

    Args:
        y_true: Бинарные метки классов.
        y_score: Предсказанные скоры / вероятности класса 1.

    Returns:
        Словарь с roc_auc, gini, ks, pr_auc, brier и ece.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    auc = float(roc_auc_score(y_true, y_score))
    return {
        "roc_auc": auc,
        "gini": gini_from_auc(auc),
        "ks": ks_statistic(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "ece": expected_calibration_error(y_true, y_score),
    }


def operating_point_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Считает precision, recall, FPR и approval rate на фиксированном пороге.

    Решение «predicted default» принимается при score >= threshold.
    Approval rate — доля заявок со score < threshold.

    Args:
        y_true: Истинные метки.
        y_score: Скоры или вероятности класса дефолта.
        threshold: Порог отсечения по score.

    Returns:
        Словарь операционных метрик, включая confusion matrix counts.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    approval_rate = float((pred == 0).mean())
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
        "approval_rate": approval_rate,
        "predicted_default_rate": float(pred.mean()),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }
