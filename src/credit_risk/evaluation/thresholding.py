"""Выбор бизнес-порога по скорам."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import roc_curve

from credit_risk.evaluation.metrics import operating_point_metrics


def threshold_max_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Выбирает порог, максимизирующий KS.

    Args:
        y_true: Истинные метки.
        y_score: Скоры / вероятности.

    Returns:
        Порог по y_score.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ks = np.abs(tpr - fpr)
    # thresholds[0] может быть inf — берём argmax на валидных.
    idx = int(np.nanargmax(ks))
    thr = float(thresholds[idx])
    if not np.isfinite(thr):
        thr = 0.5
    return thr


def threshold_for_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_recall: float = 0.7,
) -> float:
    """Выбирает порог с recall не ниже целевого.

    Args:
        y_true: Истинные метки.
        y_score: Скоры / вероятности.
        target_recall: Целевой recall по дефолтам.

    Returns:
        Максимальный порог, при котором recall ≥ target_recall.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ok = np.where(tpr >= target_recall)[0]
    if len(ok) == 0:
        return float(np.min(y_score))
    idx = int(ok[0])
    thr = float(thresholds[idx])
    return thr if np.isfinite(thr) else 0.5


def choose_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    strategy: Literal["max_ks", "recall"] = "max_ks",
    target_recall: float = 0.7,
) -> dict[str, float | str]:
    """Выбирает порог по стратегии и считает метрики operating point.

    Args:
        y_true: Истинные метки.
        y_score: Скоры или вероятности.
        strategy: Стратегия выбора: max_ks или recall.
        target_recall: Целевой recall для strategy=recall.

    Returns:
        Словарь со strategy, threshold и метриками operating point.

    Raises:
        ValueError: Если передана неизвестная strategy.
    """
    if strategy == "max_ks":
        thr = threshold_max_ks(y_true, y_score)
    elif strategy == "recall":
        thr = threshold_for_recall(
            y_true,
            y_score,
            target_recall=target_recall,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}.")

    point = operating_point_metrics(y_true, y_score, threshold=thr)
    return {
        "strategy": strategy,
        "threshold": thr,
        "target_recall": target_recall if strategy == "recall" else None,
        **point,
    }
