"""Метрики стабильности распределений (PSI)."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Считает Population Stability Index между двумя распределениями.

    Args:
        expected: Референсное распределение, обычно train scores.
        actual: Сравниваемое распределение, обычно test scores.
        n_bins: Число квантильных бинов по expected.
        eps: Сглаживание нулевых долей бинов.

    Returns:
        Неотрицательное значение PSI. Ориентиры: меньше 0.1 — stable,
        0.1–0.25 — watch, больше 0.25 — shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    breaks = np.unique(np.quantile(expected, quantiles))
    if len(breaks) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=breaks)
    act_counts, _ = np.histogram(actual, bins=breaks)
    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_report(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, float | str]:
    """Формирует краткий отчёт PSI со текстовым статусом.

    Args:
        train_scores: Скоры на обучающей выборке.
        test_scores: Скоры на тестовой выборке.
        n_bins: Число бинов для PSI.

    Returns:
        Словарь с ключами psi, status и n_bins.
    """
    psi = population_stability_index(train_scores, test_scores, n_bins=n_bins)
    if psi < 0.1:
        status = "stable"
    elif psi < 0.25:
        status = "watch"
    else:
        status = "shift"
    return {"psi": psi, "status": status, "n_bins": float(n_bins)}


def plot_score_distributions(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    *,
    title: str = "Score distribution train vs test",
) -> plt.Figure:
    """Рисует гистограммы скоров train/test.

    Args:
        train_scores: Train PD/scores.
        test_scores: Test PD/scores.
        title: Заголовок.

    Returns:
        Matplotlib-фигура.
    """
    fig, ax = plt.subplots()
    ax.hist(train_scores, bins=30, density=True, alpha=0.55, label="train", color="#4C78A8")
    ax.hist(test_scores, bins=30, density=True, alpha=0.55, label="test", color="#E45756")
    ax.set_xlabel("Score / PD")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return fig
