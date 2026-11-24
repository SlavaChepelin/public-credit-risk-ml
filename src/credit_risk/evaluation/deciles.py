"""Децильный анализ скоров (lift / capture)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def score_decile_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Строит таблицу децилей по убыванию предсказанного риска.

    Args:
        y_true: Бинарные метки.
        y_score: Скоры или вероятности; большее значение означает больший риск.
        n_bins: Число корзин, обычно 10.

    Returns:
        DataFrame с колонками count, defaults, default_rate, lift, capture.
    """
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n = len(y_sorted)
    base_rate = float(y_true.mean()) if n else 0.0

    edges = [int(n * i / n_bins) for i in range(n_bins + 1)]
    rows: list[dict[str, float | int]] = []
    captured = 0.0
    total_pos = float(y_true.sum())

    for i in range(n_bins):
        chunk = y_sorted[edges[i] : edges[i + 1]]
        count = int(len(chunk))
        defaults = float(chunk.sum())
        rate = defaults / count if count else 0.0
        lift = rate / base_rate if base_rate > 0 else 0.0
        captured += defaults
        capture = captured / total_pos if total_pos > 0 else 0.0
        rows.append(
            {
                "decile": i + 1,
                "count": count,
                "defaults": defaults,
                "default_rate": rate,
                "lift": lift,
                "capture": capture,
            }
        )
    return pd.DataFrame(rows)


def plot_decile_lift(deciles: pd.DataFrame, *, title: str = "Decile lift") -> plt.Figure:
    """Рисует столбчатый график lift по децилям риска.

    Args:
        deciles: Таблица, возвращённая score_decile_table.
        title: Заголовок графика.

    Returns:
        Matplotlib-фигура.
    """
    fig, ax = plt.subplots()
    ax.bar(deciles["decile"].astype(str), deciles["lift"], color="#E45756")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="baseline")
    ax.set_xlabel("Decile (1 = riskiest)")
    ax.set_ylabel("Lift")
    ax.set_title(title)
    ax.legend()
    return fig
