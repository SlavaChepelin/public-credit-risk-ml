"""Графики категориальных признаков и default rate."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def default_rate_by_category(
    df: pd.DataFrame,
    column: str,
    *,
    target_col: str = "TARGET",
    min_count: int = 100,
) -> pd.DataFrame:
    """Считает default rate и частоту по категориям.

    Args:
        df: Датафрейм заявок.
        column: Категориальная колонка.
        target_col: Имя таргета.
        min_count: Минимальный размер категории для отображения.

    Returns:
        Таблица с count и default_rate.
    """
    grouped = (
        df.groupby(column, dropna=False)[target_col]
        .agg(count="size", default_rate="mean")
        .reset_index()
    )
    return grouped[grouped["count"] >= min_count].sort_values("default_rate")


def plot_categorical_default_rates(
    df: pd.DataFrame,
    columns: list[str],
    *,
    target_col: str = "TARGET",
    min_count: int = 100,
) -> plt.Figure:
    """Рисует default rate по категориям для набора признаков.

    Args:
        df: Датафрейм заявок.
        columns: Категориальные колонки.
        target_col: Имя таргета.
        min_count: Порог частоты категории.

    Returns:
        Matplotlib-фигура.
    """
    cols = [c for c in columns if c in df.columns]
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n))
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes_arr, cols, strict=True):
        stats = default_rate_by_category(
            df,
            col,
            target_col=target_col,
            min_count=min_count,
        )
        labels = stats[col].fillna("Missing").astype(str).tolist()
        ax.barh(labels, stats["default_rate"].tolist(), color="#54A24B")
        ax.set_xlabel("Default rate")
        ax.set_title(f"{col} (n ≥ {min_count})")

    fig.suptitle("Default rate по категориальным признакам", y=1.01)
    fig.tight_layout()
    return fig
