"""Графики числовых признаков относительно TARGET."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _clip_for_plot(series: pd.Series, q_low: float = 0.01, q_high: float = 0.99) -> pd.Series:
    """Обрезает хвосты для читаемых гистограмм.

    Args:
        series: Числовой ряд.
        q_low: Нижний квантиль.
        q_high: Верхний квантиль.

    Returns:
        Обрезанный ряд без NaN.
    """
    clean = series.dropna()
    low, high = clean.quantile([q_low, q_high])
    return clean.clip(lower=low, upper=high)


def plot_numeric_by_target(
    df: pd.DataFrame,
    columns: list[str],
    *,
    target_col: str = "TARGET",
    bins: int = 40,
) -> plt.Figure:
    """Рисует распределения числовых признаков по классам TARGET.

    Args:
        df: Датафрейм заявок.
        columns: Список числовых колонок.
        target_col: Имя таргета.
        bins: Число бинов гистограммы.

    Returns:
        Matplotlib-фигура.
    """
    cols = [c for c in columns if c in df.columns]
    n = len(cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes_arr, cols, strict=False):
        for label, color in ((0, "#4C78A8"), (1, "#E45756")):
            values = _clip_for_plot(df.loc[df[target_col] == label, col])
            ax.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.55,
                label=f"TARGET={label}",
                color=color,
            )
        ax.set_title(col)
        ax.legend(fontsize=8)

    for ax in axes_arr[len(cols) :]:
        ax.axis("off")

    fig.suptitle("Числовые признаки по TARGET", y=1.01)
    fig.tight_layout()
    return fig
