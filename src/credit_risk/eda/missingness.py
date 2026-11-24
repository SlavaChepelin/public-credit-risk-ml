"""Графики пропусков в признаках."""

from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt


def missing_rate(df: pd.DataFrame) -> pd.Series:
    """Считает долю пропусков по колонкам.

    Args:
        df: Датафрейм.

    Returns:
        Series с долей NaN, отсортированная по убыванию.
    """
    return df.isna().mean().sort_values(ascending=False)


def plot_missing_top(
    df: pd.DataFrame,
    *,
    top_n: int = 30,
) -> plt.Figure:
    """Рисует топ колонок по доле пропусков.

    Args:
        df: Датафрейм заявок.
        top_n: Сколько колонок показать.

    Returns:
        Matplotlib-фигура.
    """
    rates = missing_rate(df)
    rates = rates[rates > 0].head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(rates))))
    ax.barh(rates.index[::-1], rates.values[::-1], color="#F58518")
    ax.set_xlabel("Доля пропусков")
    ax.set_title(f"Топ-{len(rates)} признаков по missing rate")
    ax.set_xlim(0, 1)
    return fig
