"""Графики корреляций числовых признаков с TARGET."""

from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt


def correlation_with_target(
    df: pd.DataFrame,
    *,
    target_col: str = "TARGET",
    top_n: int = 25,
) -> pd.Series:
    """Считает корреляцию Пирсона числовых признаков с TARGET.

    Args:
        df: Датафрейм заявок.
        target_col: Имя таргета.
        top_n: Сколько признаков вернуть по |corr|.

    Returns:
        Series корреляций, отсортированная по абсолютному значению.

    Raises:
        KeyError: Если target_col отсутствует или не числовой.
    """
    numeric = df.select_dtypes(include="number")
    if target_col not in numeric.columns:
        raise KeyError(f"Missing or non-numeric target column: {target_col}.")
    corr = numeric.corr(numeric_only=True)[target_col].drop(labels=[target_col])
    corr = corr.dropna()
    return corr.reindex(corr.abs().sort_values(ascending=False).head(top_n).index)


def plot_target_correlations(
    df: pd.DataFrame,
    *,
    target_col: str = "TARGET",
    top_n: int = 25,
) -> plt.Figure:
    """Рисует топ корреляций числовых признаков с TARGET.

    Args:
        df: Датафрейм заявок.
        target_col: Имя таргета.
        top_n: Число признаков на графике.

    Returns:
        Matplotlib-фигура.
    """
    corr = correlation_with_target(df, target_col=target_col, top_n=top_n)
    colors = ["#E45756" if v > 0 else "#4C78A8" for v in corr.values]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(corr))))
    ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1])
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(f"corr with {target_col}")
    ax.set_title(f"Топ-{len(corr)} корреляций с {target_col}")
    return fig
