"""Графики распределения целевой переменной."""

from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt


def plot_target_balance(df: pd.DataFrame, *, target_col: str = "TARGET") -> plt.Figure:
    """Рисует баланс классов TARGET.

    Args:
        df: Датафрейм заявок.
        target_col: Имя целевой колонки.

    Returns:
        Matplotlib-фигура.
    """
    counts = df[target_col].value_counts().sort_index()
    rates = df[target_col].value_counts(normalize=True).sort_index()

    fig, ax = plt.subplots()
    bars = ax.bar(
        ["0 — no default", "1 — default"],
        counts.values,
        color=["#4C78A8", "#E45756"],
    )
    ax.set_ylabel("Число заявок")
    ax.set_title("Баланс классов TARGET")
    for bar, rate in zip(bars, rates.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{rate:.1%}",
            ha="center",
            va="bottom",
        )
    return fig
