"""Общие настройки matplotlib для EDA-графиков."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def configure_matplotlib() -> None:
    """Настраивает единый стиль графиков."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
            "font.size": 11,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Сохраняет фигуру на диск и закрывает её.

    Args:
        fig: Matplotlib-фигура.
        path: Целевой путь к PNG.

    Returns:
        Путь сохранённого файла.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
