"""Интерпретация модели: важность признаков."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline


def _feature_names_from_pipeline(pipe: Pipeline) -> list[str]:
    """Достаёт имена признаков после препроцессора.

    Args:
        pipe: Pipeline с шагом pre.

    Returns:
        Список имён трансформированных признаков.
    """
    pre = pipe.named_steps["pre"]
    if hasattr(pre, "get_feature_names_out"):
        return list(pre.get_feature_names_out())
    return [f"f{i}" for i in range(pre.transform(np.zeros((1, 1))).shape[1])]


def extract_feature_importance(model: Any, top_n: int = 15) -> pd.DataFrame:
    """Извлекает top-N важности признаков из модели/пайплайна.

    Args:
        model: Обученный Pipeline или estimator.
        top_n: Сколько признаков вернуть.

    Returns:
        DataFrame с columns feature, importance.

    Raises:
        ValueError: Если важности недоступны.
    """
    clf = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    names: list[str]
    if isinstance(model, Pipeline):
        names = _feature_names_from_pipeline(model)
    else:
        names = [f"f{i}" for i in range(getattr(clf, "n_features_in_", 0))]

    if hasattr(clf, "feature_importances_"):
        values = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        values = np.abs(np.asarray(clf.coef_, dtype=float).ravel())
    elif hasattr(clf, "get_feature_importance"):
        values = np.asarray(clf.get_feature_importance(), dtype=float)
    else:
        raise ValueError("Model does not expose feature importances.")

    n = min(len(names), len(values))
    frame = pd.DataFrame(
        {
            "feature": names[:n],
            "importance": values[:n],
        }
    )
    frame = frame.sort_values("importance", ascending=False).head(top_n)
    return frame.reset_index(drop=True)


def plot_feature_importance(
    importance: pd.DataFrame,
    *,
    title: str = "Feature importance top-N",
) -> plt.Figure:
    """Рисует горизонтальный bar chart важностей.

    Args:
        importance: Таблица feature/importance.
        title: Заголовок.

    Returns:
        Matplotlib-фигура.
    """
    ordered = importance.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(ordered))))
    ax.barh(ordered["feature"], ordered["importance"], color="#4C78A8")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    return fig
