"""Загрузка датасета и построение X/y для сплитов."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from credit_risk.data.home_credit import PROXY_TIME_COL
from credit_risk.data.io import load_or_generate
from credit_risk.features.home_credit import build_home_credit_features, home_credit_xy
from credit_risk.features.synthetic import feature_frame

DatasetName = Literal["synthetic", "home_credit"]


def load_dataset(
    dataset: DatasetName,
    *,
    n_samples: int,
    seed: int,
    nrows: int | None,
) -> tuple[pd.DataFrame, str, str]:
    """Загружает датасет и возвращает frame, target_col и time_col.

    Args:
        dataset: Имя датасета: synthetic или home_credit.
        n_samples: Размер синтетики.
        seed: Зерно синтетики.
        nrows: Лимит строк Home Credit.

    Returns:
        Кортеж (df, target_col, time_col).

    Raises:
        ValueError: Если имя датасета неизвестно.
    """
    if dataset == "synthetic":
        df = load_or_generate(n_samples=n_samples, seed=seed)
        return df, "target", "application_date"
    if dataset == "home_credit":
        df = build_home_credit_features(nrows=nrows)
        return df, "TARGET", PROXY_TIME_COL
    raise ValueError(f"Unknown dataset: {dataset}.")


def xy_from_split(
    frame: pd.DataFrame,
    dataset: DatasetName,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Строит X, y для куска сплита.

    Args:
        frame: Кусок датафрейма.
        dataset: Тип датасета.
        target_col: Имя таргета.

    Returns:
        Кортеж (X, y).

    Raises:
        AssertionError: Если в synthetic-кадре отсутствует колонка target.
    """
    if dataset == "synthetic":
        X, y = feature_frame(frame)
        assert y is not None, "Synthetic frame is missing target column."
        return X, y
    return home_credit_xy(frame, target_col=target_col)
