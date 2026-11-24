"""Хелперы time-based валидации для скоринга."""

from __future__ import annotations

import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    time_col: str = "application_date",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Делит выборку хронологически на train, valid и test.

    Будущие заявки не попадают в обучение, что соответствует production-скорингу.

    Args:
        df: Датафрейм заявок.
        time_col: Колонка с датой / proxy-временем заявки.
        train_ratio: Доля обучающей выборки.
        valid_ratio: Доля валидационной выборки.

    Returns:
        Кортеж (train, valid, test) в хронологическом порядке.

    Raises:
        ValueError: Если сумма долей train и valid не меньше 1.
    """
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1.0.")

    ordered = df.sort_values(time_col).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    train = ordered.iloc[:train_end]
    valid = ordered.iloc[train_end:valid_end]
    test = ordered.iloc[valid_end:]
    return train, valid, test


def temporal_folds(
    df: pd.DataFrame,
    time_col: str,
    n_folds: int = 3,
    embed_valid_ratio: float = 0.2,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Строит expanding-window фолды по времени.

    Для фолда k обучающая выборка — всё до начала тестового сегмента,
    валидация — следующий сегмент. Так имитируется OOF по времени.

    Args:
        df: Датафрейм заявок.
        time_col: Колонка времени / proxy-времени.
        n_folds: Число валидационных сегментов (≥ 2).
        embed_valid_ratio: Зарезервированный параметр; сегменты равной длины.

    Returns:
        Список пар (train_df, valid_df) без пересечения valid по времени.

    Raises:
        ValueError: Если n_folds < 2 или данных слишком мало.
    """
    del embed_valid_ratio
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")

    ordered = df.sort_values(time_col).reset_index(drop=True)
    n = len(ordered)
    if n < n_folds * 10:
        raise ValueError("Not enough rows for temporal folds.")

    # Первый сегмент только в train; остальные n_folds — valid по очереди.
    edges = [int(n * i / (n_folds + 1)) for i in range(n_folds + 2)]
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(1, n_folds + 1):
        train = ordered.iloc[: edges[i]]
        valid = ordered.iloc[edges[i] : edges[i + 1]]
        if len(train) == 0 or len(valid) == 0:
            raise ValueError("Empty fold produced; decrease n_folds.")
        folds.append((train, valid))
    return folds


def assert_no_time_leakage(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    time_col: str,
) -> None:
    """Проверяет, что max(train.time) <= min(valid.time).

    Args:
        train: Обучающая выборка.
        valid: Валидационная выборка.
        time_col: Колонка времени / proxy-времени.

    Raises:
        AssertionError: Если есть пересечение по времени.
    """
    train_max = train[time_col].max()
    valid_min = valid[time_col].min()
    assert train_max <= valid_min, (
        f"Time leakage: train max {train_max} > valid min {valid_min}."
    )
