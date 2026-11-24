"""Подготовка признаков и метаданных для API-скоринга."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.features.synthetic import feature_frame

META_NAME = "model_meta.json"


def write_model_meta(
    path: Path,
    *,
    dataset: str,
    feature_columns: list[str],
    best_model: str | None,
    threshold: float | None = None,
    calibrated: bool = False,
) -> Path:
    """Пишет sidecar-метаданные для serving.

    Args:
        path: Путь к model_meta.json.
        dataset: synthetic или home_credit.
        feature_columns: Колонки X в порядке обучения.
        best_model: Имя чемпиона.
        threshold: Бизнес-порог (если есть).
        calibrated: Есть ли champion_calibrated.joblib.

    Returns:
        Путь к записанному файлу.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "feature_columns": feature_columns,
        "best_model": best_model,
        "threshold": threshold,
        "calibrated": calibrated,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_model_meta(path: Path) -> dict[str, Any]:
    """Читает model_meta.json.

    Args:
        path: Путь к метаданным.

    Returns:
        Словарь метаданных.

    Raises:
        FileNotFoundError: Если файл отсутствует.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_scoring_frame(
    payload: dict[str, Any],
    *,
    dataset: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Собирает однострочный DataFrame под колонки обученной модели.

    Для synthetic применяет тот же feature engineering, что и при обучении.
    Отсутствующие колонки заполняются NaN (импутация в препроцессоре).

    Args:
        payload: Входные поля заявки.
        dataset: Имя датасета: synthetic или home_credit.
        feature_columns: Ожидаемые колонки модели.

    Returns:
        DataFrame с одной строкой и колонками feature_columns.
    """
    if dataset == "synthetic":
        row = pd.DataFrame([payload])
        if "target" not in row.columns:
            row = row.assign(target=0)
        X, _ = feature_frame(row)
    else:
        X = pd.DataFrame([payload])

    missing = [c for c in feature_columns if c not in X.columns]
    for col in missing:
        X[col] = np.nan

    return X.reindex(columns=feature_columns)
