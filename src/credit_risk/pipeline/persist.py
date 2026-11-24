"""Сохранение моделей и отчёта метрик."""

from __future__ import annotations

import json
from typing import Any

import joblib

from credit_risk import paths


def persist_artifacts(
    models: dict[str, Any],
    best_name: str | None,
    report: dict[str, Any],
) -> None:
    """Сохраняет модели и отчёт с метриками на диск.

    Args:
        models: Обученные модели.
        best_name: Имя чемпиона.
        report: Полный отчёт метрик.
    """
    paths.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    paths.RESULTS.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, paths.ARTIFACTS / f"{name}.joblib")

    if best_name is not None:
        joblib.dump(models[best_name], paths.ARTIFACTS / "champion.joblib")

    (paths.RESULTS / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
