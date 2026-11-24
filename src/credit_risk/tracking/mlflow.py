"""Опциональное логирование прогона в MLflow."""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _mlflow_or_none():
    """Импортирует mlflow при наличии пакета.

    Returns:
        Модуль mlflow или None.
    """
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


def start_run(
    *,
    run_name: str,
    params: dict[str, Any],
) -> AbstractContextManager[Any]:
    """Открывает MLflow run или no-op контекст.

    Args:
        run_name: Имя run.
        params: Параметры для логирования.

    Returns:
        Context manager (mlflow.start_run или nullcontext).
    """
    mlflow = _mlflow_or_none()
    if mlflow is None:
        logger.info("MLflow is not installed; tracking skipped")
        return nullcontext()

    ctx = mlflow.start_run(run_name=run_name)
    return _RunWrapper(ctx, params)


class _RunWrapper(AbstractContextManager[Any]):
    """Контекст-менеджер, логирующий params при входе в MLflow run."""

    def __init__(self, ctx: AbstractContextManager[Any], params: dict[str, Any]):
        """Сохраняет вложенный контекст и параметры прогона.

        Args:
            ctx: Результат mlflow.start_run.
            params: Параметры для log_params.
        """
        self._ctx = ctx
        self._params = params
        self._mlflow = _mlflow_or_none()

    def __enter__(self) -> Any:
        """Входит в run и логирует params.

        Returns:
            Объект активного MLflow run.
        """
        run = self._ctx.__enter__()
        if self._mlflow is not None:
            flat = {k: str(v) for k, v in self._params.items()}
            self._mlflow.log_params(flat)
        return run

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        """Закрывает вложенный MLflow run.

        Args:
            exc_type: Тип исключения.
            exc: Экземпляр исключения.
            tb: Traceback.

        Returns:
            Результат __exit__ вложенного контекста.
        """
        return self._ctx.__exit__(exc_type, exc, tb)


def log_metrics(metrics: dict[str, float]) -> None:
    """Логирует метрики в активный MLflow run, если пакет доступен.

    Args:
        metrics: Плоский словарь метрик.
    """
    mlflow = _mlflow_or_none()
    if mlflow is None:
        return
    try:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to log MLflow metrics: %s", exc)


def log_artifacts(paths: list[Path]) -> None:
    """Логирует файлы-артефакты в MLflow, если пакет доступен.

    Args:
        paths: Пути к файлам.
    """
    mlflow = _mlflow_or_none()
    if mlflow is None:
        return
    for path in paths:
        if path.exists():
            try:
                mlflow.log_artifact(str(path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to log artifact %s: %s", path, exc)
