"""Тесты MLflow tracking (no-op и с моком)."""

from pathlib import Path
from unittest.mock import MagicMock

from credit_risk.tracking import mlflow as tracking


def test_mlflow_or_none_without_package(monkeypatch) -> None:
    """Без mlflow _mlflow_or_none возвращает None."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("no mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert tracking._mlflow_or_none() is None


def test_start_run_noop_without_mlflow(monkeypatch) -> None:
    """start_run без mlflow — nullcontext, не падает."""
    monkeypatch.setattr(tracking, "_mlflow_or_none", lambda: None)
    with tracking.start_run(run_name="t", params={"a": 1}) as ctx:
        assert ctx is None


def test_log_metrics_and_artifacts_noop(monkeypatch, tmp_path: Path) -> None:
    """log_* без mlflow молча выходят."""
    monkeypatch.setattr(tracking, "_mlflow_or_none", lambda: None)
    tracking.log_metrics({"auc": 0.7})
    tracking.log_artifacts([tmp_path / "missing.txt"])


def test_start_run_with_mock_mlflow(monkeypatch, tmp_path: Path) -> None:
    """С моком mlflow логируются params/metrics/artifacts."""
    mlflow = MagicMock()
    run_ctx = MagicMock()
    run_ctx.__enter__.return_value = "run"
    run_ctx.__exit__.return_value = None
    mlflow.start_run.return_value = run_ctx

    monkeypatch.setattr(tracking, "_mlflow_or_none", lambda: mlflow)

    with tracking.start_run(run_name="demo", params={"x": 1, "y": True}):
        tracking.log_metrics({"gini": 0.4})
        artifact = tmp_path / "a.txt"
        artifact.write_text("ok", encoding="utf-8")
        tracking.log_artifacts([artifact, tmp_path / "nope.txt"])

    mlflow.start_run.assert_called_once()
    mlflow.log_params.assert_called_once()
    mlflow.log_metrics.assert_called_once()
    mlflow.log_artifact.assert_called_once_with(str(artifact))


def test_log_metrics_swallows_errors(monkeypatch) -> None:
    """Ошибка log_metrics не пробрасывается."""
    mlflow = MagicMock()
    mlflow.log_metrics.side_effect = RuntimeError("boom")
    monkeypatch.setattr(tracking, "_mlflow_or_none", lambda: mlflow)
    tracking.log_metrics({"a": 1.0})


def test_log_artifacts_swallows_errors(monkeypatch, tmp_path: Path) -> None:
    """Ошибка log_artifact не пробрасывается."""
    mlflow = MagicMock()
    mlflow.log_artifact.side_effect = RuntimeError("boom")
    monkeypatch.setattr(tracking, "_mlflow_or_none", lambda: mlflow)
    path = tmp_path / "x.json"
    path.write_text("{}", encoding="utf-8")
    tracking.log_artifacts([path])
