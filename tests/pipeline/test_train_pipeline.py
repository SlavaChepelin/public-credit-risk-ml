"""Smoke-тесты пайплайна pipeline.run на маленькой синтетике."""

from pathlib import Path

import joblib

from credit_risk.pipeline import (
    _fit_models,
    _load_dataset,
    _persist_artifacts,
    _predict_scores,
    _run_oof,
    _xy_from_split,
    run,
)


def test_load_dataset_synthetic(tmp_path: Path, monkeypatch) -> None:
    """_load_dataset возвращает synthetic frame и колонки."""
    import credit_risk.pipeline.dataset as dataset_mod
    from credit_risk.data import load_or_generate
    from credit_risk.data.synthetic import generate_synthetic_credit

    path = tmp_path / "syn.csv"
    generate_synthetic_credit(n_samples=200, seed=1).to_csv(path, index=False)
    monkeypatch.setattr(
        dataset_mod,
        "load_or_generate",
        lambda **kwargs: load_or_generate(path=path, n_samples=200, seed=1),
    )
    df, target_col, time_col = _load_dataset(
        "synthetic",
        n_samples=200,
        seed=1,
        nrows=None,
    )
    assert target_col == "target"
    assert time_col == "application_date"
    assert target_col in df.columns
    assert len(df) == 200


def test_xy_from_split_synthetic() -> None:
    """_xy_from_split для synthetic отдаёт X/y."""
    from credit_risk.data.synthetic import generate_synthetic_credit

    df = generate_synthetic_credit(n_samples=150, seed=0)
    X, y = _xy_from_split(df, "synthetic", "target")
    assert len(X) == len(y) == 150
    assert "target" not in X.columns


def test_predict_scores_decision_function() -> None:
    """_predict_scores работает с decision_function без predict_proba."""
    import numpy as np
    import pandas as pd

    class _Dec:
        def decision_function(self, X):
            return np.linspace(0, 1, len(X))

    X = pd.DataFrame({"a": range(6)})
    y = pd.Series([0, 0, 1, 1, 0, 1])
    metrics = _predict_scores(_Dec(), X, y)
    assert {"roc_auc", "gini", "ks", "pr_auc", "brier", "ece"} <= set(metrics)


def test_fit_models_and_persist(tmp_path: Path, monkeypatch) -> None:
    """_fit_models + _persist_artifacts пишут joblib и metrics.json."""
    import credit_risk.paths as paths
    from credit_risk.data.synthetic import generate_synthetic_credit

    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results")

    df = generate_synthetic_credit(n_samples=300, seed=2)
    X, y = _xy_from_split(df, "synthetic", "target")
    models = _fit_models(X, y, torch_epochs=1)
    assert set(models) == {"logistic", "boosting", "torch_nn"}

    report = {"best_model": "logistic", "ok": True}
    _persist_artifacts(models, "logistic", report)
    assert (tmp_path / "artifacts" / "logistic.joblib").exists()
    assert (tmp_path / "artifacts" / "champion.joblib").exists()
    assert (tmp_path / "results" / "metrics.json").exists()
    loaded = joblib.load(tmp_path / "artifacts" / "champion.joblib")
    assert hasattr(loaded, "predict_proba")


def test_run_oof_summary() -> None:
    """_run_oof возвращает mean/std по фолдам."""
    from credit_risk.data.synthetic import generate_synthetic_credit

    df = generate_synthetic_credit(n_samples=400, seed=4)
    oof = _run_oof(
        df,
        dataset="synthetic",
        target_col="target",
        time_col="application_date",
        n_folds=3,
        torch_epochs=1,
    )
    assert oof["n_folds"] == 3
    assert "gini" in oof["summary"]
    assert "mean" in oof["summary"]["gini"]
    assert len(oof["folds"]) == 3


def test_run_synthetic_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """run() на synthetic пишет артефакты, meta и postprocess."""
    import credit_risk.paths as paths
    import credit_risk.pipeline.dataset as dataset_mod
    from credit_risk.data import load_or_generate
    from credit_risk.data.synthetic import generate_synthetic_credit

    data_path = tmp_path / "syn.csv"
    generate_synthetic_credit(n_samples=500, seed=7).to_csv(data_path, index=False)
    monkeypatch.setattr(
        dataset_mod,
        "load_or_generate",
        lambda **kwargs: load_or_generate(path=data_path, n_samples=500, seed=7),
    )
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(paths, "FIGURES", tmp_path / "results" / "figures")

    report = run(
        dataset="synthetic",
        n_samples=500,
        seed=7,
        torch_epochs=1,
        n_folds=3,
        with_oof=True,
        with_postprocess=True,
    )
    assert report["dataset"] == "synthetic"
    assert report["best_model"] in {"logistic", "boosting", "torch_nn"}
    assert report["selection_rule"] == "max_valid_gini"
    assert "oof" in report
    assert "postprocess" in report
    assert (tmp_path / "artifacts" / "champion.joblib").exists()
    assert (tmp_path / "artifacts" / "champion_calibrated.joblib").exists()
    assert (tmp_path / "artifacts" / "model_meta.json").exists()
    assert (tmp_path / "results" / "figures" / "reliability.png").exists()
    assert (tmp_path / "results" / "figures" / "decile_lift.png").exists()
    assert (tmp_path / "results" / "figures" / "score_distribution.png").exists()
    assert "operating_point_test" in report["postprocess"]
    assert "deciles_test" in report["postprocess"]
    assert "stability" in report["postprocess"]
    assert "brier" in report["models"][report["best_model"]]["test"]
