"""Оркестрация полного пайплайна обучения и оценки."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from credit_risk import paths
from credit_risk.models import resolve_boosting_backend
from credit_risk.pipeline.dataset import DatasetName, load_dataset, xy_from_split
from credit_risk.pipeline.evaluate import evaluate_models, run_oof
from credit_risk.pipeline.fit import fit_models
from credit_risk.pipeline.persist import persist_artifacts
from credit_risk.pipeline.postprocess import postprocess_champion
from credit_risk.serving import META_NAME, write_model_meta
from credit_risk.tracking import log_artifacts, log_metrics, start_run
from credit_risk.validation.leakage import check_split_leakage
from credit_risk.validation.split import time_based_split


def run(
    *,
    dataset: DatasetName = "synthetic",
    n_samples: int = 12_000,
    seed: int = 42,
    nrows: int | None = None,
    torch_epochs: int | None = None,
    n_folds: int = 3,
    with_oof: bool = True,
    with_postprocess: bool = True,
) -> dict[str, Any]:
    """Запускает полный пайплайн обучения и оценки.

    Args:
        dataset: synthetic или home_credit.
        n_samples: Число заявок при генерации синтетики.
        seed: Зерно генератора данных.
        nrows: Лимит строк Home Credit.
        torch_epochs: Эпохи torch; default зависит от датасета.
        n_folds: Число temporal OOF фолдов.
        with_oof: Считать ли OOF по бустингу.
        with_postprocess: Калибровка / importance / порог.

    Returns:
        Отчёт с метриками, размерами сплитов и именем чемпиона.
    """
    if torch_epochs is None:
        torch_epochs = 8 if dataset == "home_credit" else 20

    params = {
        "dataset": dataset,
        "n_samples": n_samples,
        "seed": seed,
        "nrows": nrows,
        "torch_epochs": torch_epochs,
        "boosting_backend": resolve_boosting_backend(),
    }

    with start_run(run_name=f"train_{dataset}", params=params):
        df, target_col, time_col = load_dataset(
            dataset,
            n_samples=n_samples,
            seed=seed,
            nrows=nrows,
        )
        train_df, valid_df, test_df = time_based_split(df, time_col=time_col)
        check_split_leakage(train_df, valid_df, time_col=time_col)
        check_split_leakage(
            pd.concat([train_df, valid_df], ignore_index=True),
            test_df,
            time_col=time_col,
        )

        X_train, y_train = xy_from_split(train_df, dataset, target_col)
        X_valid, y_valid = xy_from_split(valid_df, dataset, target_col)
        X_test, y_test = xy_from_split(test_df, dataset, target_col)

        models = fit_models(X_train, y_train, torch_epochs=torch_epochs)
        model_metrics, best_name = evaluate_models(
            models,
            (X_valid, y_valid),
            (X_test, y_test),
        )

        report: dict[str, Any] = {
            "dataset": dataset,
            "n_samples": len(df),
            "default_rate": float(df[target_col].mean()),
            "boosting_backend": resolve_boosting_backend(),
            "split": {
                "train": len(train_df),
                "valid": len(valid_df),
                "test": len(test_df),
                "strategy": "time_based_proxy" if dataset == "home_credit" else "time_based",
                "time_col": time_col,
            },
            "models": model_metrics,
            "best_model": best_name,
        }

        if with_oof:
            report["oof"] = run_oof(
                df,
                dataset=dataset,
                target_col=target_col,
                time_col=time_col,
                n_folds=n_folds,
                torch_epochs=torch_epochs,
            )

        postprocess: dict[str, Any] | None = None
        if with_postprocess and best_name is not None:
            postprocess = postprocess_champion(
                models[best_name],
                X_train,
                y_train,
                X_valid,
                y_valid,
                X_test,
                y_test,
            )
            report["postprocess"] = postprocess
            report["selection_rule"] = "max_valid_gini"

        persist_artifacts(models, best_name, report)

        threshold = None
        if postprocess and postprocess.get("threshold"):
            threshold = postprocess["threshold"].get("threshold")
        write_model_meta(
            paths.ARTIFACTS / META_NAME,
            dataset=dataset,
            feature_columns=list(X_train.columns),
            best_model=best_name,
            threshold=threshold,
            calibrated=postprocess is not None,
        )

        flat_metrics = {}
        if best_name:
            for split_name, vals in model_metrics[best_name].items():
                for k, v in vals.items():
                    flat_metrics[f"{best_name}_{split_name}_{k}"] = v
        log_metrics(flat_metrics)
        log_artifacts(
            [
                paths.RESULTS / "metrics.json",
                paths.ARTIFACTS / "champion.joblib",
                paths.ARTIFACTS / "champion_calibrated.joblib",
                paths.ARTIFACTS / META_NAME,
                paths.FIGURES / "reliability.png",
                paths.FIGURES / "importance_top15.png",
            ]
        )
        return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
