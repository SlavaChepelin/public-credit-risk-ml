"""Постпроцессинг чемпиона: калибровка, порог, децили, PSI."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from credit_risk.eda.style import save_figure
from credit_risk.evaluation.calibration import calibrate_model, plot_reliability
from credit_risk.evaluation.deciles import plot_decile_lift, score_decile_table
from credit_risk.evaluation.explain import (
    extract_feature_importance,
    plot_feature_importance,
)
from credit_risk.evaluation.metrics import (
    classification_report_scores,
    operating_point_metrics,
)
from credit_risk.evaluation.stability import plot_score_distributions, psi_report
from credit_risk.evaluation.thresholding import choose_threshold
from credit_risk import paths


def postprocess_champion(
    champion: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Готовит postprocess-отчёт для чемпиона: калибровка, порог, децили, PSI.

    Калибратор и порог подбираются на valid. Reliability, operating point и
    децили считаются на test. PSI считается по калиброванным PD train→test.

    Args:
        champion: Лучшая модель.
        X_train: Признаки обучения.
        y_train: Таргет обучения; не используется напрямую, сохранён для API.
        X_valid: Валидационные признаки.
        y_valid: Валидационный таргет.
        X_test: Тестовые признаки.
        y_test: Тестовый таргет.

    Returns:
        Блок postprocess для metrics.json.
    """
    del y_train
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    calibrated = calibrate_model(champion, X_valid, y_valid, method="sigmoid")
    proba_train = calibrated.predict_proba(X_train)[:, 1]
    proba_valid = calibrated.predict_proba(X_valid)[:, 1]
    proba_test = calibrated.predict_proba(X_test)[:, 1]
    threshold_info = choose_threshold(y_valid.to_numpy(), proba_valid, strategy="max_ks")
    thr = float(threshold_info["threshold"])

    rel_path = paths.FIGURES / "reliability.png"
    save_figure(
        plot_reliability(
            y_test.to_numpy(),
            proba_test,
            title="Reliability (test, after valid calibration)",
        ),
        rel_path,
    )

    deciles = score_decile_table(y_test.to_numpy(), proba_test, n_bins=10)
    decile_path = paths.FIGURES / "decile_lift.png"
    save_figure(
        plot_decile_lift(deciles, title="Test decile lift (calibrated PD)"),
        decile_path,
    )

    psi = psi_report(proba_train, proba_test, n_bins=10)
    score_dist_path = paths.FIGURES / "score_distribution.png"
    save_figure(
        plot_score_distributions(
            proba_train,
            proba_test,
            title="Calibrated PD: train vs test",
        ),
        score_dist_path,
    )

    importance_path = None
    importance_rows: list[dict[str, Any]] = []
    try:
        imp = extract_feature_importance(champion, top_n=15)
        importance_rows = imp.to_dict(orient="records")
        importance_path = paths.FIGURES / "importance_top15.png"
        save_figure(
            plot_feature_importance(imp, title="Champion feature importance top-15"),
            importance_path,
        )
    except ValueError:
        importance_rows = []

    paths.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, paths.ARTIFACTS / "champion_calibrated.joblib")

    return {
        "calibration": {
            "method": "sigmoid",
            "fit_split": "valid",
            "valid_fit_metrics": classification_report_scores(
                y_valid.to_numpy(),
                proba_valid,
            ),
            "test": classification_report_scores(y_test.to_numpy(), proba_test),
        },
        "threshold": threshold_info,
        "operating_point_test": operating_point_metrics(
            y_test.to_numpy(),
            proba_test,
            threshold=thr,
        ),
        "deciles_test": deciles.to_dict(orient="records"),
        "stability": {
            "psi_train_test": psi,
        },
        "feature_importance_top15": importance_rows,
        "figures": {
            "reliability": str(rel_path),
            "importance": str(importance_path) if importance_path else None,
            "decile_lift": str(decile_path),
            "score_distribution": str(score_dist_path),
        },
    }
