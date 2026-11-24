"""Метрики, калибровка, порог, децили, PSI и интерпретация."""

from credit_risk.evaluation.calibration import calibrate_model, plot_reliability
from credit_risk.evaluation.deciles import plot_decile_lift, score_decile_table
from credit_risk.evaluation.explain import (
    extract_feature_importance,
    plot_feature_importance,
)
from credit_risk.evaluation.metrics import (
    classification_report_scores,
    expected_calibration_error,
    gini_from_auc,
    ks_statistic,
    operating_point_metrics,
)
from credit_risk.evaluation.stability import (
    plot_score_distributions,
    population_stability_index,
    psi_report,
)
from credit_risk.evaluation.thresholding import (
    choose_threshold,
    threshold_for_recall,
    threshold_max_ks,
)

__all__ = [
    "calibrate_model",
    "choose_threshold",
    "classification_report_scores",
    "expected_calibration_error",
    "extract_feature_importance",
    "gini_from_auc",
    "ks_statistic",
    "operating_point_metrics",
    "plot_decile_lift",
    "plot_feature_importance",
    "plot_reliability",
    "plot_score_distributions",
    "population_stability_index",
    "psi_report",
    "score_decile_table",
    "threshold_for_recall",
    "threshold_max_ks",
]
