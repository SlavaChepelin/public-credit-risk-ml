"""Пайплайн обучения: загрузка → fit → evaluate → postprocess → persist."""

from credit_risk.pipeline.dataset import load_dataset, xy_from_split
from credit_risk.pipeline.evaluate import evaluate_models, predict_scores, run_oof
from credit_risk.pipeline.fit import fit_models
from credit_risk.pipeline.persist import persist_artifacts
from credit_risk.pipeline.postprocess import postprocess_champion
from credit_risk.pipeline.run import run
from credit_risk.paths import ARTIFACTS, FIGURES, RESULTS

# Совместимость со старыми именами в тестах.
MODELS = ARTIFACTS
_load_dataset = load_dataset
_xy_from_split = xy_from_split
_fit_models = fit_models
_predict_scores = predict_scores
_evaluate_models = evaluate_models
_run_oof = run_oof
_postprocess_champion = postprocess_champion
_persist_artifacts = persist_artifacts

__all__ = [
    "ARTIFACTS",
    "FIGURES",
    "MODELS",
    "RESULTS",
    "evaluate_models",
    "fit_models",
    "load_dataset",
    "persist_artifacts",
    "postprocess_champion",
    "predict_scores",
    "run",
    "run_oof",
    "xy_from_split",
]
