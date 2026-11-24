"""FastAPI-сервис скоринга по champion / champion_calibrated."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_risk.serving import META_NAME, build_scoring_frame, load_model_meta

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CHAMPION = ARTIFACTS / "champion.joblib"
CHAMPION_CALIBRATED = ARTIFACTS / "champion_calibrated.joblib"
MODEL_META = ARTIFACTS / META_NAME

app = FastAPI(title="credit-risk-ml", version="0.1.0")
_model = None
_model_is_calibrated = False
_meta: dict[str, Any] | None = None


class Application(BaseModel):
    """Входная заявка для typed-скоринга на синтетической схеме признаков."""

    age: int = Field(ge=18, le=100)
    income: float
    credit_amount: float
    annuity: float
    employment_years: float | None = None
    num_prev_loans: int = 0
    num_late_payments: int = 0
    ext_score_1: float = Field(ge=0, le=1)
    ext_score_2: float | None = Field(default=None, ge=0, le=1)
    contract_type: str = "Cash loans"
    education: str = "Secondary"
    family_status: str = "Married"
    housing: str = "House / apartment"


class FeaturePayload(BaseModel):
    """Универсальный payload со словарём признаков обученной модели."""

    features: dict[str, Any]


def _load_meta() -> dict[str, Any]:
    """Загружает model_meta.json или возвращает synthetic-значения по умолчанию.

    Returns:
        Словарь метаданных serving.
    """
    global _meta
    if _meta is not None:
        return _meta
    if MODEL_META.exists():
        _meta = load_model_meta(MODEL_META)
    else:
        _meta = {
            "dataset": "synthetic",
            "feature_columns": [],
            "best_model": None,
            "threshold": 0.5,
            "calibrated": False,
        }
    return _meta


def get_model():
    """Лениво загружает калиброванную модель или raw champion.

    Returns:
        Загруженная модель.

    Raises:
        HTTPException: Если артефакты модели отсутствуют.
    """
    global _model, _model_is_calibrated
    if _model is None:
        meta = _load_meta()
        want_calibrated = bool(meta.get("calibrated"))
        if want_calibrated and CHAMPION_CALIBRATED.exists():
            path = CHAMPION_CALIBRATED
            _model_is_calibrated = True
        elif CHAMPION.exists():
            path = CHAMPION
            _model_is_calibrated = False
        else:
            raise HTTPException(
                503,
                "champion.joblib missing — run scripts/run_pipeline.py first.",
            )
        _model = joblib.load(path)
    return _model


def _score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Считает вероятность дефолта по словарю признаков.

    Args:
        payload: Поля заявки.

    Returns:
        Ответ с вероятностью и метаданными serving.

    Raises:
        HTTPException: Если в meta нет feature_columns.
    """
    model = get_model()
    meta = _load_meta()
    columns = meta.get("feature_columns") or []
    if not columns:
        raise HTTPException(
            503,
            "model_meta.json missing feature_columns — re-run training pipeline.",
        )
    X = build_scoring_frame(
        payload,
        dataset=meta.get("dataset", "synthetic"),
        feature_columns=columns,
    )
    proba = float(model.predict_proba(X)[:, 1][0])
    threshold = meta.get("threshold") if _model_is_calibrated else None
    decision = None if threshold is None else int(proba >= float(threshold))
    return {
        "default_probability": proba,
        "model": meta.get("best_model") or "champion",
        "dataset": meta.get("dataset"),
        "calibrated": _model_is_calibrated,
        "threshold": threshold,
        "decision": decision,
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Проверяет доступность сервиса.

    Returns:
        Статус сервиса.
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(app_in: Application) -> dict[str, Any]:
    """Считает вероятность дефолта (typed schema для synthetic).

    Args:
        app_in: Входная заявка synthetic-формата.

    Returns:
        Словарь с default_probability и метаданными serving.

    Raises:
        HTTPException: Если champion обучен на home_credit (нужен /predict/raw).
    """
    meta = _load_meta()
    if meta.get("dataset") == "home_credit":
        raise HTTPException(
            400,
            "Champion trained on home_credit — use POST /predict/raw with feature dict.",
        )
    return _score_payload(app_in.model_dump())


@app.post("/predict/raw")
def predict_raw(body: FeaturePayload) -> dict[str, Any]:
    """Считает вероятность дефолта по словарю признаков модели.

    Args:
        body: {"features": {...}} в схеме обучения (synthetic или home_credit).

    Returns:
        Словарь с default_probability и метаданными.
    """
    return _score_payload(body.features)
