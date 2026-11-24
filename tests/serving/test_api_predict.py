"""Smoke-тесты FastAPI /health и /predict."""

from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import api.main as api_main
from credit_risk.features import build_preprocessor_for_frame, feature_frame
from credit_risk.serving import write_model_meta
from credit_risk.data.synthetic import generate_synthetic_credit


@pytest.fixture()
def client_with_champion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Поднимает API с мини-champion.joblib + meta (synthetic)."""
    df = generate_synthetic_credit(n_samples=300, seed=1)
    X, y = feature_frame(df)
    pipe = Pipeline(
        steps=[
            ("pre", build_preprocessor_for_frame(X)),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    pipe.fit(X, y)
    champion = tmp_path / "champion.joblib"
    calibrated = tmp_path / "champion_calibrated.joblib"
    joblib.dump(pipe, champion)
    joblib.dump(pipe, calibrated)
    meta = tmp_path / "model_meta.json"
    write_model_meta(
        meta,
        dataset="synthetic",
        feature_columns=list(X.columns),
        best_model="logistic",
        threshold=0.5,
        calibrated=True,
    )

    monkeypatch.setattr(api_main, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(api_main, "CHAMPION", champion)
    monkeypatch.setattr(api_main, "CHAMPION_CALIBRATED", calibrated)
    monkeypatch.setattr(api_main, "MODEL_META", meta)
    monkeypatch.setattr(api_main, "_model", None)
    monkeypatch.setattr(api_main, "_model_is_calibrated", False)
    monkeypatch.setattr(api_main, "_meta", None)
    return TestClient(api_main.app)


def test_health(client_with_champion: TestClient) -> None:
    """GET /health возвращает ok."""
    resp = client_with_champion.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_smoke(client_with_champion: TestClient) -> None:
    """POST /predict возвращает вероятность в [0, 1]."""
    payload = {
        "age": 35,
        "income": 120000,
        "credit_amount": 200000,
        "annuity": 15000,
        "employment_years": 5,
        "num_prev_loans": 1,
        "num_late_payments": 0,
        "ext_score_1": 0.5,
        "ext_score_2": 0.4,
        "contract_type": "Cash loans",
        "education": "Secondary",
        "family_status": "Married",
        "housing": "House / apartment",
    }
    resp = client_with_champion.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["calibrated"] is True
    assert body["decision"] in (0, 1)


def test_predict_raw(client_with_champion: TestClient) -> None:
    """POST /predict/raw принимает словарь фичей."""
    resp = client_with_champion.post(
        "/predict/raw",
        json={
            "features": {
                "age": 40,
                "income": 90000,
                "credit_amount": 100000,
                "annuity": 8000,
                "employment_years": 3,
                "num_prev_loans": 0,
                "num_late_payments": 0,
                "ext_score_1": 0.6,
                "ext_score_2": 0.5,
                "contract_type": "Cash loans",
                "education": "Higher",
                "family_status": "Single",
                "housing": "Rented",
            }
        },
    )
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["default_probability"] <= 1.0


def test_predict_rejects_typed_on_home_credit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed /predict запрещён, если champion обучен на home_credit."""
    write_model_meta(
        tmp_path / "model_meta.json",
        dataset="home_credit",
        feature_columns=["EXT_SOURCE_2", "age_years"],
        best_model="boosting",
        calibrated=False,
    )
    # минимальная «модель»
    from sklearn.dummy import DummyClassifier
    import numpy as np
    import pandas as pd

    X = pd.DataFrame({"EXT_SOURCE_2": [0.1, 0.9], "age_years": [20.0, 50.0]})
    y = np.array([0, 1])
    clf = DummyClassifier(strategy="prior").fit(X, y)
    joblib.dump(clf, tmp_path / "champion.joblib")

    monkeypatch.setattr(api_main, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(api_main, "CHAMPION", tmp_path / "champion.joblib")
    monkeypatch.setattr(api_main, "CHAMPION_CALIBRATED", tmp_path / "missing.joblib")
    monkeypatch.setattr(api_main, "MODEL_META", tmp_path / "model_meta.json")
    monkeypatch.setattr(api_main, "_model", None)
    monkeypatch.setattr(api_main, "_model_is_calibrated", False)
    monkeypatch.setattr(api_main, "_meta", None)
    client = TestClient(api_main.app)

    resp = client.post(
        "/predict",
        json={
            "age": 30,
            "income": 1,
            "credit_amount": 1,
            "annuity": 1,
            "ext_score_1": 0.5,
        },
    )
    assert resp.status_code == 400

    raw = client.post(
        "/predict/raw",
        json={"features": {"EXT_SOURCE_2": 0.2, "age_years": 33}},
    )
    assert raw.status_code == 200


def test_predict_raw_when_calibrated_artifact_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Meta говорит calibrated, но файла нет → raw model и calibrated=false."""
    df = generate_synthetic_credit(n_samples=200, seed=2)
    X, y = feature_frame(df)
    pipe = Pipeline(
        steps=[
            ("pre", build_preprocessor_for_frame(X)),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
        ]
    )
    pipe.fit(X, y)
    joblib.dump(pipe, tmp_path / "champion.joblib")
    write_model_meta(
        tmp_path / "model_meta.json",
        dataset="synthetic",
        feature_columns=list(X.columns),
        best_model="logistic",
        threshold=0.12,
        calibrated=True,
    )
    monkeypatch.setattr(api_main, "CHAMPION", tmp_path / "champion.joblib")
    monkeypatch.setattr(api_main, "CHAMPION_CALIBRATED", tmp_path / "no_calibrated.joblib")
    monkeypatch.setattr(api_main, "MODEL_META", tmp_path / "model_meta.json")
    monkeypatch.setattr(api_main, "_model", None)
    monkeypatch.setattr(api_main, "_model_is_calibrated", False)
    monkeypatch.setattr(api_main, "_meta", None)
    client = TestClient(api_main.app)

    resp = client.post(
        "/predict",
        json={
            "age": 35,
            "income": 100000,
            "credit_amount": 50000,
            "annuity": 5000,
            "ext_score_1": 0.5,
            "ext_score_2": 0.4,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["calibrated"] is False
    assert body["threshold"] is None
    assert body["decision"] is None


def test_predict_missing_champion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Без артефакта /predict даёт 503."""
    write_model_meta(
        tmp_path / "model_meta.json",
        dataset="synthetic",
        feature_columns=["age"],
        best_model="logistic",
        calibrated=False,
    )
    monkeypatch.setattr(api_main, "CHAMPION", tmp_path / "missing.joblib")
    monkeypatch.setattr(api_main, "CHAMPION_CALIBRATED", tmp_path / "missing2.joblib")
    monkeypatch.setattr(api_main, "MODEL_META", tmp_path / "model_meta.json")
    monkeypatch.setattr(api_main, "_model", None)
    monkeypatch.setattr(api_main, "_model_is_calibrated", False)
    monkeypatch.setattr(api_main, "_meta", None)
    client = TestClient(api_main.app)
    resp = client.post(
        "/predict",
        json={
            "age": 30,
            "income": 1,
            "credit_amount": 1,
            "annuity": 1,
            "ext_score_1": 0.5,
        },
    )
    assert resp.status_code == 503
