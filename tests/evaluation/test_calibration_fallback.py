"""Тесты fallback-калибровки _ScorePlatform."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_risk.evaluation.calibration import _ScorePlatform, calibrate_model


def test_score_platform_sigmoid_and_predict() -> None:
    """_ScorePlatform sigmoid даёт proba в [0,1] и predict."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(160, 3)), columns=list("abc"))
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=160) > 0).astype(int))
    base = LogisticRegression(max_iter=400).fit(X.iloc[:100], y.iloc[:100])
    cal = _ScorePlatform(base=base, method="sigmoid").fit(X.iloc[100:], y.iloc[100:])
    proba = cal.predict_proba(X.iloc[100:])
    assert proba.shape[1] == 2
    assert ((proba >= 0) & (proba <= 1)).all()
    preds = cal.predict(X.iloc[100:])
    assert set(np.unique(preds)).issubset({0, 1})


def test_score_platform_isotonic() -> None:
    """_ScorePlatform isotonic отрабатывает."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(160, 2)), columns=list("ab"))
    y = pd.Series((X["a"] > 0).astype(int))
    base = LogisticRegression(max_iter=400).fit(X.iloc[:100], y.iloc[:100])
    cal = _ScorePlatform(base=base, method="isotonic").fit(X.iloc[100:], y.iloc[100:])
    p1 = cal.predict_proba(X.iloc[100:])[:, 1]
    assert ((p1 >= 0) & (p1 <= 1)).all()


def test_calibrate_model_forces_fallback(monkeypatch) -> None:
    """Если FrozenEstimator недоступен — используется fallback."""
    import credit_risk.evaluation.calibration as cal_mod

    def boom(*args, **kwargs):
        raise ImportError("no frozen")

    monkeypatch.setattr(cal_mod, "CalibratedClassifierCV", boom)

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    y = pd.Series((X["a"] > 0).astype(int))
    base = LogisticRegression(max_iter=300).fit(X.iloc[:70], y.iloc[:70])
    calibrated = calibrate_model(base, X.iloc[70:], y.iloc[70:], method="sigmoid")
    assert isinstance(calibrated, _ScorePlatform)
    assert calibrated.predict_proba(X.iloc[70:]).shape[0] == 50
