"""Тесты калибровки и выбора порога."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from credit_risk.evaluation.calibration import calibrate_model, plot_reliability
from credit_risk.evaluation.thresholding import choose_threshold, threshold_max_ks


def test_threshold_max_ks_finite() -> None:
    """Порог max KS конечен."""
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    thr = threshold_max_ks(y, s)
    assert np.isfinite(thr)


def test_choose_threshold_recall() -> None:
    """Стратегия recall возвращает метаданные."""
    y = np.array([0, 0, 1, 1, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.4, 0.9, 0.3, 0.7, 0.8])
    info = choose_threshold(y, s, strategy="recall", target_recall=0.5)
    assert info["strategy"] == "recall"
    assert 0.0 <= float(info["recall"]) <= 1.0


def test_calibrate_model_proba_range() -> None:
    """После калибровки вероятности в [0, 1]."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    base = LogisticRegression(max_iter=500).fit(X[:150], y[:150])
    calibrated = calibrate_model(base, X[150:], y[150:], method="sigmoid")
    proba = calibrated.predict_proba(X[150:])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    fig = plot_reliability(y[150:], proba)
    assert fig is not None
