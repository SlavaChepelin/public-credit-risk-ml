"""Регрессии калибровки."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_risk.evaluation.calibration import calibrate_model


def test_calibrate_model_fallback_works() -> None:
    """Калибровка отрабатывает (FrozenEstimator или fallback)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    y = pd.Series((X["a"] + rng.normal(scale=0.4, size=120) > 0).astype(int))
    base = LogisticRegression(max_iter=400).fit(X.iloc[:80], y.iloc[:80])
    calibrated = calibrate_model(base, X.iloc[80:], y.iloc[80:], method="sigmoid")
    proba = calibrated.predict_proba(X.iloc[80:])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
