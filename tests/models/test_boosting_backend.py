"""Тесты выбора backend бустинга."""

from credit_risk.features import feature_frame
from credit_risk.models.boosting import (
    BOOSTING_BACKEND_HIST_GB,
    _build_boosting_classifier,
    fit_boosting,
    resolve_boosting_backend,
)
from credit_risk.data.synthetic import generate_synthetic_credit


def test_resolve_boosting_backend_known() -> None:
    """resolve_boosting_backend возвращает catboost или hist_gb."""
    assert resolve_boosting_backend() in {"catboost", "hist_gb"}


def test_force_hist_gb_backend() -> None:
    """Принудительный hist_gb создаёт HistGradientBoosting."""
    clf = _build_boosting_classifier(BOOSTING_BACKEND_HIST_GB)
    assert clf.__class__.__name__ == "HistGradientBoostingClassifier"


def test_fit_boosting_forced_hist_gb() -> None:
    """fit_boosting с backend=hist_gb обучается и ставит атрибут."""
    df = generate_synthetic_credit(n_samples=250, seed=0)
    X, y = feature_frame(df)
    pipe = fit_boosting(X, y, backend=BOOSTING_BACKEND_HIST_GB)
    assert pipe.boosting_backend_ == BOOSTING_BACKEND_HIST_GB
    assert pipe.predict_proba(X.head(3)).shape == (3, 2)
