"""Smoke-тесты обучения моделей на маленькой выборке."""

from credit_risk.features import feature_frame
from credit_risk.models import fit_boosting, fit_logistic, fit_torch_nn
from credit_risk.data.synthetic import generate_synthetic_credit


def _toy_xy(n_samples: int = 400, seed: int = 0):
    """Готовит маленькую обучающую выборку."""
    df = generate_synthetic_credit(n_samples=n_samples, seed=seed)
    return feature_frame(df)


def test_fit_logistic_predicts_proba() -> None:
    """Логистическая регрессия обучается и отдаёт вероятности."""
    X, y = _toy_xy()
    model = fit_logistic(X, y)
    proba = model.predict_proba(X.head(10))
    assert proba.shape == (10, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_fit_boosting_predicts_proba() -> None:
    """Бустинг обучается и отдаёт вероятности."""
    X, y = _toy_xy()
    model = fit_boosting(X, y)
    proba = model.predict_proba(X.head(10))
    assert proba.shape == (10, 2)


def test_fit_torch_nn_predicts_proba() -> None:
    """Нейросеть обучается за малое число эпох и отдаёт вероятности."""
    X, y = _toy_xy(n_samples=300, seed=2)
    model = fit_torch_nn(X, y, epochs=2, batch_size=64)
    proba = model.predict_proba(X.head(8))
    assert proba.shape == (8, 2)
    assert abs(proba.sum(axis=1) - 1.0).max() < 1e-5
