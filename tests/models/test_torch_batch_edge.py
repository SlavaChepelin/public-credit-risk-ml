"""Регрессии моделей."""

from credit_risk.data.synthetic import generate_synthetic_credit
from credit_risk.features import feature_frame
from credit_risk.models.torch_nn import fit_torch_nn


def test_torch_survives_batch_size_edge() -> None:
    """Torch не падает, когда n % batch_size == 1."""
    df = generate_synthetic_credit(n_samples=257, seed=0)
    X, y = feature_frame(df)
    model = fit_torch_nn(X, y, epochs=1, batch_size=256)
    proba = model.predict_proba(X.head(5))
    assert proba.shape == (5, 2)
