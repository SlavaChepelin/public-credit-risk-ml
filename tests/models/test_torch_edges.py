"""Edge-cases TorchTabularClassifier."""

import numpy as np
import pytest

from credit_risk.models.torch_nn import TorchTabularClassifier


def test_predict_proba_before_fit_raises() -> None:
    """predict_proba до fit → RuntimeError."""
    clf = TorchTabularClassifier(epochs=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_proba(np.zeros((2, 3), dtype=np.float32))


def test_predict_after_fit() -> None:
    """predict возвращает 0/1 после обучения."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    clf = TorchTabularClassifier(epochs=2, batch_size=16).fit(X, y)
    preds = clf.predict(X[:10])
    assert preds.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})
