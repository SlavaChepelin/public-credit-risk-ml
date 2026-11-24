"""Расширенные тесты explain / importance."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_risk.eda.style import save_figure
from credit_risk.evaluation.explain import (
    extract_feature_importance,
    plot_feature_importance,
)
from credit_risk.features import build_preprocessor_for_frame, feature_frame
from credit_risk.data.synthetic import generate_synthetic_credit


def test_extract_importance_from_forest() -> None:
    """feature_importances_ из tree-модели извлекаются."""
    df = generate_synthetic_credit(n_samples=250, seed=1)
    X, y = feature_frame(df)
    pipe = Pipeline(
        steps=[
            ("pre", build_preprocessor_for_frame(X)),
            ("clf", RandomForestClassifier(n_estimators=20, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    imp = extract_feature_importance(pipe, top_n=5)
    assert len(imp) == 5
    assert list(imp.columns) == ["feature", "importance"]


def test_extract_importance_bare_estimator() -> None:
    """Importance без Pipeline (голый estimator с coef_)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression(max_iter=300).fit(X, y)
    imp = extract_feature_importance(clf, top_n=3)
    assert 1 <= len(imp) <= 3


def test_extract_importance_raises_without_attr() -> None:
    """Без importances/coef_ — ValueError."""

    class _NoImp:
        n_features_in_ = 2

    with pytest.raises(ValueError, match="does not expose"):
        extract_feature_importance(_NoImp(), top_n=2)


def test_plot_feature_importance(tmp_path: Path) -> None:
    """plot_feature_importance сохраняет PNG."""
    imp = pd.DataFrame(
        {
            "feature": ["f1", "f2", "f3"],
            "importance": [0.5, 0.3, 0.2],
        }
    )
    path = save_figure(plot_feature_importance(imp), tmp_path / "imp.png")
    assert path.exists() and path.stat().st_size > 0
