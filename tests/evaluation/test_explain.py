"""Тесты feature importance."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_risk.evaluation.explain import extract_feature_importance
from credit_risk.features import build_preprocessor_for_frame, feature_frame
from credit_risk.data.synthetic import generate_synthetic_credit


def test_extract_importance_logistic() -> None:
    """Importance для logistic pipeline не пустой и ограничен top_n."""
    df = generate_synthetic_credit(n_samples=250, seed=0)
    X, y = feature_frame(df)
    pipe = Pipeline(
        steps=[
            ("pre", build_preprocessor_for_frame(X)),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
        ]
    )
    pipe.fit(X, y)
    imp = extract_feature_importance(pipe, top_n=10)
    assert 1 <= len(imp) <= 10
    assert imp["feature"].astype(str).str.len().min() > 0
    assert (imp["importance"] >= 0).all()
