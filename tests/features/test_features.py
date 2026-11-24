"""Тесты feature engineering."""

from credit_risk.features import (
    CATEGORICAL,
    NUMERIC,
    build_preprocessor,
    engineer,
    feature_frame,
)
from credit_risk.data.synthetic import generate_synthetic_credit


def test_engineer_adds_derived_columns() -> None:
    """engineer добавляет производные признаки."""
    df = generate_synthetic_credit(n_samples=100, seed=0)
    out = engineer(df)
    for col in (
        "credit_income_ratio",
        "annuity_income_ratio",
        "ext_score_mean",
        "ext_score_missing",
    ):
        assert col in out.columns
    assert out["ext_score_missing"].isin([0.0, 1.0]).all()


def test_feature_frame_with_and_without_target() -> None:
    """feature_frame возвращает y только при наличии target."""
    df = generate_synthetic_credit(n_samples=50, seed=0)
    X, y = feature_frame(df)
    assert list(X.columns) == NUMERIC + CATEGORICAL
    assert y is not None
    assert len(X) == len(y)

    X2, y2 = feature_frame(df.drop(columns=["target"]))
    assert y2 is None
    assert len(X2) == len(df)


def test_preprocessor_fit_transform_shape() -> None:
    """Препроцессор отрабатывает без ошибок и сохраняет число строк."""
    df = generate_synthetic_credit(n_samples=80, seed=0)
    X, _ = feature_frame(df)
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(NUMERIC)
