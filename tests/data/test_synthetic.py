"""Тесты генерации синтетического датасета."""

import numpy as np

from credit_risk.data.synthetic import generate_synthetic_credit


REQUIRED_COLUMNS = {
    "application_date",
    "age",
    "income",
    "credit_amount",
    "annuity",
    "employment_years",
    "num_prev_loans",
    "num_late_payments",
    "ext_score_1",
    "ext_score_2",
    "contract_type",
    "education",
    "family_status",
    "housing",
    "target",
}


def test_synthetic_shape_and_columns() -> None:
    """Генератор возвращает ожидаемые колонки и размер."""
    df = generate_synthetic_credit(n_samples=500, seed=1)
    assert len(df) == 500
    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_synthetic_is_reproducible() -> None:
    """Одинаковый seed даёт одинаковый датасет."""
    a = generate_synthetic_credit(n_samples=200, seed=7)
    b = generate_synthetic_credit(n_samples=200, seed=7)
    assert a.equals(b)


def test_synthetic_has_missingness_and_imbalance() -> None:
    """Есть пропуски и умеренный дисбаланс target."""
    df = generate_synthetic_credit(n_samples=2000, default_rate=0.08, seed=42)
    assert df["ext_score_2"].isna().any()
    assert df["employment_years"].isna().any()
    rate = float(df["target"].mean())
    assert 0.02 < rate < 0.20
    assert set(df["target"].unique()) <= {0, 1}


def test_synthetic_numeric_ranges() -> None:
    """Числовые признаки лежат в разумных диапазонах."""
    df = generate_synthetic_credit(n_samples=1000, seed=3)
    assert df["age"].between(21, 69).all()
    assert (df["income"] > 0).all()
    assert (df["credit_amount"] > 0).all()
    assert np.isfinite(df["ext_score_1"]).all()
