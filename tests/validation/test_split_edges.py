"""Граничные случаи temporal_folds / split."""

import pytest

from credit_risk.validation.split import temporal_folds, time_based_split
from credit_risk.data.synthetic import generate_synthetic_credit


def test_temporal_folds_rejects_small_n_folds() -> None:
    """n_folds < 2 → ValueError."""
    df = generate_synthetic_credit(n_samples=100, seed=0)
    with pytest.raises(ValueError, match="n_folds"):
        temporal_folds(df, time_col="application_date", n_folds=1)


def test_temporal_folds_rejects_too_few_rows() -> None:
    """Слишком мало строк → ValueError."""
    df = generate_synthetic_credit(n_samples=20, seed=0)
    with pytest.raises(ValueError, match="Not enough rows"):
        temporal_folds(df, time_col="application_date", n_folds=3)


def test_time_based_split_sizes() -> None:
    """Размеры сплита примерно соответствуют долям."""
    df = generate_synthetic_credit(n_samples=1000, seed=1)
    train, valid, test = time_based_split(
        df,
        train_ratio=0.6,
        valid_ratio=0.2,
    )
    assert abs(len(train) / 1000 - 0.6) < 0.02
    assert abs(len(valid) / 1000 - 0.2) < 0.02
    assert len(train) + len(valid) + len(test) == 1000
