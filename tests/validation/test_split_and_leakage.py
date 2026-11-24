"""Тесты temporal folds и контроля leakage."""

import pandas as pd
import pytest

from credit_risk.validation.leakage import check_split_leakage
from credit_risk.validation.split import temporal_folds, time_based_split
from credit_risk.data.synthetic import generate_synthetic_credit


def test_temporal_folds_no_overlap() -> None:
    """Valid-сегменты фолдов не пересекаются по времени."""
    df = generate_synthetic_credit(n_samples=500, seed=0)
    folds = temporal_folds(df, time_col="application_date", n_folds=3)
    assert len(folds) == 3

    valid_ranges = []
    for train, valid in folds:
        assert train["application_date"].max() <= valid["application_date"].min()
        valid_ranges.append(
            (valid["application_date"].min(), valid["application_date"].max())
        )

    for i in range(len(valid_ranges) - 1):
        assert valid_ranges[i][1] <= valid_ranges[i + 1][0]


def test_check_split_leakage_detects_overlap() -> None:
    """Пересечение id ловится assertion-ом."""
    train = pd.DataFrame({"t": [1, 2], "SK_ID_CURR": [1, 2]})
    valid = pd.DataFrame({"t": [3, 4], "SK_ID_CURR": [2, 3]})
    with pytest.raises(AssertionError, match="ID overlap"):
        check_split_leakage(train, valid, time_col="t")


def test_time_based_split_home_credit_proxy() -> None:
    """Proxy-колонка DAYS_ID_PUBLISH работает в time_based_split."""
    df = pd.DataFrame(
        {
            "DAYS_ID_PUBLISH": [-10, -30, -20, -5, -40],
            "TARGET": [0, 1, 0, 1, 0],
            "SK_ID_CURR": [1, 2, 3, 4, 5],
        }
    )
    train, valid, test = time_based_split(df, time_col="DAYS_ID_PUBLISH", train_ratio=0.4, valid_ratio=0.2)
    assert train["DAYS_ID_PUBLISH"].max() <= valid["DAYS_ID_PUBLISH"].min()
    assert valid["DAYS_ID_PUBLISH"].max() <= test["DAYS_ID_PUBLISH"].min()
