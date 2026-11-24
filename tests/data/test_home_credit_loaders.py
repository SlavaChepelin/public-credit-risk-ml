"""Тесты загрузчиков home_credit и xy helpers."""

from pathlib import Path

import pandas as pd
import pytest

from credit_risk.features.home_credit import (
    home_credit_xy,
    numeric_and_categorical_columns,
)
from credit_risk.data.home_credit import (
    HOME_CREDIT_DIR,
    add_derived_application_features,
    home_credit_root,
    load_application_test,
    load_column_descriptions,
    require_file,
)


def test_home_credit_root_default_and_override(tmp_path: Path) -> None:
    """home_credit_root уважает override."""
    assert home_credit_root() == HOME_CREDIT_DIR
    assert home_credit_root(tmp_path) == tmp_path


def test_require_file_missing(tmp_path: Path) -> None:
    """require_file кидает FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        require_file(tmp_path, "nope.csv")


def test_load_application_test_and_descriptions() -> None:
    """load_application_test / column descriptions на реальных данных."""
    if not (HOME_CREDIT_DIR / "application_test.csv").exists():
        pytest.skip("Home Credit data not unpacked")
    test = load_application_test(nrows=5)
    assert len(test) == 5
    assert "TARGET" not in test.columns
    desc = load_column_descriptions()
    assert len(desc) > 0


def test_add_derived_annuity_and_ext() -> None:
    """Производные annuity_income_ratio и ext_source_*."""
    df = add_derived_application_features(
        pd.DataFrame(
            {
                "DAYS_BIRTH": [-10000],
                "DAYS_EMPLOYED": [-500],
                "AMT_CREDIT": [100.0],
                "AMT_INCOME_TOTAL": [50.0],
                "AMT_ANNUITY": [10.0],
                "EXT_SOURCE_1": [0.1],
                "EXT_SOURCE_2": [None],
                "EXT_SOURCE_3": [0.3],
            }
        )
    )
    assert abs(df.loc[0, "annuity_income_ratio"] - 0.2) < 1e-9
    assert df.loc[0, "ext_source_missing_cnt"] == 1.0
    assert abs(df.loc[0, "ext_source_mean"] - 0.2) < 1e-9


def test_numeric_and_categorical_columns() -> None:
    """Деление колонок на num/cat."""
    X = pd.DataFrame(
        {
            "age_years": [20.0, 30.0],
            "CODE_GENDER": ["M", "F"],
            "AMT_CREDIT": [1.0, 2.0],
        }
    )
    num, cat = numeric_and_categorical_columns(X)
    assert "age_years" in num and "AMT_CREDIT" in num
    assert "CODE_GENDER" in cat


def test_home_credit_xy_drop_cols() -> None:
    """home_credit_xy уважает drop_cols."""
    features = pd.DataFrame(
        {
            "SK_ID_CURR": [1],
            "TARGET": [0],
            "DAYS_ID_PUBLISH": [-10],
            "DAYS_BIRTH": [-1000],
            "DAYS_EMPLOYED": [-100],
            "age_years": [30.0],
            "CODE_GENDER": ["M"],
            "extra_drop": [1.0],
        }
    )
    X, y = home_credit_xy(features, drop_cols=["extra_drop"])
    assert "extra_drop" not in X.columns
    assert "CODE_GENDER" in X.columns
    assert len(y) == 1
