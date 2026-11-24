"""Тесты агрегатов и сборки фичей Home Credit."""

from pathlib import Path

import pandas as pd

from credit_risk.features.home_credit import (
    aggregate_bureau,
    aggregate_installments,
    aggregate_previous_application,
    build_home_credit_features,
    home_credit_xy,
)
from credit_risk.data.home_credit import (
    DAYS_EMPLOYED_SENTINEL,
    add_derived_application_features,
)

from tests.features.fixtures import _write_mini_home_credit


def test_aggregate_bureau_counts() -> None:
    """Агрегаты bureau считают count/active корректно."""
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1],
            "SK_ID_BUREAU": [1, 2],
            "CREDIT_ACTIVE": ["Active", "Closed"],
            "AMT_CREDIT_SUM": [10.0, 20.0],
            "AMT_CREDIT_SUM_DEBT": [1.0, 0.0],
            "DAYS_CREDIT": [-1, -2],
            "CREDIT_DAY_OVERDUE": [0, 0],
        }
    )
    out = aggregate_bureau(bureau)
    row = out.iloc[0]
    assert row["bureau_count"] == 2
    assert row["bureau_active_count"] == 1
    assert row["bureau_amt_credit_sum"] == 30.0


def test_aggregate_previous_and_installments() -> None:
    """Агрегаты previous/installments считают rate и late."""
    prev = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1],
            "SK_ID_PREV": [1, 2],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused"],
            "AMT_CREDIT": [1.0, 2.0],
            "AMT_ANNUITY": [0.5, 0.5],
            "DAYS_DECISION": [-1, -2],
        }
    )
    prev_agg = aggregate_previous_application(prev).iloc[0]
    assert prev_agg["prev_count"] == 2
    assert abs(prev_agg["prev_approved_rate"] - 0.5) < 1e-9

    inst = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1],
            "SK_ID_PREV": [1, 1, 1],
            "DAYS_INSTALMENT": [-10.0, -20.0, -30.0],
            "DAYS_ENTRY_PAYMENT": [-5.0, -25.0, None],
            "AMT_INSTALMENT": [10.0, 10.0, 10.0],
            "AMT_PAYMENT": [10.0, 8.0, 10.0],
        }
    )
    inst_agg = aggregate_installments(inst).iloc[0]
    # Одна просрочка, один вовремя, один NaN (не считается вовремя).
    assert inst_agg["inst_late_count"] == 1
    assert abs(inst_agg["inst_late_rate"] - 0.5) < 1e-9
    assert abs(inst_agg["inst_payment_diff_sum"] - (-2.0)) < 1e-9


def test_employment_sentinel() -> None:
    """Sentinel DAYS_EMPLOYED становится NaN в employment_years."""
    df = add_derived_application_features(
        pd.DataFrame(
            {
                "DAYS_BIRTH": [-10000],
                "DAYS_EMPLOYED": [DAYS_EMPLOYED_SENTINEL],
                "AMT_CREDIT": [1.0],
                "AMT_INCOME_TOTAL": [1.0],
            }
        )
    )
    assert pd.isna(df.loc[0, "employment_years"])


def test_build_home_credit_features_fixture(tmp_path: Path) -> None:
    """build_home_credit_features собирает TARGET и агрегаты на фикстуре."""
    _write_mini_home_credit(tmp_path)
    features = build_home_credit_features(root=tmp_path)
    assert len(features) == 2
    assert "TARGET" in features.columns
    assert "bureau_count" in features.columns
    assert "prev_count" in features.columns
    assert "inst_late_count" in features.columns

    X, y = home_credit_xy(features)
    assert "SK_ID_CURR" not in X.columns
    assert "TARGET" not in X.columns
    assert len(X) == len(y) == 2
