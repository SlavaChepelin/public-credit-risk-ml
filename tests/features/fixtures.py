"""Общие фикстуры CSV Home Credit для тестов features."""

from pathlib import Path

import pandas as pd

from credit_risk.data.home_credit import DAYS_EMPLOYED_SENTINEL


def _write_mini_home_credit(root: Path) -> None:
    """Пишет мини-фикстуры CSV Home Credit."""
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans"],
            "CODE_GENDER": ["M", "F"],
            "FLAG_OWN_CAR": ["Y", "N"],
            "FLAG_OWN_REALTY": ["Y", "Y"],
            "CNT_CHILDREN": [0, 1],
            "AMT_INCOME_TOTAL": [100000.0, 200000.0],
            "AMT_CREDIT": [50000.0, 150000.0],
            "AMT_ANNUITY": [5000.0, 12000.0],
            "AMT_GOODS_PRICE": [45000.0, 140000.0],
            "NAME_INCOME_TYPE": ["Working", "Working"],
            "NAME_EDUCATION_TYPE": ["Secondary", "Higher"],
            "NAME_FAMILY_STATUS": ["Married", "Single"],
            "NAME_HOUSING_TYPE": ["House / apartment", "House / apartment"],
            "DAYS_BIRTH": [-10000, -15000],
            "DAYS_EMPLOYED": [-1000, DAYS_EMPLOYED_SENTINEL],
            "DAYS_REGISTRATION": [-2000.0, -3000.0],
            "DAYS_ID_PUBLISH": [-500, -800],
            "OWN_CAR_AGE": [3.0, None],
            "OCCUPATION_TYPE": ["Laborers", None],
            "CNT_FAM_MEMBERS": [2.0, 1.0],
            "REGION_RATING_CLIENT": [2, 1],
            "EXT_SOURCE_1": [0.4, None],
            "EXT_SOURCE_2": [0.5, 0.2],
            "EXT_SOURCE_3": [0.6, 0.1],
            "OBS_30_CNT_SOCIAL_CIRCLE": [0.0, 1.0],
            "DEF_30_CNT_SOCIAL_CIRCLE": [0.0, 0.0],
            "DAYS_LAST_PHONE_CHANGE": [-100.0, -200.0],
        }
    ).to_csv(root / "application_train.csv", index=False)

    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [10, 11, 12],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "AMT_CREDIT_SUM": [1000.0, 2000.0, 3000.0],
            "AMT_CREDIT_SUM_DEBT": [100.0, 0.0, 500.0],
            "DAYS_CREDIT": [-100, -200, -50],
            "CREDIT_DAY_OVERDUE": [0, 0, 2],
        }
    ).to_csv(root / "bureau.csv", index=False)

    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 2],
            "SK_ID_PREV": [100, 101, 102],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
            "AMT_CREDIT": [10000.0, 20000.0, 15000.0],
            "AMT_ANNUITY": [1000.0, 2000.0, 1500.0],
            "DAYS_DECISION": [-30, -40, -10],
        }
    ).to_csv(root / "previous_application.csv", index=False)

    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_PREV": [100, 100, 101],
            "DAYS_INSTALMENT": [-10.0, -20.0, -5.0],
            "DAYS_ENTRY_PAYMENT": [-8.0, -25.0, -1.0],
            "AMT_INSTALMENT": [100.0, 100.0, 200.0],
            "AMT_PAYMENT": [100.0, 80.0, 200.0],
        }
    ).to_csv(root / "installments_payments.csv", index=False)

