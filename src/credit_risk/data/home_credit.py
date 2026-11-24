"""Пути и загрузка таблиц Home Credit Default Risk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk.paths import DATA_DIR

HOME_CREDIT_DIR = DATA_DIR / "home_credit"

APPLICATION_TRAIN = "application_train.csv"
APPLICATION_TEST = "application_test.csv"
BUREAU = "bureau.csv"
PREVIOUS_APPLICATION = "previous_application.csv"
INSTALLMENTS_PAYMENTS = "installments_payments.csv"
COLUMN_DESCRIPTION = "HomeCredit_columns_description.csv"

# Sentinel «без занятости» в DAYS_EMPLOYED у Home Credit.
DAYS_EMPLOYED_SENTINEL = 365243

# Proxy-время заявки (более отрицательные значения — более ранние события).
# В application_train нет календарной application_date.
PROXY_TIME_COL = "DAYS_ID_PUBLISH"

APPLICATION_FEATURE_COLS = [
    "SK_ID_CURR",
    "TARGET",
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
]


def home_credit_root(root: Path | None = None) -> Path:
    """Возвращает каталог с CSV Home Credit.

    Args:
        root: Явный путь. По умолчанию data/home_credit.

    Returns:
        Путь к каталогу данных.
    """
    return root or HOME_CREDIT_DIR


def require_file(root: Path, name: str) -> Path:
    """Проверяет наличие файла в каталоге Home Credit.

    Args:
        root: Каталог данных.
        name: Имя файла.

    Returns:
        Путь к существующему файлу.

    Raises:
        FileNotFoundError: Если файл отсутствует.
    """
    path = root / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}. Unpack the Kaggle zip into {root}."
        )
    return path


def _read_csv(
    root: Path | None,
    name: str,
    *,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Читает CSV из каталога Home Credit.

    Args:
        root: Каталог Home Credit.
        name: Имя файла.
        usecols: Подмножество колонок.
        nrows: Ограничение числа строк.

    Returns:
        Загруженный датафрейм.
    """
    path = require_file(home_credit_root(root), name)
    return pd.read_csv(path, usecols=usecols, nrows=nrows)


def load_application_train(
    root: Path | None = None,
    *,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Загружает application_train.csv.

    Args:
        root: Каталог Home Credit.
        usecols: Подмножество колонок.
        nrows: Ограничение числа строк для быстрых прогонов.

    Returns:
        Датафрейм заявок с TARGET.
    """
    return _read_csv(root, APPLICATION_TRAIN, usecols=usecols, nrows=nrows)


def load_application_test(
    root: Path | None = None,
    *,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Загружает application_test.csv.

    Args:
        root: Каталог Home Credit.
        usecols: Подмножество колонок.
        nrows: Ограничение числа строк.

    Returns:
        Датафрейм тестовых заявок без TARGET.
    """
    return _read_csv(root, APPLICATION_TEST, usecols=usecols, nrows=nrows)


def load_bureau(
    root: Path | None = None,
    *,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Загружает bureau.csv.

    Args:
        root: Каталог Home Credit.
        usecols: Подмножество колонок.
        nrows: Ограничение числа строк.

    Returns:
        Датафрейм кредитных бюро-историй.
    """
    return _read_csv(root, BUREAU, usecols=usecols, nrows=nrows)


def load_previous_application(
    root: Path | None = None,
    *,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Загружает previous_application.csv.

    Args:
        root: Каталог Home Credit.
        usecols: Подмножество колонок.
        nrows: Ограничение числа строк.

    Returns:
        Датафрейм предыдущих заявок.
    """
    return _read_csv(root, PREVIOUS_APPLICATION, usecols=usecols, nrows=nrows)


def load_installments_payments(
    root: Path | None = None,
    *,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Загружает installments_payments.csv.

    Args:
        root: Каталог Home Credit.
        usecols: Подмножество колонок.
        nrows: Ограничение числа строк.

    Returns:
        Датафрейм платежей по рассрочкам.
    """
    return _read_csv(root, INSTALLMENTS_PAYMENTS, usecols=usecols, nrows=nrows)


def load_column_descriptions(root: Path | None = None) -> pd.DataFrame:
    """Загружает описание колонок Home Credit.

    Args:
        root: Каталог Home Credit.

    Returns:
        Датафрейм с описаниями признаков.
    """
    path = require_file(home_credit_root(root), COLUMN_DESCRIPTION)
    return pd.read_csv(path, encoding="latin-1")


def add_derived_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные признаки заявки.

    Args:
        df: application_train или application_test.

    Returns:
        Копия датафрейма с age_years, employment_years и ratios.
    """
    out = df.copy()
    if "DAYS_BIRTH" in out.columns:
        out["age_years"] = (-out["DAYS_BIRTH"] / 365.25).astype(float)
    if "DAYS_EMPLOYED" in out.columns:
        employed = out["DAYS_EMPLOYED"].mask(
            out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
        )
        out["employment_years"] = pd.to_numeric(-employed / 365.25, errors="coerce")
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"} <= set(out.columns):
        income = out["AMT_INCOME_TOTAL"].clip(lower=1.0)
        out["credit_income_ratio"] = out["AMT_CREDIT"] / income
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"} <= set(out.columns):
        income = out["AMT_INCOME_TOTAL"].clip(lower=1.0)
        out["annuity_income_ratio"] = out["AMT_ANNUITY"] / income
    if {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"} <= set(out.columns):
        ext = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
        out["ext_source_mean"] = ext.mean(axis=1)
        out["ext_source_missing_cnt"] = ext.isna().sum(axis=1).astype(float)
    return out
