"""Инженерия признаков для табличного кредитного скоринга."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "age",
    "income",
    "credit_amount",
    "annuity",
    "employment_years",
    "num_prev_loans",
    "num_late_payments",
    "ext_score_1",
    "ext_score_2",
    "credit_income_ratio",
    "annuity_income_ratio",
    "ext_score_mean",
    "ext_score_missing",
]

CATEGORICAL = [
    "contract_type",
    "education",
    "family_status",
    "housing",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные признаки к синтетическому датафрейму.

    Args:
        df: Исходный датафрейм.

    Returns:
        Копия датафрейма с новыми признаками.
    """
    out = df.copy()
    out["credit_income_ratio"] = out["credit_amount"] / out["income"].clip(lower=1.0)
    out["annuity_income_ratio"] = out["annuity"] / out["income"].clip(lower=1.0)
    out["ext_score_missing"] = out["ext_score_2"].isna().astype(float)
    out["ext_score_mean"] = out[["ext_score_1", "ext_score_2"]].mean(axis=1)
    return out


def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Определяет числовые и категориальные колонки.

    Args:
        X: Матрица признаков.

    Returns:
        Кортеж (numeric_cols, categorical_cols).
    """
    categorical = [
        c
        for c in X.columns
        if X[c].dtype == object
        or str(X[c].dtype) in {"string", "category", "boolean"}
        or pd.api.types.is_string_dtype(X[c])
    ]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def build_preprocessor(
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> ColumnTransformer:
    """Собирает препроцессор для числовых и категориальных признаков.

    Args:
        numeric: Числовые колонки. По умолчанию синтетический NUMERIC.
        categorical: Категориальные колонки. По умолчанию CATEGORICAL.

    Returns:
        ColumnTransformer с импутацией, масштабированием и one-hot.
    """
    numeric = NUMERIC if numeric is None else numeric
    categorical = CATEGORICAL if categorical is None else categorical

    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers)


def build_preprocessor_for_frame(X: pd.DataFrame) -> ColumnTransformer:
    """Строит препроцессор по фактическим колонкам X.

    Args:
        X: Матрица признаков.

    Returns:
        ColumnTransformer под типы колонок X.
    """
    numeric, categorical = infer_column_types(X)
    return build_preprocessor(numeric=numeric, categorical=categorical)


def feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Строит матрицу признаков и опциональный таргет (синтетика).

    Args:
        df: Датафрейм заявок (с target или без).

    Returns:
        Кортеж (X, y), где y равен None, если колонки target нет.
    """
    fe = engineer(df)
    cols = NUMERIC + CATEGORICAL
    X = fe[cols]
    y = fe["target"] if "target" in fe.columns else None
    return X, y
