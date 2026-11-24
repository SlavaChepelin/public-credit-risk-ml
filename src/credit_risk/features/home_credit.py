"""Инженерия признаков Home Credit: application и агрегаты связанных таблиц."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.data.home_credit import (
    APPLICATION_FEATURE_COLS,
    PROXY_TIME_COL,
    add_derived_application_features,
    load_application_train,
    load_bureau,
    load_installments_payments,
    load_previous_application,
)

# В Home Credit поля DAYS_* уже относительно даты заявки (отрицательные дни).
# Календарного timestamp заявки нет, поэтому строгий as-of join недоступен.


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Строит агрегаты bureau на уровне SK_ID_CURR.

    Args:
        bureau: Таблица bureau.

    Returns:
        Датафрейм агрегатов с префиксом bureau_.
    """
    frame = bureau.copy()
    frame["bureau_is_active"] = (frame["CREDIT_ACTIVE"] == "Active").astype(float)

    grouped = frame.groupby("SK_ID_CURR").agg(
        bureau_count=("SK_ID_BUREAU", "count"),
        bureau_amt_credit_sum=("AMT_CREDIT_SUM", "sum"),
        bureau_amt_credit_mean=("AMT_CREDIT_SUM", "mean"),
        bureau_amt_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_amt_debt_mean=("AMT_CREDIT_SUM_DEBT", "mean"),
        bureau_active_count=("bureau_is_active", "sum"),
        bureau_days_credit_mean=("DAYS_CREDIT", "mean"),
        bureau_days_credit_max=("DAYS_CREDIT", "max"),
        bureau_overdue_max=("CREDIT_DAY_OVERDUE", "max"),
    )
    return grouped.reset_index()


def aggregate_previous_application(previous: pd.DataFrame) -> pd.DataFrame:
    """Строит агрегаты previous_application на уровне SK_ID_CURR.

    Args:
        previous: Таблица previous_application.

    Returns:
        Датафрейм агрегатов с префиксом prev_.
    """
    frame = previous.copy()
    frame["prev_approved"] = (frame["NAME_CONTRACT_STATUS"] == "Approved").astype(float)
    frame["prev_refused"] = (frame["NAME_CONTRACT_STATUS"] == "Refused").astype(float)

    grouped = frame.groupby("SK_ID_CURR").agg(
        prev_count=("SK_ID_PREV", "count"),
        prev_approved_rate=("prev_approved", "mean"),
        prev_refused_rate=("prev_refused", "mean"),
        prev_amt_credit_mean=("AMT_CREDIT", "mean"),
        prev_amt_annuity_mean=("AMT_ANNUITY", "mean"),
        prev_days_decision_mean=("DAYS_DECISION", "mean"),
    )
    return grouped.reset_index()


def aggregate_installments(installments: pd.DataFrame) -> pd.DataFrame:
    """Строит агрегаты installments_payments на уровне SK_ID_CURR.

    Args:
        installments: Таблица installments_payments.

    Returns:
        Датафрейм агрегатов с префиксом inst_.
    """
    frame = installments.copy()
    frame["inst_payment_diff"] = frame["AMT_PAYMENT"] - frame["AMT_INSTALMENT"]
    # NaN в датах ≠ «вовремя»: иначе late_rate занижается.
    delay = frame["DAYS_ENTRY_PAYMENT"] - frame["DAYS_INSTALMENT"]
    frame["inst_is_late"] = np.where(
        frame["DAYS_ENTRY_PAYMENT"].isna() | frame["DAYS_INSTALMENT"].isna(),
        np.nan,
        (delay > 0).astype(float),
    )

    grouped = frame.groupby("SK_ID_CURR").agg(
        inst_count=("SK_ID_PREV", "count"),
        inst_payment_diff_mean=("inst_payment_diff", "mean"),
        inst_payment_diff_sum=("inst_payment_diff", "sum"),
        inst_late_count=("inst_is_late", "sum"),
        inst_late_rate=("inst_is_late", "mean"),
        inst_amt_payment_mean=("AMT_PAYMENT", "mean"),
        inst_amt_instalment_mean=("AMT_INSTALMENT", "mean"),
    )
    return grouped.reset_index()


def _select_application_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет поддерживаемый набор колонок application.

    Args:
        df: Сырой application_train.

    Returns:
        Датафрейм с пересечением доступных колонок.
    """
    cols = [c for c in APPLICATION_FEATURE_COLS if c in df.columns]
    return df[cols].copy()


def build_home_credit_features(
    root: Path | None = None,
    *,
    nrows: int | None = None,
    app_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Собирает train-ready таблицу Home Credit с агрегатами.

    Args:
        root: Каталог CSV Home Credit.
        nrows: Опциональный лимит строк application_train.
        app_ids: Если задан, оставляет только эти SK_ID_CURR
            (удобно для тестов и subsample).

    Returns:
        Датафрейм с TARGET, proxy-временем и агрегированными фичами.
        SK_ID_CURR сохраняется для join/отладки; в модель его не подают.

    Raises:
        KeyError: Если в application отсутствует proxy-колонка времени.
    """
    app = load_application_train(root=root, nrows=nrows)
    app = _select_application_columns(app)
    if app_ids is not None:
        app = app[app["SK_ID_CURR"].isin(app_ids)].copy()

    ids = app["SK_ID_CURR"].tolist()
    bureau = load_bureau(root=root)
    previous = load_previous_application(root=root)
    installments = load_installments_payments(root=root)

    if app_ids is not None or nrows is not None:
        id_set = set(ids)
        bureau = bureau[bureau["SK_ID_CURR"].isin(id_set)]
        previous = previous[previous["SK_ID_CURR"].isin(id_set)]
        installments = installments[installments["SK_ID_CURR"].isin(id_set)]

    app = add_derived_application_features(app)
    features = (
        app.merge(aggregate_bureau(bureau), on="SK_ID_CURR", how="left")
        .merge(aggregate_previous_application(previous), on="SK_ID_CURR", how="left")
        .merge(aggregate_installments(installments), on="SK_ID_CURR", how="left")
    )

    # Отсутствие истории = нулевые count/sum/rate, не «дыра» в join.
    for col in features.columns:
        if col.startswith(("bureau_", "prev_", "inst_")) and (
            col.endswith(("_count", "_sum", "_rate"))
        ):
            features[col] = features[col].fillna(0.0)

    if PROXY_TIME_COL not in features.columns:
        raise KeyError(f"Missing proxy time column: {PROXY_TIME_COL}.")

    return features


def home_credit_xy(
    features: pd.DataFrame,
    *,
    target_col: str = "TARGET",
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Отделяет матрицу признаков от таргета.

    Args:
        features: Результат build_home_credit_features.
        target_col: Имя таргета.
        drop_cols: Дополнительные колонки для исключения из X.

    Returns:
        Кортеж (X, y). SK_ID_CURR и proxy-время не входят в X по умолчанию.
    """
    exclude = {"SK_ID_CURR", target_col, PROXY_TIME_COL, "DAYS_BIRTH", "DAYS_EMPLOYED"}
    if drop_cols:
        exclude.update(drop_cols)
    y = features[target_col]
    X = features.drop(columns=[c for c in exclude if c in features.columns]).copy()
    # Категории как object + np.nan: SimpleImputer не любит pandas NA.
    for col in X.columns:
        if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_string_dtype(X[col]):
            X[col] = X[col].astype(object).where(X[col].notna(), other=np.nan)
    return X, y


def numeric_and_categorical_columns(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Делит колонки X на числовые и категориальные.

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
