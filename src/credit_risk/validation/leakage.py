"""Проверки на утечки во времени и по ключам сплита."""

from __future__ import annotations

import pandas as pd

from credit_risk.validation.split import assert_no_time_leakage


LEAKAGE_NOTES = {
    "DAYS_ID_PUBLISH": (
        "Proxy-время заявки: в Home Credit нет calendar application_date. "
        "Сплит по DAYS_ID_PUBLISH — приближение, не идеальный as-of."
    ),
    "bureau_days_*": (
        "Поля DAYS_* в bureau отрицательны относительно даты заявки в постановке "
        "соревнования; calendar as-of join без application timestamp невозможен."
    ),
    "SK_ID_CURR": (
        "Идентификатор заявки исключается из X, чтобы не учить id-меморизацию."
    ),
}


def check_split_leakage(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    time_col: str,
    id_col: str | None = "SK_ID_CURR",
) -> None:
    """Проверяет отсутствие пересечения по времени и id.

    Args:
        train: Train-часть.
        valid: Valid/test-часть.
        time_col: Колонка времени / proxy.
        id_col: Опциональный id для проверки пересечения ключей.

    Raises:
        AssertionError: При обнаружении утечки.
    """
    assert_no_time_leakage(train, valid, time_col)
    if id_col and id_col in train.columns and id_col in valid.columns:
        overlap = set(train[id_col]).intersection(set(valid[id_col]))
        assert not overlap, (
            f"ID overlap between train and valid: {len(overlap)} ids."
        )
