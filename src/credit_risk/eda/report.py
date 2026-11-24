"""Сборка полного набора EDA-графиков Home Credit."""

from __future__ import annotations

import pandas as pd

from credit_risk.eda.categoricals import plot_categorical_default_rates
from credit_risk.eda.correlations import plot_target_correlations
from credit_risk.eda.missingness import plot_missing_top
from credit_risk.eda.numerics import plot_numeric_by_target
from credit_risk.eda.style import configure_matplotlib, save_figure
from credit_risk.eda.target import plot_target_balance
from credit_risk.data.home_credit import (
    add_derived_application_features,
    load_application_train,
)

from credit_risk.paths import FIGURES

DEFAULT_FIGURES_DIR = FIGURES / "home_credit"

NUMERIC_FOR_DIST = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "age_years",
    "employment_years",
    "credit_income_ratio",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

CATEGORICAL_FOR_RATES = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
]


def prepare_application_frame(
    root: Path | None = None,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Загружает application_train и добавляет производные признаки.

    Args:
        root: Каталог Home Credit.
        nrows: Опциональный лимит строк.

    Returns:
        Датафрейм, готовый к построению графиков.
    """
    df = load_application_train(root=root, nrows=nrows)
    return add_derived_application_features(df)


def run_eda_report(
    *,
    root: Path | None = None,
    output_dir: Path | None = None,
    nrows: int | None = None,
) -> list[Path]:
    """Строит и сохраняет набор EDA-графиков.

    Args:
        root: Каталог Home Credit.
        output_dir: Куда сохранять PNG.
        nrows: Опциональный лимит строк для быстрых прогонов.

    Returns:
        Список путей сохранённых файлов.
    """
    configure_matplotlib()
    out_dir = output_dir or DEFAULT_FIGURES_DIR
    df = prepare_application_frame(root=root, nrows=nrows)

    jobs = [
        ("01_target_balance.png", plot_target_balance(df)),
        ("02_missing_top.png", plot_missing_top(df, top_n=30)),
        (
            "03_numeric_by_target.png",
            plot_numeric_by_target(df, NUMERIC_FOR_DIST),
        ),
        (
            "04_categorical_default_rates.png",
            plot_categorical_default_rates(df, CATEGORICAL_FOR_RATES),
        ),
        (
            "05_corr_with_target.png",
            plot_target_correlations(df, top_n=25),
        ),
    ]

    saved: list[Path] = []
    for name, fig in jobs:
        saved.append(save_figure(fig, out_dir / name))
    return saved
