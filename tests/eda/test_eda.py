"""Тесты EDA-графиков на мини-датафрейме."""

from pathlib import Path

import pandas as pd

from credit_risk.eda.categoricals import default_rate_by_category, plot_categorical_default_rates
from credit_risk.eda.correlations import correlation_with_target, plot_target_correlations
from credit_risk.eda.missingness import missing_rate, plot_missing_top
from credit_risk.eda.numerics import plot_numeric_by_target
from credit_risk.eda.report import run_eda_report
from credit_risk.eda.style import save_figure
from credit_risk.eda.target import plot_target_balance
from credit_risk.data.home_credit import add_derived_application_features


def _toy_application(n: int = 200) -> pd.DataFrame:
    """Собирает мини-датафрейм в духе application_train."""
    rng_target = [0] * int(n * 0.9) + [1] * (n - int(n * 0.9))
    return pd.DataFrame(
        {
            "TARGET": rng_target,
            "AMT_INCOME_TOTAL": [100_000 + 500 * i for i in range(n)],
            "AMT_CREDIT": [200_000 + 700 * i for i in range(n)],
            "AMT_ANNUITY": [10_000 + 20 * i for i in range(n)],
            "AMT_GOODS_PRICE": [180_000 + 600 * i for i in range(n)],
            "DAYS_BIRTH": [-10_000 - 20 * i for i in range(n)],
            "DAYS_EMPLOYED": [-1_000 - 3 * i if i % 10 else 365243 for i in range(n)],
            "EXT_SOURCE_1": [0.1 + (i % 50) / 100 for i in range(n)],
            "EXT_SOURCE_2": [0.2 + (i % 40) / 100 for i in range(n)],
            "EXT_SOURCE_3": [None if i % 7 == 0 else 0.3 + (i % 30) / 100 for i in range(n)],
            "NAME_CONTRACT_TYPE": ["Cash loans" if i % 2 == 0 else "Revolving loans" for i in range(n)],
            "CODE_GENDER": ["M" if i % 3 else "F" for i in range(n)],
            "NAME_EDUCATION_TYPE": ["Secondary" if i % 2 else "Higher" for i in range(n)],
            "NAME_FAMILY_STATUS": ["Married" if i % 2 else "Single" for i in range(n)],
            "NAME_HOUSING_TYPE": ["House / apartment"] * n,
            "NAME_INCOME_TYPE": ["Working" if i % 4 else "Pensioner" for i in range(n)],
            "OCCUPATION_TYPE": ["Laborers" if i % 5 else None for i in range(n)],
        }
    )


def test_add_derived_features() -> None:
    """Производные age/employment/ratio появляются корректно."""
    df = add_derived_application_features(_toy_application(20))
    assert "age_years" in df.columns
    assert "employment_years" in df.columns
    assert "credit_income_ratio" in df.columns
    assert df["age_years"].between(0, 100).all()
    assert df["employment_years"].isna().any()


def test_plot_functions_return_figures(tmp_path: Path) -> None:
    """Базовые plot-* функции сохраняют PNG."""
    df = add_derived_application_features(_toy_application())
    paths = [
        save_figure(plot_target_balance(df), tmp_path / "target.png"),
        save_figure(plot_missing_top(df, top_n=5), tmp_path / "miss.png"),
        save_figure(
            plot_numeric_by_target(df, ["AMT_CREDIT", "age_years"]),
            tmp_path / "num.png",
        ),
        save_figure(
            plot_categorical_default_rates(
                df,
                ["CODE_GENDER", "NAME_CONTRACT_TYPE"],
                min_count=1,
            ),
            tmp_path / "cat.png",
        ),
        save_figure(plot_target_correlations(df, top_n=5), tmp_path / "corr.png"),
    ]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_helpers_shapes() -> None:
    """Вспомогательные таблицы EDA имеют ожидаемую форму."""
    df = _toy_application()
    rates = missing_rate(df)
    assert rates.iloc[0] >= 0
    corr = correlation_with_target(df, top_n=3)
    assert len(corr) <= 3
    cats = default_rate_by_category(df, "CODE_GENDER", min_count=1)
    assert {"CODE_GENDER", "count", "default_rate"} <= set(cats.columns)


def test_run_eda_report_with_local_csv(tmp_path: Path) -> None:
    """run_eda_report пишет полный набор графиков из локального CSV."""
    data_dir = tmp_path / "home_credit"
    data_dir.mkdir()
    _toy_application(150).to_csv(data_dir / "application_train.csv", index=False)
    out = tmp_path / "figures"
    paths = run_eda_report(root=data_dir, output_dir=out)
    assert len(paths) == 5
    assert all(p.exists() for p in paths)
