"""Генерация синтетического Home-Credit-подобного датасета."""

from __future__ import annotations

import numpy as np
import pandas as pd

_CONTRACT_TYPES = ("Cash loans", "Revolving loans")
_EDUCATION_LEVELS = ("Secondary", "Higher", "Incomplete higher", "Lower secondary")
_FAMILY_STATUSES = ("Married", "Single", "Civil marriage", "Separated")
_HOUSING_TYPES = ("House / apartment", "With parents", "Municipal", "Rented")


def _sample_application_dates(
    rng: np.random.Generator,
    n_samples: int,
    start: str = "2023-01-01",
    span_days: int = 730,
) -> np.ndarray:
    """Сэмплирует даты заявок на заданном горизонте.

    Args:
        rng: Генератор случайных чисел.
        n_samples: Число наблюдений.
        start: Начало периода заявок.
        span_days: Длина горизонта в днях.

    Returns:
        Массив дат заявок.
    """
    start_ts = np.datetime64(start)
    offsets = rng.integers(0, span_days, size=n_samples)
    return start_ts + offsets.astype("timedelta64[D]")


def _sample_numeric_features(
    rng: np.random.Generator,
    n_samples: int,
) -> dict[str, np.ndarray]:
    """Сэмплирует числовые признаки заявки.

    Args:
        rng: Генератор случайных чисел.
        n_samples: Число наблюдений.

    Returns:
        Словарь числовых признаков.
    """
    credit_amount = rng.lognormal(mean=11.2, sigma=0.6, size=n_samples)
    return {
        "age": rng.integers(21, 70, size=n_samples),
        "income": rng.lognormal(mean=10.5, sigma=0.55, size=n_samples),
        "credit_amount": credit_amount,
        "annuity": credit_amount * rng.uniform(0.02, 0.08, size=n_samples),
        "employment_years": np.clip(rng.normal(5, 4, size=n_samples), 0, 40),
        "num_prev_loans": rng.poisson(2.0, size=n_samples),
        "num_late_payments": rng.poisson(0.4, size=n_samples),
        "ext_score_1": rng.beta(2, 2, size=n_samples),
        "ext_score_2": rng.beta(2.5, 2, size=n_samples),
    }


def _sample_categorical_features(
    rng: np.random.Generator,
    n_samples: int,
) -> dict[str, np.ndarray]:
    """Сэмплирует категориальные признаки заявки.

    Args:
        rng: Генератор случайных чисел.
        n_samples: Число наблюдений.

    Returns:
        Словарь категориальных признаков.
    """
    return {
        "contract_type": rng.choice(
            _CONTRACT_TYPES,
            size=n_samples,
            p=[0.85, 0.15],
        ),
        "education": rng.choice(
            _EDUCATION_LEVELS,
            size=n_samples,
            p=[0.55, 0.30, 0.10, 0.05],
        ),
        "family_status": rng.choice(
            _FAMILY_STATUSES,
            size=n_samples,
            p=[0.55, 0.25, 0.12, 0.08],
        ),
        "housing": rng.choice(
            _HOUSING_TYPES,
            size=n_samples,
            p=[0.6, 0.2, 0.1, 0.1],
        ),
    }


def _default_logit(
    numeric: dict[str, np.ndarray],
    categorical: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Считает латентный логит риска дефолта.

    Args:
        numeric: Числовые признаки.
        categorical: Категориальные признаки.
        rng: Генератор случайных чисел для шума.

    Returns:
        Вектор логитов.
    """
    ratio = numeric["credit_amount"] / np.clip(numeric["income"], 1.0, None)
    emp = np.nan_to_num(numeric["employment_years"], nan=0.0)
    late = numeric["num_late_payments"]
    low_ext = (numeric["ext_score_1"] < 0.4).astype(float)

    return (
        -3.2
        + 1.1 * ratio
        + 0.35 * (ratio**2)
        + 0.04 * (numeric["age"] - 40)
        - 0.08 * emp
        + 0.35 * late
        + 0.20 * late * low_ext
        + 0.15 * numeric["num_prev_loans"]
        - 2.0 * numeric["ext_score_1"]
        - 1.5 * np.nan_to_num(numeric["ext_score_2"], nan=0.5)
        + 0.45 * np.sin(numeric["age"] / 8.0)
        + 0.25 * (categorical["education"] == "Lower secondary").astype(float)
        + 0.15 * (categorical["contract_type"] == "Revolving loans").astype(float)
        + rng.normal(0, 0.45, size=len(ratio))
    )


def _calibrate_default_probs(
    logit: np.ndarray,
    default_rate: float,
) -> np.ndarray:
    """Калибрует вероятности дефолта к целевой доле.

    Args:
        logit: Латентные логиты риска.
        default_rate: Целевая доля дефолтов.

    Returns:
        Вектор вероятностей в диапазоне [0, 1].
    """
    probs = 1.0 / (1.0 + np.exp(-logit))
    scale = default_rate / max(float(probs.mean()), 1e-6)
    return np.clip(probs * scale, 0.0, 1.0)


def _apply_missingness(
    df: pd.DataFrame,
    rng: np.random.Generator,
    ext_score_miss_rate: float = 0.12,
    employment_miss_rate: float = 0.05,
) -> pd.DataFrame:
    """Добавляет пропуски в выбранные колонки.

    Args:
        df: Датафрейм заявок.
        rng: Генератор случайных чисел.
        ext_score_miss_rate: Доля пропусков в ext_score_2.
        employment_miss_rate: Доля пропусков в employment_years.

    Returns:
        Датафрейм с добавленными пропусками.
    """
    out = df.copy()
    n = len(out)
    ext_idx = rng.choice(n, size=int(ext_score_miss_rate * n), replace=False)
    emp_idx = rng.choice(n, size=int(employment_miss_rate * n), replace=False)
    out.loc[ext_idx, "ext_score_2"] = np.nan
    out.loc[emp_idx, "employment_years"] = np.nan
    return out


def generate_synthetic_credit(
    n_samples: int = 12_000,
    default_rate: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """Генерирует синтетический датасет кредитных заявок.

    Датасет имитирует типичные свойства банковских табличных данных:
    дисбаланс классов, временную ось, пропуски и категориальные признаки.

    Args:
        n_samples: Число заявок.
        default_rate: Целевая доля дефолтов.
        seed: Зерно генератора случайных чисел.

    Returns:
        Датафрейм с признаками и колонкой target.
    """
    rng = np.random.default_rng(seed)
    numeric = _sample_numeric_features(rng, n_samples)
    categorical = _sample_categorical_features(rng, n_samples)
    probs = _calibrate_default_probs(
        _default_logit(numeric, categorical, rng),
        default_rate=default_rate,
    )

    df = pd.DataFrame(
        {
            "application_date": pd.to_datetime(
                _sample_application_dates(rng, n_samples)
            ),
            **numeric,
            **categorical,
            "target": rng.binomial(1, probs),
        }
    )
    return _apply_missingness(df, rng)
