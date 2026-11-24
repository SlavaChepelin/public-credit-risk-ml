"""Загрузка и сохранение датасетов кредитного скоринга."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk.data.synthetic import generate_synthetic_credit

from credit_risk.paths import DATA_DIR


def _read_table(path: Path) -> pd.DataFrame:
    """Читает таблицу из parquet или csv.

    Args:
        path: Путь к файлу.

    Returns:
        Загруженный датафрейм.

    Raises:
        ValueError: Если расширение файла не поддерживается.
    """
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, parse_dates=["application_date"])
    raise ValueError(f"Unsupported data format: {path.suffix}. Expected .parquet or .csv.")


def _write_table(df: pd.DataFrame, path: Path) -> Path:
    """Сохраняет таблицу в parquet или csv.

    Для .parquet при ошибке записи делается fallback на .csv.

    Args:
        df: Датафрейм для сохранения.
        path: Целевой путь (.parquet или .csv).

    Returns:
        Фактический путь сохранённого файла.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return path
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def _resolve_existing_table(path: Path) -> Path | None:
    """Ищет существующий файл таблицы с учётом csv-fallback.

    Args:
        path: Запрошенный путь (.parquet или .csv).

    Returns:
        Путь к существующему файлу или None.
    """
    if path.exists():
        return path
    if path.suffix == ".parquet":
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return csv_path
    return None


def load_or_generate(
    path: Path | None = None,
    n_samples: int = 12_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Загружает датасет с диска или генерирует синтетический.

    Args:
        path: Путь к файлу. По умолчанию data/synthetic_credit.parquet.
        n_samples: Число заявок при генерации.
        seed: Зерно генератора при генерации.

    Returns:
        Датафрейм заявок с колонкой target.
    """
    path = path or (DATA_DIR / "synthetic_credit.parquet")
    existing = _resolve_existing_table(path)
    if existing is not None:
        return _read_table(existing)

    df = generate_synthetic_credit(n_samples=n_samples, seed=seed)
    _write_table(df, path)
    return df


def load_home_credit(root: Path | None = None) -> pd.DataFrame:
    """Загружает Home Credit application_train с диска.

    Args:
        root: Каталог с CSV Home Credit. По умолчанию data/home_credit.

    Returns:
        Датафрейм заявок Home Credit.

    Raises:
        FileNotFoundError: Если application_train.csv отсутствует.
    """
    from credit_risk.data.home_credit import load_application_train

    return load_application_train(root=root)
