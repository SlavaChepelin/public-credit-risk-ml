"""Доп. тесты data IO."""

from pathlib import Path

import pandas as pd
import pytest

from credit_risk.data.io import _read_table, _write_table
from credit_risk.data.synthetic import generate_synthetic_credit


def test_read_table_unsupported_suffix(tmp_path: Path) -> None:
    """Неподдерживаемый формат → ValueError."""
    path = tmp_path / "x.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        _read_table(path)


def test_write_table_csv_direct(tmp_path: Path) -> None:
    """_write_table для .csv пишет csv."""
    df = generate_synthetic_credit(n_samples=30, seed=0)
    path = _write_table(df, tmp_path / "out.csv")
    assert path.suffix == ".csv"
    loaded = _read_table(path)
    assert len(loaded) == 30


def test_write_table_parquet_or_csv_fallback(tmp_path: Path) -> None:
    """_write_table parquet либо пишет parquet, либо fallback csv."""
    df = generate_synthetic_credit(n_samples=20, seed=1)
    path = _write_table(df, tmp_path / "out.parquet")
    assert path.exists()
    assert path.suffix in {".parquet", ".csv"}
