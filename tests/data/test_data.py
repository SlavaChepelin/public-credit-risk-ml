"""Тесты загрузки и сохранения данных."""

from pathlib import Path

import pytest

from credit_risk.data import load_home_credit, load_or_generate
from credit_risk.data.synthetic import generate_synthetic_credit


def test_load_or_generate_writes_and_reads(tmp_path: Path) -> None:
    """При отсутствии файла генерирует и сохраняет датасет."""
    path = tmp_path / "sample.csv"
    df = load_or_generate(path=path, n_samples=120, seed=11)
    assert path.exists()
    assert len(df) == 120

    again = load_or_generate(path=path, n_samples=999, seed=0)
    assert len(again) == 120


def test_load_or_generate_reads_existing_csv(tmp_path: Path) -> None:
    """Читает уже сохранённый csv без перегенерации."""
    path = tmp_path / "ready.csv"
    original = generate_synthetic_credit(n_samples=60, seed=5)
    original.to_csv(path, index=False)
    loaded = load_or_generate(path=path, n_samples=10_000, seed=0)
    assert len(loaded) == 60
    assert "target" in loaded.columns


def test_load_home_credit_missing(tmp_path: Path) -> None:
    """Отсутствие application_train.csv даёт FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_home_credit(root=tmp_path)


def test_load_home_credit_reads_csv(tmp_path: Path) -> None:
    """При наличии CSV загрузчик возвращает датафрейм."""
    (tmp_path / "application_train.csv").write_text(
        "SK_ID_CURR,TARGET\n1,0\n2,1\n",
        encoding="utf-8",
    )
    df = load_home_credit(root=tmp_path)
    assert list(df.columns) == ["SK_ID_CURR", "TARGET"]
    assert len(df) == 2
