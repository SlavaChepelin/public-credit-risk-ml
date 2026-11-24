"""Регрессии data I/O."""

from pathlib import Path

import pandas as pd

from credit_risk.data import load_or_generate


def test_load_or_generate_reads_csv_fallback(tmp_path: Path, monkeypatch) -> None:
    """Если parquet не записался, следующий load читает .csv fallback."""
    path = tmp_path / "sample.parquet"

    def boom(*args, **kwargs):
        raise OSError("no parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    df1 = load_or_generate(path=path, n_samples=80, seed=3)
    assert (tmp_path / "sample.csv").exists()
    assert not path.exists()
    df2 = load_or_generate(path=path, n_samples=999, seed=0)
    assert len(df2) == len(df1) == 80
