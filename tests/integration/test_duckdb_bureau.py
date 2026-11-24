"""Сверка pandas vs DuckDB агрегатов bureau."""

from pathlib import Path

import pandas as pd
import pytest

from credit_risk.features.home_credit import aggregate_bureau

duckdb = pytest.importorskip("duckdb")


def test_pandas_vs_duckdb_bureau(tmp_path: Path) -> None:
    """Агрегаты bureau совпадают с SQL на фикстуре."""
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [10, 11, 12],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "AMT_CREDIT_SUM": [100.0, 200.0, 300.0],
            "AMT_CREDIT_SUM_DEBT": [10.0, 0.0, 30.0],
            "DAYS_CREDIT": [-1, -2, -3],
            "CREDIT_DAY_OVERDUE": [0, 1, 0],
        }
    )
    csv_path = tmp_path / "bureau.csv"
    bureau.to_csv(csv_path, index=False)

    sql = (Path(__file__).resolve().parents[2] / "sql" / "bureau_aggregates.sql").read_text(
        encoding="utf-8"
    )
    con = duckdb.connect()
    con.execute(
        f"CREATE OR REPLACE VIEW bureau AS SELECT * FROM read_csv_auto('{csv_path}')"
    )
    sql_df = con.execute(sql).fetchdf().sort_values("SK_ID_CURR").reset_index(drop=True)
    pandas_df = aggregate_bureau(bureau).sort_values("SK_ID_CURR").reset_index(drop=True)

    for col in [
        "bureau_count",
        "bureau_amt_credit_sum",
        "bureau_active_count",
        "bureau_days_credit_max",
    ]:
        assert (
            (sql_df[col].to_numpy() - pandas_df[col].to_numpy()).max() < 1e-6
        )
