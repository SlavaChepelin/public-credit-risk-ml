#!/usr/bin/env python3
"""Сборка агрегатов bureau через DuckDB (SQL как в проде)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.data.home_credit import HOME_CREDIT_DIR, BUREAU, require_file


def build_bureau_aggregates_duckdb(
    data_dir: Path,
    sql_path: Path,
    output_path: Path,
) -> Path:
    """Считает агрегаты bureau SQL-ом и сохраняет parquet/csv.

    Args:
        data_dir: Каталог Home Credit CSV.
        sql_path: Путь к sql/bureau_aggregates.sql.
        output_path: Куда сохранить результат.

    Returns:
        Путь к сохранённому файлу.

    Raises:
        ImportError: Если duckdb не установлен.
    """
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "duckdb is required: pip install duckdb"
        ) from exc

    bureau_path = require_file(data_dir, BUREAU)
    sql = sql_path.read_text(encoding="utf-8")
    con = duckdb.connect()
    con.execute(
        f"CREATE OR REPLACE VIEW bureau AS SELECT * FROM read_csv_auto('{bureau_path}')"
    )
    frame = con.execute(sql).fetchdf()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        frame.to_csv(output_path, index=False)
    else:
        try:
            frame.to_parquet(output_path, index=False)
        except Exception:
            csv_path = output_path.with_suffix(".csv")
            frame.to_csv(csv_path, index=False)
            return csv_path
    return output_path


def main() -> None:
    """CLI для DuckDB-агрегатов bureau."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HOME_CREDIT_DIR)
    parser.add_argument(
        "--sql",
        type=Path,
        default=ROOT / "sql" / "bureau_aggregates.sql",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "bureau_aggregates.parquet",
    )
    args = parser.parse_args()
    path = build_bureau_aggregates_duckdb(args.data_dir, args.sql, args.output)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
