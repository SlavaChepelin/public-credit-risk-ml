#!/usr/bin/env python3
"""CLI: построение EDA-графиков по application_train Home Credit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.eda import run_eda_report
from credit_risk.eda.report import DEFAULT_FIGURES_DIR
from credit_risk.data.home_credit import HOME_CREDIT_DIR


def main() -> None:
    """Запускает EDA и печатает пути к PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HOME_CREDIT_DIR,
        help="Каталог с CSV Home Credit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Куда сохранять PNG",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Опциональный лимит строк для быстрого прогона",
    )
    args = parser.parse_args()

    paths = run_eda_report(
        root=args.data_dir,
        output_dir=args.output_dir,
        nrows=args.nrows,
    )
    print(f"Saved {len(paths)} figures to {args.output_dir}")
    for path in paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
