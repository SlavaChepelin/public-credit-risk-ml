#!/usr/bin/env python3
"""CLI-вход для запуска пайплайна обучения."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.pipeline import run


def main() -> None:
    """Запускает обучение и печатает сводку метрик."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "home_credit"],
        default="synthetic",
        help="Источник данных",
    )
    parser.add_argument("--n-samples", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Лимит строк Home Credit (для быстрых прогонов)",
    )
    parser.add_argument("--torch-epochs", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--no-oof", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true")
    args = parser.parse_args()

    report = run(
        dataset=args.dataset,
        n_samples=args.n_samples,
        seed=args.seed,
        nrows=args.nrows,
        torch_epochs=args.torch_epochs,
        n_folds=args.n_folds,
        with_oof=not args.no_oof,
        with_postprocess=not args.no_postprocess,
    )
    print("Dataset:", report["dataset"])
    print("Best model:", report["best_model"])
    print("Boosting backend:", report.get("boosting_backend"))
    for name, block in report["models"].items():
        t = block["test"]
        print(
            f"  {name:10s}  test AUC={t['roc_auc']:.4f}  "
            f"Gini={t['gini']:.4f}  KS={t['ks']:.4f}  PR-AUC={t['pr_auc']:.4f}"
        )
    if "oof" in report:
        gini = report["oof"]["summary"]["gini"]
        print(
            f"OOF boosting Gini: {gini['mean']:.4f} ± {gini['std']:.4f} "
            f"(n_folds={report['oof']['n_folds']})"
        )


if __name__ == "__main__":
    main()
