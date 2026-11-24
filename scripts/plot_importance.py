#!/usr/bin/env python3
"""Строит top-15 feature importance для champion.joblib."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.eda.style import configure_matplotlib, save_figure
from credit_risk.evaluation.explain import extract_feature_importance, plot_feature_importance


def main() -> None:
    """CLI для importance plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "champion.joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "figures" / "importance_top15.png",
    )
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    configure_matplotlib()
    model = joblib.load(args.model)
    importance = extract_feature_importance(model, top_n=args.top_n)
    path = save_figure(
        plot_feature_importance(importance),
        args.output,
    )
    print(importance.to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
