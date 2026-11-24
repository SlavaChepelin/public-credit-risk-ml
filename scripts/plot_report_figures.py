#!/usr/bin/env python3
"""Сводные графики для описания результатов (из results/metrics.json)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.eda.style import configure_matplotlib, save_figure
from credit_risk.paths import FIGURES, RESULTS


def _load_metrics(path: Path) -> dict:
    """Читает metrics.json.

    Args:
        path: Путь к отчёту.

    Returns:
        Словарь метрик.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def plot_model_comparison(report: dict) -> plt.Figure:
    """Сравнивает модели по test Gini / ROC-AUC / KS / PR-AUC.

    Args:
        report: Отчёт пайплайна.

    Returns:
        Matplotlib-фигура.
    """
    models = report["models"]
    names = list(models.keys())
    metrics = ["gini", "roc_auc", "ks", "pr_auc"]
    labels = ["Gini", "ROC-AUC", "KS", "PR-AUC"]
    best = report.get("best_model")

    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [models[n]["test"][metric] for n in names]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
        for bar, name in zip(bars, names):
            if name == best:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.6)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Test score")
    ax.set_title(
        f"Сравнение моделей на test (champion={best}, selection=max valid Gini)"
    )
    ax.legend(loc="lower right", ncol=2)
    ax.set_ylim(0.0, max(0.85, ax.get_ylim()[1]))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_oof_stability(report: dict) -> plt.Figure | None:
    """Рисует Gini по temporal OOF-фолдам и mean±std.

    Args:
        report: Отчёт пайплайна.

    Returns:
        Matplotlib-фигура или None, если блока oof нет.
    """
    oof = report.get("oof")
    if not oof:
        return None
    folds = oof["folds"]
    ginies = [f["gini"] for f in folds]
    mean = float(oof["summary"]["gini"]["mean"])
    std = float(oof["summary"]["gini"]["std"])

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(1, len(ginies) + 1)
    ax.bar(xs, ginies, color="#4C78A8", label="fold Gini")
    ax.axhline(mean, color="#E45756", linewidth=1.8, label=f"mean={mean:.3f}")
    ax.fill_between(
        [0.5, len(ginies) + 0.5],
        mean - std,
        mean + std,
        color="#E45756",
        alpha=0.15,
        label=f"±std ({std:.3f})",
    )
    ax.set_xticks(xs)
    ax.set_xlabel("Temporal fold")
    ax.set_ylabel("Gini")
    ax.set_title(f"OOF boosting stability ({oof.get('n_folds')} expanding-window folds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_operating_point(report: dict) -> plt.Figure | None:
    """Визуализирует operating point на test после калибровки.

    Args:
        report: Отчёт пайплайна.

    Returns:
        Matplotlib-фигура или None.
    """
    post = report.get("postprocess") or {}
    point = post.get("operating_point_test")
    if not point:
        return None

    keys = ["precision", "recall", "fpr", "approval_rate", "predicted_default_rate"]
    labels = ["Precision", "Recall", "FPR", "Approval", "Pred. default"]
    vals = [float(point[k]) for k in keys]
    thr = float(point.get("threshold", post.get("threshold", {}).get("threshold", 0.0)))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#4C78A8", "#54A24B", "#E45756", "#F58518", "#B279A2"]
    bars = ax.bar(labels, vals, color=colors)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title(f"Operating point on test (threshold={thr:.3f}, strategy=max KS on valid)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_decile_capture(report: dict) -> plt.Figure | None:
    """Рисует накопленный capture дефолтов по децилям риска.

    Args:
        report: Отчёт пайплайна.

    Returns:
        Matplotlib-фигура или None.
    """
    post = report.get("postprocess") or {}
    rows = post.get("deciles_test")
    if not rows:
        return None

    deciles = [int(r["decile"]) for r in rows]
    capture = [float(r["capture"]) for r in rows]
    lift = [float(r["lift"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.plot(deciles, capture, marker="o", color="#E45756", label="capture (cumsum)")
    ax1.plot(
        [1, 10],
        [0.1, 1.0],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="random baseline",
    )
    ax1.set_xlabel("Decile (1 = riskiest)")
    ax1.set_ylabel("Capture of defaults")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(deciles)

    ax2 = ax1.twinx()
    ax2.bar(deciles, lift, alpha=0.25, color="#4C78A8", label="lift")
    ax2.set_ylabel("Lift")
    ax2.axhline(1.0, color="#4C78A8", linestyle=":", linewidth=1)

    ax1.set_title("Test risk ranking: capture curve + lift by decile")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    fig.tight_layout()
    return fig


def main() -> None:
    """Строит сводные PNG для README / презентации."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=RESULTS / "metrics.json",
        help="Путь к metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES,
        help="Каталог для PNG",
    )
    args = parser.parse_args()

    configure_matplotlib()
    report = _load_metrics(args.metrics)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    paths.append(save_figure(plot_model_comparison(report), out / "model_comparison.png"))

    oof_fig = plot_oof_stability(report)
    if oof_fig is not None:
        paths.append(save_figure(oof_fig, out / "oof_stability.png"))

    op_fig = plot_operating_point(report)
    if op_fig is not None:
        paths.append(save_figure(op_fig, out / "operating_point.png"))

    cap_fig = plot_decile_capture(report)
    if cap_fig is not None:
        paths.append(save_figure(cap_fig, out / "decile_capture.png"))

    print(f"Wrote {len(paths)} figures to {out}")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
