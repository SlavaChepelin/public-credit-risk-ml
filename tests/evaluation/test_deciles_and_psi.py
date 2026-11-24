"""Тесты децильного lift и PSI."""

import numpy as np

from credit_risk.evaluation.deciles import plot_decile_lift, score_decile_table
from credit_risk.eda.style import save_figure
from credit_risk.evaluation.stability import (
    plot_score_distributions,
    population_stability_index,
    psi_report,
)


def test_score_decile_table_monotonic_capture() -> None:
    """Capture растёт по децилям, top-1 lift > 1 при хорошем скоре."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    # скор коррелирует с y
    s = y * 0.7 + rng.random(500) * 0.3
    table = score_decile_table(y, s, n_bins=10)
    assert len(table) == 10
    assert table["capture"].is_monotonic_increasing
    assert table.iloc[0]["lift"] >= table.iloc[-1]["lift"]


def test_psi_identical_near_zero() -> None:
    """Одинаковые распределения → PSI ≈ 0."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    assert population_stability_index(x, x.copy()) < 0.05


def test_psi_report_status() -> None:
    """psi_report возвращает status."""
    rng = np.random.default_rng(2)
    train = rng.normal(loc=0.0, size=800)
    test = rng.normal(loc=1.5, size=800)
    report = psi_report(train, test)
    assert report["status"] in {"stable", "watch", "shift"}
    assert report["psi"] >= 0.0


def test_decile_and_psi_plots(tmp_path) -> None:
    """Графики децилей и score distribution сохраняются."""
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 20)
    s = np.linspace(0.05, 0.95, len(y))
    table = score_decile_table(y, s, n_bins=10)
    p1 = save_figure(plot_decile_lift(table), tmp_path / "decile.png")
    p2 = save_figure(
        plot_score_distributions(s[:100], s[100:]),
        tmp_path / "dist.png",
    )
    assert p1.exists() and p2.exists()
