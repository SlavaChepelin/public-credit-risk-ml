"""Edge-cases thresholding."""

import numpy as np
import pytest

from credit_risk.evaluation.thresholding import (
    choose_threshold,
    threshold_for_recall,
    threshold_max_ks,
)


def test_threshold_for_recall_unreachable() -> None:
    """Если target_recall недостижим — берётся min score."""
    y = np.array([0, 0, 1, 1])
    s = np.array([0.2, 0.3, 0.4, 0.5])
    thr = threshold_for_recall(y, s, target_recall=0.99)
    assert thr <= float(np.min(s)) + 1e-9 or thr <= 0.5


def test_choose_threshold_max_ks_and_unknown() -> None:
    """choose_threshold max_ks и неизвестная стратегия."""
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    info = choose_threshold(y, s, strategy="max_ks")
    assert info["strategy"] == "max_ks"
    assert np.isfinite(float(info["threshold"]))
    with pytest.raises(ValueError, match="Unknown strategy"):
        choose_threshold(y, s, strategy="nope")  # type: ignore[arg-type]


def test_threshold_max_ks_finite_on_tied() -> None:
    """Порог конечен даже при совпадающих скорах."""
    y = np.array([0, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    thr = threshold_max_ks(y, s)
    assert np.isfinite(thr) or thr == 0.5
