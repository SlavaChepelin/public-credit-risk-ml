"""Доп. тесты serving frame / meta errors."""

from pathlib import Path

import pytest

from credit_risk.serving import build_scoring_frame, load_model_meta


def test_load_model_meta_missing(tmp_path: Path) -> None:
    """Отсутствие meta → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model_meta(tmp_path / "nope.json")


def test_build_scoring_frame_home_credit_fills_missing() -> None:
    """Для home_credit недостающие колонки дополняются NaN."""
    X = build_scoring_frame(
        {"EXT_SOURCE_2": 0.3, "age_years": 40},
        dataset="home_credit",
        feature_columns=["EXT_SOURCE_2", "age_years", "bureau_count"],
    )
    assert list(X.columns) == ["EXT_SOURCE_2", "age_years", "bureau_count"]
    assert X.loc[0, "bureau_count"] != X.loc[0, "bureau_count"]  # NaN
