"""Тест build_home_credit_features с app_ids на фикстуре."""

from pathlib import Path

import pandas as pd

from credit_risk.features.home_credit import build_home_credit_features
from credit_risk.data.home_credit import DAYS_EMPLOYED_SENTINEL
from tests.features.fixtures import _write_mini_home_credit


def test_build_with_app_ids_filter(tmp_path: Path) -> None:
    """app_ids оставляет только выбранные заявки."""
    _write_mini_home_credit(tmp_path)
    features = build_home_credit_features(root=tmp_path, app_ids=[2])
    assert len(features) == 1
    assert int(features.iloc[0]["SK_ID_CURR"]) == 2
