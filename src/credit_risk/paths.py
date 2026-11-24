"""Корневые пути репозитория (data, results, artifacts)."""

from __future__ import annotations

from pathlib import Path

# src/credit_risk/paths.py → parents[2] = корень репозитория.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"
ARTIFACTS = REPO_ROOT / "artifacts"
FIGURES = RESULTS / "figures"
