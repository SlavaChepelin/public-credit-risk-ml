"""Регрессии serving meta / scoring frame."""

from pathlib import Path

from credit_risk.serving import build_scoring_frame, load_model_meta, write_model_meta


def test_build_scoring_frame_aligns_columns() -> None:
    """Serving frame выравнивает колонки под модель."""
    cols = ["age", "income", "credit_amount", "annuity", "employment_years",
            "num_prev_loans", "num_late_payments", "ext_score_1", "ext_score_2",
            "credit_income_ratio", "annuity_income_ratio", "ext_score_mean",
            "ext_score_missing", "contract_type", "education", "family_status",
            "housing"]
    X = build_scoring_frame(
        {
            "age": 30,
            "income": 1000,
            "credit_amount": 2000,
            "annuity": 100,
            "employment_years": 2,
            "num_prev_loans": 0,
            "num_late_payments": 0,
            "ext_score_1": 0.5,
            "ext_score_2": 0.4,
            "contract_type": "Cash loans",
            "education": "Secondary",
            "family_status": "Married",
            "housing": "House / apartment",
        },
        dataset="synthetic",
        feature_columns=cols,
    )
    assert list(X.columns) == cols


def test_model_meta_roundtrip(tmp_path: Path) -> None:
    """model_meta пишется и читается."""
    path = write_model_meta(
        tmp_path / "model_meta.json",
        dataset="home_credit",
        feature_columns=["EXT_SOURCE_2"],
        best_model="boosting",
        threshold=0.12,
        calibrated=True,
    )
    meta = load_model_meta(path)
    assert meta["dataset"] == "home_credit"
    assert meta["calibrated"] is True
