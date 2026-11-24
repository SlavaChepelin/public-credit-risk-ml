"""Градиентный бустинг для табличного скоринга."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from credit_risk.features.synthetic import build_preprocessor_for_frame

BOOSTING_BACKEND_CATBOOST = "catboost"
BOOSTING_BACKEND_HIST_GB = "hist_gb"


def resolve_boosting_backend() -> str:
    """Определяет доступный backend бустинга.

    Returns:
        Имя backend: catboost или hist_gb.
    """
    try:
        import catboost  # noqa: F401

        return BOOSTING_BACKEND_CATBOOST
    except ImportError:
        return BOOSTING_BACKEND_HIST_GB


def _build_boosting_classifier(backend: str | None = None):
    """Создаёт классификатор бустинга.

    CatBoost — default, если пакет установлен; иначе HistGradientBoosting.

    Args:
        backend: Принудительный backend. По умолчанию auto.

    Returns:
        Необученный классификатор бустинга.
    """
    backend = backend or resolve_boosting_backend()
    if backend == BOOSTING_BACKEND_CATBOOST:
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            depth=6,
            learning_rate=0.05,
            iterations=400,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            auto_class_weights="Balanced",
            random_seed=42,
        )
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        class_weight="balanced",
        random_state=42,
    )


def fit_boosting(X, y, *, backend: str | None = None) -> Pipeline:
    """Обучает бустинг с препроцессором.

    Args:
        X: Матрица признаков.
        y: Бинарный таргет.
        backend: Опциональный backend (catboost | hist_gb).

    Returns:
        Обученный sklearn Pipeline.
    """
    chosen = backend or resolve_boosting_backend()
    pipe = Pipeline(
        steps=[
            ("pre", build_preprocessor_for_frame(X)),
            ("clf", _build_boosting_classifier(chosen)),
        ]
    )
    pipe.fit(X, y)
    pipe.boosting_backend_ = chosen
    return pipe
