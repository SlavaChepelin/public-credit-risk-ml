"""Калибровка вероятностей дефолта."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression


class _ScorePlatform(BaseEstimator, ClassifierMixin):
    """Калибрует вероятности по скору уже обученной базовой модели."""

    def __init__(self, base: Any, method: Literal["sigmoid", "isotonic"] = "sigmoid"):
        """Сохраняет базовую модель и метод калибровки.

        Args:
            base: Обученный estimator с predict_proba.
            method: Метод калибровки: sigmoid или isotonic.
        """
        self.base = base
        self.method = method
        self.classes_ = np.array([0, 1])
        self._calibrator: Any | None = None

    def fit(self, X, y):
        """Обучает калибратор на скорах base.predict_proba.

        Args:
            X: Признаки.
            y: Метки.

        Returns:
            Self после обучения калибратора.
        """
        scores = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        if self.method == "sigmoid":
            self._calibrator = LogisticRegression(max_iter=1000)
        else:
            from sklearn.isotonic import IsotonicRegression

            self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(scores, y)
        return self

    def predict_proba(self, X):
        """Возвращает калиброванные вероятности классов.

        Args:
            X: Признаки.

        Returns:
            Массив формы (n_samples, 2).

        Raises:
            RuntimeError: Если калибратор ещё не обучен.
        """
        if self._calibrator is None:
            raise RuntimeError("Calibrator is not fitted.")
        scores = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        if hasattr(self._calibrator, "predict_proba"):
            p1 = self._calibrator.predict_proba(scores)[:, 1]
        else:
            p1 = np.asarray(self._calibrator.predict(scores), dtype=float)
        p1 = np.clip(p1, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """Предсказывает метки классов по порогу 0.5.

        Args:
            X: Признаки.

        Returns:
            Вектор меток 0/1.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calibrate_model(
    model: Any,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    method: Literal["sigmoid", "isotonic"] = "sigmoid",
) -> Any:
    """Калибрует уже обученную модель на validation-выборке.

    Предпочитает CalibratedClassifierCV с FrozenEstimator при sklearn >= 1.6,
    иначе использует fallback на Platt или isotonic по скорам.

    Args:
        model: Обученный estimator с predict_proba.
        X_valid: Валидационные признаки.
        y_valid: Валидационный таргет.
        method: Метод калибровки: sigmoid или isotonic.

    Returns:
        Калиброванный estimator.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
        calibrated.fit(X_valid, y_valid)
        return calibrated
    except Exception:
        wrapper = _ScorePlatform(base=model, method=method)
        wrapper.fit(X_valid, y_valid)
        return wrapper


def plot_reliability(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
    title: str = "Reliability plot",
) -> plt.Figure:
    """Строит reliability-график калибровки вероятностей.

    Args:
        y_true: Истинные метки.
        y_proba: Предсказанные вероятности класса 1.
        n_bins: Число бинов калибровочной кривой.
        title: Заголовок графика.

    Returns:
        Matplotlib-фигура.
    """
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_proba,
        n_bins=n_bins,
        strategy="quantile",
    )
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
    ax.plot(mean_pred, frac_pos, marker="o", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    return fig
