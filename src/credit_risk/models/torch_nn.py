"""Табличная нейросеть на PyTorch со sklearn-совместимой обёрткой."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from credit_risk.features.synthetic import build_preprocessor_for_frame


class TabularNet(nn.Module):
    """Многослойный перцептрон для табличных признаков."""

    def __init__(self, in_dim: int, hidden: int = 64):
        """Инициализирует архитектуру сети.

        Args:
            in_dim: Размерность входа.
            hidden: Размер скрытого слоя.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Считает логиты класса дефолта.

        Args:
            x: Батч признаков.

        Returns:
            Вектор логитов.
        """
        return self.net(x).squeeze(-1)


@dataclass
class _TorchState:
    """Хранит обученную сеть и устройство вычислений."""

    model: TabularNet
    device: torch.device


def _pos_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    """Считает вес положительного класса для дисбаланса.

    Args:
        y: Бинарные метки.
        device: Устройство вычислений.

    Returns:
        Тензор pos_weight для BCEWithLogitsLoss.
    """
    n_pos = max(float(y.sum()), 1.0)
    n_neg = max(float(len(y) - n_pos), 1.0)
    return torch.tensor([n_neg / n_pos], device=device)


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> DataLoader:
    """Создаёт DataLoader для обучения.

    drop_last=True при n > batch_size, чтобы BatchNorm не падал
    на последнем батче размера 1.

    Args:
        X: Матрица признаков.
        y: Вектор меток.
        batch_size: Размер батча.

    Returns:
        DataLoader с перемешиванием.
    """
    n = len(X)
    safe_bs = max(1, min(batch_size, n))
    drop_last = n > safe_bs
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=safe_bs,
        shuffle=True,
        drop_last=drop_last,
    )


def _train_epochs(
    model: TabularNet,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    pos_weight: torch.Tensor,
    device: torch.device,
) -> None:
    """Обучает сеть заданное число эпох.

    Args:
        model: Модель TabularNet.
        loader: DataLoader с обучающими батчами.
        epochs: Число эпох.
        lr: Learning rate Adam.
        pos_weight: Вес положительного класса.
        device: Устройство вычислений.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()


class TorchTabularClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-совместимая обёртка над табличным MLP на PyTorch."""

    def __init__(
        self,
        epochs: int = 25,
        batch_size: int = 256,
        lr: float = 1e-3,
        hidden: int = 64,
        seed: int = 42,
    ):
        """Сохраняет гиперпараметры обучения.

        Args:
            epochs: Число эпох.
            batch_size: Размер батча.
            lr: Learning rate.
            hidden: Размер скрытого слоя.
            seed: Зерно инициализации.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden = hidden
        self.seed = seed
        self.classes_ = np.array([0, 1])
        self._state: _TorchState | None = None

    def fit(self, X, y):
        """Обучает нейросеть на матрице признаков.

        Args:
            X: Матрица признаков.
            y: Бинарный таргет.

        Returns:
            Self после обучения.
        """
        torch.manual_seed(self.seed)
        device = torch.device("cpu")
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        model = TabularNet(X_arr.shape[1], hidden=self.hidden).to(device)
        loader = _make_loader(X_arr, y_arr, self.batch_size)
        _train_epochs(
            model,
            loader,
            epochs=self.epochs,
            lr=self.lr,
            pos_weight=_pos_weight(y_arr, device),
            device=device,
        )
        self._state = _TorchState(model=model, device=device)
        return self

    def predict_proba(self, X):
        """Предсказывает вероятности классов.

        Args:
            X: Матрица признаков.

        Returns:
            Массив формы (n_samples, 2) с вероятностями классов 0 и 1.

        Raises:
            RuntimeError: Если модель ещё не обучена.
        """
        if self._state is None:
            raise RuntimeError("Model is not fitted.")
        model, device = self._state.model, self._state.device
        model.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = model(torch.from_numpy(X_arr).to(device))
            p1 = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """Предсказывает метки классов по порогу 0.5.

        Args:
            X: Матрица признаков.

        Returns:
            Вектор меток 0/1.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_torch_nn(X, y, **kwargs) -> Pipeline:
    """Обучает табличную нейросеть с препроцессором.

    Args:
        X: Матрица признаков.
        y: Бинарный таргет.
        **kwargs: Гиперпараметры TorchTabularClassifier.

    Returns:
        Обученный sklearn Pipeline.
    """
    pipe = Pipeline(
        steps=[
            ("pre", build_preprocessor_for_frame(X)),
            ("clf", TorchTabularClassifier(**kwargs)),
        ]
    )
    pipe.fit(X, y)
    return pipe
