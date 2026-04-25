"""Co-teaching for label-noise-robust training.

Two networks teach each other: each epoch, the network with lower loss
selects its small-loss samples as "clean" and feeds them to the other
network. This is robust to 20-30% label noise.

Implemented as a sklearn-compatible wrapper around PyTorch MLPs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


if TORCH_AVAILABLE:

    class _SimpleMLP(nn.Module):
        """2-layer MLP for co-teaching."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, 1)

        def forward(self, x: Any) -> Any:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return torch.sigmoid(self.fc2(x)).squeeze(-1)

else:

    class _SimpleMLP:  # type: ignore[no-redef]  # pragma: no cover - exercised only when torch is unavailable
        """Fallback placeholder when torch is unavailable."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise ImportError("torch not available")


class CoTeachingTrainer:
    """Co-teaching trainer for noisy binary classification.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int
        MLP hidden layer size (default 64).
    dropout : float
        Dropout rate (default 0.3).
    learning_rate : float
        Adam learning rate (default 1e-3).
    forget_rate : float
        Estimated noise rate; determines how many samples are trusted
        each epoch (default 0.2).
    num_gradual : int
        Epochs over which to ramp the forget rate from 0 to ``forget_rate``.
    num_epochs : int
        Total training epochs.
    batch_size : int
        Training batch size.
    device : str
        torch device.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        forget_rate: float = 0.2,
        num_gradual: int = 10,
        num_epochs: int = 50,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for CoTeachingTrainer. Install with: uv pip install torch"
            )
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.lr = float(learning_rate)
        self.forget_rate = float(forget_rate)
        self.num_gradual = int(num_gradual)
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.device = str(device)
        self._model_a: Any | None = None
        self._model_b: Any | None = None
        self._is_fitted = False

    def _build_models(self) -> tuple[Any, Any]:
        if nn is None:
            raise ImportError("torch not available")
        model_a = _SimpleMLP(self.input_dim, self.hidden_dim, self.dropout).to(self.device)
        model_b = _SimpleMLP(self.input_dim, self.hidden_dim, self.dropout).to(self.device)
        return model_a, model_b

    def _adjust_forget_rate(self, epoch: int) -> float:
        """Linear ramp from 0 to forget_rate over num_gradual epochs."""
        if epoch < self.num_gradual:
            return self.forget_rate * (epoch / self.num_gradual)
        return self.forget_rate

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "CoTeachingTrainer":
        """Fit co-teaching on potentially noisy labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
            Binary labels (may contain noise).
        sample_weight : np.ndarray | None
            Optional per-sample weights.

        Returns
        -------
        self
        """
        if not TORCH_AVAILABLE or torch is None:
            raise ImportError("torch not available")

        n_samples = len(X)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        if sample_weight is not None:
            w_tensor = torch.tensor(sample_weight, dtype=torch.float32, device=self.device)
        else:
            w_tensor = torch.ones(n_samples, device=self.device)

        model_a, model_b = self._build_models()
        optimizer_a = torch.optim.Adam(model_a.parameters(), lr=self.lr)
        optimizer_b = torch.optim.Adam(model_b.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            # Shuffle
            indices = torch.randperm(n_samples, device=self.device)
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]
            w_shuffled = w_tensor[indices]

            forget_rate = self._adjust_forget_rate(epoch)
            n_remember = int((1.0 - forget_rate) * n_samples)

            model_a.train()
            model_b.train()

            epoch_loss_a = 0.0
            epoch_loss_b = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.batch_size):
                batch_x = X_shuffled[i : i + self.batch_size]
                batch_y = y_shuffled[i : i + self.batch_size]
                batch_w = w_shuffled[i : i + self.batch_size]

                # Forward
                pred_a = model_a(batch_x)
                pred_b = model_b(batch_x)

                # Loss per sample
                loss_a = F.binary_cross_entropy(pred_a, batch_y, reduction="none")
                loss_b = F.binary_cross_entropy(pred_b, batch_y, reduction="none")

                # Co-teaching sample selection
                # A selects small-loss for B, B selects small-loss for A
                loss_a_sorted = torch.argsort(loss_a)
                loss_b_sorted = torch.argsort(loss_b)
                remember_idx_a = loss_a_sorted[: max(1, min(n_remember, len(loss_a)))]
                remember_idx_b = loss_b_sorted[: max(1, min(n_remember, len(loss_b)))]

                # Update A with B's clean samples
                optimizer_a.zero_grad()
                pred_a_clean = model_a(batch_x[remember_idx_b])
                y_clean_b = batch_y[remember_idx_b]
                w_clean_b = batch_w[remember_idx_b]
                loss_a_update = F.binary_cross_entropy(
                    pred_a_clean,
                    y_clean_b,
                    weight=w_clean_b,
                )
                loss_a_update.backward()  # type: ignore[no-untyped-call]
                optimizer_a.step()

                # Update B with A's clean samples
                optimizer_b.zero_grad()
                pred_b_clean = model_b(batch_x[remember_idx_a])
                y_clean_a = batch_y[remember_idx_a]
                w_clean_a = batch_w[remember_idx_a]
                loss_b_update = F.binary_cross_entropy(
                    pred_b_clean,
                    y_clean_a,
                    weight=w_clean_a,
                )
                loss_b_update.backward()  # type: ignore[no-untyped-call]
                optimizer_b.step()

                epoch_loss_a += float(loss_a_update.item())
                epoch_loss_b += float(loss_b_update.item())
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                _log.info(
                    "Co-teaching epoch %d/%d: loss_a=%.4f, loss_b=%.4f, forget=%.3f",
                    epoch + 1,
                    self.num_epochs,
                    epoch_loss_a / max(n_batches, 1),
                    epoch_loss_b / max(n_batches, 1),
                    forget_rate,
                )

        self._model_a = model_a
        self._model_b = model_b
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble probability by averaging both networks."""
        if not self._is_fitted or self._model_a is None or self._model_b is None:
            raise RuntimeError("Model must be fitted before prediction.")
        if torch is None:
            raise ImportError("torch not available")

        self._model_a.eval()
        self._model_b.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred_a = self._model_a(X_tensor).cpu().numpy()
            pred_b = self._model_b(X_tensor).cpu().numpy()
        return (pred_a + pred_b) / 2.0

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
