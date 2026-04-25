"""Multi-task learning: shared encoder + three task heads.

Trains geo_spread, bio_transfer, clinical_hazard jointly with
uncertainty-weighted loss (Kendall et al. 2018).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

_log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


_TORCH_INSTALL_HINT = "Install it with `pip install torch`."
if TYPE_CHECKING:
    _TorchModuleBase: TypeAlias = nn.Module
else:
    _TorchModuleBase = nn.Module if nn is not None else object


def _require_torch(context: str) -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(f"torch is required for {context}. {_TORCH_INSTALL_HINT}")


class MultiTaskPlasmidNet(_TorchModuleBase):
    """Shared MLP encoder + three binary classification heads.

    Uncertainty weighting learns log(sigma^2) per task automatically.
    """

    TASK_NAMES = ["geo_spread", "bio_transfer", "clinical_hazard"]

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        _require_torch("multi-task learning")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({t: nn.Linear(hidden_dim, 1) for t in self.TASK_NAMES})
        # Learnable log-variance per task for uncertainty weighting
        self.log_vars = nn.Parameter(torch.zeros(len(self.TASK_NAMES)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        return {t: torch.sigmoid(self.heads[t](h)).squeeze(-1) for t in self.TASK_NAMES}


class MultiTaskTrainer:
    """Train MultiTaskPlasmidNet with uncertainty-weighted loss."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
        batch_size: int = 64,
        device: str = "cpu",
        patience: int = 10,
    ) -> None:
        _require_torch("multi-task learning")
        self.device = str(device)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self._model = MultiTaskPlasmidNet(input_dim, hidden_dim, dropout).to(self.device)
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_dict: dict[str, np.ndarray],
    ) -> "MultiTaskTrainer":
        _require_torch("multi-task learning")
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        y_t = {
            t: torch.tensor(np.asarray(y_dict[t], dtype=np.float32), device=self.device)
            for t in MultiTaskPlasmidNet.TASK_NAMES
            if t in y_dict
        }

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-3, weight_decay=1e-5)
        n = len(X_t)
        best_loss = float("inf")
        best_state: Any = None
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self._model.train()
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                batch_x = X_t[idx]
                preds = self._model(batch_x)
                loss: torch.Tensor = torch.zeros(1, device=self.device)
                for j, (t, y) in enumerate(y_t.items()):
                    precision = torch.exp(-self._model.log_vars[j])
                    loss = loss + (
                        precision * nn.functional.binary_cross_entropy(preds[t], y[idx])
                        + self._model.log_vars[j]
                    )
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                _log.info("Multi-task early stop at epoch %d", epoch)
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model.to(self.device)
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        _require_torch("multi-task learning")
        self._model.eval()
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        with torch.no_grad():
            preds = self._model(X_t)
        return {t: p.cpu().numpy() for t, p in preds.items()}

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> dict[str, np.ndarray]:
        proba = self.predict_proba(X)
        return {t: (p >= threshold).astype(int) for t, p in proba.items()}
