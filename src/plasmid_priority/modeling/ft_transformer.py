"""FT-Transformer: Feature Tokenizer Transformer for tabular data.

Reference: Gorishniy et al. 2021. Replaces tree ensembles with a transformer
that treats each feature as a token. CPU-optimized small variant.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd

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


class _FeatureTokenizer(_TorchModuleBase):
    """Tokenize continuous features into d_model dimensions."""

    def __init__(self, n_features: int, d_model: int) -> None:
        _require_torch("FT-Transformer")
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, n_features, d_model)
        return x.unsqueeze(-1) * self.weight + self.bias


class _TransformerBlock(_TorchModuleBase):
    """Pre-norm transformer block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        _require_torch("FT-Transformer")
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(_TorchModuleBase):
    """Small FT-Transformer for binary classification.

    Parameters
    ----------
    n_features : int
    d_model : int
    n_heads : int
    n_blocks : int
    dropout : float
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        _require_torch("FT-Transformer")
        super().__init__()
        self.tokenizer = _FeatureTokenizer(n_features, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.blocks = nn.Sequential(
            *[_TransformerBlock(d_model, n_heads, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        tokens = self.tokenizer(x)  # (batch, n_features, d_model)
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # prepend CLS
        features = self.blocks(tokens)
        cls_out = features[:, 0, :]  # (batch, d_model)
        return torch.sigmoid(self.head(cls_out)).squeeze(-1)


class FTTransformerClassifier:
    """Sklearn-compatible wrapper around FT-Transformer.

    Parameters
    ----------
    d_model, n_heads, n_blocks, dropout : see FTTransformer
    learning_rate : float
    num_epochs : int
    batch_size : int
    device : str
    patience : int
        Early stopping patience.
    """

    def __init__(
        self,
        *,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        batch_size: int = 128,
        device: str = "cpu",
        patience: int = 10,
    ) -> None:
        _require_torch("FT-Transformer")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = str(device)
        self.patience = patience
        self._model: FTTransformer | None = None
        self._feature_names: list[str] | None = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "FTTransformerClassifier":
        _require_torch("FT-Transformer")

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        n_features = X_arr.shape[1]
        self._model = FTTransformer(
            n_features=n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
        ).to(self.device)

        X_tensor = torch.tensor(X_arr, device=self.device)
        y_tensor = torch.tensor(y_arr, device=self.device)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.BCELoss()

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        patience_counter = 0

        n = len(X_tensor)
        for epoch in range(self.num_epochs):
            self._model.train()
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                batch_x = X_tensor[idx]
                batch_y = y_tensor[idx]
                optimizer.zero_grad()
                pred = self._model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
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
                _log.info("FT-Transformer early stop at epoch %d", epoch)
                break

        if best_state is not None and self._model is not None:
            self._model.load_state_dict(best_state)
            self._model.to(self.device)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted.")
        _require_torch("FT-Transformer")
        X_arr = (
            X.to_numpy(dtype=np.float32)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X, dtype=np.float32)
        )
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_arr, device=self.device)
            out = self._model(X_t).cpu().numpy()
        return np.column_stack([1 - out, out])

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
