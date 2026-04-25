"""Attention-based dynamic consensus fusion.

Learns per-sample weights for each branch prediction using a small
attention network. AMR-rich → clinical weight up; geo-spread → clinical weight down.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from sklearn.neural_network import MLPRegressor

_log = logging.getLogger(__name__)


class AttentionConsensus:
    """Learned attention weights for branch ensemble fusion.

    Given branch predictions (geo, bio, clinical) and optional meta-features
    (e.g. AMR richness, host diversity), learns attention scores per sample.

    Parameters
    ----------
    meta_dim : int
        Dimension of meta-features for attention context (e.g. AMR count).
    hidden_dim : int
        Attention MLP hidden size.
    use_meta : bool
        If False, falls back to uniform + learned residual.
    """

    def __init__(
        self,
        *,
        meta_dim: int = 0,
        hidden_dim: int = 16,
        use_meta: bool = True,
    ) -> None:
        self.meta_dim = meta_dim
        self.hidden_dim = hidden_dim
        self.use_meta = use_meta
        self._attention_net: MLPRegressor | None = None
        self._fallback_weights: np.ndarray = np.ones(3) / 3.0
        self._is_fitted = False

    def fit(
        self,
        branch_probs: np.ndarray,
        y_true: np.ndarray,
        meta_features: np.ndarray | None = None,
    ) -> "AttentionConsensus":
        """Fit attention weights on validation data.

        Parameters
        ----------
        branch_probs : np.ndarray, shape (n_samples, 3)
            Columns: [geo_spread, bio_transfer, clinical_hazard] probabilities.
        y_true : np.ndarray, shape (n_samples,)
            Binary ground truth.
        meta_features : np.ndarray | None, shape (n_samples, meta_dim)
            Optional context features.
        """
        branch_probs = np.asarray(branch_probs, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        if self.use_meta and meta_features is not None and self.meta_dim > 0:
            X_attn = np.hstack([branch_probs, np.asarray(meta_features, dtype=float)])
        else:
            X_attn = branch_probs.copy()

        # Target: optimal weights per sample (approximated by Ridge on squared error)
        # We learn attention scores that minimize BCE-weighted MSE
        best_weights = np.zeros((len(y_true), 3))
        for i in range(len(y_true)):
            # Optimal weight = proportional to accuracy of each branch on this sample
            errs = np.abs(branch_probs[i] - y_true[i])
            acc = 1.0 - errs
            w = acc / acc.sum() if acc.sum() > 0 else self._fallback_weights
            best_weights[i] = w

        self._attention_net = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim,),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        )
        self._attention_net.fit(X_attn, best_weights)
        self._is_fitted = True
        _log.info("Attention consensus fitted on %d samples", len(y_true))
        return self

    def predict(
        self, branch_probs: np.ndarray, meta_features: np.ndarray | None = None
    ) -> np.ndarray:
        """Return consensus probability per sample."""
        if not self._is_fitted or self._attention_net is None:
            raise RuntimeError("Call fit() first.")
        branch_probs = np.asarray(branch_probs, dtype=float)
        if self.use_meta and meta_features is not None and self.meta_dim > 0:
            X_attn = np.hstack([branch_probs, np.asarray(meta_features, dtype=float)])
        else:
            X_attn = branch_probs.copy()
        weights = self._attention_net.predict(X_attn)
        weights = np.clip(weights, 0.01, 1.0)
        weights = weights / weights.sum(axis=1, keepdims=True)
        consensus = (branch_probs * weights).sum(axis=1)
        return cast(np.ndarray, np.clip(consensus, 0.0, 1.0))

    @property
    def fallback_weights(self) -> np.ndarray:
        return self._fallback_weights.copy()
