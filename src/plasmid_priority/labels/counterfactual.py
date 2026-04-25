"""Causal counterfactual label estimation.

Answers: "If this backbone had NOT been observed in country X, what would
its spread risk be?" Uses propensity score weighting within a potential
outcomes framework.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_log = logging.getLogger(__name__)


class CausalLabelEstimator:
    """Estimate counterfactual spread labels via propensity score weighting.

    For each backbone, defines a "treatment" (e.g. observed in a high-risk
    country) and estimates the Average Treatment Effect (ATE) on the spread
    probability. The counterfactual label is the estimated spread probability
    under treatment vs control.

    Parameters
    ----------
    treatment_col : str
        Binary treatment indicator column.
    outcome_col : str
        Binary outcome column (spread label).
    covariate_cols : list[str]
        Confounding adjustment variables.
    """

    def __init__(
        self,
        *,
        treatment_col: str = "observed_in_high_risk_country",
        outcome_col: str = "spread_label",
        covariate_cols: list[str] | None = None,
    ) -> None:
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.covariate_cols = covariate_cols or []
        self._propensity_model: LogisticRegression | None = None

    def fit(self, df: pd.DataFrame) -> "CausalLabelEstimator":
        """Fit propensity score model P(treatment=1 | covariates).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``treatment_col``, ``outcome_col``, and covariates.
        """
        if self.treatment_col not in df.columns:
            raise KeyError(f"Missing treatment column: {self.treatment_col}")
        if not self.covariate_cols:
            # Auto-select numeric covariates
            self.covariate_cols = [
                c
                for c in df.select_dtypes(include="number").columns
                if c not in {self.treatment_col, self.outcome_col}
            ]

        X = df[self.covariate_cols].fillna(0)
        T = df[self.treatment_col].fillna(0).astype(int)

        self._propensity_model = LogisticRegression(max_iter=1000, solver="lbfgs")
        self._propensity_model.fit(X, T)
        _log.info("Propensity model fitted on %d samples, %d covariates", len(X), len(X.columns))
        return self

    def estimate_ate(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Estimate Average Treatment Effect using inverse propensity weighting.

        Returns a DataFrame with:
        - ``backbone_id``
        - ``observed_outcome``
        - ``counterfactual_outcome_treated``
        - ``counterfactual_outcome_control``
        - ``ate`` (Average Treatment Effect)
        """
        if self._propensity_model is None:
            raise RuntimeError("Model must be fitted before ATE estimation.")

        X = df[self.covariate_cols].fillna(0)
        T = df[self.treatment_col].fillna(0).astype(int).to_numpy()
        Y = pd.to_numeric(df[self.outcome_col], errors="coerce").fillna(0).to_numpy()

        # Propensity scores
        ps = self._propensity_model.predict_proba(X)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)  # trimming

        # IPW weights
        w1 = T / ps
        w0 = (1 - T) / (1 - ps)

        ate = (w1 * Y).sum() / w1.sum() - (w0 * Y).sum() / w0.sum()

        result = pd.DataFrame(
            {
                "observed_outcome": Y,
                "propensity_score": ps,
                "ipw_weight_treated": w1,
                "ipw_weight_control": w0,
                "ate": ate,
            }
        )

        _log.info("ATE estimate: %.4f", ate)
        return result

    def estimate_counterfactual_labels(
        self,
        df: pd.DataFrame,
        backbone_col: str = "backbone_id",
    ) -> pd.DataFrame:
        """Return counterfactual spread labels per backbone.

        For each backbone, computes the spread probability under:
        - treatment=1 (observed in high-risk setting)
        - treatment=0 (not observed in high-risk setting)

        The difference is the causal effect of the treatment on spread.
        """
        ate_df = self.estimate_ate(df)
        out = df[[backbone_col]].copy()
        out["counterfactual_spread_prob_treated"] = np.clip(
            ate_df["observed_outcome"] + ate_df["ate"] * 0.5,
            0,
            1,
        )
        out["counterfactual_spread_prob_control"] = np.clip(
            ate_df["observed_outcome"] - ate_df["ate"] * 0.5,
            0,
            1,
        )
        out["causal_effect"] = ate_df["ate"]
        return out
