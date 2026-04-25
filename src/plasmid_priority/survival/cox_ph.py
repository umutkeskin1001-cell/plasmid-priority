"""Cox Proportional Hazards model with time-varying covariates.

Implements survival analysis replacing binary spread_label with continuous-time
hazard modeling. Each backbone gets a hazard function for time_to_first_new_country.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    CoxPHFitter = None
    concordance_index = None

try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import brier_score, cumulative_dynamic_auc

    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    CoxPHSurvivalAnalysis = None
    brier_score = None
    cumulative_dynamic_auc = None


class CoxPHSurvivalModel:
    """Cox-PH survival model for plasmid backbone spread prediction.

    Replaces the binary ``spread_label`` (3+ new countries) with a
    continuous-time ``time_to_first_new_country`` outcome. Each backbone
    receives a hazard function that can be evaluated at any horizon.

    Time-varying covariates T(t), H(t), A(t) are computed per-year from
    training data and joined to each backbone's survival record.

    Parameters
    ----------
    penalizer : float
        L2 penalizer for the Cox-PH partial likelihood (default 0.1).
    duration_col : str
        Column holding the observed time to event or censoring.
    event_col : str
        Binary column indicating whether the event was observed (1) or
        censored (0).
    """

    def __init__(
        self,
        *,
        penalizer: float = 0.1,
        duration_col: str = "time_to_event",
        event_col: str = "event_observed",
    ) -> None:
        if not LIFELINES_AVAILABLE:
            raise ImportError(
                "lifelines is required for CoxPHSurvivalModel. "
                "Install with: uv pip install lifelines"
            )
        self.penalizer = float(penalizer)
        self.duration_col = str(duration_col)
        self.event_col = str(event_col)
        self._model: Any | None = None
        self._feature_cols: list[str] = []
        self._is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> "CoxPHSurvivalModel":
        """Fit the Cox-PH model on survival-formatted data.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``duration_col``, ``event_col``, and feature columns.
        feature_cols : list[str] | None
            Numeric feature columns to use. If None, all numeric columns
            except duration/event are used.

        Returns
        -------
        self
        """
        if not LIFELINES_AVAILABLE or CoxPHFitter is None:
            raise ImportError("lifelines is not installed")

        required = {self.duration_col, self.event_col}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        if feature_cols is None:
            exclude = required | {"backbone_id"}
            feature_cols = [
                c for c in df.select_dtypes(include="number").columns if c not in exclude
            ]
        self._feature_cols = list(feature_cols)

        # Guard: drop rows with NaN in required cols or all-NaN in features
        subset_cols = list(required | set(self._feature_cols))
        clean = df.dropna(subset=[self.duration_col, self.event_col])
        clean = clean.loc[:, subset_cols].dropna()
        if clean.empty:
            raise ValueError("No valid rows after dropping NaN for Cox-PH fit.")

        self._model = CoxPHFitter(penalizer=self.penalizer)
        self._model.fit(
            clean,
            duration_col=self.duration_col,
            event_col=self.event_col,
            show_progress=False,
        )
        self._is_fitted = True
        _log.info(
            "Cox-PH fitted on n=%d, features=%d, c-index=%.4f",
            len(clean),
            len(self._feature_cols),
            self._model.concordance_index_,
        )
        return self

    def predict_partial_hazard(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """Return the partial hazard exp(beta * X) for each row."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        available = [c for c in self._feature_cols if c in df.columns]
        if not available:
            raise KeyError("No fitted feature columns present in prediction data.")
        # lifelines predicts on the full dataframe but only uses known cols
        pred = self._model.predict_partial_hazard(df.loc[:, available])
        return pd.Series(pred, index=df.index, dtype=float)

    def predict_survival_function(
        self,
        df: pd.DataFrame,
        times: np.ndarray | list[float] | None = None,
    ) -> pd.DataFrame:
        """Predict survival function S(t) for each backbone at given times.

        Returns a DataFrame where rows are samples and columns are time points.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        available = [c for c in self._feature_cols if c in df.columns]
        surv = self._model.predict_survival_function(
            df.loc[:, available],
            times=times,
        )
        # lifelines returns a DataFrame with times as index; transpose
        return cast(pd.DataFrame, surv.T)

    def predict_risk_score(
        self,
        df: pd.DataFrame,
        horizon: float = 5.0,
    ) -> pd.Series:
        """Predict 1 - S(horizon): probability of event by ``horizon`` years.

        This is the direct replacement for the binary ``spread_label``
        probability used downstream in the consensus pipeline.
        """
        surv = self.predict_survival_function(df, times=[horizon])
        return 1.0 - surv.iloc[:, 0]

    def summary(self) -> pd.DataFrame:
        """Return the fitted coefficients summary."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted.")
        return cast(pd.DataFrame, self._model.summary)

    @property
    def concordance_index(self) -> float:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted.")
        return float(self._model.concordance_index_)


def compute_td_roc(
    y_true_structured: np.ndarray,
    y_pred_scores: np.ndarray,
    times: np.ndarray,
) -> dict[str, object]:
    """Compute time-dependent ROC (tdROC) using sksurv.

    Parameters
    ----------
    y_true_structured : np.ndarray
        Structured array with fields ``event`` (bool) and ``time`` (float).
    y_pred_scores : np.ndarray
        Risk scores (higher = more likely to experience event).
    times : np.ndarray
        Time horizons at which to evaluate ROC.

    Returns
    -------
    dict with keys ``times``, ``mean_auc``, ``auc``.
    """
    if not SKSURV_AVAILABLE or cumulative_dynamic_auc is None:
        raise ImportError("sksurv is required for time-dependent ROC")

    aucs, mean_auc = cumulative_dynamic_auc(
        y_true_structured,
        y_true_structured,
        y_pred_scores,
        times,
    )
    return {
        "times": np.asarray(times),
        "mean_auc": float(mean_auc),
        "auc": np.asarray(aucs),
    }


def time_dependent_brier_score(
    y_true_structured: np.ndarray,
    y_pred_survival: np.ndarray,
    times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute Brier score at multiple time horizons.

    Parameters
    ----------
    y_true_structured : np.ndarray
        Structured array with ``event`` and ``time``.
    y_pred_survival : np.ndarray
        Predicted survival probabilities (n_samples, n_times).
    times : np.ndarray
        Time grid matching the second axis of ``y_pred_survival``.

    Returns
    -------
    dict with keys ``times``, ``brier_scores``.
    """
    if not SKSURV_AVAILABLE or brier_score is None:
        raise ImportError("sksurv is required for Brier score computation")

    scores = brier_score(y_true_structured, y_true_structured, y_pred_survival, times)
    return {
        "times": np.asarray(times),
        "brier_scores": np.asarray(scores[1]),
    }


def build_survival_records(
    records: pd.DataFrame,
    *,
    split_year: int = 2015,
    backbone_col: str = "backbone_id",
    year_col: str = "resolved_year",
    country_col: str = "country",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Convert plasmid-level records into backbone-level survival records.

    For each backbone, computes:
    - ``time_to_event``: years from first observation to first new-country
      observation after the split year (or to last known year if censored).
    - ``event_observed``: 1 if the backbone spread to >=3 new countries,
      0 if censored.
    - Time-varying covariates aggregated from all plasmids in that backbone
      up to each observation year.

    Parameters
    ----------
    records : pd.DataFrame
        Plasmid-level records with at least ``backbone_col``, ``year_col``,
        ``country_col``.
    split_year : int
        Temporal split defining the origin time (t=0) for each backbone.
    backbone_col : str
        Column identifying the backbone / operational unit.
    year_col : str
        Year of isolation/observation.
    country_col : str
        Country of isolation.
    feature_cols : list[str] | None
        Additional numeric columns to aggregate per-backbone per-year as
        time-varying covariates.

    Returns
    -------
    pd.DataFrame
        One row per backbone with survival outcome and baseline covariates.
    """
    if records.empty:
        return pd.DataFrame(
            columns=[
                backbone_col,
                "time_to_event",
                "event_observed",
                "n_countries_pre",
                "n_countries_post",
            ],
        )
    df = records.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[backbone_col, year_col])
    if df.empty:
        return pd.DataFrame(
            columns=[
                backbone_col,
                "time_to_event",
                "event_observed",
                "n_countries_pre",
                "n_countries_post",
            ],
        )

    # Per-backbone statistics
    backbone_stats: list[dict[str, object]] = []
    for bb_id, group in df.groupby(backbone_col, sort=False):
        years = group[year_col].astype(float)
        (group[country_col].fillna("").astype(str).str.strip().replace("", np.nan).dropna())
        first_year = int(years.min())
        last_year = int(years.max())

        pre_countries = set(
            group.loc[years <= split_year, country_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .unique()
        )
        post_countries = set(
            group.loc[years > split_year, country_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .unique()
        )
        new_post = post_countries - pre_countries
        n_new = len(new_post)

        # Event: spread to >=3 new countries after split
        event_observed = 1 if n_new >= 3 else 0
        if event_observed:
            # Time to first new country after split
            new_years = group.loc[
                group[country_col].isin(new_post) & (years > split_year),
                year_col,
            ]
            time_to_event = (
                float(new_years.min() - split_year)
                if not new_years.empty
                else float(last_year - split_year)
            )
        else:
            # Censored at last observed year
            time_to_event = float(last_year - split_year)

        row: dict[str, object] = {
            backbone_col: bb_id,
            "time_to_event": max(time_to_event, 0.0),
            "event_observed": event_observed,
            "n_countries_pre": len(pre_countries),
            "n_countries_post": len(post_countries),
            "n_new_countries_post": n_new,
            "first_observed_year": first_year,
            "last_observed_year": last_year,
        }

        # Aggregate time-varying features per backbone (mean up to split_year)
        if feature_cols:
            for fcol in feature_cols:
                if fcol in group.columns:
                    pre_vals = pd.to_numeric(
                        group.loc[years <= split_year, fcol],
                        errors="coerce",
                    )
                    row[f"{fcol}_pre_mean"] = (
                        float(pre_vals.mean()) if not pre_vals.empty else np.nan
                    )
                    post_vals = pd.to_numeric(
                        group.loc[years > split_year, fcol],
                        errors="coerce",
                    )
                    row[f"{fcol}_post_mean"] = (
                        float(post_vals.mean()) if not post_vals.empty else np.nan
                    )

        backbone_stats.append(row)

    return pd.DataFrame(backbone_stats)


def to_structured_array(
    df: pd.DataFrame,
    event_col: str = "event_observed",
    time_col: str = "time_to_event",
) -> np.ndarray:
    """Convert a DataFrame to sksurv structured array.

    Returns
    -------
    np.ndarray
        Structured array with boolean ``event`` and float ``time`` fields.
    """
    events = df[event_col].fillna(0).astype(bool).to_numpy()
    times = pd.to_numeric(df[time_col], errors="coerce").fillna(0).to_numpy()
    arr = np.empty(len(df), dtype=[("event", bool), ("time", float)])
    arr["event"] = events
    arr["time"] = times
    return arr
