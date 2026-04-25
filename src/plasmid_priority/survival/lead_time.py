"""Lead-time bias correction using Pathogen Detection metadata timestamps.

Reporting lag means ``resolved_year`` may be later than the true emergence
year. This module estimates ``true_emergence_year`` from metadata and
adjusts survival times accordingly.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


class LeadTimeBiasCorrector:
    """Correct lead-time bias in temporal outcomes.

    Uses Pathogen Detection metadata (or other timestamp sources) to estimate
    the gap between true emergence and database resolution year.

    Parameters
    ----------
    correction_method : {"median_lag", "quantile_lag", "pd_metadata", "none"}
        - ``median_lag``: subtract median lag per-country from resolved_year.
        - ``quantile_lag``: subtract a quantile of the lag distribution.
        - ``pd_metadata``: use Pathogen Detection ``collection_year`` directly.
        - ``none``: passthrough (no correction).
    lag_quantile : float
        Quantile of the lag distribution to use when ``quantile_lag`` is selected.
    min_observations_for_lag : int
        Minimum observations required to estimate a country-specific lag.
    """

    def __init__(
        self,
        *,
        correction_method: Literal[
            "median_lag",
            "quantile_lag",
            "pd_metadata",
            "none",
        ] = "median_lag",
        lag_quantile: float = 0.75,
        min_observations_for_lag: int = 5,
    ) -> None:
        self.correction_method = correction_method
        self.lag_quantile = float(np.clip(lag_quantile, 0.1, 0.99))
        self.min_observations = int(min_observations_for_lag)
        self._lag_estimates: dict[str, float] = {}
        self._global_lag: float = 0.0

    def fit(
        self,
        df: pd.DataFrame,
        resolved_year_col: str = "resolved_year",
        true_year_col: str = "collection_year",
        country_col: str = "country",
    ) -> "LeadTimeBiasCorrector":
        """Learn lag distribution from records where both years are known.

        Parameters
        ----------
        df : pd.DataFrame
            Records with ``resolved_year`` and ``true_year_col`` (e.g. collection_year).
        resolved_year_col : str
            Year when the record entered the database.
        true_year_col : str
            Year when the sample was actually collected.
        country_col : str
            Country for stratified lag estimation.

        Returns
        -------
        self
        """
        if self.correction_method == "none":
            return self

        working = df.copy()
        working["_resolved"] = pd.to_numeric(working[resolved_year_col], errors="coerce")
        working["_true"] = pd.to_numeric(working[true_year_col], errors="coerce")
        working = working.dropna(subset=["_resolved", "_true"])
        working["_lag"] = working["_resolved"] - working["_true"]
        working = working.loc[working["_lag"] >= 0]  # only positive lags make sense

        if working.empty:
            _log.warning("No valid lag observations; global lag=0")
            self._global_lag = 0.0
            return self

        # Global lag
        if self.correction_method == "median_lag":
            self._global_lag = float(working["_lag"].median())
        elif self.correction_method == "quantile_lag":
            self._global_lag = float(working["_lag"].quantile(self.lag_quantile))
        elif self.correction_method == "pd_metadata":
            # For pd_metadata, lag is record-specific; we still compute global fallback
            self._global_lag = float(working["_lag"].median())
        else:
            self._global_lag = 0.0  # type: ignore[unreachable]

        # Country-specific lags
        if country_col in working.columns:
            for country, group in working.groupby(country_col, sort=False):
                if len(group) >= self.min_observations:
                    if self.correction_method == "median_lag":
                        self._lag_estimates[str(country)] = float(group["_lag"].median())
                    elif self.correction_method == "quantile_lag":
                        self._lag_estimates[str(country)] = float(
                            group["_lag"].quantile(self.lag_quantile),
                        )

        _log.info(
            "Lead-time bias: global_lag=%.1f years, country-specific=%d countries",
            self._global_lag,
            len(self._lag_estimates),
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        resolved_year_col: str = "resolved_year",
        country_col: str = "country",
        true_year_col: str = "collection_year",
    ) -> pd.DataFrame:
        """Apply lead-time correction to a DataFrame.

        Adds:
        - ``true_emergence_year``: corrected year estimate.
        - ``lead_time_lag_years``: estimated lag used.
        """
        out = df.copy()
        resolved = pd.to_numeric(out[resolved_year_col], errors="coerce")

        if self.correction_method == "none":
            out["true_emergence_year"] = resolved
            out["lead_time_lag_years"] = 0.0
            return out

        if self.correction_method == "pd_metadata" and true_year_col in out.columns:
            # Prefer direct metadata, fallback to lag subtraction
            direct_true = pd.to_numeric(out[true_year_col], errors="coerce")
            lags = pd.Series(self._global_lag, index=out.index, dtype=float)
            if country_col in out.columns:
                for country, lag in self._lag_estimates.items():
                    mask = out[country_col].astype(str).str.strip().str.lower() == country.lower()
                    lags.loc[mask] = lag
            out["lead_time_lag_years"] = lags
            out["true_emergence_year"] = direct_true.fillna(resolved - lags)
        else:
            # Lag subtraction method
            lags = pd.Series(self._global_lag, index=out.index, dtype=float)
            if country_col in out.columns:
                for country, lag in self._lag_estimates.items():
                    mask = out[country_col].astype(str).str.strip().str.lower() == country.lower()
                    lags.loc[mask] = lag
            out["lead_time_lag_years"] = lags
            out["true_emergence_year"] = resolved - lags

        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        resolved_year_col: str = "resolved_year",
        true_year_col: str = "collection_year",
        country_col: str = "country",
    ) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(df, resolved_year_col, true_year_col, country_col)
        return self.transform(df, resolved_year_col, country_col, true_year_col)
