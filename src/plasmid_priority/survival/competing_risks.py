"""Fine-Gray competing risks model for multi-outcome survival analysis.

Models three competing outcomes simultaneously:
- country-spread (geo)
- host-expansion (bio_transfer)
- clinical-escalation (clinical_hazard)

A backbone can experience multiple event types; Fine-Gray estimates the
subdistribution hazard for each.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

try:
    from lifelines import CoxPHFitter

    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    CoxPHFitter = None


class FineGrayCompetingRisks:
    """Fine-Gray subdistribution hazards for three competing outcomes.

    Each outcome type k gets its own Cox-PH model fitted on a
    modified dataset where:
    - Event type k => event=1
    - Any other event type => event=0 (censored at event time)
    - No event by end => event=0 (administratively censored)

    This yields three subdistribution hazard functions that can be
    combined into a unified risk score per backbone.

    Parameters
    ----------
    penalizer : float
        L2 penalizer passed to each Cox-PH fitter.
    duration_col : str
        Time column name.
    event_col : str
        Column holding event type codes (0=censored, 1=geo, 2=bio, 3=clinical).
    """

    OUTCOME_CODES = {
        "geo_spread": 1,
        "host_expansion": 2,
        "clinical_escalation": 3,
    }

    def __init__(
        self,
        *,
        penalizer: float = 0.1,
        duration_col: str = "time_to_event",
        event_col: str = "event_type",
    ) -> None:
        if not LIFELINES_AVAILABLE:
            raise ImportError(
                "lifelines is required for FineGrayCompetingRisks. "
                "Install with: uv pip install lifelines"
            )
        self.penalizer = float(penalizer)
        self.duration_col = str(duration_col)
        self.event_col = str(event_col)
        self._models: dict[str, Any] = {}
        self._feature_cols: list[str] = []
        self._is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> "FineGrayCompetingRisks":
        """Fit one Cox-PH per outcome type using Fine-Gray censoring.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``duration_col``, ``event_col``, and feature columns.
            ``event_col`` should hold integer codes (0=censored, 1=geo, ...).
        feature_cols : list[str] | None
            Numeric features. If None, auto-detected.

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

        for outcome_name, code in self.OUTCOME_CODES.items():
            # Fine-Gray censoring: other events treated as censored at their time
            working = df.copy()
            working["_fg_event"] = (
                working[self.event_col].fillna(0).astype(int).eq(code).astype(int)
            )
            # Drop rows missing required fields
            working = working.dropna(subset=[self.duration_col, self.event_col])
            cols = list({self.duration_col, "_fg_event"} | set(self._feature_cols))
            working = working.loc[:, cols].dropna()
            if working.empty or working["_fg_event"].sum() < 3:
                _log.warning(
                    "Insufficient events for %s (n=%d); skipping model.",
                    outcome_name,
                    working["_fg_event"].sum(),
                )
                continue

            model = CoxPHFitter(penalizer=self.penalizer)
            model.fit(
                working,
                duration_col=self.duration_col,
                event_col="_fg_event",
                show_progress=False,
            )
            self._models[outcome_name] = model
            _log.info(
                "Fine-Gray %s fitted: n=%d, events=%d, c-index=%.4f",
                outcome_name,
                len(working),
                working["_fg_event"].sum(),
                model.concordance_index_,
            )

        self._is_fitted = bool(self._models)
        return self

    def predict_cumulative_incidence(
        self,
        df: pd.DataFrame,
        outcome: str,
        times: np.ndarray | list[float] | None = None,
    ) -> pd.DataFrame:
        """Predict cumulative incidence F_k(t) = 1 - S_k(t) for outcome k."""
        if outcome not in self._models:
            raise KeyError(f"Outcome '{outcome}' not fitted.")
        model = self._models[outcome]
        available = [c for c in self._feature_cols if c in df.columns]
        surv = model.predict_survival_function(df.loc[:, available], times=times)
        return cast(pd.DataFrame, 1.0 - surv.T)

    def predict_unified_risk_score(
        self,
        df: pd.DataFrame,
        horizon: float = 5.0,
        weights: dict[str, float] | None = None,
    ) -> pd.Series:
        """Combine cumulative incidences into a single risk score.

        Default weights reflect the consensus importance:
        geo_spread=0.5, host_expansion=0.25, clinical_escalation=0.25.
        """
        if weights is None:
            weights = {
                "geo_spread": 0.5,
                "host_expansion": 0.25,
                "clinical_escalation": 0.25,
            }
        scores = pd.Series(0.0, index=df.index, dtype=float)
        for outcome, w in weights.items():
            if outcome not in self._models:
                continue
            inc = self.predict_cumulative_incidence(df, outcome, times=[horizon])
            scores += w * inc.iloc[:, 0]
        return scores

    def summary(self, outcome: str) -> pd.DataFrame:
        """Return coefficient summary for a specific outcome model."""
        if outcome not in self._models:
            raise KeyError(f"Outcome '{outcome}' not fitted.")
        return cast(pd.DataFrame, self._models[outcome].summary)


def build_competing_risk_records(
    records: pd.DataFrame,
    *,
    split_year: int = 2015,
    backbone_col: str = "backbone_id",
    year_col: str = "resolved_year",
    country_col: str = "country",
    host_genus_col: str = "host_genus",
    clinical_context_col: str = "clinical_context",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build competing-risk survival records per backbone.

    Each backbone gets:
    - ``time_to_event``: years from split to first qualifying event.
    - ``event_type``: 0=censored, 1=geo_spread, 2=host_expansion, 3=clinical_escalation.
        If multiple events occur in the same year, the most severe is chosen
        (clinical > host > geo) to avoid ties.
    """
    df = records.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[backbone_col, year_col])
    if df.empty:
        return pd.DataFrame(
            columns=[
                backbone_col,
                "time_to_event",
                "event_type",
                "event_description",
            ],
        )

    rows: list[dict[str, object]] = []
    for bb_id, group in df.groupby(backbone_col, sort=False):
        years = group[year_col].astype(float)
        last_year = int(years.max())

        # Geo spread: >=3 new countries after split
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
        new_countries = post_countries - pre_countries
        geo_year = None
        if len(new_countries) >= 3:
            geo_years = group.loc[
                group[country_col].isin(new_countries) & (years > split_year),
                year_col,
            ]
            if not geo_years.empty:
                geo_year = int(geo_years.min())

        # Host expansion: >=2 new genera or >=1 new family after split
        host_genus_year = None
        if host_genus_col in group.columns:
            pre_genera = set(
                group.loc[years <= split_year, host_genus_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", np.nan)
                .dropna()
                .unique()
            )
            post_genera = set(
                group.loc[years > split_year, host_genus_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", np.nan)
                .dropna()
                .unique()
            )
            new_genera = post_genera - pre_genera
            if len(new_genera) >= 2:
                host_years = group.loc[
                    group[host_genus_col].isin(new_genera) & (years > split_year),
                    year_col,
                ]
                if not host_years.empty:
                    host_genus_year = int(host_years.min())

        # Clinical escalation: simple proxy based on clinical context fraction gain
        clinical_year = None
        if clinical_context_col in group.columns:
            pre_clinical = (
                group.loc[years <= split_year, clinical_context_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"clinical", "hospital", "patient", "human"})
                .mean()
            )
            post_clinical = (
                group.loc[years > split_year, clinical_context_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"clinical", "hospital", "patient", "human"})
                .mean()
            )
            if post_clinical - pre_clinical >= 0.15:
                clinical_years = group.loc[years > split_year, year_col]
                if not clinical_years.empty:
                    clinical_year = int(clinical_years.min())

        # Resolve first event (clinical > host > geo) by earliest year
        events = []
        if geo_year is not None:
            events.append((geo_year, 1, "geo_spread"))
        if host_genus_year is not None:
            events.append((host_genus_year, 2, "host_expansion"))
        if clinical_year is not None:
            events.append((clinical_year, 3, "clinical_escalation"))

        if events:
            events.sort(key=lambda x: (x[0], -x[1]))  # earliest year, highest severity
            first_year, event_type, event_desc = events[0]
            time_to_event = max(float(first_year - split_year), 0.0)
        else:
            event_type = 0
            event_desc = "censored"
            time_to_event = max(float(last_year - split_year), 0.0)

        row: dict[str, object] = {
            backbone_col: bb_id,
            "time_to_event": time_to_event,
            "event_type": event_type,
            "event_description": event_desc,
            "last_observed_year": last_year,
        }

        # Aggregate features
        if feature_cols:
            for fcol in feature_cols:
                if fcol in group.columns:
                    vals = pd.to_numeric(group[fcol], errors="coerce")
                    row[f"{fcol}_mean"] = float(vals.mean()) if not vals.empty else np.nan

        rows.append(row)

    return pd.DataFrame(rows)
