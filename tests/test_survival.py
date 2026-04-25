"""Tests for the Phase 1 survival analysis module."""

from __future__ import annotations

import pandas as pd
import pytest

from plasmid_priority.survival.adaptive_split import AdaptiveTemporalSplitter
from plasmid_priority.survival.competing_risks import (
    FineGrayCompetingRisks,
    build_competing_risk_records,
)
from plasmid_priority.survival.cox_ph import (
    CoxPHSurvivalModel,
    build_survival_records,
    to_structured_array,
)
from plasmid_priority.survival.lead_time import LeadTimeBiasCorrector


class TestBuildSurvivalRecords:
    def test_basic(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["BB1", "BB1", "BB1", "BB2", "BB2"],
                "resolved_year": [2010, 2012, 2018, 2014, 2019],
                "country": ["USA", "CAN", "GBR", "DEU", "FRA"],
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        surv = build_survival_records(
            records,
            split_year=2015,
            backbone_col="backbone_id",
            year_col="resolved_year",
            country_col="country",
            feature_cols=["feature_a"],
        )
        assert len(surv) == 2
        # BB1: pre=USA,CAN (2); post=GBR (1 new) -> event=0, censored at 2018
        bb1 = surv.loc[surv["backbone_id"] == "BB1"]
        assert bb1["event_observed"].iloc[0] == 0
        assert bb1["time_to_event"].iloc[0] == 3.0  # 2018 - 2015
        # BB2: pre=DEU (1); post=FRA (1 new) -> event=0
        bb2 = surv.loc[surv["backbone_id"] == "BB2"]
        assert bb2["event_observed"].iloc[0] == 0

    def test_event_observed(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["BB1"] * 6,
                "resolved_year": [2010, 2011, 2012, 2016, 2017, 2018],
                "country": ["USA", "CAN", "MEX", "GBR", "FRA", "DEU"],
            }
        )
        surv = build_survival_records(records, split_year=2015)
        # 3 new countries after 2015 -> event=1, time_to_event=1 (2016-2015)
        assert surv["event_observed"].iloc[0] == 1
        assert surv["time_to_event"].iloc[0] == 1.0

    def test_empty(self) -> None:
        surv = build_survival_records(pd.DataFrame())
        assert surv.empty


class TestCoxPHSurvivalModel:
    def test_fit_predict(self) -> None:
        pytest.importorskip("lifelines")
        df = pd.DataFrame(
            {
                "time_to_event": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "event_observed": [1, 1, 0, 1, 0, 1, 0, 1],
                "feature_x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            }
        )
        model = CoxPHSurvivalModel(penalizer=0.1)
        model.fit(df, feature_cols=["feature_x"])
        assert model.concordance_index > 0.0

        risk = model.predict_risk_score(df, horizon=5.0)
        assert len(risk) == len(df)
        assert risk.between(0, 1).all()

    def test_missing_columns_raises(self) -> None:
        pytest.importorskip("lifelines")
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        model = CoxPHSurvivalModel()
        with pytest.raises(KeyError):
            model.fit(df)


class TestFineGrayCompetingRisks:
    def test_fit_predict(self) -> None:
        pytest.importorskip("lifelines")
        # Need >=3 events per outcome type for fitting
        df = pd.DataFrame(
            {
                "time_to_event": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "event_type": [1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0],
                "feature_x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            }
        )
        model = FineGrayCompetingRisks(penalizer=0.1)
        model.fit(df, feature_cols=["feature_x"])
        assert model._is_fitted

        risk = model.predict_unified_risk_score(df, horizon=5.0)
        assert len(risk) == len(df)
        assert risk.between(0, 1).all()


class TestBuildCompetingRiskRecords:
    def test_geo_spread_only(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["BB1"] * 6,
                "resolved_year": [2010, 2011, 2012, 2016, 2017, 2018],
                "country": ["USA", "CAN", "MEX", "GBR", "FRA", "DEU"],
                "host_genus": ["Escherichia"] * 6,
                "clinical_context": ["environmental"] * 6,
            }
        )
        cr = build_competing_risk_records(records, split_year=2015)
        assert len(cr) == 1
        assert cr["event_type"].iloc[0] == 1  # geo_spread
        assert cr["event_description"].iloc[0] == "geo_spread"

    def test_censored(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["BB1"] * 3,
                "resolved_year": [2010, 2011, 2012],
                "country": ["USA"] * 3,
            }
        )
        cr = build_competing_risk_records(records, split_year=2015)
        assert cr["event_type"].iloc[0] == 0
        assert cr["event_description"].iloc[0] == "censored"


class TestAdaptiveTemporalSplitter:
    def test_quantile(self) -> None:
        years = pd.Series([2000] * 5 + [2010] * 20 + [2020] * 10)
        splitter = AdaptiveTemporalSplitter(quantile=0.6, strategy="quantile")
        splitter.fit(years)
        # 60th percentile of 35 obs = around 2010
        assert splitter.split_year == 2010

    def test_density_peak(self) -> None:
        years = pd.Series(
            list(range(2000, 2005)) * 2 + list(range(2008, 2018)) * 6 + list(range(2018, 2023)) * 2
        )
        splitter = AdaptiveTemporalSplitter(strategy="density_peak")
        splitter.fit(years)
        # Should split at or before the dense 2008-2017 block
        assert splitter.split_year <= 2008

    def test_transform(self) -> None:
        df = pd.DataFrame(
            {
                "resolved_year": [2008, 2012, 2018],
                "value": [1, 2, 3],
            }
        )
        splitter = AdaptiveTemporalSplitter(quantile=0.5)
        out = splitter.fit_transform(df, year_col="resolved_year")
        assert "pre_split" in out.columns
        assert "post_split" in out.columns
        assert out["pre_split"].sum() >= 1
        assert out["post_split"].sum() >= 1

    def test_too_few_years_fallback(self) -> None:
        years = pd.Series([2010, 2015, 2020])
        splitter = AdaptiveTemporalSplitter()
        splitter.fit(years)
        assert splitter.split_year == 2015  # median


class TestLeadTimeBiasCorrector:
    def test_median_lag(self) -> None:
        df = pd.DataFrame(
            {
                "resolved_year": [2020, 2021, 2022, 2023],
                "collection_year": [2018, 2019, 2020, 2021],
                "country": ["USA", "USA", "CAN", "CAN"],
            }
        )
        corrector = LeadTimeBiasCorrector(correction_method="median_lag")
        out = corrector.fit_transform(df)
        # median lag = 2 years
        assert out["lead_time_lag_years"].iloc[0] == 2.0
        assert out["true_emergence_year"].iloc[0] == 2018.0

    def test_none(self) -> None:
        df = pd.DataFrame(
            {
                "resolved_year": [2020],
            }
        )
        corrector = LeadTimeBiasCorrector(correction_method="none")
        out = corrector.fit_transform(df)
        assert out["true_emergence_year"].iloc[0] == 2020.0
        assert out["lead_time_lag_years"].iloc[0] == 0.0

    def test_country_specific(self) -> None:
        df = pd.DataFrame(
            {
                "resolved_year": [2020, 2021, 2020, 2021],
                "collection_year": [2018, 2019, 2019, 2020],
                "country": ["USA", "USA", "CAN", "CAN"],
            }
        )
        corrector = LeadTimeBiasCorrector(
            correction_method="median_lag",
            min_observations_for_lag=2,
        )
        out = corrector.fit_transform(df)
        # USA lag = 2, CAN lag = 1
        assert out.loc[out["country"] == "USA", "lead_time_lag_years"].iloc[0] == 2.0
        assert out.loc[out["country"] == "CAN", "lead_time_lag_years"].iloc[0] == 1.0


class TestToStructuredArray:
    def test_basic(self) -> None:
        df = pd.DataFrame(
            {
                "event_observed": [1, 0, 1],
                "time_to_event": [2.5, 5.0, 1.0],
            }
        )
        arr = to_structured_array(df)
        assert arr.dtype.names == ("event", "time")
        assert arr["event"].tolist() == [True, False, True]
        assert arr["time"].tolist() == [2.5, 5.0, 1.0]
