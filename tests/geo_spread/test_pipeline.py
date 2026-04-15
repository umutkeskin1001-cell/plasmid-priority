from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from plasmid_priority.geo_spread import (
    build_geo_spread_model_summary,
    build_geo_spread_prediction_table,
    evaluate_geo_spread_branch,
    fit_geo_spread_model,
    fit_geo_spread_model_predictions,
    prepare_geo_spread_dataset,
)


class GeoSpreadPipelineTests(unittest.TestCase):
    def _scored(self) -> pd.DataFrame:
        n = 12
        labels = np.array([0, 1] * (n // 2), dtype=float)
        return pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": labels,
                "n_new_countries": np.where(labels > 0, 4, 1),
                "split_year": [2015] * n,
                "backbone_assignment_mode": ["training_only"] * n,
                "max_resolved_year_train": [2014] * n,
                "min_resolved_year_test": [2017] * n,
                "training_only_future_unseen_backbone_flag": [False] * n,
                "log1p_member_count_train": np.linspace(0.2, 1.8, n),
                "log1p_n_countries_train": np.linspace(0.1, 1.4, n),
                "refseq_share_train": np.linspace(0.0, 1.0, n),
                "T_eff_norm": np.linspace(0.15, 0.95, n),
                "H_obs_specialization_norm": np.linspace(0.9, 0.1, n),
                "A_eff_norm": np.linspace(0.2, 0.8, n),
                "coherence_score": np.linspace(0.1, 0.9, n),
                "H_phylogenetic_specialization_norm": np.linspace(0.88, 0.12, n),
                "host_phylogenetic_dispersion_norm": np.linspace(0.2, 0.85, n),
                "host_taxon_evenness_norm": np.linspace(0.25, 0.75, n),
                "ecology_context_diversity_norm": np.linspace(0.1, 0.8, n),
                "backbone_purity_norm": np.linspace(0.2, 0.7, n),
                "assignment_confidence_norm": np.linspace(0.3, 0.85, n),
                "mash_neighbor_distance_train_norm": np.linspace(0.8, 0.2, n),
                "orit_support": np.linspace(0.05, 0.65, n),
                "H_external_host_range_norm": np.linspace(0.85, 0.15, n),
                "geo_country_record_count_train": np.linspace(1.0, 8.0, n),
                "geo_country_entropy_train": np.linspace(0.1, 0.9, n),
                "geo_macro_region_entropy_train": np.linspace(0.15, 0.85, n),
                "geo_dominant_region_share_train": np.linspace(0.9, 0.2, n),
            }
        )

    def test_prepare_geo_spread_dataset_returns_expected_surface(self) -> None:
        dataset = prepare_geo_spread_dataset(
            self._scored(),
            model_name="geo_parsimonious_priority",
        )
        self.assertEqual(dataset.model_name, "geo_parsimonious_priority")
        self.assertEqual(dataset.label_column, "spread_label")
        self.assertEqual(len(dataset.eligible), 12)
        self.assertEqual(
            tuple(dataset.feature_columns),
            tuple(dataset.config.feature_sets["geo_parsimonious_priority"]),
        )

    def test_fit_geo_spread_model_returns_model_result(self) -> None:
        result = fit_geo_spread_model(
            self._scored(),
            model_name="geo_parsimonious_priority",
            n_splits=3,
            n_repeats=1,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.predictions), 12)
        self.assertIn("roc_auc", result.metrics)

    def test_fit_geo_spread_model_predictions_scores_all_rows(self) -> None:
        predictions = fit_geo_spread_model_predictions(
            self._scored(),
            model_name="geo_counts_baseline",
        )
        self.assertEqual(len(predictions), 12)
        self.assertIn("prediction", predictions.columns)

    def test_evaluate_geo_spread_branch_returns_summary_tables(self) -> None:
        results = evaluate_geo_spread_branch(
            self._scored(),
            model_names=[
                "geo_context_hybrid_priority",
                "geo_support_light_priority",
                "geo_phylo_ecology_priority",
            ],
            n_splits=3,
            n_repeats=1,
            n_jobs=1,
        )
        summary = build_geo_spread_model_summary(results)
        predictions = build_geo_spread_prediction_table(results)
        self.assertEqual(
            set(results),
            {
                "geo_context_hybrid_priority",
                "geo_support_light_priority",
                "geo_phylo_ecology_priority",
                "geo_reliability_blend",
                "geo_adaptive_knownness_priority",
                "geo_meta_knownness_priority",
            },
        )
        self.assertEqual(
            set(summary["model_name"]),
            {
                "geo_context_hybrid_priority",
                "geo_support_light_priority",
                "geo_phylo_ecology_priority",
                "geo_reliability_blend",
                "geo_adaptive_knownness_priority",
                "geo_meta_knownness_priority",
            },
        )
        self.assertEqual(
            set(predictions["model_name"]),
            {
                "geo_context_hybrid_priority",
                "geo_support_light_priority",
                "geo_phylo_ecology_priority",
                "geo_reliability_blend",
                "geo_adaptive_knownness_priority",
                "geo_meta_knownness_priority",
            },
        )

    def test_fit_geo_spread_branch_prepares_shared_surface_once(self) -> None:
        from plasmid_priority.geo_spread import train as geo_train

        prepared = self._scored().copy()
        with (
            mock.patch.object(
                geo_train,
                "prepare_geo_spread_scored_table",
                return_value=prepared,
            ) as prepared_mock,
            mock.patch.object(
                geo_train,
                "build_geo_spread_dataset_from_prepared",
                side_effect=lambda *args, **kwargs: mock.Mock(
                    scored=prepared,
                    feature_columns=("T_eff_norm", "A_eff_norm"),
                    model_name=kwargs["model_name"],
                    fit_config={"l2": 1.0, "max_iter": 100, "sample_weight_mode": None},
                ),
            ),
            mock.patch.object(
                geo_train,
                "_fit_geo_spread_model_on_dataset",
                return_value=mock.Mock(
                    status="ok", metrics={"roc_auc": 0.5}, predictions=pd.DataFrame()
                ),
            ),
        ):
            geo_train.fit_geo_spread_branch(
                self._scored(),
                model_names=["geo_counts_baseline", "geo_parsimonious_priority"],
                n_jobs=1,
            )

        self.assertEqual(prepared_mock.call_count, 1)

    def test_fit_geo_spread_branch_captures_model_failures(self) -> None:
        from plasmid_priority.geo_spread import train as geo_train

        prepared = self._scored().copy()
        with (
            mock.patch.object(
                geo_train,
                "prepare_geo_spread_scored_table",
                return_value=prepared,
            ),
            mock.patch.object(
                geo_train,
                "build_geo_spread_dataset_from_prepared",
                side_effect=RuntimeError("missing feature surface"),
            ),
        ):
            results = geo_train.fit_geo_spread_branch(
                self._scored(),
                model_names=["geo_support_light_priority"],
                n_jobs=1,
            )

        self.assertEqual(results["geo_support_light_priority"].status, "failed")
        self.assertIn(
            "missing feature surface", results["geo_support_light_priority"].error_message
        )


if __name__ == "__main__":
    unittest.main()
