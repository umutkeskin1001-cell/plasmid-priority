from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from plasmid_priority.bio_transfer.contracts import build_bio_transfer_input_contract
from plasmid_priority.bio_transfer.specs import load_bio_transfer_config
from plasmid_priority.shared.branching import fit_branch, prepare_branch_scored_table


class SharedBranchingTests(unittest.TestCase):
    def test_label_merge_preserves_required_metadata_columns(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "split_year": [2015],
                "backbone_assignment_mode": ["training_only"],
                "max_resolved_year_train": [2014],
                "min_resolved_year_test": [2016],
                "training_only_future_unseen_backbone_flag": [False],
                "future_new_host_genera_count": [2],
                "future_new_host_families_count": [1],
            }
        )

        def label_builder(
            _scored: pd.DataFrame,
            _records: pd.DataFrame | None,
            split_year: int,
            horizon_years: int,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "backbone_id": ["bb1"],
                    "split_year": [split_year],
                    "horizon_years": [horizon_years],
                    "max_resolved_year_train": [2014],
                    "min_resolved_year_test": [2016],
                    "training_only_future_unseen_backbone_flag": [False],
                    "bio_transfer_label": [1.0],
                    "bio_transfer_label_reason": ["ok"],
                    "n_new_host_genera_future": [2],
                    "n_new_host_families_future": [1],
                }
            )

        prepared = prepare_branch_scored_table(
            scored,
            config=load_bio_transfer_config(),
            contract=build_bio_transfer_input_contract(),
            label_builder=label_builder,
            branch_label_column="bio_transfer_label",
        )

        self.assertIn("split_year", prepared.columns)
        self.assertNotIn("split_year_x", prepared.columns)
        self.assertNotIn("split_year_y", prepared.columns)
        self.assertIn("bio_transfer_label", prepared.columns)
        self.assertEqual(int(prepared.loc[0, "bio_transfer_label"]), 1)

    def test_fit_branch_prepares_surface_once_for_multiple_models(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "split_year": [2015, 2015],
                "backbone_assignment_mode": ["training_only", "training_only"],
                "max_resolved_year_train": [2014, 2014],
                "min_resolved_year_test": [2016, 2016],
                "training_only_future_unseen_backbone_flag": [False, False],
                "bio_transfer_label": [0.0, 1.0],
                "spread_label": [0.0, 1.0],
                "log1p_member_count_train": [0.1, 0.2],
                "log1p_n_countries_train": [0.2, 0.3],
                "orit_support": [0.3, 0.4],
                "T_eff_norm": [0.2, 0.8],
                "H_obs_specialization_norm": [0.4, 0.6],
                "A_eff_norm": [0.3, 0.7],
                "coherence_score": [0.5, 0.6],
                "mobility_support_norm": [0.2, 0.7],
                "backbone_purity_norm": [0.8, 0.7],
                "assignment_confidence_norm": [0.9, 0.6],
            }
        )
        config = load_bio_transfer_config()
        contract = build_bio_transfer_input_contract()
        prepared = scored.copy()
        with (
            mock.patch(
                "plasmid_priority.shared.branching.prepare_branch_scored_table",
                return_value=prepared,
            ) as prepared_mock,
            mock.patch(
                "plasmid_priority.shared.branching._fit_branch_model_on_dataset",
                return_value=mock.Mock(status="ok", metrics={}, predictions=pd.DataFrame()),
            ),
        ):
            fit_branch(
                scored,
                model_names=["bio_transfer_baseline", "bio_transfer_parsimonious"],
                config=config,
                contract=contract,
                branch_label_column="bio_transfer_label",
                n_jobs=1,
            )
        self.assertEqual(prepared_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
