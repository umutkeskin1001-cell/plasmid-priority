from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from plasmid_priority.geo_spread import cli as geo_spread_cli


class _FakeGeoSpreadRun:
    def __init__(self) -> None:
        self.inputs: list[Path] = []
        self.outputs: list[Path] = []
        self.metrics: dict[str, object] = {}
        self.notes: list[str] = []

    def __enter__(self) -> _FakeGeoSpreadRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def record_input(self, path: Path) -> None:
        self.inputs.append(Path(path))

    def record_output(self, path: Path) -> None:
        self.outputs.append(Path(path))

    def note(self, message: str) -> None:
        self.notes.append(message)

    def set_metric(self, key: str, value: object) -> None:
        self.metrics[key] = value

    def set_rows_out(self, key: str, value: int) -> None:
        self.metrics[key] = value


class _FakePipelineSettings:
    split_year = 2015
    min_new_countries_for_spread = 3


class _FakeGeoContext:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_dir = root / "data"
        self.pipeline_settings = _FakePipelineSettings()
        self.config = {"geo_spread": {}}


class GeoSpreadScriptTests(unittest.TestCase):
    def test_main_writes_geo_spread_outputs(self) -> None:
        fake_run = _FakeGeoSpreadRun()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            context = _FakeGeoContext(root)
            scored = pd.DataFrame(
                {
                    "backbone_id": ["bb1", "bb2"],
                    "spread_label": [1, 0],
                    "n_new_countries": [4, 1],
                    "split_year": [2015, 2015],
                    "backbone_assignment_mode": ["training_only", "training_only"],
                    "max_resolved_year_train": [2014, 2015],
                    "min_resolved_year_test": [2017, 2018],
                    "training_only_future_unseen_backbone_flag": [False, False],
                    "T_eff_norm": [0.9, 0.1],
                    "H_obs_specialization_norm": [0.1, 0.9],
                    "A_eff_norm": [0.8, 0.2],
                    "coherence_score": [0.8, 0.2],
                    "log1p_member_count_train": [1.0, 1.5],
                    "log1p_n_countries_train": [0.5, 0.2],
                    "refseq_share_train": [0.4, 0.6],
                    "backbone_purity_norm": [0.7, 0.3],
                    "assignment_confidence_norm": [0.8, 0.2],
                    "mash_neighbor_distance_train_norm": [0.2, 0.8],
                    "orit_support": [0.4, 0.1],
                    "H_external_host_range_norm": [0.6, 0.4],
                }
            )
            fake_results = {
                "geo_counts_baseline": mock.Mock(
                    metrics={"roc_auc": 0.9},
                    status="ok",
                    error_message=None,
                    predictions=pd.DataFrame({"backbone_id": ["bb1"], "prediction": [0.9]}),
                )
            }
            with (
                mock.patch.object(geo_spread_cli, "build_context", return_value=context),
                mock.patch.object(geo_spread_cli, "ManagedScriptRun", return_value=fake_run),
                mock.patch.object(geo_spread_cli, "project_python_source_paths", return_value=[]),
                mock.patch.object(geo_spread_cli, "load_signature_manifest", return_value=False),
                mock.patch.object(geo_spread_cli, "read_tsv", return_value=scored),
                mock.patch.object(
                    geo_spread_cli,
                    "resolve_geo_spread_model_names",
                    return_value=("geo_counts_baseline",),
                ),
                mock.patch.object(
                    geo_spread_cli,
                    "evaluate_geo_spread_branch",
                    return_value=fake_results,
                ),
                mock.patch.object(
                    geo_spread_cli,
                    "build_geo_spread_calibration_summary",
                    return_value=pd.DataFrame(
                        {
                            "model_name": ["geo_counts_baseline"],
                            "calibration_method": ["none"],
                            "abstain_rate": [0.0],
                        }
                    ),
                ),
                mock.patch.object(
                    geo_spread_cli,
                    "build_geo_spread_calibrated_prediction_table",
                    return_value=pd.DataFrame(
                        {
                            "backbone_id": ["bb1"],
                            "model_name": ["geo_counts_baseline"],
                            "calibrated_prediction": [0.9],
                            "confidence_band": ["high"],
                        }
                    ),
                ),
                mock.patch.object(
                    geo_spread_cli,
                    "build_geo_spread_run_provenance",
                    return_value={
                        "run_signature": "abc123",
                        "primary_model_name": "geo_support_light_priority",
                        "recommended_primary_model_name": "geo_support_light_priority",
                    },
                ),
                mock.patch.object(
                    geo_spread_cli,
                    "build_geo_spread_inventory",
                    return_value=(
                        pd.DataFrame({"path": ["src/plasmid_priority/geo_spread/report.py"], "category": ["code"]}),
                        pd.DataFrame({"path": ["data/analysis/other.tsv"], "category": ["data"]}),
                        {"used_file_count": 1, "unused_file_count": 1},
                    ),
                ),
            ):
                result = geo_spread_cli.main([])

        self.assertEqual(result, 0)
        self.assertIn(Path(tmp_dir) / "data/geo_spread/analysis/geo_spread_metrics.json", fake_run.outputs)
        self.assertIn(Path(tmp_dir) / "data/geo_spread/analysis/geo_spread_model_summary.tsv", fake_run.outputs)
        self.assertIn(Path(tmp_dir) / "data/geo_spread/analysis/geo_spread_predictions.tsv", fake_run.outputs)
        self.assertIn(Path(tmp_dir) / "data/geo_spread/analysis/geo_spread_calibration_summary.tsv", fake_run.outputs)
        self.assertIn(Path(tmp_dir) / "data/geo_spread/analysis/geo_spread_calibrated_predictions.tsv", fake_run.outputs)
        self.assertIn(Path(tmp_dir) / "data/geo_spread/analysis/geo_spread_provenance.json", fake_run.outputs)
        self.assertIn(Path(tmp_dir) / "data/geo_spread/analysis/geo_spread_report_card.tsv", fake_run.outputs)
        self.assertIn(Path(tmp_dir) / "data/geo_spread/analysis/geo_spread_report.md", fake_run.outputs)
        self.assertIn(Path(tmp_dir) / "data/geo_spread/inventory/used_paths.tsv", fake_run.outputs)
        self.assertIn(Path(tmp_dir) / "data/geo_spread/inventory/unused_paths.tsv", fake_run.outputs)
        self.assertEqual(fake_run.metrics.get("cache_hit"), False)
        self.assertEqual(fake_run.metrics.get("models_run"), 1)

    def test_main_returns_early_on_cache_hit(self) -> None:
        fake_run = _FakeGeoSpreadRun()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            context = _FakeGeoContext(root)
            with (
                mock.patch.object(geo_spread_cli, "build_context", return_value=context),
                mock.patch.object(geo_spread_cli, "ManagedScriptRun", return_value=fake_run),
                mock.patch.object(geo_spread_cli, "project_python_source_paths", return_value=[]),
                mock.patch.object(geo_spread_cli, "load_signature_manifest", return_value=True),
                mock.patch.object(
                    geo_spread_cli,
                    "resolve_geo_spread_model_names",
                    return_value=("geo_counts_baseline",),
                ),
                mock.patch.object(geo_spread_cli, "evaluate_geo_spread_branch") as evaluate_mock,
            ):
                result = geo_spread_cli.main([])

        self.assertEqual(result, 0)
        evaluate_mock.assert_not_called()
        self.assertEqual(fake_run.metrics.get("cache_hit"), True)


if __name__ == "__main__":
    unittest.main()
