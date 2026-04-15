from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from plasmid_priority import config as config_module
from plasmid_priority.modeling import module_a_support as module_a_support_impl

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_root = Path(tmp_dir)
        (data_root / "scores").mkdir(parents=True, exist_ok=True)
        (data_root / "analysis").mkdir(parents=True, exist_ok=True)

        fake_features = [f"f{i}" for i in range(10)]
        frame = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)],
                "spread_label": [0, 1] * 6,
                **{feature: [0.1, 0.9] * 6 for feature in fake_features},
            }
        )
        frame.to_csv(data_root / "scores" / "backbone_scored.tsv", sep="\t", index=False)
        (data_root / "analysis" / "module_a_metrics.json").write_text("{}", encoding="utf-8")

        spec = importlib.util.spec_from_file_location(
            "train_sovereign_script",
            PROJECT_ROOT / "scripts/29_train_sovereign.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)

        fake_feature_sets = dict(module_a_support_impl.MODULE_A_FEATURE_SETS)
        fake_feature_sets["sovereign_precision_priority"] = fake_features

        with (
            mock.patch.object(
                config_module,
                "build_context",
                return_value=SimpleNamespace(data_dir=data_root),
            ),
            mock.patch.object(
                module_a_support_impl,
                "MODULE_A_FEATURE_SETS",
                fake_feature_sets,
            ),
        ):
            spec.loader.exec_module(module)
        return module


def test_choose_best_candidate_prefers_reliability_when_auc_is_close() -> None:
    train_sovereign_script = _load_script_module()
    scorecard = pd.DataFrame(
        [
            {
                "model_name": "sovereign_precision_priority",
                "roc_auc": 0.8294,
                "average_precision": 0.7640,
                "brier_score": 0.1656,
                "expected_calibration_error": 0.060,
                "knownness_matched_gap": -0.014,
                "feature_count": 37,
            },
            {
                "model_name": "sovereign_v2_priority",
                "roc_auc": 0.8292,
                "average_precision": 0.7642,
                "brier_score": 0.1610,
                "expected_calibration_error": 0.045,
                "knownness_matched_gap": -0.006,
                "feature_count": 33,
            },
        ]
    )

    winner = train_sovereign_script._choose_best_candidate(scorecard)

    assert winner["model_name"] == "sovereign_v2_priority"


def test_main_reuses_cached_outputs_without_recomputing() -> None:
    train_sovereign_script = _load_script_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_root = Path(tmp_dir)
        (data_root / "scores").mkdir(parents=True, exist_ok=True)
        (data_root / "analysis").mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)],
                "spread_label": [0, 1] * 6,
                "f0": [0.1, 0.9] * 6,
            }
        )
        frame.to_csv(data_root / "scores" / "backbone_scored.tsv", sep="\t", index=False)
        (data_root / "analysis" / "module_a_metrics.json").write_text("{}", encoding="utf-8")

        fake_feature_sets = {
            "sovereign_precision_priority": ["f0"],
            "sovereign_v2_priority": ["f0"],
        }
        fake_result = SimpleNamespace(
            metrics={
                "roc_auc": 0.82,
                "average_precision": 0.75,
                "brier_score": 0.16,
                "expected_calibration_error": 0.08,
                "knownness_matched_gap": 0.0,
            },
            predictions=pd.DataFrame(
                {
                    "backbone_id": frame["backbone_id"],
                    "oof_prediction": [0.2, 0.8] * 6,
                }
            ),
            status="ok",
            error_message=None,
        )

        evaluate_model_name_mock = mock.Mock(return_value=fake_result)
        fit_full_model_predictions_mock = mock.Mock(
            return_value=pd.DataFrame(
                {
                    "backbone_id": frame["backbone_id"],
                    "prediction": [0.2, 0.8] * 6,
                }
            )
        )

        with (
            mock.patch.object(
                train_sovereign_script,
                "build_context",
                return_value=SimpleNamespace(
                    root=PROJECT_ROOT, data_dir=data_root, logs_dir=data_root / "logs"
                ),
            ),
            mock.patch.object(train_sovereign_script, "MODULE_A_FEATURE_SETS", fake_feature_sets),
            mock.patch.object(
                train_sovereign_script.module_a_support_impl,
                "MODULE_A_FEATURE_SETS",
                fake_feature_sets,
            ),
            mock.patch.object(
                train_sovereign_script.module_a_support_impl,
                "_model_fit_kwargs",
                side_effect=lambda _name: {"sample_weight_mode": "class_balanced", "l2": 1.5},
            ),
            mock.patch.object(
                train_sovereign_script, "evaluate_model_name", evaluate_model_name_mock
            ),
            mock.patch.object(
                train_sovereign_script,
                "fit_full_model_predictions",
                fit_full_model_predictions_mock,
            ),
            mock.patch.object(
                train_sovereign_script,
                "project_python_source_paths",
                return_value=[PROJECT_ROOT / "scripts" / "29_train_sovereign.py"],
            ),
        ):
            assert train_sovereign_script.main(["--jobs", "1"]) == 0
            assert train_sovereign_script.main(["--jobs", "1"]) == 0

        assert evaluate_model_name_mock.call_count == 2
        assert fit_full_model_predictions_mock.call_count == 1
        for call in evaluate_model_name_mock.call_args_list:
            assert call.kwargs["include_ci"] is False
            assert len(call.args[0]) == 12

        summary_path = data_root / "analysis" / "sovereign_v2_priority_trained.json"
        manifest_path = data_root / "analysis" / "29_train_sovereign.manifest.json"
        assert summary_path.exists()
        assert manifest_path.exists()

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["selected_model_name"] == "sovereign_v2_priority"
        assert payload["jobs"] == 1
