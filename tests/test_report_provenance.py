from __future__ import annotations

import importlib.util
from pathlib import Path

from plasmid_priority.protocol import (
    ScientificProtocol,
    build_protocol_hash,
    build_protocol_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_BUILD_RELEASE_BUNDLE_SPEC = importlib.util.spec_from_file_location(
    "build_release_bundle_script",
    PROJECT_ROOT / "scripts/28_build_release_bundle.py",
)
assert _BUILD_RELEASE_BUNDLE_SPEC is not None and _BUILD_RELEASE_BUNDLE_SPEC.loader is not None
build_release_bundle_script = importlib.util.module_from_spec(_BUILD_RELEASE_BUNDLE_SPEC)
_BUILD_RELEASE_BUNDLE_SPEC.loader.exec_module(build_release_bundle_script)


def test_protocol_snapshot_includes_single_model_selection_weights() -> None:
    protocol = ScientificProtocol.from_config(
        {
            "pipeline": {"split_year": 2015, "min_new_countries_for_spread": 3},
            "models": {
                "primary_model_name": "discovery_12f_source",
                "primary_model_fallback": "parsimonious_priority",
                "conservative_model_name": "parsimonious_priority",
                "governance_model_name": "phylo_support_fusion_priority",
                "governance_model_fallback": "support_synergy_priority",
                "core_model_names": [
                    "discovery_12f_source",
                    "phylo_support_fusion_priority",
                    "baseline_both",
                ],
                "research_model_names": [],
                "ablation_model_names": [],
            },
        }
    )

    snapshot = build_protocol_snapshot(protocol)

    assert snapshot["benchmark_contract_version"] == "2026-04-10"
    assert snapshot["benchmark_scope"]["split_year"] == 2015
    assert snapshot["benchmark_scope"]["required_assignment_mode"] == "training_only"
    assert snapshot["benchmark_scope"]["accepted_audit_gates"] == [
        "matched_knownness_gap_min",
        "source_holdout_gap_min",
        "spatial_holdout_gap_min",
        "ece_max",
        "selection_adjusted_p_max",
    ]
    assert snapshot["single_model_objective_weights"] == {
        "reliability": 0.4,
        "predictive_power": 0.4,
        "compute_efficiency": 0.2,
    }


def test_release_manifest_includes_provenance_fields() -> None:
    context = build_release_bundle_script.build_context(PROJECT_ROOT)
    manifest = build_release_bundle_script._build_release_manifest(context)
    protocol = build_release_bundle_script._release_protocol(context)

    provenance = manifest["provenance"]
    assert provenance["git_commit"] != ""
    assert provenance["python_version"] != ""
    assert provenance["project_version"] != ""
    assert provenance["benchmark_contract_version"] == "2026-04-10"
    assert provenance["benchmark_scope"]["split_year"] == protocol.split_year
    assert provenance["benchmark_contract_hash"] != ""
    assert provenance["scientific_acceptance_status"] != ""
    assert isinstance(provenance["official_model_names"], list)
    assert provenance["benchmark_contract_hash"] == build_protocol_hash(protocol)
