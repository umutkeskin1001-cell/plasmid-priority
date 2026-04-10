from __future__ import annotations

from plasmid_priority.protocol import ScientificProtocol, build_protocol_snapshot


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

    assert snapshot["single_model_objective_weights"] == {
        "reliability": 0.4,
        "predictive_power": 0.4,
        "compute_efficiency": 0.2,
    }
