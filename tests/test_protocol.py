from __future__ import annotations

from plasmid_priority.protocol import ScientificProtocol, build_protocol_hash


def test_protocol_exposes_single_model_selection_weights() -> None:
    protocol = ScientificProtocol.from_config(
        {
            "pipeline": {
                "split_year": 2015,
                "min_new_countries_for_spread": 3,
                "horizon_years": 5,
                "min_new_host_genera_for_transfer": 2,
                "min_new_host_families_for_transfer": 1,
                "clinical_escalation_thresholds": {"clinical_fraction_gain": 0.15},
                "forbidden_features": ["n_countries_test"],
                "label_proxy_caveats": {"clinical_hazard": "metadata bias"},
                "eligibility_rules": {"required_assignment_mode": "training_only"},
            },
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

    assert protocol.single_model_objective_weights == {
        "reliability": 0.4,
        "predictive_power": 0.4,
        "compute_efficiency": 0.2,
    }
    assert protocol.horizon_years == 5
    assert protocol.min_new_host_genera_for_transfer == 2
    assert protocol.min_new_host_families_for_transfer == 1
    assert protocol.clinical_escalation_thresholds == {"clinical_fraction_gain": 0.15}
    assert protocol.forbidden_features == ("n_countries_test",)
    assert protocol.label_proxy_caveats == {"clinical_hazard": "metadata bias"}
    assert protocol.eligibility_rules == {"required_assignment_mode": "training_only"}


def test_protocol_hash_changes_when_horizon_changes() -> None:
    base = ScientificProtocol.from_config(
        {
            "pipeline": {"split_year": 2015, "min_new_countries_for_spread": 3, "horizon_years": 5},
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
    changed = ScientificProtocol.from_config(
        {
            "pipeline": {"split_year": 2015, "min_new_countries_for_spread": 3, "horizon_years": 7},
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

    assert build_protocol_hash(base) != build_protocol_hash(changed)
