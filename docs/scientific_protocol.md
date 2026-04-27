# Scientific Protocol

This protocol is generated from the canonical authority surface:
`config/benchmarks.yaml` + `src/plasmid_priority/protocol.py`.

```json
{
  "ablation_model_names": [
    "T_only",
    "H_only",
    "A_only",
    "T_plus_H_plus_A"
  ],
  "acceptance_thresholds": {
    "ece_max": 0.05,
    "matched_knownness_gap_min": -0.005,
    "selection_adjusted_p_max": 0.01,
    "source_holdout_gap_min": -0.005,
    "spatial_holdout_gap_min": -0.03
  },
  "benchmark_contract_version": "2025-07-15",
  "benchmark_scope": {
    "accepted_audit_gates": [
      "matched_knownness_gap_min",
      "source_holdout_gap_min",
      "spatial_holdout_gap_min",
      "ece_max",
      "selection_adjusted_p_max"
    ],
    "calibration_metric_definitions": {
      "ece_max": "Expected calibration error upper bound.",
      "selection_adjusted_p_max": "Selection-adjusted empirical p-value upper bound."
    },
    "eligibility_rules": {
      "require_temporal_metadata": true,
      "require_training_only_assignment": true,
      "required_assignment_mode": "training_only"
    },
    "eligible_cohort": {
      "forbidden_features": [],
      "horizon_years": 5,
      "min_new_countries_for_spread": 3,
      "min_new_host_families_for_transfer": 1,
      "min_new_host_genera_for_transfer": 2,
      "require_training_only_assignment": true,
      "requires_temporal_metadata": true,
      "temporal_split_year": 2015
    },
    "official_model_names": [
      "discovery_boosted",
      "governance_linear",
      "baseline_both"
    ],
    "outcome_definition": {
      "horizon_years": 5,
      "min_new_countries_for_spread": 3,
      "min_new_host_families_for_transfer": 1,
      "min_new_host_genera_for_transfer": 2,
      "outcome_label": "spread_label",
      "split_year": 2015
    },
    "required_assignment_mode": "training_only",
    "split_year": 2015
  },
  "clinical_escalation_thresholds": {
    "clinical_fraction_gain": 0.15,
    "last_resort_fraction_gain": 0.1,
    "mdr_proxy_fraction_gain": 0.1,
    "pd_clinical_support_gain": 0.1
  },
  "conservative_model_name": "parsimonious_priority",
  "core_model_names": [
    "baseline_both",
    "governance_linear",
    "discovery_boosted",
    "discovery_graph_boosted",
    "T_only",
    "H_only",
    "A_only",
    "T_plus_H_plus_A",
    "parsimonious_priority",
    "support_synergy_priority"
  ],
  "ece_max": 0.05,
  "eligibility_rules": {
    "require_temporal_metadata": true,
    "require_training_only_assignment": true,
    "required_assignment_mode": "training_only"
  },
  "forbidden_features": [],
  "governance_model_fallback": "support_synergy_priority",
  "governance_model_name": "governance_linear",
  "governance_model_policy": {
    "conservative_model_name": "parsimonious_priority",
    "governance_model_fallback": "support_synergy_priority",
    "governance_model_name": "governance_linear",
    "primary_model_fallback": "parsimonious_priority",
    "primary_model_name": "discovery_boosted"
  },
  "horizon_years": 5,
  "label_proxy_caveats": {},
  "matched_knownness_gap_min": -0.005,
  "min_new_countries_for_spread": 3,
  "min_new_host_families_for_transfer": 1,
  "min_new_host_genera_for_transfer": 2,
  "official_model_names": [
    "discovery_boosted",
    "governance_linear",
    "baseline_both"
  ],
  "outcome_definition": {
    "horizon_years": 5,
    "min_new_countries_for_spread": 3,
    "min_new_host_families_for_transfer": 1,
    "min_new_host_genera_for_transfer": 2,
    "outcome_label": "spread_label",
    "split_year": 2015
  },
  "primary_model_fallback": "parsimonious_priority",
  "primary_model_name": "discovery_boosted",
  "research_model_names": [
    "H_plus_A",
    "T_plus_A",
    "T_plus_H",
    "balanced_evidence_priority",
    "baseline_country_count",
    "baseline_member_count",
    "bio_clean_priority",
    "bio_residual_synergy_priority",
    "contextual_bio_priority",
    "discovery_12f_class_balanced",
    "discovery_12f_source",
    "discovery_15f_synergy",
    "discovery_4f_hybrid",
    "discovery_5f_phylo",
    "discovery_9f_source",
    "ecology_clinical_priority",
    "enhanced_priority",
    "evidence_aware_priority",
    "firth_parsimonious_priority",
    "full_priority",
    "governance_15f_pruned",
    "graph_evidence_priority",
    "host_transfer_synergy_priority",
    "hybrid_agreement_priority",
    "knownness_calibrated_fusion_priority",
    "knownness_robust_priority",
    "label_noise_aware_priority",
    "monotonic_latent_priority",
    "natural_auc_priority",
    "nonlinear_deconfounded_priority",
    "pairwise_rank_priority",
    "parsimonious_priority",
    "phylo_support_fusion_priority",
    "phylogeny_aware_priority",
    "proxy_light_priority",
    "regime_stability_priority",
    "source_only",
    "sovereign_precision_priority",
    "sovereign_v2_priority",
    "structured_signal_priority",
    "support_calibrated_priority",
    "support_synergy_priority",
    "threat_architecture_priority",
    "visibility_adjusted_priority"
  ],
  "selection_adjusted_p_max": 0.01,
  "single_model_objective_weights": {
    "compute_efficiency": 0.2,
    "predictive_power": 0.4,
    "reliability": 0.4
  },
  "source_holdout_gap_min": -0.005,
  "spatial_holdout_gap_min": -0.03,
  "split_year": 2015
}
```
