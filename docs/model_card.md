# Model Card

## Official Surface

- official_model_names: `['discovery_boosted', 'governance_linear', 'baseline_both']`
- core_model_names: `['baseline_both', 'governance_linear', 'discovery_boosted', 'discovery_graph_boosted', 'T_only', 'H_only', 'A_only', 'T_plus_H_plus_A', 'parsimonious_priority', 'support_synergy_priority']`
- research_model_names: `['H_plus_A', 'T_plus_A', 'T_plus_H', 'balanced_evidence_priority', 'baseline_country_count', 'baseline_member_count', 'bio_clean_priority', 'bio_residual_synergy_priority', 'contextual_bio_priority', 'discovery_12f_class_balanced', 'discovery_12f_source', 'discovery_15f_synergy', 'discovery_4f_hybrid', 'discovery_5f_phylo', 'discovery_9f_source', 'ecology_clinical_priority', 'enhanced_priority', 'evidence_aware_priority', 'firth_parsimonious_priority', 'full_priority', 'governance_15f_pruned', 'graph_evidence_priority', 'host_transfer_synergy_priority', 'hybrid_agreement_priority', 'knownness_calibrated_fusion_priority', 'knownness_robust_priority', 'label_noise_aware_priority', 'monotonic_latent_priority', 'natural_auc_priority', 'nonlinear_deconfounded_priority', 'pairwise_rank_priority', 'parsimonious_priority', 'phylo_support_fusion_priority', 'phylogeny_aware_priority', 'proxy_light_priority', 'regime_stability_priority', 'source_only', 'sovereign_precision_priority', 'sovereign_v2_priority', 'structured_signal_priority', 'support_calibrated_priority', 'support_synergy_priority', 'threat_architecture_priority', 'visibility_adjusted_priority']`
- ablation_model_names: `['T_only', 'H_only', 'A_only', 'T_plus_H_plus_A']`

## Governance

- primary_model_name: `discovery_boosted`
- governance_model_name: `governance_linear`
- conservative_model_name: `parsimonious_priority`

## Legacy Status

- `config.yaml#models` is legacy/research surface and not official benchmark truth.
