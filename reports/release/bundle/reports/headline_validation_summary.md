# Headline Validation Summary

This is the canonical one-page validation surface for jury review.

- Discovery primary: `discovery_12f_source`
- Governance watch-only: `phylo_support_fusion_priority`
- Baseline comparator: `baseline_both`
- Permutation entries below include the selection-adjusted official-model null; the fixed-score label-permutation audit is retained only as an exploratory appendix diagnostic.
- The explicit leakage canary is exported separately in `future_sentinel_audit.tsv`.
- The frozen acceptance audit is exported separately in `frozen_scientific_acceptance_audit.tsv`.
- The nonlinear deconfounding audit is exported separately in `nonlinear_deconfounding_audit.tsv`.
- Alternative endpoint audits are exported separately in `ordinal_outcome_audit.tsv`, `exposure_adjusted_event_outcomes.tsv`, and `macro_region_jump_outcome.tsv`.
- The prospective freeze audits are exported separately in `prospective_candidate_freeze.tsv` and `annual_candidate_freeze_summary.tsv`.
- The graph, counterfactual, geographic-jump, and AMR-uncertainty diagnostics are exported separately in `mash_similarity_graph.tsv`, `counterfactual_shortlist_comparison.tsv`, `geographic_jump_distance_outcome.tsv`, and `amr_uncertainty_summary.tsv`.
- Frozen scientific acceptance combines matched-knownness, source holdout, spatial holdout, calibration, selection-adjusted null, and leakage review.

| Surface | Model | ROC AUC | ROC AUC 95% CI | AP | AP 95% CI | Brier | Brier Skill | ECE | Max CE | Frozen Acceptance | Frozen Acceptance Reason | Selection-adjusted p | Fixed-score p | Delta vs baseline | Spatial holdout AUC | n | Positives |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| discovery_primary | discovery_12f_source | 0.804 | [0.775, 0.831] | 0.723 | [0.678, 0.770] | 0.169 | 0.270 | 0.049 | NA | fail | fail:matched_knownness,source_holdout | <0.001 | <0.001 | 0.081 ([0.046, 0.116]) | 0.789 | 989 | 362 |
| governance_watch_only | phylo_support_fusion_priority | 0.827 | [0.800, 0.852] | 0.767 | [0.725, 0.803] | 0.166 | 0.284 | 0.085 | NA | fail | fail:matched_knownness,source_holdout,calibration | <0.001 | NA | NA | 0.821 | 989 | 362 |
| counts_baseline | baseline_both | 0.722 | [0.689, 0.756] | 0.647 | [0.596, 0.698] | 0.186 | 0.198 | 0.039 | NA | fail | fail:matched_knownness | <0.001 | <0.001 | NA | 0.740 | 989 | 362 |
| single_model_pareto_official | discovery_12f_source | 0.804 | [0.775, 0.831] | 0.723 | [0.678, 0.770] | 0.169 | 0.270 | 0.049 | NA | fail | fail:matched_knownness | 0.005 | <0.001 | 0.081 ([0.046, 0.116]) | 0.789 | 989 | 362 |

## Single-Model Pareto Decision

- Official single-model candidate: `discovery_12f_source`; status `fail`; reason `lowest_failure_severity_with_competitive_auc`.
- Selected from `3` Pareto finalists after finalist-heavy audit.
- Weighted objective `0.500` with failure severity `1.993`.
- Full Stage A screen time for the winning candidate family row was `0.20` seconds.

## Rolling-Origin Validation

Nested rolling-origin validation spans outer split years 2012 to 2018 across horizons 1,3,5,8 years and assignment modes all_records,training_only.
Across 54 successful outer-split rows, ROC AUC mean 0.749 (range 0.636 to 0.914) and AP mean 0.326 (range 0.024 to 0.680).
Mean Brier score across the successful outer splits is 0.095.

## Blocked Holdout Audit

- discovery_12f_source blocked holdout audit (dominant_region_train + dominant_source): weighted ROC AUC `0.786` across `7` blocked groups; hardest group `dominant_source:insd_leaning` at ROC AUC `0.684`.

## Country Missingness

- discovery_12f_source country-missingness audit (`country_missingness_bounds.tsv`, `country_missingness_sensitivity.tsv`): observed labels mark 362/989 eligible backbones positive; midpoint / optimistic / weighted interpretations shift 75/89/42 labels and yield 437/451/404 positives. Sensitivity across those label variants spans ROC AUC 0.793 to 0.804 and AP 0.723 to 0.780..

## Ranking Stability

- `candidate_rank_stability.tsv` records candidate rank stability across bootstrap resamples; the strongest stable backbone `AA175` remains in the top-`25` set at frequency `1.00`.
- `candidate_variant_consistency.tsv` records candidate rank stability across model variants; the strongest stable backbone `AA324` remains in the top-`25` set at frequency `0.88`.
