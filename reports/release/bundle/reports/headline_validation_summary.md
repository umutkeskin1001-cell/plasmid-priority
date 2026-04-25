# Headline Validation Summary

This is the canonical one-page validation surface for jury review.

- Discovery primary benchmark candidate: `discovery_boosted`
- Governance watch-only: `governance_linear`
- Baseline comparator: `baseline_both`
- Benchmark scope note: the headline benchmark does not clear the frozen scientific acceptance gate, so the narrative remains conditional and benchmark-limited.
- Permutation entries below include the selection-adjusted official-model null; the fixed-score label-permutation audit is retained only as an exploratory appendix diagnostic.
- The explicit leakage canary is exported separately in `future_sentinel_audit.tsv`.
- The frozen acceptance audit is exported separately in `frozen_scientific_acceptance_audit.tsv`.
- The nonlinear deconfounding audit is exported separately in `nonlinear_deconfounding_audit.tsv`.
- Calibration metrics below are fixed-bin diagnostics: ECE, max calibration error, calibration slope, and calibration intercept are reported with their binning semantics made explicit.
- Alternative endpoint audits are exported separately in `ordinal_outcome_audit.tsv`, `exposure_adjusted_event_outcomes.tsv`, and `macro_region_jump_outcome.tsv`.
- The prospective freeze audits are exported separately in `prospective_candidate_freeze.tsv` and `annual_candidate_freeze_summary.tsv`.
- The graph, counterfactual, geographic-jump, and AMR-uncertainty diagnostics are exported separately in `mash_similarity_graph.tsv`, `counterfactual_shortlist_comparison.tsv`, `geographic_jump_distance_outcome.tsv`, and `amr_uncertainty_summary.tsv`.
- Frozen scientific acceptance combines matched-knownness, source holdout, spatial holdout, calibration, selection-adjusted null, and leakage review.

| Surface | Model | ROC AUC | ROC AUC 95% CI | AP | AP 95% CI | Brier | Brier Skill | Fixed-bin ECE | Fixed-bin Max CE | Calibration Slope | Calibration Intercept | Frozen Acceptance | Frozen Acceptance Reason | Selection-adjusted p | Fixed-score p | Delta vs baseline | Spatial holdout AUC | n | Positives |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| discovery_primary | discovery_boosted | 0.811 | [0.780, 0.839] | 0.721 | [0.675, 0.768] | 0.172 | 0.257 | 0.092 | NA | NA | NA | fail | fail:matched_knownness,source_holdout,spatial_holdout,calibration,selection_adjusted_null | 0.091 | 0.005 | 0.074 ([0.040, 0.107]) | 0.710 | 989 | 362 |
| governance_watch_only | governance_linear | 0.692 | [0.655, 0.723] | 0.604 | [0.549, 0.652] | 0.211 | 0.092 | 0.100 | NA | NA | NA | fail | fail:source_holdout,calibration,selection_adjusted_null | 0.091 | NA | NA | 0.679 | 989 | 362 |
| counts_baseline | baseline_both | 0.736 | [0.702, 0.768] | 0.648 | [0.599, 0.700] | 0.186 | 0.199 | 0.040 | NA | NA | NA | fail | fail:matched_knownness,selection_adjusted_null | 0.091 | 0.005 | NA | 0.737 | 989 | 362 |
| single_model_pareto_official | governance_linear__pruned | 0.709 | NA | 0.616 | NA | NA | NA | 0.095 | NA | NA | NA | not_scored | missing:calibration_intercept,calibration_slope | 0.091 | NA | NA | NA | 0 | 0 |

## Single-Model Pareto Decision

- Official single-model candidate: `governance_linear__pruned`; status `not_scored`; reason `best_available_weighted_tradeoff_pending_full_acceptance`.
- Selected from `4` Pareto finalists after finalist-heavy audit.
- Weighted objective `0.111` with failure severity `4.009`.
- Full Stage A screen time for the winning candidate family row was `0.35` seconds.

## Rolling-Origin Validation

Nested rolling-origin validation spans outer split years 2016 to 2017 across horizons 5 years and assignment modes all_records,training_only.
Across 4 successful outer-split rows, ROC AUC mean 0.790 (range 0.776 to 0.805) and AP mean 0.572 (range 0.538 to 0.605).
Mean Brier score across the successful outer splits is 0.175.

## Blocked Holdout Audit

- discovery-boosted primary model blocked holdout audit (dominant_source): weighted ROC AUC `0.512` across `2` blocked groups; hardest group `dominant_source:refseq_leaning` at ROC AUC `0.494`. discovery-boosted primary model blocked holdout audit (dominant_region_train): weighted ROC AUC `0.710` across `5` blocked groups; hardest group `dominant_region_train:Oceania` at ROC AUC `0.614`.

## Country Missingness

- discovery-boosted primary model country-missingness audit (`country_missingness_bounds.tsv`, `country_missingness_sensitivity.tsv`): observed labels mark 362/989 eligible backbones positive; midpoint / optimistic / weighted interpretations shift 75/89/42 labels and yield 437/451/404 positives. Sensitivity across those label variants spans ROC AUC 0.803 to 0.814 and AP 0.721 to 0.790..

## Ranking Stability

- `candidate_rank_stability.tsv` records candidate rank stability across bootstrap resamples; the strongest stable backbone `AA316` remains in the top-`10` set at frequency `1.00`.
- `candidate_variant_consistency.tsv` records candidate rank stability across model variants; the strongest stable backbone `AA171` remains in the top-`10` set at frequency `1.00`.
