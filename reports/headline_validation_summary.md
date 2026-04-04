# Headline Validation Summary

This is the canonical one-page validation surface for jury review.

- Discovery primary: `bio_clean_priority`
- Governance watch-only: `phylo_support_fusion_priority`
- Baseline comparator: `baseline_both`
- Permutation entries below include the selection-adjusted official-model null; the fixed-score label-permutation audit is retained only as an exploratory appendix diagnostic.
- The explicit leakage canary is exported separately in `future_sentinel_audit.tsv`.
- The frozen acceptance audit is exported separately in `frozen_scientific_acceptance_audit.tsv`.
- Frozen scientific acceptance combines matched-knownness, source holdout, spatial holdout, calibration, selection-adjusted null, and leakage review.

| Surface | Model | ROC AUC | ROC AUC 95% CI | AP | AP 95% CI | Brier | Brier Skill | ECE | Max CE | Frozen Acceptance | Frozen Acceptance Reason | Selection-adjusted p | Fixed-score p | Delta vs baseline | Spatial holdout AUC | n | Positives |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| discovery_primary | bio_clean_priority | 0.747 | [0.713, 0.778] | 0.660 | [0.611, 0.715] | 0.187 | 0.193 | 0.042 | NA | fail | fail:matched_knownness,source_holdout | 0.005 | <0.001 | 0.024 ([-0.019, 0.065]) | 0.735 | 989 | 362 |
| governance_watch_only | phylo_support_fusion_priority | 0.827 | [0.800, 0.852] | 0.766 | [0.726, 0.804] | 0.166 | 0.284 | 0.085 | NA | fail | fail:matched_knownness,source_holdout,calibration | 0.005 | NA | NA | 0.818 | 989 | 362 |
| counts_baseline | baseline_both | 0.722 | [0.689, 0.756] | 0.647 | [0.596, 0.698] | 0.186 | 0.198 | 0.039 | NA | fail | fail:matched_knownness | 0.005 | <0.001 | NA | 0.740 | 989 | 362 |

## Rolling-Origin Validation

Nested rolling-origin validation spans outer split years 2012 to 2018 across horizons 1,3,5,8 years and assignment modes all_records,training_only.
Across 54 successful outer-split rows, ROC AUC mean 0.749 (range 0.636 to 0.914) and AP mean 0.326 (range 0.024 to 0.680).
Mean Brier score across the successful outer splits is 0.095.
