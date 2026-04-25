# Benchmark Contract

This document is generated from canonical scientific metadata.

## Version

- protocol_hash: `c6ede6432592ee3725d17a697567d39a381f7156d817702f6fe53b757144410a`
- benchmarks_hash: `93256ff0207e262313a2852d29ddddf9296ac1d152584dffa6e9fb0da01e3201`
- data_contract_sha: `da7b9523c273aa42b72b1e3e057d5755763a424f5e9b3982ff3736de301a9c5b`
- config_sha: `4607bb0d0c4752b169f34cbb264f9832c5069aaabe8a93b28f2efb84a0470eee`

## Canonical Authority

- primary_source: `config/benchmarks.yaml`
- protocol_resolver: `src/plasmid_priority/protocol.py`
- generated_doc: `docs/benchmark_contract.md`

## Global Protocol Scope

- split_year: `2015`
- horizon_years: `5`
- min_new_countries_for_spread: `3`
- min_new_host_genera_for_transfer: `2`
- min_new_host_families_for_transfer: `1`
- primary_model_name: `discovery_boosted`
- governance_model_name: `governance_linear`
- conservative_model_name: `parsimonious_priority`

## Branch Benchmarks

### `bio_transfer`
- name: `bio_transfer_v1`
- split_year: `2015`
- horizon_years: `5`
- assignment_mode: `training_only`
- label_column: `bio_transfer_label`
- outcome_column: `future_new_host_genera_count`

### `clinical_hazard`
- name: `clinical_hazard_v1`
- split_year: `2015`
- horizon_years: `5`
- assignment_mode: `training_only`
- label_column: `clinical_hazard_label`
- outcome_column: `clinical_fraction_future`

### `consensus`
- name: `consensus_v1`
- split_year: `2015`
- horizon_years: `5`
- assignment_mode: `training_only`
- label_column: `spread_label`
- outcome_column: `n_new_countries_future`

### `geo_spread`
- name: `geo_spread_v1`
- split_year: `2015`
- horizon_years: `5`
- assignment_mode: `training_only`
- label_column: `spread_label`
- outcome_column: `n_new_countries`

## Acceptance Thresholds

- matched_knownness_gap_min: `-0.005`
- source_holdout_gap_min: `-0.005`
- spatial_holdout_gap_min: `-0.03`
- ece_max: `0.05`
- selection_adjusted_p_max: `0.01`

## Claim Guardrails

- If strict acceptance fails, claims must remain conditional and benchmark-limited.
- Missing branch predictions must remain explicit and cannot be silently imputed.
- Uncertainty and instability must be surfaced in the official consensus output.
