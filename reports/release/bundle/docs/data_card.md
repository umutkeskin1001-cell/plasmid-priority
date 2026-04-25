# Data Card

## Branch Data Contracts

### `bio_transfer`
- benchmark_name: `bio_transfer_v1`
- label_column: `bio_transfer_label`
- outcome_column: `future_new_host_genera_count`
- required_columns: `['backbone_id', 'bio_transfer_label', 'future_new_host_genera_count', 'future_new_host_families_count', 'split_year', 'backbone_assignment_mode', 'max_resolved_year_train', 'min_resolved_year_test', 'training_only_future_unseen_backbone_flag']`

### `clinical_hazard`
- benchmark_name: `clinical_hazard_v1`
- label_column: `clinical_hazard_label`
- outcome_column: `clinical_fraction_future`
- required_columns: `['backbone_id', 'clinical_hazard_label', 'clinical_fraction_future', 'last_resort_fraction_future', 'mdr_proxy_fraction_future', 'pd_clinical_support_future', 'split_year', 'backbone_assignment_mode', 'max_resolved_year_train', 'min_resolved_year_test', 'training_only_future_unseen_backbone_flag']`

### `consensus`
- benchmark_name: `consensus_v1`
- label_column: `spread_label`
- outcome_column: `n_new_countries_future`
- required_columns: `['backbone_id', 'spread_label', 'split_year', 'backbone_assignment_mode', 'max_resolved_year_train', 'min_resolved_year_test', 'training_only_future_unseen_backbone_flag']`

### `geo_spread`
- benchmark_name: `geo_spread_v1`
- label_column: `spread_label`
- outcome_column: `n_new_countries`
- required_columns: `['backbone_id', 'spread_label', 'n_new_countries', 'split_year', 'backbone_assignment_mode', 'max_resolved_year_train', 'min_resolved_year_test', 'training_only_future_unseen_backbone_flag']`

## Provenance Pins

- data_contract_sha: `da7b9523c273aa42b72b1e3e057d5755763a424f5e9b3982ff3736de301a9c5b`
- config_sha: `4607bb0d0c4752b169f34cbb264f9832c5069aaabe8a93b28f2efb84a0470eee`
- protocol_hash: `c6ede6432592ee3725d17a697567d39a381f7156d817702f6fe53b757144410a`
