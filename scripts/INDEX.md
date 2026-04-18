# Script Index

This index documents every pipeline entry point. Scripts are numbered by execution
order; `b`-suffixed scripts are optional or parallel alternatives to the same slot.

| Script | Description | Reads | Writes | Est. Time |
|--------|-------------|-------|--------|-----------|
| `00_fetch_external_data.py` | Fetch PLSDB, RefSeq, and WHO data | — | `data/raw/*` | 5-30 min |
| `01_check_inputs.py` | Validate all required inputs exist and are non-empty | `data/manifests/data_contract.json`, `data/raw/*` | run summary | <5 s |
| `02_build_all_plasmids_fasta.py` | Merge PLSDB + RefSeq FASTA into unified file | `data/raw/*.fasta` | `data/raw/all_plasmids.fasta` | 1-5 min |
| `03_build_bronze_table.py` | Parse metadata into canonical bronze table | `data/raw/*` | `data/bronze/*.tsv` | 1-3 min |
| `04_harmonize_metadata.py` | Harmonize country/year/host fields | `data/bronze/*.tsv` | `data/silver/harmonized_plasmids.tsv` | 1-2 min |
| `05_deduplicate.py` | Deduplicate by accession + metadata criteria | `data/silver/harmonized_plasmids.tsv` | `data/silver/deduplicated_plasmids.tsv` | 30 s |
| `06_annotate_mobility.py` | Run MOB-suite (or load cached) to annotate mobility/MPF/replicon | `data/silver/deduplicated_plasmids.tsv` | `data/silver/mobility_annotations.tsv` | 10-60 min |
| `07_annotate_amr.py` | Run AMRFinderPlus (or load cached) | `data/raw/all_plasmids.fasta` | `data/silver/amr_annotations.tsv` | 10-60 min |
| `08_build_amr_consensus.py` | Aggregate per-plasmid AMR calls | `data/silver/amr_annotations.tsv` | `data/silver/plasmid_amr_consensus.tsv` | 30 s |
| `09_assign_backbones.py` | Group plasmids into operational backbone classes | `data/silver/deduplicated_plasmids.tsv`, `data/silver/mobility_annotations.tsv` | `data/silver/plasmid_backbones.tsv` | 30 s |
| `10_compute_coherence.py` | Compute within-backbone coherence scores | `data/silver/plasmid_backbones.tsv` | `data/silver/backbone_coherence.tsv` | 30 s |
| `11_compute_feature_T.py` | Transfer potential (T) feature engineering | `data/silver/plasmid_backbones.tsv` | `data/features/feature_T.tsv` | 1-2 min |
| `12_compute_feature_H.py` | Host diversity (H) feature engineering | `data/silver/plasmid_backbones.tsv` | `data/features/feature_H.tsv` | 1-5 min |
| `13_compute_feature_A.py` | AMR burden (A) feature engineering | `data/silver/plasmid_backbones.tsv`, `data/silver/plasmid_amr_consensus.tsv` | `data/features/feature_A.tsv` | 1-2 min |
| `14_build_backbone_table.py` | Merge all features into master backbone table | `data/features/*.tsv`, `data/silver/backbone_coherence.tsv` | `data/silver/backbone_table.tsv` | 30 s |
| `15_normalize_and_score.py` | Rank-percentile normalization → priority index | `data/silver/backbone_table.tsv` | `data/scores/backbone_scored.tsv` | 30 s |
| `16_run_module_A.py` | OOF cross-validation for all priority models | `data/scores/backbone_scored.tsv` | `data/analysis/module_a_metrics.json`, `data/analysis/module_a_predictions.tsv` | 5-30 min |
| `17_run_module_B.py` | Exploratory AMR comparison module | `data/scores/backbone_scored.tsv`, `data/silver/plasmid_amr_consensus.tsv` | `data/analysis/module_b_*.tsv` | 5-10 min |
| `18_run_module_C_pathogen_detection.py` | Pathogen Detection metadata probe (supportive) | `data/raw/pathogen_detection/*.tsv` | `data/analysis/module_c_*.tsv` | 2-10 min |
| `19_run_module_D_external_support.py` | External literature/db support layer | `data/raw/literature_support/*` | `data/analysis/module_d_*.tsv` | 1-5 min |
| `20_run_module_E_amrfinder_concordance.py` | AMRFinder concordance probe (skipped if binary absent) | `data/silver/plasmid_amr_consensus.tsv` | `data/analysis/module_e_*.tsv` | 1-5 min |
| `21_run_validation.py` | Full validation: calibration, subgroup, holdout, permutation tests, VIF | `data/scores/backbone_scored.tsv`, `data/analysis/module_a_*.{json,tsv}` | `data/analysis/*_validation*.tsv`, `data/analysis/vif_audit*.tsv` | 5-20 min |
| `22_run_sensitivity.py` | Sensitivity analysis: consensus weight sweep ±0.1 | `data/scores/backbone_scored.tsv` | `data/analysis/sensitivity_*.tsv` | 5-15 min |
| `23_run_module_f_enrichment.py` | Plasmid enrichment scoring | `data/scores/backbone_scored.tsv` | `data/analysis/enrichment_*.tsv` | 1-5 min |
| `24_build_reports.py` | Build all reports, tables, and figures | `data/analysis/*.{json,tsv}` | `reports/**/*` | 5-20 min |
| `25_export_tubitak_summary.py` | Export TÜBİTAK-ready headline metrics | `data/analysis/module_a_metrics.json`, `data/analysis/permutation_null_summary.tsv` | `reports/tubitak_final_metrics.txt` | <5 s |
| `26_run_tests_or_smoke.py` | Manual/Optional: run test suite or smoke check (CI entry point) | — | run summary | 30-120 s |
| `27_generate_full_fit_predictions.py` | Generate predictions from full-data model fit | `data/scores/backbone_scored.tsv` | `data/analysis/full_fit_predictions.tsv` | 5-15 min |
| `27b_run_advanced_audits.py` | Advanced audit suite (lead-time bias, spatial holdout) | `data/scores/backbone_scored.tsv`, `data/analysis/*.tsv` | `data/analysis/advanced_audit_*.tsv` | 10-30 min |
| `28_build_release_bundle.py` | Build signed release bundle for archiving | `reports/**/*` | `reports/release/bundle/*` | 1-5 min |
| `28b_run_sovereign_ensemble.py` | Sovereign ensemble model run (experimental) | `data/scores/backbone_scored.tsv` | `data/experiments/sovereign_*.tsv` | 5-30 min |
| `29_build_experiment_registry.py` | Build experiment hash registry | `data/experiments/*.tsv` | `data/experiments/registry.json` | 30 s |
| `29_train_sovereign.py` | Train sovereign model checkpoint | `data/scores/backbone_scored.tsv` | `data/experiments/sovereign_checkpoint.*` | 10-30 min |
| `30_train_ensemble.py` | Train full ensemble | `data/scores/backbone_scored.tsv` | `data/experiments/ensemble_*.pkl` | 10-60 min |
| `31_generate_scientific_contracts.py` | Generate canonical benchmark/protocol/model/data contract docs | `config/benchmarks.yaml`, `config*.yaml`, `data/manifests/data_contract.json` | `docs/{benchmark_contract,scientific_protocol,model_card,data_card}.md`, `reports/reviewer_pack/*` | <10 s |
| `32_freeze_and_invariants.py` | Build freeze snapshot and evaluate invariant drift | `reports/core_tables/*.tsv` | `reports/freeze/*` | <10 s |
| `33_scientific_equivalence.py` | Scientific equivalence harness between baseline and candidate freeze snapshots | `reports/freeze/*.json` | `reports/freeze/scientific_equivalence.json` | <5 s |
| `34_generate_disposition_ledger.py` | Generate keep/absorb/replace/delete ledger for legacy surfaces | source tree inventory | `docs/disposition_ledger.md` | <5 s |
| `35_generate_code_size_contract.py` | Generate file/function size contract report for god-file/god-function cleanup tracking | `src/plasmid_priority/**/*.py`, `scripts/**/*.py` | `reports/quality/code_size_contract.json` | <5 s |

## Utility / Non-Pipeline Scripts

| Script | Description |
|--------|-------------|
| `build_phase_52_report.py` | Phase 5.2 detailed discovery report (archival) |
| `check_code_review_graph.py` | Validate code review graph integrity |
| `generate_hardening_snapshot.py` | Generate hardening audit snapshot |
| `run_bio_transfer_branch.py` | Standalone bio-transfer branch evaluation |
| `run_clinical_hazard_branch.py` | Standalone clinical-hazard branch evaluation |
| `run_consensus_branch.py` | Standalone consensus fusion evaluation |
| `run_geo_spread_branch.py` | Standalone geo-spread branch evaluation |
| `run_branch.py` | Canonical unified branch CLI (`--branch geo_spread|bio_transfer|clinical_hazard|consensus`) |
| `run_governance_temporal_evidence.py` | Governance track temporal evidence audit |
| `run_hardening_summary.py` | Hardening audit summary |
| `run_missingness_audit.py` | Missingness / metadata completeness audit |
| `run_mode.py` | Pipeline mode router (`fast-local`, `full-local`) |
| `run_phase_52_discovery.py` | Phase 5.2 discovery track run (archival) |
| `run_phase_61_governance_pruning.py` | Phase 6.1 governance pruning (archival) |
| `run_schema_validation.py` | Standalone schema validation |
| `run_workflow.py` | Orchestrator for `make pipeline` |

## Key Dependencies

```
Required env vars:
  PLASMID_PRIORITY_DATA_ROOT  — path to large data directory (USB disk / NAS)
  PLASMID_PRIORITY_FIRTHLOGIST_PYTHON  — path to Python with firthlogist installed

Required extras (for primary model):
  pip install -e ".[analysis,dev,tree-models]"
```
