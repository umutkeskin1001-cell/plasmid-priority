# Progress Snapshot (2026-04-22)

## Live Pipeline State
19566 Ss   02:41:48   0.0  0.1 uv run python scripts/run_workflow.py pipeline --max-workers 4
19569 S    02:41:48   0.0  0.5 /Users/umut/Projeler/plasmid-priority/.venv/bin/python3 scripts/run_workflow.py pipeline --max-workers 4
19609 R    02:41:46  99.3  2.7 /Users/umut/Projeler/plasmid-priority/.venv/bin/python3 /Users/umut/Projeler/plasmid-priority/scripts/21_run_validation.py --jobs 2 --single-model-screen-splits 1 --single-model-screen-repeats 1 --single-model-finalist-splits 1 --single-model-finalist-repeats 1 --selection-adjusted-permutations 10

## Git Status Summary
modified_or_staged=163
untracked=85

## Full Changed Paths (Tracked)
.github/workflows/ci.yml
Makefile
data/manifests/data_contract.json
docs/benchmark_contract.md
docs/data_card.md
pyproject.toml
reports/release/bundle/RELEASE_INFO.txt
reports/release/bundle/reports/diagnostic_tables/single_model_pareto_finalists.tsv
reports/release/bundle/reports/diagnostic_tables/single_model_pareto_screen.tsv
reports/release/plasmid_priority_release_bundle.zip
reports/release/plasmid_priority_release_manifest.json
reports/reviewer_pack/README.md
reports/reviewer_pack/canonical_metadata.json
scripts/00_fetch_external_data.py
scripts/01_check_inputs.py
scripts/02_build_all_plasmids_fasta.py
scripts/03_build_bronze_table.py
scripts/04_harmonize_metadata.py
scripts/06_annotate_mobility.py
scripts/07_annotate_amr.py
scripts/08_build_amr_consensus.py
scripts/09_assign_backbones.py
scripts/14_build_backbone_table.py
scripts/15_normalize_and_score.py
scripts/16_run_module_A.py
scripts/17_run_module_B.py
scripts/18_run_module_C_pathogen_detection.py
scripts/19_run_module_D_external_support.py
scripts/20_run_module_E_amrfinder_concordance.py
scripts/21_run_validation.py
scripts/22_run_sensitivity.py
scripts/23_run_module_f_enrichment.py
scripts/24_build_reports.py
scripts/25_export_tubitak_summary.py
scripts/26_run_tests_or_smoke.py
scripts/27_generate_full_fit_predictions.py
scripts/27b_run_advanced_audits.py
scripts/28_build_release_bundle.py
scripts/28b_run_sovereign_ensemble.py
scripts/29_build_experiment_registry.py
scripts/29_train_sovereign.py
scripts/30_train_ensemble.py
scripts/31_generate_scientific_contracts.py
scripts/34_generate_disposition_ledger.py
scripts/35_generate_code_size_contract.py
scripts/archive/build_phase_52_report.py
scripts/archive/check_code_review_graph.py
scripts/archive/run_phase_52_discovery.py
scripts/check_import_contracts.py
scripts/generate_hardening_snapshot.py
scripts/run_governance_temporal_evidence.py
scripts/run_hardening_summary.py
scripts/run_mode.py
scripts/run_phase_61_governance_pruning.py
scripts/run_rolling_origin_validation.py
scripts/run_workflow.py
src/plasmid_priority/annotate/__init__.py
src/plasmid_priority/annotate/tables.py
src/plasmid_priority/api/app.py
src/plasmid_priority/backbone/core.py
src/plasmid_priority/bio_transfer/calibration.py
src/plasmid_priority/bio_transfer/cli.py
src/plasmid_priority/bio_transfer/evaluate.py
src/plasmid_priority/bio_transfer/features.py
src/plasmid_priority/bio_transfer/report.py
src/plasmid_priority/clinical_hazard/calibration.py
src/plasmid_priority/clinical_hazard/cli.py
src/plasmid_priority/clinical_hazard/dataset.py
src/plasmid_priority/clinical_hazard/evaluate.py
src/plasmid_priority/clinical_hazard/features.py
src/plasmid_priority/clinical_hazard/report.py
src/plasmid_priority/config.py
src/plasmid_priority/consensus/calibration.py
src/plasmid_priority/consensus/cli.py
src/plasmid_priority/consensus/evaluate.py
src/plasmid_priority/consensus/fuse.py
src/plasmid_priority/consensus/report.py
src/plasmid_priority/dedup/canonicalize.py
src/plasmid_priority/features/__init__.py
src/plasmid_priority/features/core.py
src/plasmid_priority/geo_spread/calibration.py
src/plasmid_priority/geo_spread/cli.py
src/plasmid_priority/geo_spread/contracts.py
src/plasmid_priority/geo_spread/enrichment.py
src/plasmid_priority/geo_spread/evaluate.py
src/plasmid_priority/geo_spread/features.py
src/plasmid_priority/geo_spread/provenance.py
src/plasmid_priority/geo_spread/report.py
src/plasmid_priority/geo_spread/select.py
src/plasmid_priority/geo_spread/specs.py
src/plasmid_priority/geo_spread/train.py
src/plasmid_priority/governance/canonical_metadata.py
src/plasmid_priority/governance/freeze.py
src/plasmid_priority/harmonize/metadata.py
src/plasmid_priority/harmonize/records.py
src/plasmid_priority/io/__init__.py
src/plasmid_priority/io/fasta.py
src/plasmid_priority/io/tabular.py
src/plasmid_priority/modeling/__init__.py
src/plasmid_priority/modeling/discovery_contract.py
src/plasmid_priority/modeling/ensemble_strategies.py
src/plasmid_priority/modeling/firthlogist_bridge.py
src/plasmid_priority/modeling/module_a.py
src/plasmid_priority/modeling/module_a_support.py
src/plasmid_priority/modeling/nested_cv.py
src/plasmid_priority/modeling/plugin_system.py
src/plasmid_priority/modeling/shap_explainer.py
src/plasmid_priority/modeling/single_model_pareto.py
src/plasmid_priority/modeling/tree_models.py
src/plasmid_priority/pipeline/step_contract.py
src/plasmid_priority/protocol.py
src/plasmid_priority/qc/input_checks.py
src/plasmid_priority/reporting/__init__.py
src/plasmid_priority/reporting/advanced_audits.py
src/plasmid_priority/reporting/amrfinder_support.py
src/plasmid_priority/reporting/candidate_tables.py
src/plasmid_priority/reporting/enrichment.py
src/plasmid_priority/reporting/epv_audit.py
src/plasmid_priority/reporting/external_support.py
src/plasmid_priority/reporting/figure_style.py
src/plasmid_priority/reporting/figures.py
src/plasmid_priority/reporting/hardening_summary.py
src/plasmid_priority/reporting/lead_time_bias_audit.py
src/plasmid_priority/reporting/model_audit.py
src/plasmid_priority/reporting/narrative_utils.py
src/plasmid_priority/reporting/overview.py
src/plasmid_priority/reporting/pathogen_support.py
src/plasmid_priority/reporting/report_build_helpers.py
src/plasmid_priority/reporting/report_candidate_helpers.py
src/plasmid_priority/reporting/report_surface_helpers.py
src/plasmid_priority/runtime.py
src/plasmid_priority/schemas/contracts.py
src/plasmid_priority/scoring/core.py
src/plasmid_priority/shared/branching.py
src/plasmid_priority/shared/calibration.py
src/plasmid_priority/shared/contracts.py
src/plasmid_priority/shared/data_inventory.py
src/plasmid_priority/shared/explanations.py
src/plasmid_priority/shared/labels.py
src/plasmid_priority/shared/provenance.py
src/plasmid_priority/shared/selection.py
src/plasmid_priority/shared/specs.py
src/plasmid_priority/snapshots.py
src/plasmid_priority/utils/__init__.py
src/plasmid_priority/utils/benchmarking.py
src/plasmid_priority/utils/dataframe.py
src/plasmid_priority/utils/files.py
src/plasmid_priority/utils/geography.py
src/plasmid_priority/utils/managed_run.py
src/plasmid_priority/utils/parallel.py
src/plasmid_priority/validation/__init__.py
src/plasmid_priority/validation/boundaries.py
src/plasmid_priority/validation/falsification.py
src/plasmid_priority/validation/metrics.py
src/plasmid_priority/validation/missingness.py
src/plasmid_priority/validation/rolling_origin.py
src/plasmid_priority/validation/schemas.py
src/plasmid_priority/validation/vif.py
tests/test_hardening_batch_1.py
tests/test_run_mode.py
tests/test_smoke_runner.py
tests/test_utils_dataframe.py
tests/test_workflow.py

## Full Untracked Paths
?? .cursor/
?? .windsurfrules
?? AGENTS.md
?? CLAUDE.md
?? config/model_compute_tiers.yaml
?? config/performance_budgets.yaml
?? config/scoring_weights.yaml
?? plan.md
?? reports/audits/
?? reports/core_tables/literature_validation_matrix.tsv
?? reports/performance/
?? reports/release/bundle/docs/
?? reports/release/bundle/reports/performance/
?? reports/release/bundle/reports/release/
?? reports/release/bundle/reports/reviewer_pack/
?? reports/release/release_readiness_report.json
?? reports/release/release_readiness_report.md
?? reports/reviewer_pack/candidate_evidence_dossiers/
?? reports/reviewer_pack/run_reproducibility.sh
?? scripts/36_report_raw_data_usage.py
?? scripts/38_artifact_integrity_gate.py
?? scripts/39_runtime_budget_gate.py
?? scripts/40_scientific_contract_gate.py
?? scripts/41_build_performance_dashboard.py
?? scripts/42_release_readiness_report.py
?? scripts/profile_workflow.py
?? src/plasmid_priority/annotate/sequence_cache.py
?? src/plasmid_priority/api/artifact_registry.py
?? src/plasmid_priority/cache/
?? src/plasmid_priority/consensus/optuna_weights.py
?? src/plasmid_priority/evidence/
?? src/plasmid_priority/features/core_impl.py
?? src/plasmid_priority/features/host.py
?? src/plasmid_priority/features/interaction.py
?? src/plasmid_priority/features/temporal_leak.py
?? src/plasmid_priority/features/transmission.py
?? src/plasmid_priority/io/table_io.py
?? src/plasmid_priority/modeling/fit_config.py
?? src/plasmid_priority/modeling/fold_cache.py
?? src/plasmid_priority/modeling/folds.py
?? src/plasmid_priority/modeling/knownness.py
?? src/plasmid_priority/modeling/matrix_cache.py
?? src/plasmid_priority/modeling/module_a_impl.py
?? src/plasmid_priority/modeling/oof_cache.py
?? src/plasmid_priority/modeling/task.py
?? src/plasmid_priority/performance/
?? src/plasmid_priority/reporting/advanced_audits_impl.py
?? src/plasmid_priority/reporting/build_reports_script_impl.py
?? src/plasmid_priority/reporting/cache.py
?? src/plasmid_priority/reporting/figures_impl.py
?? src/plasmid_priority/reporting/literature_validation.py
?? src/plasmid_priority/reporting/model_audit_impl.py
?? src/plasmid_priority/sensitivity/
?? src/plasmid_priority/shared/branch_base.py
?? src/plasmid_priority/shared/branches.py
?? src/plasmid_priority/utils/coercion.py
?? src/plasmid_priority/utils/numeric_ops.py
?? src/plasmid_priority/utils/parquet_utils.py
?? src/plasmid_priority/utils/sensitivity_cache.py
?? src/plasmid_priority/utils/threshold_sensitivity.py
?? src/plasmid_priority/validation/artifact_integrity.py
?? src/plasmid_priority/validation/label_cards.py
?? src/plasmid_priority/validation/release_readiness.py
?? src/plasmid_priority/validation/scientific_contract.py
?? tests/test_api_artifact_registry.py
?? tests/test_artifact_cache.py
?? tests/test_artifact_integrity_validation.py
?? tests/test_evidence_claim_rules.py
?? tests/test_label_cards.py
?? tests/test_literature_validation.py
?? tests/test_model_caches.py
?? tests/test_module_a_compute_tiers.py
?? tests/test_performance_dashboard.py
?? tests/test_performance_telemetry.py
?? tests/test_release_readiness.py
?? tests/test_reporting_cache.py
?? tests/test_scientific_contract.py
?? tests/test_sensitivity_cache.py
?? tests/test_sequence_annotation_cache.py
?? tests/test_table_io.py
?? tests/test_utils_benchmarking.py
?? tests/test_utils_math.py
?? tests/test_validation_boundaries.py
?? tests/test_validation_schemas.py
?? tests/test_validation_vif.py

## Recent Step Results
2026-04-22 19:12:35	01_check_inputs_step_result.json	status=ok	cache=miss	duration=24.034875125000326
2026-04-22 18:57:32	22_run_sensitivity_step_result.json	status=ok	cache=miss	duration=1384.9810704590004
2026-04-21 22:56:36	21_run_validation_step_result.json	status=ok	cache=miss	duration=53.20490858400444
2026-04-21 22:56:08	18_run_module_C_pathogen_detection_step_result.json	status=ok	cache=miss	duration=27.207737000004272
2026-04-21 22:55:47	19_run_module_D_external_support_step_result.json	status=ok	cache=miss	duration=6.512215542003105
2026-04-21 22:55:43	20_run_module_E_amrfinder_concordance_step_result.json	status=ok	cache=miss	duration=2.5329670419960166
2026-04-21 22:55:40	16_run_module_A_step_result.json	status=ok	cache=miss	duration=17.290576624996902
2026-04-21 22:55:27	23_run_module_f_enrichment_step_result.json	status=ok	cache=miss	duration=4.133945875000791
2026-04-21 22:55:25	17_run_module_B_step_result.json	status=ok	cache=miss	duration=2.0025079999977606
2026-04-21 22:55:23	15_normalize_and_score_step_result.json	status=ok	cache=miss	duration=2.237288417003583
2026-04-21 22:55:20	14_build_backbone_table_step_result.json	status=ok	cache=miss	duration=2.598505583999213
2026-04-21 22:55:18	13_compute_feature_A_step_result.json	status=ok	cache=miss	duration=2.531341457994131
2026-04-21 22:55:15	12_compute_feature_H_step_result.json	status=ok	cache=miss	duration=2.3050259169976925
2026-04-21 22:55:13	11_compute_feature_T_step_result.json	status=ok	cache=miss	duration=2.2440892919985345
2026-04-21 22:55:10	10_compute_coherence_step_result.json	status=ok	cache=miss	duration=2.1350384580000537
2026-04-21 22:55:07	09_assign_backbones_step_result.json	status=ok	cache=miss	duration=3.258670915995026
2026-04-21 22:55:04	08_build_amr_consensus_step_result.json	status=ok	cache=miss	duration=4.2928967909974745
2026-04-21 22:55:00	07_annotate_amr_step_result.json	status=ok	cache=miss	duration=6.562816166995617
2026-04-21 22:54:53	06_annotate_mobility_step_result.json	status=ok	cache=miss	duration=3.98813579200214
2026-04-21 22:54:48	05_deduplicate_step_result.json	status=ok	cache=miss	duration=3.291371416002221
2026-04-21 22:54:44	04_harmonize_metadata_step_result.json	status=ok	cache=miss	duration=16.216271624994988
2026-04-21 22:54:27	03_build_bronze_table_step_result.json	status=ok	cache=miss	duration=3.67093400000158
2026-04-21 22:52:47	02_build_all_plasmids_fasta_step_result.json	status=ok	cache=miss	duration=1.5827335840003798
2026-04-17 13:54:12	29_build_experiment_registry_step_result.json	status=ok	cache=None	duration=None
2026-04-17 13:54:11	28_build_release_bundle_step_result.json	status=ok	cache=None	duration=None
2026-04-17 13:53:44	31_generate_benchmark_contract_step_result.json	status=ok	cache=None	duration=None
2026-04-17 13:53:43	24_build_reports_step_result.json	status=ok	cache=None	duration=None

## Workflow Profile Snapshot
status=failed
mode=pipeline
started_at=2026-04-21T19:52:24+00:00
finished_at=2026-04-21T19:56:36+00:00
duration_seconds=252.555662
steps_recorded=21

## Key Analysis Artifacts Timestamps
2026-04-22 18:57:32	/Users/umut/Projeler/plasmid-priority/data/analysis/sensitivity_summary.json
2026-04-22 18:57:04	/Users/umut/Projeler/plasmid-priority/data/analysis/rolling_temporal_validation.tsv
2026-04-22 18:57:32	/Users/umut/Projeler/plasmid-priority/data/analysis/candidate_rank_stability.tsv
2026-04-22 18:57:07	/Users/umut/Projeler/plasmid-priority/data/analysis/candidate_variant_consistency.tsv
2026-04-22 18:57:32	/Users/umut/Projeler/plasmid-priority/data/analysis/prospective_candidate_freeze.tsv
2026-04-22 18:57:04	/Users/umut/Projeler/plasmid-priority/data/analysis/annual_candidate_freeze_summary.tsv
2026-04-21 22:56:36	/Users/umut/Projeler/plasmid-priority/data/tmp/logs/21_run_validation_step_result.json
2026-04-22 18:57:32	/Users/umut/Projeler/plasmid-priority/data/tmp/logs/22_run_sensitivity_step_result.json
2026-04-22 19:12:35	/Users/umut/Projeler/plasmid-priority/data/tmp/logs/01_check_inputs_step_result.json

## Sensitivity Cache Footprint
     237
161M	/Users/umut/Projeler/plasmid-priority/data/analysis/sensitivity_cache

## Reports/Release/Reviewer Pack Files
/Users/umut/Projeler/plasmid-priority/reports/performance/workflow_performance_dashboard.json
/Users/umut/Projeler/plasmid-priority/reports/performance/workflow_performance_dashboard.md
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/README.txt
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/RELEASE_INFO.txt
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/docs/benchmark_contract.md
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/docs/data_card.md
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/docs/label_card_bundle.md
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/docs/model_card.md
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/docs/reproducibility_manifest.json
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/docs/scientific_protocol.md
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/reports/executive_summary.md
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/reports/headline_validation_summary.md
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/reports/jury_brief.md
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/reports/ozet_tr.md
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/reports/pitch_notes.md
/Users/umut/Projeler/plasmid-priority/reports/release/bundle/reports/tubitak_final_metrics.txt
/Users/umut/Projeler/plasmid-priority/reports/release/plasmid_priority_release_bundle.zip
/Users/umut/Projeler/plasmid-priority/reports/release/plasmid_priority_release_manifest.json
/Users/umut/Projeler/plasmid-priority/reports/release/release_readiness_report.json
/Users/umut/Projeler/plasmid-priority/reports/release/release_readiness_report.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/README.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AA086.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AA279.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AA282.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AA316.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AA319.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AA324.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AA331.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AA409.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AA411.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AA434.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AA514.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AA764.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AA859.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AC030.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AC202.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AC203.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AC301.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AD100.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/AE264.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/candidate_evidence_dossiers/index.md
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/canonical_metadata.json
/Users/umut/Projeler/plasmid-priority/reports/reviewer_pack/run_reproducibility.sh
