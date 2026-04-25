# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 Optimization: Makefile
# - TIMESTAMP tracking per target
# - JOBS env var forwarded to all Python scripts
# - test-fast: smoke-only critical path (2-3 min)
# - pipeline-fast: skip validation + reports (dev iteration)
# - uv cache dir for reproducible fast installs
# ═══════════════════════════════════════════════════════════════════════════════

PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
NJOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)
TIMESTAMP = $(shell date '+%H:%M:%S')
# Forward JOBS to all Python scripts that accept --jobs
PYTHON_JOBS = $(if $(JOBS),$(JOBS),$(NJOBS))
PYTHON_WITH_JOBS = PLASMID_PRIORITY_MAX_JOBS=$(PYTHON_JOBS) $(PYTHON)
# UV cache for fast reinstalls
export UV_CACHE_DIR ?= $(PWD)/.uv_cache
# Security gate allowlist for vulnerabilities with no upstream patch yet.
PIP_AUDIT_IGNORES ?= CVE-2025-69872 CVE-2026-3219
PIP_AUDIT_IGNORE_ARGS = $(foreach vuln,$(PIP_AUDIT_IGNORES),--ignore-vuln $(vuln))

.PHONY: check-inputs build-bronze-fasta build-bronze-table module-c module-f reports tubitak-summary fast-local full-local demo test test-fast test-cov critical-coverage docs-check smoke code-review-graph-check pipeline pipeline-fast pipeline-sequential clean-generated lint lint-fix typecheck check verify-pipeline verify-release quality ci security generate-scientific-contracts protocol-freshness import-contract freeze-baseline freeze-current scientific-equivalence generate-disposition-ledger generate-code-size-contract raw-data-usage-audit run-branch reviewer-package runtime-budget-gate scientific-contract-gate artifact-integrity profile-workflow performance-dashboard release-readiness generate-sample-data

define log_start
	@echo "[$(TIMESTAMP)] ▶️  Starting: $(1)"
endef

define log_done
	@echo "[$(TIMESTAMP)] ✅ Done: $(1)"
endef

check-inputs:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/01_check_inputs.py
	$(call log_done,$@)

build-bronze-fasta:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/02_build_all_plasmids_fasta.py
	$(call log_done,$@)

build-bronze-table:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/03_build_bronze_table.py
	$(call log_done,$@)

module-c:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/18_run_module_C_pathogen_detection.py --jobs $(PYTHON_JOBS)
	$(call log_done,$@)

module-f:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/23_run_module_f_enrichment.py --jobs $(PYTHON_JOBS)
	$(call log_done,$@)

reports:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/24_build_reports.py
	$(call log_done,$@)

tubitak-summary:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/25_export_tubitak_summary.py
	$(call log_done,$@)

fast-local:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/run_mode.py fast-local $(if $(SOURCE_DATA_ROOT),--source-data-root $(SOURCE_DATA_ROOT),) --jobs $(PYTHON_JOBS)
	$(call log_done,$@)

full-local:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/run_mode.py full-local $(if $(DATA_ROOT),--data-root $(DATA_ROOT),) --jobs $(PYTHON_JOBS)
	$(call log_done,$@)

generate-sample-data:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/generate_sample_data.py --output-dir data/sample
	$(call log_done,$@)

demo: generate-sample-data
	@echo "Running demo pipeline with sample data..."
	$(PYTHON_WITH_JOBS) scripts/run_mode.py demo --no-profile-after-run --jobs $(PYTHON_JOBS)

test:
	$(PYTHON_WITH_JOBS) -m pytest tests/ -x -q --tb=short $(if $(JOBS),-n $(JOBS),)

test-fast:
	@echo "🚀 Fast smoke test (critical path only, ~2-3 min)..."
	$(PYTHON_WITH_JOBS) -m pytest tests/test_config.py tests/test_fasta.py tests/test_features.py tests/test_modeling.py tests/test_reporting.py tests/test_leakage.py tests/test_embedding.py -x -q --tb=short $(if $(JOBS),-n $(JOBS),)

test-cov:
	$(PYTHON_WITH_JOBS) -m pytest tests/ -x -q --tb=short --cov=src/plasmid_priority --cov-report=term-missing --cov-report=html:htmlcov --cov-fail-under=70 $(if $(JOBS),-n $(JOBS),)

critical-coverage:
	$(PYTHON_WITH_JOBS) -m coverage erase
	$(PYTHON_WITH_JOBS) -m coverage run --source=src/plasmid_priority -m pytest -q \
		tests/test_temporal_contracts.py \
		tests/test_probabilistic_labels.py \
		tests/test_modeling_temporal_cv.py \
		tests/test_leakage.py \
		tests/test_config.py \
		tests/test_model_caches.py \
		-o addopts=''
	$(PYTHON_WITH_JOBS) -m coverage report \
		--include='src/plasmid_priority/shared/temporal.py,src/plasmid_priority/labels/probabilistic.py,src/plasmid_priority/modeling/temporal_cv.py,src/plasmid_priority/config.py' \
		--fail-under=90 \
		--show-missing

lint:
	$(PYTHON_WITH_JOBS) -m ruff check .

lint-fix:
	$(PYTHON_WITH_JOBS) -m ruff check src/ scripts/ tests/ --fix

typecheck:
	$(PYTHON_WITH_JOBS) -m mypy src/plasmid_priority/

check: lint test
	@echo "All checks passed."

verify-pipeline:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/01_check_inputs.py
	$(PYTHON_WITH_JOBS) -m pytest tests/ -x -q --tb=short $(if $(JOBS),-n $(JOBS),)
	@echo "Pipeline verification complete."
	$(call log_done,$@)

docs-check:
	$(PYTHON_WITH_JOBS) -m mkdocs build --strict

verify-release:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/46_verify_release.py
	$(call log_done,$@)

smoke:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/26_run_tests_or_smoke.py --with-tests
	$(call log_done,$@)

code-review-graph-check:
	$(PYTHON_WITH_JOBS) scripts/check_code_review_graph.py

security:
	$(PYTHON_WITH_JOBS) -m pip_audit --desc $(PIP_AUDIT_IGNORE_ARGS)

generate-scientific-contracts:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/31_generate_scientific_contracts.py
	$(call log_done,$@)

reviewer-package: generate-scientific-contracts
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/41_build_performance_dashboard.py
	$(PYTHON_WITH_JOBS) scripts/42_release_readiness_report.py
	$(PYTHON_WITH_JOBS) scripts/28_build_release_bundle.py
	$(call log_done,$@)

runtime-budget-gate:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/39_runtime_budget_gate.py --mode smoke-local
	$(call log_done,$@)

profile-workflow:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/profile_workflow.py --budget-mode smoke-local --top-n 10
	$(call log_done,$@)

performance-dashboard:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/41_build_performance_dashboard.py
	$(call log_done,$@)

release-readiness:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/42_release_readiness_report.py
	$(call log_done,$@)

scientific-contract-gate: generate-scientific-contracts
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/40_scientific_contract_gate.py
	$(call log_done,$@)

artifact-integrity:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/38_artifact_integrity_gate.py
	$(call log_done,$@)

protocol-freshness: generate-scientific-contracts
	git diff --exit-code -- docs/benchmark_contract.md docs/scientific_protocol.md docs/model_card.md docs/data_card.md docs/label_card_bundle.md docs/reproducibility_manifest.json reports/reviewer_pack/README.md reports/reviewer_pack/canonical_metadata.json

import-contract:
	$(PYTHON_WITH_JOBS) scripts/check_import_contracts.py

freeze-baseline:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/32_freeze_and_invariants.py --promote-baseline --run-quality-checks
	$(call log_done,$@)

freeze-current:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/32_freeze_and_invariants.py --run-quality-checks
	$(call log_done,$@)

scientific-equivalence:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/33_scientific_equivalence.py --baseline reports/freeze/baseline_freeze.json --candidate reports/freeze/current_freeze.json
	$(call log_done,$@)

generate-disposition-ledger:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/34_generate_disposition_ledger.py
	$(call log_done,$@)

generate-code-size-contract:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/35_generate_code_size_contract.py
	$(call log_done,$@)

raw-data-usage-audit:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/36_report_raw_data_usage.py
	$(call log_done,$@)

run-branch:
	$(PYTHON_WITH_JOBS) scripts/run_branch.py $(ARGS)

quality: check typecheck smoke security
	@echo "All quality gates passed."

ci: protocol-freshness import-contract runtime-budget-gate scientific-contract-gate artifact-integrity release-readiness lint typecheck test-cov critical-coverage security smoke docs-check

pipeline:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/run_workflow.py pipeline
	$(call log_done,$@)

pipeline-fast:
	@echo "🚀 Fast pipeline (skip heavy validation + reports, dev iteration)..."
	PLASMID_PRIORITY_MAX_JOBS=$(PYTHON_JOBS) $(PYTHON) scripts/run_workflow.py pipeline --skip-reports --skip-validation --jobs $(PYTHON_JOBS)

pipeline-sequential:
	$(call log_start,$@)
	$(PYTHON_WITH_JOBS) scripts/run_workflow.py pipeline-sequential
	$(call log_done,$@)

clean-generated:
	find . -name '.DS_Store' -delete
	rm -rf src/*.egg-info reports/figures reports/logs reports/diagnostic_figures
	rm -rf data/tmp/logs
	rm -rf reports/release
	rm -f data/analysis/headline_model_candidate_audit.tsv data/analysis/tmp_model_search.tsv reports/tubitak_detayli_proje_ozeti_tr.txt reports/final_summary.json reports/tubitak_final_metrics.json reports/outbreak_audit.txt

# ═══════════════════════════════════════════════════════════════════════════════
# Convenience aliases for maximum efficiency
# ═══════════════════════════════════════════════════════════════════════════════
fast: pipeline-fast
	@echo "🚀 Fast pipeline complete! Use 'make full' for complete run."

full: pipeline
	@echo "✅ Full pipeline complete!"
