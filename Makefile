PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
NJOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)

.PHONY: check-inputs build-bronze-fasta build-bronze-table module-c module-f reports tubitak-summary fast-local full-local test test-cov smoke code-review-graph-check pipeline pipeline-sequential clean-generated lint lint-fix typecheck check verify-pipeline quality ci security generate-scientific-contracts protocol-freshness import-contract freeze-baseline freeze-current scientific-equivalence generate-disposition-ledger generate-code-size-contract run-branch

check-inputs:
	$(PYTHON) scripts/01_check_inputs.py

build-bronze-fasta:
	$(PYTHON) scripts/02_build_all_plasmids_fasta.py

build-bronze-table:
	$(PYTHON) scripts/03_build_bronze_table.py

module-c:
	$(PYTHON) scripts/18_run_module_C_pathogen_detection.py

module-f:
	$(PYTHON) scripts/23_run_module_f_enrichment.py

reports:
	$(PYTHON) scripts/24_build_reports.py

tubitak-summary:
	$(PYTHON) scripts/25_export_tubitak_summary.py

fast-local:
	$(PYTHON) scripts/run_mode.py fast-local $(if $(SOURCE_DATA_ROOT),--source-data-root $(SOURCE_DATA_ROOT),)

full-local:
	$(PYTHON) scripts/run_mode.py full-local $(if $(DATA_ROOT),--data-root $(DATA_ROOT),)

test:
	$(PYTHON) -m pytest tests/ -x -q --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -x -q --tb=short --cov=src/plasmid_priority --cov-report=term-missing --cov-report=html:htmlcov --cov-fail-under=80

lint:
	$(PYTHON) -m ruff check src/ scripts/ tests/

lint-fix:
	$(PYTHON) -m ruff check src/ scripts/ tests/ --fix

typecheck:
	$(PYTHON) -m mypy src/plasmid_priority/

check: lint test
	@echo "All checks passed."

verify-pipeline:
	$(PYTHON) scripts/01_check_inputs.py
	$(PYTHON) -m pytest tests/ -x -q --tb=short
	@echo "Pipeline verification complete."

smoke:
	$(PYTHON) scripts/26_run_tests_or_smoke.py --with-tests

code-review-graph-check:
	$(PYTHON) scripts/check_code_review_graph.py

security:
	$(PYTHON) -m pip_audit --desc

generate-scientific-contracts:
	$(PYTHON) scripts/31_generate_scientific_contracts.py

protocol-freshness: generate-scientific-contracts
	git diff --exit-code -- docs/benchmark_contract.md docs/scientific_protocol.md docs/model_card.md docs/data_card.md reports/reviewer_pack/README.md reports/reviewer_pack/canonical_metadata.json

import-contract:
	$(PYTHON) scripts/check_import_contracts.py

freeze-baseline:
	$(PYTHON) scripts/32_freeze_and_invariants.py --promote-baseline --run-quality-checks

freeze-current:
	$(PYTHON) scripts/32_freeze_and_invariants.py --run-quality-checks

scientific-equivalence:
	$(PYTHON) scripts/33_scientific_equivalence.py --baseline reports/freeze/baseline_freeze.json --candidate reports/freeze/current_freeze.json

generate-disposition-ledger:
	$(PYTHON) scripts/34_generate_disposition_ledger.py

generate-code-size-contract:
	$(PYTHON) scripts/35_generate_code_size_contract.py

run-branch:
	$(PYTHON) scripts/run_branch.py $(ARGS)

quality: check typecheck smoke security
	@echo "All quality gates passed."

ci: protocol-freshness import-contract check typecheck

pipeline:
	$(PYTHON) scripts/run_workflow.py pipeline

pipeline-sequential:
	$(PYTHON) scripts/run_workflow.py pipeline-sequential

clean-generated:
	find . -name '.DS_Store' -delete
	rm -rf src/*.egg-info reports/figures reports/logs reports/diagnostic_figures
	rm -rf data/tmp/logs
	rm -rf reports/release
	rm -f data/analysis/headline_model_candidate_audit.tsv data/analysis/tmp_model_search.tsv reports/tubitak_detayli_proje_ozeti_tr.txt reports/final_summary.json reports/tubitak_final_metrics.json reports/outbreak_audit.txt
