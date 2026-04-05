PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: check-inputs build-bronze-fasta build-bronze-table module-c module-f reports tubitak-summary fast-local full-local test smoke pipeline pipeline-sequential clean-generated lint lint-fix typecheck check verify-pipeline quality ci

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
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

lint:
	$(PYTHON) -m ruff check src/ scripts/ tests/

lint-fix:
	$(PYTHON) -m ruff check src/ scripts/ tests/ --fix

typecheck:
	$(PYTHON) -m mypy src/plasmid_priority/

check: lint test
	@echo "Tüm kontroller geçti."

verify-pipeline:
	$(PYTHON) scripts/01_check_inputs.py
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v
	@echo "Pipeline doğrulama tamamlandı."

smoke:
	$(PYTHON) scripts/26_run_tests_or_smoke.py

quality: check typecheck smoke
	@echo "Kalite kapilari gecti."

ci: check typecheck

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
