set shell := ["bash", "-cu"]

default: quality

quality:
    uv run ruff check .
    PYTHONPATH=src pytest -q -o addopts=''

format:
    uv run ruff format .
    uv run ruff check . --fix --unsafe-fixes

test:
    PYTHONPATH=src pytest -q -o addopts=''

test-cov:
    PYTHONPATH=src pytest -q --cov=src/plasmid_priority --cov-report=term-missing --cov-report=html:htmlcov --cov-fail-under=70 -o addopts=''

critical-coverage:
    uv run python -m coverage erase
    PYTHONPATH=src uv run python -m coverage run --source=src/plasmid_priority -m pytest -q \
      tests/test_temporal_contracts.py \
      tests/test_probabilistic_labels.py \
      tests/test_modeling_temporal_cv.py \
      tests/test_leakage.py \
      tests/test_config.py \
      tests/test_model_caches.py \
      -o addopts=''
    uv run python -m coverage report \
      --include='src/plasmid_priority/shared/temporal.py,src/plasmid_priority/labels/probabilistic.py,src/plasmid_priority/modeling/temporal_cv.py,src/plasmid_priority/config.py' \
      --fail-under=90 \
      --show-missing

docs-check:
    uv run mkdocs build --strict

verify-release:
    PYTHONPATH=src uv run python scripts/46_verify_release.py

workflow mode="core-refresh" workers="4":
    uv run python scripts/run_workflow.py {{mode}} --max-workers {{workers}}

prefect-plan include_fetch="false" release="false":
    cmd=(uv run --extra engineering python scripts/43_run_prefect_workflow.py --dry-run)
    if [[ "{{include_fetch}}" == "true" ]]; then cmd+=(--include-fetch); fi
    if [[ "{{release}}" == "true" ]]; then cmd+=(--release); fi
    "${cmd[@]}"

prefect include_fetch="false" release="false" workers="4":
    cmd=(uv run --extra engineering python scripts/43_run_prefect_workflow.py --max-workers {{workers}})
    if [[ "{{include_fetch}}" == "true" ]]; then cmd+=(--include-fetch); fi
    if [[ "{{release}}" == "true" ]]; then cmd+=(--release); fi
    "${cmd[@]}"

mutation:
    uv run --extra dev mutmut run
    uv run --extra dev mutmut results

phase5:
    PYTHONPATH=src uv run python scripts/41_build_performance_dashboard.py
    PYTHONPATH=src uv run python scripts/42_release_readiness_report.py
    PYTHONPATH=src uv run python scripts/44_build_jury_dashboard.py
    PYTHONPATH=src uv run python scripts/45_generate_independent_audit_packet.py
