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
