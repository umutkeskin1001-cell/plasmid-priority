from __future__ import annotations

import time

from plasmid_priority.performance.telemetry import (
    build_step_telemetry,
    capture_resource_snapshot,
    infer_cache_status_from_summary,
    upsert_summary_telemetry,
)


def test_build_step_telemetry_populates_required_fields() -> None:
    summary = {
        "n_rows_in": {"input_rows": 10},
        "n_rows_out": {"output_rows": 4},
        "input_manifest": {"a": {"size": 100}},
        "output_manifest": {"b": {"size": 55}},
    }
    started = capture_resource_snapshot()
    time.sleep(0.01)
    telemetry = build_step_telemetry(
        step_name="example_step",
        status="ok",
        cache_status="miss",
        summary=summary,
        started=started,
    )
    assert telemetry.duration_seconds > 0
    assert telemetry.rows_in == 10
    assert telemetry.rows_out == 4
    assert telemetry.bytes_read == 100
    assert telemetry.bytes_written == 55
    assert telemetry.input_hash is not None
    assert telemetry.output_hash is not None


def test_upsert_summary_telemetry_writes_top_level_contract_fields() -> None:
    summary = {"script_name": "sample_step", "status": "ok"}
    started = capture_resource_snapshot()
    telemetry = build_step_telemetry(
        step_name="sample_step",
        status="ok",
        cache_status="partial",
        summary={},
        started=started,
    )
    enriched = upsert_summary_telemetry(summary, telemetry)
    for key in (
        "duration_seconds",
        "cache_status",
        "rows_in",
        "rows_out",
        "bytes_read",
        "bytes_written",
        "peak_rss_mb",
        "cpu_time_seconds",
        "io_wait_hint",
        "input_hash",
        "output_hash",
    ):
        assert key in enriched
    assert enriched["cache_status"] == "partial"
    assert isinstance(enriched["telemetry"], dict)


def test_infer_cache_status_from_summary_metrics() -> None:
    assert infer_cache_status_from_summary({"metrics": {"cache_hit": True}}) == "partial"
    assert infer_cache_status_from_summary({"metrics": {"cache_hit": False}}) == "miss"
