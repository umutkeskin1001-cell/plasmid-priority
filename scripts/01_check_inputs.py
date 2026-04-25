#!/usr/bin/env python3
"""Validate repository inputs against the local data contract."""

from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.logging_utils import configure_logging
from plasmid_priority.qc import run_input_checks
from plasmid_priority.reporting import ManagedScriptRun


def main() -> int:
    configure_logging()
    logger = logging.getLogger("01_check_inputs")
    context = build_context(PROJECT_ROOT)

    with ManagedScriptRun(context, "01_check_inputs") as run:
        report = run_input_checks(context)

        for asset in context.contract.assets:
            run.record_input(asset.resolved_path(context.root, context.data_dir))

        for note in context.contract.notes:
            run.note(note)

        run.set_metric("assets_checked", len(report.results))
        run.set_metric("error_count", len(report.errors))
        run.set_metric("warning_count", len(report.warnings))

        for result in report.results:
            logger.info("%s | %s | %s", result.status.upper(), result.key, result.path)
            for detail in result.details:
                logger.info("  %s", detail)
            if result.status == "warning":
                for detail in result.details:
                    run.warn(f"{result.key}: {detail}")

        if not report.ok:
            raise RuntimeError(
                "Input validation failed. See data/tmp/logs/01_check_inputs_summary.json.",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
