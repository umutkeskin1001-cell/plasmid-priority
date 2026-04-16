from __future__ import annotations

import io
import json
import logging

from plasmid_priority.logging_utils import configure_logging, pop_logging_context, push_logging_context


def test_structured_logging_includes_correlation_metadata() -> None:
    stream = io.StringIO()
    tokens = push_logging_context(correlation_id="corr-123", run_id="run-123", script_name="demo")
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    try:
        root.handlers.clear()
        configure_logging(structured=True, stream=stream)
        logger = logging.getLogger("plasmid_priority.tests")
        logger.info("hello", extra={"sample": 1})
    finally:
        pop_logging_context(tokens)
        root.handlers[:] = original_handlers

    payload = json.loads(stream.getvalue().strip())
    assert payload["correlation_id"] == "corr-123"
    assert payload["run_id"] == "run-123"
    assert payload["script_name"] == "demo"
    assert payload["sample"] == 1
