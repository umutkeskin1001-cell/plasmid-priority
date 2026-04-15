"""Logging helpers for consistent pipeline output.

Provides:
- ``configure_logging`` – one-call setup for the root logger.
- ``get_logger`` – factory that returns a named child logger.
- ``StructuredFormatter`` – JSON-lines formatter for machine-parseable logs.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Emit each log record as a single JSON line.

    Useful for centralized log aggregation (ELK, CloudWatch, etc.).
    Extra keyword arguments passed to the logger are merged into the
    JSON payload automatically.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc"] = self.formatException(record.exc_info)
        # Merge structured extra fields (skip internal dunder keys)
        for key, value in record.__dict__.items():
            if key not in payload and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    level: int = logging.INFO,
    *,
    structured: bool = False,
    stream: Any | None = None,
) -> None:
    """Configure the root logger with a compact, deterministic format.

    Args:
        level: Logging verbosity (e.g. ``logging.DEBUG``).
        structured: If True, emit JSON-lines via ``StructuredFormatter``.
        stream: Output stream (defaults to ``sys.stderr``).
    """
    handler = logging.StreamHandler(stream or sys.stderr)
    if structured:
        handler.setFormatter(StructuredFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers on repeated calls
    if not root.handlers:
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the plasmid_priority namespace.

    Usage::

        from plasmid_priority.logging_utils import get_logger
        _log = get_logger(__name__)
    """
    return logging.getLogger(name)
