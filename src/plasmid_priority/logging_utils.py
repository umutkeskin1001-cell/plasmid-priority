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
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_correlation_id: ContextVar[str] = ContextVar("plasmid_priority_correlation_id", default="-")
_run_id: ContextVar[str] = ContextVar("plasmid_priority_run_id", default="-")
_script_name: ContextVar[str] = ContextVar("plasmid_priority_script_name", default="-")


@dataclass(frozen=True)
class _LoggingContextTokens:
    correlation_id: object | None = None
    run_id: object | None = None
    script_name: object | None = None


def current_correlation_id() -> str:
    return _correlation_id.get()


def current_run_id() -> str:
    return _run_id.get()


def current_script_name() -> str:
    return _script_name.get()


def push_logging_context(
    *,
    correlation_id: str | None = None,
    run_id: str | None = None,
    script_name: str | None = None,
) -> _LoggingContextTokens:
    """Set per-run logging metadata and return tokens for restoration."""
    return _LoggingContextTokens(
        correlation_id=_correlation_id.set(correlation_id or "-"),
        run_id=_run_id.set(run_id or "-"),
        script_name=_script_name.set(script_name or "-"),
    )


def pop_logging_context(tokens: _LoggingContextTokens | None) -> None:
    """Restore the previous logging metadata context."""
    if tokens is None:
        return
    if tokens.script_name is not None:
        _script_name.reset(tokens.script_name)
    if tokens.run_id is not None:
        _run_id.reset(tokens.run_id)
    if tokens.correlation_id is not None:
        _correlation_id.reset(tokens.correlation_id)


class _ContextInjectionFilter(logging.Filter):
    """Attach correlation metadata to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "correlation_id"):
            record.correlation_id = current_correlation_id()
        if not hasattr(record, "run_id"):
            record.run_id = current_run_id()
        if not hasattr(record, "script_name"):
            record.script_name = current_script_name()
        return True


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
            "correlation_id": getattr(record, "correlation_id", current_correlation_id()),
            "run_id": getattr(record, "run_id", current_run_id()),
            "script_name": getattr(record, "script_name", current_script_name()),
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
    root = logging.getLogger()
    root.setLevel(level)
    formatter: logging.Formatter
    if structured:
        formatter = StructuredFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(correlation_id)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    if not root.handlers:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.addFilter(_ContextInjectionFilter())
        handler.setFormatter(formatter)
        root.addHandler(handler)
        return

    for handler in root.handlers:
        handler.addFilter(_ContextInjectionFilter())
        handler.setFormatter(formatter)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the plasmid_priority namespace.

    Usage::

        from plasmid_priority.logging_utils import get_logger
        _log = get_logger(__name__)
    """
    return logging.getLogger(name)
