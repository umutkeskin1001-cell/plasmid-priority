"""Managed JSON summaries for numbered pipeline scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from plasmid_priority.config import ProjectContext
from plasmid_priority.utils.files import (
    atomic_write_json,
    ensure_directory,
    path_signature_with_hash,
    relative_path_str,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ManagedScriptRun:
    """Context manager that writes a structured summary JSON for a script run."""

    context: ProjectContext
    script_name: str
    start_time: str = field(default_factory=_utc_now)
    end_time: str | None = None
    status: str = "running"
    input_files_checked: list[str] = field(default_factory=list)
    output_files_written: list[str] = field(default_factory=list)
    input_manifest: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    n_rows_in: dict[str, int] = field(default_factory=dict)
    n_rows_out: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __enter__(self) -> "ManagedScriptRun":
        ensure_directory(self.context.logs_dir)
        return self

    def __exit__(self, exc_type, exc, _tb) -> Literal[False]:
        self.end_time = _utc_now()
        if exc is None and self.status == "running":
            self.status = "ok"
        elif exc is not None:
            self.status = "failed"
            self.errors.append(str(exc))

        atomic_write_json(self.summary_path, self.to_dict())
        return False

    @property
    def summary_path(self) -> Path:
        return self.context.logs_dir / f"{self.script_name}_summary.json"

    def record_input(self, path: Path, *, include_hash: bool = True) -> None:
        """Record an input file path with optional cryptographic hash.

        The cryptographic hash is computed using path_signature_with_hash which
        applies a 100 MB default size limit. Files larger than this limit are
        recorded with metadata only (size, mtime) without the sha256 hash.
        This is a practical trade-off to avoid excessive I/O for large files.
        """
        rel_path = relative_path_str(path, self.context.root)
        self.input_files_checked.append(rel_path)
        if include_hash:
            try:
                sig = path_signature_with_hash(path, include_file_hash=True)
                self.input_manifest[rel_path] = {
                    "path": sig.get("path", str(path)),
                    "size": sig.get("size"),
                    "sha256": sig.get("sha256"),
                }
            except (OSError, ValueError):
                pass  # Hashing optional; skip if it fails

    def record_input_manifest(self, paths: list[Path], *, include_hash: bool = True) -> None:
        """Record multiple input files with their cryptographic hashes."""
        for path in paths:
            self.record_input(path, include_hash=include_hash)

    def record_output(self, path: Path) -> None:
        self.output_files_written.append(relative_path_str(path, self.context.root))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def set_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def set_rows_in(self, key: str, value: int) -> None:
        self.n_rows_in[key] = value

    def set_rows_out(self, key: str, value: int) -> None:
        self.n_rows_out[key] = value

    def set_provenance(self, provenance: dict[str, Any]) -> None:
        self.provenance = dict(provenance)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "script_name": self.script_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "input_files_checked": self.input_files_checked,
            "output_files_written": self.output_files_written,
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "warnings": self.warnings,
            "errors": self.errors,
            "metrics": self.metrics,
            "notes": self.notes,
        }
        if self.input_manifest:
            result["input_manifest"] = self.input_manifest
        if self.provenance:
            result["provenance"] = self.provenance
        return result
