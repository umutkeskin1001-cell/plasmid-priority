"""Step contract helpers for workflow executions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Literal

from plasmid_priority.utils.files import atomic_write_json, ensure_directory


def _hash_payload(payload: Any) -> str:
    digest = hashlib.sha256(repr(payload).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class StepResult:
    step_name: str
    status: Literal["ok", "warning", "failed"]
    inputs: dict[str, str]
    outputs: dict[str, str]
    protocol_hash: str
    rows_in: int | None = None
    rows_out: int | None = None
    duration_seconds: float | None = None
    warnings: list[str] = field(default_factory=list)
    scientific_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract_hash"] = _hash_payload(
            {
                "step_name": self.step_name,
                "status": self.status,
                "inputs": self.inputs,
                "outputs": self.outputs,
                "protocol_hash": self.protocol_hash,
                "rows_in": self.rows_in,
                "rows_out": self.rows_out,
                "duration_seconds": self.duration_seconds,
                "warnings": self.warnings,
                "scientific_notes": self.scientific_notes,
                "metadata": self.metadata,
            }
        )
        return payload


def write_step_result(result: StepResult, logs_dir: Path) -> Path:
    ensure_directory(logs_dir)
    path = logs_dir / f"{result.step_name}_step_result.json"
    atomic_write_json(path, result.to_dict())
    return path
