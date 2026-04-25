"""Artifact-backed registry for API scoring and evidence lookups."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasmid_priority.config import build_context
from plasmid_priority.evidence import derive_claim_level
from plasmid_priority.io.table_io import read_table


class ArtifactUnavailableError(RuntimeError):
    """Raised when required scoring artifacts are not available."""


@dataclass(frozen=True)
class ArtifactRegistry:
    project_root: Path | None = None

    def _resolve_project_root(self) -> Path:
        if self.project_root is not None:
            return self.project_root.resolve()
        return Path(__file__).resolve().parents[3]

    def _resolve_data_dir(self) -> Path:
        if self.project_root is not None:
            return (self._resolve_project_root() / "data").resolve()
        root = self._resolve_project_root()
        context = build_context(root)
        return Path(context.data_dir).resolve()

    def _resolve_scored_path(self) -> Path:
        data_dir = self._resolve_data_dir()
        candidates = [
            data_dir / "scores" / "backbone_scored.parquet",
            data_dir / "scores" / "backbone_scored.tsv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise ArtifactUnavailableError(
            "Scored artifact missing: expected data/scores/backbone_scored.parquet or .tsv",
        )

    def _resolve_manifest_path(self) -> Path:
        project_root = self._resolve_project_root()
        manifest = project_root / "docs" / "reproducibility_manifest.json"
        if not manifest.exists():
            raise ArtifactUnavailableError(
                "Reproducibility manifest missing: expected docs/reproducibility_manifest.json",
            )
        return manifest

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self._resolve_manifest_path()
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactUnavailableError(
                "Reproducibility manifest is unreadable JSON",
            ) from exc
        if not isinstance(payload, dict):
            raise ArtifactUnavailableError("Reproducibility manifest must be a JSON object")
        return payload

    def list_models(self) -> list[dict[str, Any]]:
        manifest = self._load_manifest()
        artifacts = manifest.get("artifacts", {})
        artifacts = artifacts if isinstance(artifacts, dict) else {}
        protocol_hash = str(
            artifacts.get("protocol_hash", manifest.get("protocol_hash", ""))
        ).strip()
        model_version = str(artifacts.get("model_version", protocol_hash or "unversioned")).strip()
        return [
            {
                "model_version": model_version,
                "protocol_hash": protocol_hash,
                "training_data_hash": str(
                    artifacts.get("training_data_hash", manifest.get("data_contract_sha", "")),
                ),
                "feature_schema_hash": str(
                    artifacts.get("feature_schema_hash", manifest.get("benchmarks_hash", "")),
                ),
                "calibration_artifact_hash": str(
                    artifacts.get("calibration_artifact_hash", ""),
                ),
            },
        ]

    def score_backbones(self, backbone_ids: list[str]) -> list[dict[str, Any]]:
        if not backbone_ids:
            return []
        scored_path = self._resolve_scored_path()
        columns = [
            "backbone_id",
            "priority_index",
            "operational_priority_index",
            "bio_priority_index",
            "evidence_support_index",
        ]
        frame = read_table(scored_path, columns=columns)
        if "backbone_id" not in frame.columns or "priority_index" not in frame.columns:
            raise ArtifactUnavailableError("Scored artifact is missing required score columns")
        subset = frame.loc[frame["backbone_id"].astype(str).isin(set(backbone_ids))].copy()
        if subset.empty:
            return []
        subset = subset.drop_duplicates(subset=["backbone_id"], keep="first")
        results: list[dict[str, Any]] = []
        for _, row in subset.iterrows():
            results.append(
                {
                    "backbone_id": str(row.get("backbone_id", "")),
                    "priority_index": float(row.get("priority_index", 0.0)),
                    "operational_priority_index": float(row.get("operational_priority_index", 0.0)),
                    "bio_priority_index": float(row.get("bio_priority_index", 0.0)),
                    "evidence_support_index": float(row.get("evidence_support_index", 0.0)),
                },
            )
        return results

    def get_evidence(self, backbone_id: str) -> dict[str, Any]:
        manifest = self._load_manifest()
        scores = self.score_backbones([backbone_id])
        if not scores:
            raise KeyError(backbone_id)
        label_cards = manifest.get("label_cards", [])
        first_card = label_cards[0] if isinstance(label_cards, list) and label_cards else {}
        score_row = scores[0]
        evidence_support = float(score_row.get("evidence_support_index", 0.0))
        claim_level = derive_claim_level(
            observed_signal=float(score_row.get("priority_index", 0.0)) > 0.0,
            proxy_only=True,
            literature_support=evidence_support >= 0.5,
            external_validation=False,
        )
        return {
            "backbone_id": backbone_id,
            "claim_level": claim_level,
            "scores": score_row,
            "label_card": first_card if isinstance(first_card, dict) else {},
            "protocol_hash": str(manifest.get("protocol_hash", "")),
        }

    def explain_backbone(self, backbone_id: str) -> dict[str, Any]:
        scored_path = self._resolve_scored_path()
        columns = [
            "backbone_id",
            "T_eff",
            "H_eff",
            "A_eff",
            "priority_index",
            "operational_priority_index",
            "bio_priority_index",
        ]
        frame = read_table(scored_path, columns=columns)
        subset = frame.loc[frame["backbone_id"].astype(str) == str(backbone_id)]
        if subset.empty:
            raise KeyError(backbone_id)
        row = subset.iloc[0]
        return {
            "backbone_id": str(row.get("backbone_id", "")),
            "components": {
                "T_eff": float(row.get("T_eff", 0.0)),
                "H_eff": float(row.get("H_eff", 0.0)),
                "A_eff": float(row.get("A_eff", 0.0)),
            },
            "scores": {
                "priority_index": float(row.get("priority_index", 0.0)),
                "operational_priority_index": float(row.get("operational_priority_index", 0.0)),
                "bio_priority_index": float(row.get("bio_priority_index", 0.0)),
            },
        }
