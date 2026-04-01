"""Pydantic models for the repository data contract."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class AssetKind(str, Enum):
    FILE = "file"
    DIRECTORY = "directory"


class Stage(str, Enum):
    CORE = "core"
    SUPPORTING = "supporting"
    OPTIONAL = "optional"
    DERIVED = "derived"


class DataAssetSpec(BaseModel):
    """Declarative specification for a repository data asset."""

    key: str
    relative_path: str
    kind: AssetKind
    stage: Stage
    required: bool
    description: str
    notes: list[str] = Field(default_factory=list)
    expected_columns: list[str] = Field(default_factory=list)
    expected_children: list[str] = Field(default_factory=list)
    expected_glob: str | None = None
    min_matches: int | None = None
    expected_release_files: list[str] = Field(default_factory=list)

    def resolved_path(self, project_root: Path) -> Path:
        return project_root / self.relative_path


class DataContract(BaseModel):
    """Full repository data contract."""

    version: int
    created_on: str
    download_date: str
    notes: list[str] = Field(default_factory=list)
    assets: list[DataAssetSpec]

    @model_validator(mode="after")
    def validate_unique_keys(self) -> "DataContract":
        keys = [asset.key for asset in self.assets]
        duplicates = {key for key in keys if keys.count(key) > 1}
        if duplicates:
            joined = ", ".join(sorted(duplicates))
            raise ValueError(f"Duplicate asset keys in data contract: {joined}")
        return self

    def asset_map(self) -> dict[str, DataAssetSpec]:
        return {asset.key: asset for asset in self.assets}
