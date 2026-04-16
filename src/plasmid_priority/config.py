"""Project configuration and manifest loading."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from plasmid_priority.protocol import _coerce_float, _coerce_int
from plasmid_priority.schemas import DataAssetSpec, DataContract

DEFAULT_CONTRACT_PATH = Path("data/manifests/data_contract.json")
DEFAULT_PIPELINE_SPLIT_YEAR = 2015
DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD = 3
DEFAULT_HOST_EVENNESS_BIAS_POWER = 0.5
DEFAULT_HOST_PHYLO_BREADTH_WEIGHT = 0.65
DEFAULT_HOST_PHYLO_DISPERSION_WEIGHT = 0.35
DATA_ROOT_ENV_VAR = "PLASMID_PRIORITY_DATA_ROOT"
ROOT_MARKERS = (
    Path("pyproject.toml"),
    Path("data/manifests/data_contract.json"),
)


@dataclass(frozen=True)
class PipelineSettings:
    """Canonical pipeline parameters loaded from config.yaml."""

    split_year: int
    min_new_countries_for_spread: int
    host_evenness_bias_power: float
    host_phylo_breadth_weight: float
    host_phylo_dispersion_weight: float


class PipelineConfig(BaseModel):
    """Typed pipeline configuration surface."""

    model_config = ConfigDict(extra="allow")

    split_year: int = DEFAULT_PIPELINE_SPLIT_YEAR
    min_new_countries_for_spread: int = DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD
    horizon_years: int = 5
    min_new_host_genera_for_transfer: int = 2
    min_new_host_families_for_transfer: int = 1
    clinical_escalation_thresholds: dict[str, float] = Field(default_factory=dict)
    consensus_weights: dict[str, float] = Field(default_factory=dict)
    ood_thresholds: dict[str, float] = Field(default_factory=dict)
    host_evenness_bias_power: float = DEFAULT_HOST_EVENNESS_BIAS_POWER
    host_phylo_breadth_weight: float = DEFAULT_HOST_PHYLO_BREADTH_WEIGHT
    host_phylo_dispersion_weight: float = DEFAULT_HOST_PHYLO_DISPERSION_WEIGHT


class ProjectConfig(BaseModel):
    """Typed layered project configuration."""

    model_config = ConfigDict(extra="allow")

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    models: dict[str, Any] = Field(default_factory=dict)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if isinstance(merged.get(key), dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _project_config_layers(project_root: Path) -> list[Path]:
    config_dir = project_root / "config"
    paths = [project_root / "config.yaml"]
    if config_dir.exists():
        paths.extend(sorted(config_dir.glob("*.yaml")))
    return paths


def load_project_config(project_root: Path | None = None) -> ProjectConfig:
    """Load and merge layered project configuration files."""
    root = find_project_root(project_root)
    payload: dict[str, Any] = {}
    for config_path in _project_config_layers(root):
        payload = _deep_merge(payload, _read_yaml(config_path))
    return ProjectConfig.model_validate(payload)


def context_config_paths(context: Any) -> tuple[Path, ...]:
    """Return the config paths associated with a project-like context.

    Older test doubles and lightweight callers may only expose ``root``.
    """
    config_paths = getattr(context, "config_paths", None)
    if config_paths:
        return tuple(Path(path) for path in config_paths)
    root = getattr(context, "root", None)
    if root is None:
        return tuple()
    return (Path(root) / "config.yaml",)


def _pipeline_settings_from_config(config: dict | None) -> PipelineSettings:
    pipeline = config.get("pipeline", {}) if isinstance(config, dict) else {}
    if not isinstance(pipeline, dict):
        pipeline = {}
    host_phylo_breadth_weight = _coerce_float(
        pipeline.get("host_phylo_breadth_weight"),
        default=DEFAULT_HOST_PHYLO_BREADTH_WEIGHT,
    )
    host_phylo_dispersion_weight = _coerce_float(
        pipeline.get("host_phylo_dispersion_weight"),
        default=DEFAULT_HOST_PHYLO_DISPERSION_WEIGHT,
    )
    total_host_phylo_weight = host_phylo_breadth_weight + host_phylo_dispersion_weight
    if not total_host_phylo_weight > 0.0:
        host_phylo_breadth_weight = DEFAULT_HOST_PHYLO_BREADTH_WEIGHT
        host_phylo_dispersion_weight = DEFAULT_HOST_PHYLO_DISPERSION_WEIGHT
        total_host_phylo_weight = (
            DEFAULT_HOST_PHYLO_BREADTH_WEIGHT + DEFAULT_HOST_PHYLO_DISPERSION_WEIGHT
        )
    host_phylo_breadth_weight /= total_host_phylo_weight
    host_phylo_dispersion_weight /= total_host_phylo_weight
    return PipelineSettings(
        split_year=_coerce_int(pipeline.get("split_year"), default=DEFAULT_PIPELINE_SPLIT_YEAR),
        min_new_countries_for_spread=_coerce_int(
            pipeline.get("min_new_countries_for_spread"),
            default=DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD,
        ),
        host_evenness_bias_power=_coerce_float(
            pipeline.get("host_evenness_bias_power"),
            default=DEFAULT_HOST_EVENNESS_BIAS_POWER,
        ),
        host_phylo_breadth_weight=host_phylo_breadth_weight,
        host_phylo_dispersion_weight=host_phylo_dispersion_weight,
    )


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking upward from the starting path."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in ROOT_MARKERS):
            return candidate
        if (candidate / "data").exists() and (candidate / "pyproject.toml").exists():
            return candidate

    markers = ", ".join(str(marker) for marker in ROOT_MARKERS)
    raise FileNotFoundError(
        f"Could not locate project root containing the expected repository markers: {markers}."
    )


def load_data_contract(project_root: Path | None = None) -> DataContract:
    """Load the machine-readable data contract from disk."""
    root = find_project_root(project_root)
    contract_path = root / DEFAULT_CONTRACT_PATH
    with contract_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return DataContract.model_validate(payload)


@dataclass(frozen=True)
class ProjectContext:
    """Convenient project paths and loaded data contract."""

    root: Path
    contract: DataContract
    data_root: Path | None = None

    @property
    def data_dir(self) -> Path:
        return (self.data_root or (self.root / "data")).resolve()

    @cached_property
    def config(self) -> dict:
        return self.project_config.model_dump(mode="python")

    @cached_property
    def project_config(self) -> ProjectConfig:
        return load_project_config(self.root)

    @cached_property
    def pipeline_settings(self) -> PipelineSettings:
        return _pipeline_settings_from_config(self.config)

    @property
    def config_paths(self) -> tuple[Path, ...]:
        return tuple(_project_config_layers(self.root))

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def experiments_dir(self) -> Path:
        return self.data_dir / "experiments"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "tmp" / "logs"

    @property
    def release_dir(self) -> Path:
        return self.reports_dir / "release"

    def resolve_path(self, relative_path: str | Path) -> Path:
        return _resolve_project_path(self.root, self.data_dir, relative_path)

    def asset(self, key: str) -> DataAssetSpec:
        try:
            return self.contract.asset_map()[key]
        except KeyError as exc:
            raise KeyError(f"Unknown asset key: {key}") from exc

    def asset_path(self, key: str) -> Path:
        return self.asset(key).resolved_path(self.root, self.data_dir)


def _resolve_project_path(project_root: Path, data_root: Path, relative_path: str | Path) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == "data":
        return (data_root / Path(*candidate.parts[1:])).resolve()
    return (project_root / candidate).resolve()


def resolve_data_root(
    project_root: Path,
    data_root: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> Path | None:
    raw_value = data_root
    if raw_value is None:
        raw_value = (env or os.environ).get(DATA_ROOT_ENV_VAR)
    if raw_value in (None, ""):
        return None
    candidate = Path(str(raw_value)).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def build_context(
    project_root: Path | None = None,
    *,
    data_root: str | Path | None = None,
) -> ProjectContext:
    """Construct a project context rooted at the repository root."""
    root = find_project_root(project_root)
    resolved_data_root = resolve_data_root(root, data_root)
    return ProjectContext(
        root=root,
        contract=load_data_contract(root),
        data_root=resolved_data_root,
    )
