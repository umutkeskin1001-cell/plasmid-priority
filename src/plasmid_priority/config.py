"""Project configuration and manifest loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from plasmid_priority.schemas import DataAssetSpec, DataContract

DEFAULT_CONTRACT_PATH = Path("data/manifests/data_contract.json")
DEFAULT_PIPELINE_SPLIT_YEAR = 2015
DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD = 3
DEFAULT_HOST_EVENNESS_BIAS_POWER = 0.5
DEFAULT_HOST_PHYLO_BREADTH_WEIGHT = 0.65
DEFAULT_HOST_PHYLO_DISPERSION_WEIGHT = 0.35
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


def _coerce_int(value: object, *, default: int) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _coerce_float(value: object, *, default: float) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


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
        split_year=_coerce_int(
            pipeline.get("split_year"), default=DEFAULT_PIPELINE_SPLIT_YEAR
        ),
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

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def config(self) -> dict:
        import yaml

        config_path = self.root / "config.yaml"
        if not config_path.exists():
            return {}
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def pipeline_settings(self) -> PipelineSettings:
        return _pipeline_settings_from_config(self.config)

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

    def asset(self, key: str) -> DataAssetSpec:
        try:
            return self.contract.asset_map()[key]
        except KeyError as exc:
            raise KeyError(f"Unknown asset key: {key}") from exc

    def asset_path(self, key: str) -> Path:
        return self.asset(key).resolved_path(self.root)


def build_context(project_root: Path | None = None) -> ProjectContext:
    """Construct a project context rooted at the repository root."""
    root = find_project_root(project_root)
    return ProjectContext(root=root, contract=load_data_contract(root))
