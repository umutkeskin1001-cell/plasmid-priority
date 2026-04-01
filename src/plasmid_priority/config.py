"""Project configuration and manifest loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from plasmid_priority.schemas import DataContract, DataAssetSpec

DEFAULT_CONTRACT_PATH = Path("data/manifests/data_contract.json")
DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD = 3
ROOT_MARKERS = (
    Path("pyproject.toml"),
    Path("data/manifests/data_contract.json"),
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
