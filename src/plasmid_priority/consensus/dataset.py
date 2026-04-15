"""Dataset assembly helpers for the consensus branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from plasmid_priority.consensus.specs import ConsensusConfig, load_consensus_config


@dataclass(slots=True)
class ConsensusDataset:
    """Merged branch prediction surface for consensus fusion."""

    table: pd.DataFrame
    config: ConsensusConfig
    label_column: str = "spread_label"

    @property
    def y(self) -> pd.Series:
        if self.label_column not in self.table.columns:
            return pd.Series(dtype=int)
        return pd.to_numeric(self.table[self.label_column], errors="coerce").fillna(-1).astype(int)


def prepare_consensus_dataset(
    geo_predictions: pd.DataFrame,
    bio_predictions: pd.DataFrame,
    clinical_predictions: pd.DataFrame,
    *,
    config: ConsensusConfig | dict[str, Any] | None = None,
) -> ConsensusDataset:
    consensus_config = (
        config if isinstance(config, ConsensusConfig) else load_consensus_config(config)
    )
    from plasmid_priority.consensus.fuse import merge_branch_predictions

    table = merge_branch_predictions(
        geo_predictions,
        bio_predictions,
        clinical_predictions,
    )
    return ConsensusDataset(table=table, config=consensus_config)
