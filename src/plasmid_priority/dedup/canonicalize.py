"""Stable canonical identifiers for identical plasmid records."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class _UnionFind:
    parents: dict[str, str]

    def __init__(self) -> None:
        self.parents = {}

    def find(self, item: str) -> str:
        if item not in self.parents:
            self.parents[item] = item
            return item
        # Iterative path compression to avoid stack overflow on large groups.
        root = item
        while self.parents.get(root, root) != root:
            root = self.parents[root]
        # Second pass: compress path for all intermediate nodes.
        current = item
        while current != root:
            next_parent = self.parents[current]
            self.parents[current] = root
            current = next_parent
        return root

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            self.parents[root_right] = root_left
        else:
            self.parents[root_left] = root_right


def annotate_canonical_ids(records: pd.DataFrame, identical_map: pd.DataFrame) -> pd.DataFrame:
    """Annotate each sequence accession with a deterministic canonical identifier."""
    union_find = _UnionFind()

    accessions = records["sequence_accession"].dropna().astype(str)
    for accession in accessions:
        union_find.find(accession)

    identical_accessions = (
        identical_map.get("NUCCORE_ACC", pd.Series("", index=identical_map.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    identical_targets = (
        identical_map.get("NUCCORE_Identical", pd.Series("", index=identical_map.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    for accession, identical in zip(
        identical_accessions.to_numpy(), identical_targets.to_numpy(), strict=False
    ):
        if accession and identical and accession.lower() != "nan" and identical.lower() != "nan":
            union_find.union(accession, identical)

    components: dict[str, list[str]] = {}
    for accession in accessions:
        root = union_find.find(accession)
        components.setdefault(root, []).append(accession)

    canonical_lookup: dict[str, tuple[str, int]] = {}
    for members in components.values():
        canonical = min(members)
        group_size = len(set(members))
        for accession in members:
            canonical_lookup[accession] = (canonical, group_size)

    annotated = records.copy()
    annotated["canonical_id"] = annotated["sequence_accession"].map(
        lambda accession: canonical_lookup.get(str(accession), (str(accession), 1))[0]
    )
    annotated["duplicate_group_size"] = annotated["sequence_accession"].map(
        lambda accession: canonical_lookup.get(str(accession), (str(accession), 1))[1]
    )
    annotated["is_canonical_representative"] = annotated["sequence_accession"].eq(
        annotated["canonical_id"]
    )
    return annotated
