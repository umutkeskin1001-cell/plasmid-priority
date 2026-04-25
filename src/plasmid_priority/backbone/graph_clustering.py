"""Graph-based sequence-aware backbone clustering using Mash + Leiden.

Replaces the operational ``OP::mobility::mpf::replicon::length_bin`` fallback
with a sequence-similarity graph. Nodes = plasmids, edges = Mash distance
below threshold. Leiden community detection yields backbones independent of
MOB-suite annotation quality.

Hierarchical naming:
- super-backbone: 90% ANI (coarse)
- backbone: 95% ANI (operational)
- sub-backbone: 99% ANI (fine-grained)

Temporal versioning: ``AA175_v2000_2010`` captures SNP accumulation and
gene gain/loss across time windows.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

try:
    import igraph as ig

    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    ig = None

try:
    import leidenalg

    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    leidenalg = None


class MashLeidenClustering:
    """Sequence-aware backbone clustering via Mash all-against-all + Leiden.

    Parameters
    ----------
    mash_threshold : float
        Mash distance threshold for edge creation (default 0.05 = ~95% ANI).
    super_threshold : float
        Coarser threshold for super-backbone (default 0.10 = ~90% ANI).
    sub_threshold : float
        Finer threshold for sub-backbone (default 0.01 = ~99% ANI).
    leiden_resolution : float
        Resolution parameter for Leiden algorithm (higher = more communities).
    temporal_window_years : int
        Width of temporal windows for versioning (default 5).
    random_state : int
        Seed for Leiden randomization.
    """

    def __init__(
        self,
        *,
        mash_threshold: float = 0.05,
        super_threshold: float = 0.10,
        sub_threshold: float = 0.01,
        leiden_resolution: float = 1.0,
        temporal_window_years: int = 5,
        random_state: int = 42,
    ) -> None:
        self.mash_threshold = float(mash_threshold)
        self.super_threshold = float(super_threshold)
        self.sub_threshold = float(sub_threshold)
        self.leiden_resolution = float(leiden_resolution)
        self.temporal_window_years = int(temporal_window_years)
        self.random_state = int(random_state)
        self._cluster_map: dict[str, dict[str, str]] = {}
        self._graph: Any | None = None

    def _run_mash_triangle(
        self,
        fasta_path: Path,
        sketch_path: Path,
        output_prefix: Path,
    ) -> pd.DataFrame:
        """Run Mash sketch + dist and return triangle distance matrix as DataFrame."""
        # Sketch
        cmd_sketch = [
            "mash",
            "sketch",
            "-o",
            str(sketch_path),
            str(fasta_path),
        ]
        _log.info("Running: %s", " ".join(cmd_sketch))
        result = subprocess.run(
            cmd_sketch,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            _log.error("mash sketch failed: %s", result.stderr)
            raise RuntimeError(f"mash sketch failed: {result.stderr}")

        # Dist
        dist_tsv = output_prefix.with_suffix(".dist.tsv")
        cmd_dist = [
            "mash",
            "dist",
            str(sketch_path) + ".msh",
            str(sketch_path) + ".msh",
            "-t",  # tabular output
        ]
        _log.info("Running: mash dist ...")
        with dist_tsv.open("w") as fh:
            result = subprocess.run(
                cmd_dist,
                stdout=fh,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            _log.error("mash dist failed: %s", result.stderr)
            raise RuntimeError(f"mash dist failed: {result.stderr}")

        # Parse triangle TSV (mash -t gives upper-triangle with IDs in first column)
        df = pd.read_csv(dist_tsv, sep="\t", index_col=0)
        return df

    def _build_graph_from_distances(
        self,
        dist_df: pd.DataFrame,
        threshold: float,
        id_to_meta: dict[str, dict[str, Any]] | None = None,
    ) -> Any:
        """Build an igraph from a Mash distance DataFrame."""
        if not IGRAPH_AVAILABLE or ig is None:
            raise ImportError("igraph is required for graph clustering")

        ids = list(dist_df.index)
        n = len(ids)

        edges: list[tuple[int, int]] = []
        weights: list[float] = []

        # Extract upper triangle edges below threshold
        for i, row_id in enumerate(ids):
            for j, col_id in enumerate(ids):
                if i >= j:
                    continue
                d = dist_df.at[row_id, col_id]
                if pd.isna(d):
                    continue
                d_float = float(str(d))
                if d_float <= threshold:
                    edges.append((i, j))
                    # Weight = 1 - normalized distance (closer = heavier edge)
                    weights.append(max(0.0, 1.0 - d_float / threshold))

        g = ig.Graph(n=n, edges=edges, directed=False)
        g.vs["name"] = ids
        if edges:
            g.es["weight"] = weights

        # Add vertex attributes from metadata if provided
        if id_to_meta:
            for vid, sid in enumerate(ids):
                meta = id_to_meta.get(sid, {})
                for key, value in meta.items():
                    if key not in g.vs.attributes():
                        g.vs[key] = [None] * n
                    g.vs[vid][key] = value

        return g

    def _run_leiden(self, graph: Any) -> list[int]:
        """Run Leiden community detection and return membership list."""
        if not LEIDEN_AVAILABLE or leidenalg is None:
            raise ImportError("leidenalg is required for Leiden clustering")

        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=self.leiden_resolution,
            seed=self.random_state,
        )
        return list(partition.membership)

    def fit(
        self,
        fasta_path: Path | str,
        metadata: pd.DataFrame | None = None,
        id_col: str = "sequence_accession",
        year_col: str = "resolved_year",
    ) -> "MashLeidenClustering":
        """Cluster plasmids from a FASTA file into sequence-aware backbones.

        Parameters
        ----------
        fasta_path : Path | str
            Path to the multi-FASTA file containing all plasmid sequences.
        metadata : pd.DataFrame | None
            Optional metadata indexed by ``id_col`` for temporal versioning.
        id_col : str
            Column in metadata matching FASTA headers.
        year_col : str
            Year column for temporal window assignment.

        Returns
        -------
        self
        """
        fasta_path = Path(fasta_path)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")

        with tempfile.TemporaryDirectory(prefix="plasmid_priority_mash_") as tmpdir:
            tmp = Path(tmpdir)
            sketch = tmp / "sketch"
            prefix = tmp / "dist"

            dist_df = self._run_mash_triangle(fasta_path, sketch, prefix)

        # Build metadata mapping
        id_to_meta: dict[str, dict[str, Any]] = {}
        if metadata is not None and id_col in metadata.columns:
            for _, row in metadata.iterrows():
                sid = str(row[id_col]).strip()
                id_to_meta[sid] = {k: row[k] for k in metadata.columns if pd.notna(row[k])}

        # Hierarchical clustering at three thresholds
        for level, threshold in (
            ("super", self.super_threshold),
            ("backbone", self.mash_threshold),
            ("sub", self.sub_threshold),
        ):
            _log.info("Building %s graph at threshold=%.3f", level, threshold)
            graph = self._build_graph_from_distances(dist_df, threshold, id_to_meta)
            membership = self._run_leiden(graph)

            # Assign cluster IDs
            cluster_map: dict[str, str] = {}
            for sid, memb in zip(dist_df.index, membership):
                cluster_map[sid] = f"{level}_{memb:05d}"

            self._cluster_map[level] = cluster_map

        # Temporal versioning: add year window suffix to backbones
        if metadata is not None and year_col in metadata.columns:
            self._apply_temporal_versioning(metadata, id_col, year_col)

        _log.info(
            "Mash+Leiden complete: super=%d, backbone=%d, sub=%d clusters",
            len(set(self._cluster_map["super"].values())),
            len(set(self._cluster_map["backbone"].values())),
            len(set(self._cluster_map["sub"].values())),
        )
        return self

    def _apply_temporal_versioning(
        self,
        metadata: pd.DataFrame,
        id_col: str,
        year_col: str,
    ) -> None:
        """Append temporal window suffix to backbone IDs."""
        window_size = int(self.temporal_window_years)
        if window_size <= 0:
            raise ValueError("temporal_window_years must be a positive integer")

        years = pd.to_numeric(metadata[year_col], errors="coerce")
        id_to_window: dict[str, str] = {}
        valid_years = years.dropna()
        if valid_years.empty:
            for _, row in metadata.iterrows():
                sid = str(row[id_col]).strip()
                id_to_window[sid] = "unknown"
        else:
            min_year = int(valid_years.min())
            max_year = int(valid_years.max())
            windows = list(range(min_year, max_year + 1, window_size))
            for _, row in metadata.iterrows():
                sid = str(row[id_col]).strip()
                raw_year = pd.to_numeric(pd.Series([row.get(year_col)]), errors="coerce").iloc[0]
                if pd.isna(raw_year):
                    id_to_window[sid] = "unknown"
                    continue
                idx = max(0, int((int(raw_year) - min_year) // window_size))
                window_start = windows[min(idx, len(windows) - 1)]
                id_to_window[sid] = f"{window_start}_{window_start + window_size - 1}"

        for level in self._cluster_map:
            for sid in self._cluster_map[level]:
                base = self._cluster_map[level][sid]
                window = id_to_window.get(sid, "unknown")
                self._cluster_map[level][sid] = f"{base}_v{window}"

    def get_backbone_id(self, sequence_id: str, level: str = "backbone") -> str:
        """Get the cluster ID for a sequence at the specified level."""
        if level not in self._cluster_map:
            raise KeyError(f"Level '{level}' not available. Choose from {list(self._cluster_map)}")
        return self._cluster_map[level].get(sequence_id, f"{level}_singleton_{sequence_id}")

    def get_cluster_table(self, level: str = "backbone") -> pd.DataFrame:
        """Return a DataFrame mapping sequence_id -> cluster_id."""
        if level not in self._cluster_map:
            raise KeyError(f"Level '{level}' not available")
        return pd.DataFrame(
            list(self._cluster_map[level].items()),
            columns=["sequence_accession", f"{level}_cluster_id"],
        )

    def compute_pangenome_features(
        self,
        metadata: pd.DataFrame,
        gene_presence: pd.DataFrame,
        id_col: str = "sequence_accession",
        level: str = "backbone",
    ) -> pd.DataFrame:
        """Compute core/accessory gene ratio per cluster as adaptation proxy.

        Requires a gene presence/absence matrix (genes x samples) where
        values are binary or counts.

        Returns
        -------
        pd.DataFrame
            One row per cluster with core_ratio, accessory_richness, shell_stability.
        """
        cluster_map = self._cluster_map.get(level, {})
        if not cluster_map:
            return pd.DataFrame()

        # Invert: cluster -> list of sequence IDs
        cluster_to_ids: dict[str, list[str]] = {}
        for sid, cid in cluster_map.items():
            cluster_to_ids.setdefault(cid, []).append(sid)

        # Subset gene matrix to known IDs
        valid_ids = [sid for sid in gene_presence.columns if sid in cluster_map]
        gp = gene_presence.loc[:, valid_ids]

        results: list[dict[str, object]] = []
        for cid, sids in cluster_to_ids.items():
            sub = gp.loc[:, [s for s in sids if s in gp.columns]]
            if sub.empty:
                continue
            # Core: present in >= 95% of cluster members
            presence_frac = (sub > 0).mean(axis=1)
            n_core = int((presence_frac >= 0.95).sum())
            n_accessory = int(((presence_frac > 0.0) & (presence_frac < 0.95)).sum())
            n_shell = int((presence_frac > 0.0).sum())
            n_total = len(presence_frac)

            core_ratio = n_core / n_total if n_total > 0 else 0.0
            accessory_richness = n_accessory / n_total if n_total > 0 else 0.0
            shell_stability = n_shell / n_total if n_total > 0 else 0.0

            results.append(
                {
                    f"{level}_cluster_id": cid,
                    "n_members": len(sids),
                    "n_core_genes": n_core,
                    "n_accessory_genes": n_accessory,
                    "core_ratio": core_ratio,
                    "accessory_richness": accessory_richness,
                    "shell_stability": shell_stability,
                }
            )

        return pd.DataFrame(results)


def integrate_mash_backbones(
    records: pd.DataFrame,
    cluster_table: pd.DataFrame,
    id_col: str = "sequence_accession",
    level: str = "backbone",
) -> pd.DataFrame:
    """Merge Mash-Leiden cluster IDs into the plasmid record table.

    Falls back to the existing ``primary_cluster_id`` or operational
    fallback if Mash clustering is unavailable for a record.
    """
    out = records.copy()
    cluster_col = f"{level}_cluster_id"
    if cluster_col not in cluster_table.columns:
        raise KeyError(f"Cluster table missing '{cluster_col}'")

    merged = out.merge(
        cluster_table[["sequence_accession", cluster_col]],
        left_on=id_col,
        right_on="sequence_accession",
        how="left",
    )

    # Use Mash cluster as primary when available
    merged["primary_cluster_id"] = merged[cluster_col].fillna(
        merged.get("primary_cluster_id", pd.Series(np.nan, index=merged.index)),
    )
    merged["backbone_assignment_rule"] = np.where(
        merged[cluster_col].notna(),
        f"mash_leiden_{level}",
        merged.get("backbone_assignment_rule", "unknown"),
    )
    return merged
