"""Tests for Mash + Leiden graph-based backbone clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plasmid_priority.backbone.graph_clustering import (
    MashLeidenClustering,
    integrate_mash_backbones,
)


class TestMashLeidenClusteringMocked:
    """Unit tests using mocked distance data (no actual Mash binary needed)."""

    def test_graph_building(self) -> None:
        pytest.importorskip("igraph")
        pytest.importorskip("leidenalg")

        clusterer = MashLeidenClustering(
            mash_threshold=0.05,
            leiden_resolution=1.0,
            random_state=42,
        )

        # Simulate a distance DataFrame
        ids = ["seq_A", "seq_B", "seq_C", "seq_D"]
        dist = pd.DataFrame(
            [
                [0.0, 0.01, 0.02, 0.50],
                [0.01, 0.0, 0.03, 0.55],
                [0.02, 0.03, 0.0, 0.60],
                [0.50, 0.55, 0.60, 0.0],
            ],
            index=ids,
            columns=ids,
        )

        graph = clusterer._build_graph_from_distances(dist, threshold=0.05)
        # seq_A, seq_B, seq_C should be connected; seq_D isolated
        assert graph.vcount() == 4
        assert graph.ecount() >= 2  # A-B, A-C, B-C edges

    def test_leiden_mock(self) -> None:
        pytest.importorskip("igraph")
        pytest.importorskip("leidenalg")

        clusterer = MashLeidenClustering(random_state=42)
        ids = ["seq_A", "seq_B", "seq_C", "seq_D"]
        dist = pd.DataFrame(
            [
                [0.0, 0.01, 0.02, 0.50],
                [0.01, 0.0, 0.03, 0.55],
                [0.02, 0.03, 0.0, 0.60],
                [0.50, 0.55, 0.60, 0.0],
            ],
            index=ids,
            columns=ids,
        )

        graph = clusterer._build_graph_from_distances(dist, threshold=0.05)
        membership = clusterer._run_leiden(graph)
        assert len(membership) == 4
        # A, B, C should be in the same community; D in another
        assert membership[0] == membership[1] == membership[2]
        assert membership[3] != membership[0]

    def test_get_cluster_table(self) -> None:
        pytest.importorskip("igraph")
        pytest.importorskip("leidenalg")

        clusterer = MashLeidenClustering(random_state=42)
        ids = ["seq_A", "seq_B", "seq_C"]
        dist = pd.DataFrame(
            [
                [0.0, 0.01, 0.02],
                [0.01, 0.0, 0.03],
                [0.02, 0.03, 0.0],
            ],
            index=ids,
            columns=ids,
        )

        graph = clusterer._build_graph_from_distances(dist, threshold=0.05)
        membership = clusterer._run_leiden(graph)
        clusterer._cluster_map["backbone"] = {
            sid: f"backbone_{m:05d}" for sid, m in zip(ids, membership)
        }

        table = clusterer.get_cluster_table(level="backbone")
        assert len(table) == 3
        assert "sequence_accession" in table.columns
        assert "backbone_cluster_id" in table.columns
        assert table["backbone_cluster_id"].nunique() == 1

    def test_temporal_versioning(self) -> None:
        pytest.importorskip("igraph")
        pytest.importorskip("leidenalg")

        clusterer = MashLeidenClustering(
            temporal_window_years=5,
            random_state=42,
        )
        clusterer._cluster_map["backbone"] = {
            "seq_A": "backbone_00001",
            "seq_B": "backbone_00001",
        }
        metadata = pd.DataFrame(
            {
                "sequence_accession": ["seq_A", "seq_B"],
                "resolved_year": [2006, 2012],
            }
        )
        clusterer._apply_temporal_versioning(
            metadata,
            id_col="sequence_accession",
            year_col="resolved_year",
        )

        # Algorithm starts windows from min_year (2006) with window_size=5
        # windows = [2006, 2011]; 2006 -> window 2006_2010, 2012 -> window 2011_2015
        assert "v2006_2010" in clusterer._cluster_map["backbone"]["seq_A"]
        assert "v2011_2015" in clusterer._cluster_map["backbone"]["seq_B"]

    def test_pangenome_features(self) -> None:
        pytest.importorskip("igraph")
        pytest.importorskip("leidenalg")

        clusterer = MashLeidenClustering(random_state=42)
        clusterer._cluster_map["backbone"] = {
            "seq_A": "backbone_00001",
            "seq_B": "backbone_00001",
            "seq_C": "backbone_00002",
        }
        gene_presence = pd.DataFrame(
            {
                "seq_A": [1, 1, 1, 0],
                "seq_B": [1, 1, 0, 0],
                "seq_C": [0, 0, 1, 1],
            },
            index=["gene_1", "gene_2", "gene_3", "gene_4"],
        )

        metadata = pd.DataFrame(
            {
                "sequence_accession": ["seq_A", "seq_B", "seq_C"],
            }
        )

        pg = clusterer.compute_pangenome_features(
            metadata,
            gene_presence,
            id_col="sequence_accession",
            level="backbone",
        )
        assert len(pg) == 2
        # backbone_00001: gene_1 and gene_2 in both = core; gene_3 in one = accessory
        bb1 = pg.loc[pg["backbone_cluster_id"] == "backbone_00001"]
        assert bb1["n_core_genes"].iloc[0] == 2
        assert bb1["n_accessory_genes"].iloc[0] == 1
        assert 0.0 < bb1["core_ratio"].iloc[0] < 1.0


class TestIntegrateMashBackbones:
    def test_merge(self) -> None:
        records = pd.DataFrame(
            {
                "sequence_accession": ["seq_A", "seq_B", "seq_C"],
                "resolved_year": [2010, 2012, 2014],
                "primary_cluster_id": ["old_1", "old_2", np.nan],
            }
        )
        cluster_table = pd.DataFrame(
            {
                "sequence_accession": ["seq_A", "seq_B"],
                "backbone_cluster_id": ["mash_1", "mash_2"],
            }
        )
        merged = integrate_mash_backbones(records, cluster_table)
        assert merged["primary_cluster_id"].iloc[0] == "mash_1"
        assert merged["primary_cluster_id"].iloc[1] == "mash_2"
        # seq_C has no mash cluster, keeps old or gets NaN
        assert pd.isna(merged["primary_cluster_id"].iloc[2])

    def test_missing_column_raises(self) -> None:
        records = pd.DataFrame({"sequence_accession": ["seq_A"]})
        cluster_table = pd.DataFrame({"wrong_col": ["x"]})
        with pytest.raises(KeyError):
            integrate_mash_backbones(records, cluster_table)


def test_temporal_versioning_all_missing_years_uses_unknown_window() -> None:
    clustering = MashLeidenClustering(temporal_window_years=5)
    clustering._cluster_map = {"backbone": {"seq1": "B1", "seq2": "B1"}}
    metadata = pd.DataFrame(
        {"sequence_accession": ["seq1", "seq2"], "resolved_year": [None, "bad"]}
    )

    clustering._apply_temporal_versioning(metadata, "sequence_accession", "resolved_year")

    assert clustering._cluster_map["backbone"]["seq1"] == "B1_vunknown"
    assert clustering._cluster_map["backbone"]["seq2"] == "B1_vunknown"


def test_temporal_versioning_rejects_non_positive_window() -> None:
    clustering = MashLeidenClustering(temporal_window_years=0)
    clustering._cluster_map = {"backbone": {"seq1": "B1"}}
    metadata = pd.DataFrame({"sequence_accession": ["seq1"], "resolved_year": [2010]})

    with pytest.raises(ValueError, match="temporal_window_years"):
        clustering._apply_temporal_versioning(metadata, "sequence_accession", "resolved_year")
