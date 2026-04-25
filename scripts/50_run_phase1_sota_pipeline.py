"""Phase 1 SOTA Pipeline Runner: Bilimsel Temel.

Orchestrates the four Phase 1 components:
1. Cox-PH survival analysis + Fine-Gray competing risks
2. Mash + Leiden backbone clustering
3. K-mer baseline sequence features
4. Probabilistic labels + Co-teaching + Causal counterfactuals

Usage:
    uv run python scripts/50_run_phase1_sota_pipeline.py \
        --records data/silver/plasmid_harmonized.tsv \
        --fasta data/bronze/all_plasmids.fasta \
        --output data/phase1/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is on path when running standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plasmid_priority.backbone.graph_clustering import (
    MashLeidenClustering,
    integrate_mash_backbones,
)
from plasmid_priority.embedding.kmer_baseline import KmerFeatureExtractor
from plasmid_priority.labels.coteaching import CoTeachingTrainer
from plasmid_priority.labels.counterfactual import CausalLabelEstimator
from plasmid_priority.labels.probabilistic import build_probabilistic_labels
from plasmid_priority.survival.adaptive_split import AdaptiveTemporalSplitter
from plasmid_priority.survival.competing_risks import (
    FineGrayCompetingRisks,
    build_competing_risk_records,
)
from plasmid_priority.survival.cox_ph import (
    CoxPHSurvivalModel,
    build_survival_records,
)
from plasmid_priority.survival.lead_time import LeadTimeBiasCorrector

_log = logging.getLogger("phase1")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def run_survival_analysis(
    records: pd.DataFrame,
    output_dir: Path,
    split_year: int = 2015,
) -> dict[str, object]:
    """Run Cox-PH and Fine-Gray on plasmid records."""
    _log.info("=== Survival Analysis ===")

    # Adaptive split (optional; can override with fixed)
    splitter = AdaptiveTemporalSplitter(quantile=0.6, strategy="quantile")
    splitter.fit(records.get("resolved_year", pd.Series([split_year])))
    effective_split = splitter.split_year
    _log.info("Effective split year: %d", effective_split)

    # Lead-time correction
    corrector = LeadTimeBiasCorrector(correction_method="median_lag")
    if "collection_year" in records.columns:
        records = corrector.fit_transform(records)

    # Cox-PH
    surv_records = build_survival_records(
        records,
        split_year=effective_split,
        feature_cols=["sequence_length"] if "sequence_length" in records.columns else None,
    )
    cox: CoxPHSurvivalModel | None = None
    if not surv_records.empty and surv_records["event_observed"].nunique() > 1:
        cox = CoxPHSurvivalModel(penalizer=0.1)
        feature_cols = [
            c
            for c in surv_records.columns
            if c
            not in {
                "backbone_id",
                "time_to_event",
                "event_observed",
                "first_observed_year",
                "last_observed_year",
            }
        ]
        cox.fit(surv_records, feature_cols=feature_cols)
        risk_scores = cox.predict_risk_score(surv_records, horizon=5.0)
        surv_records["cox_risk_score_5yr"] = risk_scores.to_numpy()
        _log.info("Cox-PH c-index: %.4f", cox.concordance_index)
    else:
        _log.warning("Insufficient survival events for Cox-PH fit.")

    # Fine-Gray competing risks
    cr_records = build_competing_risk_records(records, split_year=effective_split)
    if not cr_records.empty and cr_records["event_type"].nunique() > 1:
        fg = FineGrayCompetingRisks(penalizer=0.1)
        feature_cols = [
            c
            for c in cr_records.columns
            if c
            not in {
                "backbone_id",
                "time_to_event",
                "event_type",
                "event_description",
                "last_observed_year",
            }
        ]
        fg.fit(cr_records, feature_cols=feature_cols)
        unified = fg.predict_unified_risk_score(cr_records, horizon=5.0)
        cr_records["fine_gray_unified_risk_5yr"] = unified.to_numpy()
        _log.info("Fine-Gray fitted for %d outcomes", len(fg._models))
    else:
        _log.warning("Insufficient competing risk events for Fine-Gray.")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    surv_records.to_csv(output_dir / "survival_records.tsv", sep="\t", index=False)
    cr_records.to_csv(output_dir / "competing_risk_records.tsv", sep="\t", index=False)

    return {
        "split_year": effective_split,
        "n_survival_records": len(surv_records),
        "n_competing_risk_records": len(cr_records),
        "cox_c_index": float(cox.concordance_index) if cox is not None and cox._is_fitted else None,
    }


def run_backbone_clustering(
    fasta_path: Path | None,
    records: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    """Run Mash + Leiden backbone clustering."""
    _log.info("=== Backbone Clustering ===")

    if fasta_path is None or not fasta_path.exists():
        _log.warning("FASTA not found; skipping Mash+Leiden clustering.")
        return {"status": "skipped", "reason": "fasta_missing"}

    clusterer = MashLeidenClustering(
        mash_threshold=0.05,
        super_threshold=0.10,
        sub_threshold=0.01,
        temporal_window_years=5,
        random_state=42,
    )

    metadata = (
        records[["sequence_accession", "resolved_year"]].copy()
        if {
            "sequence_accession",
            "resolved_year",
        }.issubset(records.columns)
        else None
    )

    clusterer.fit(
        fasta_path, metadata=metadata, id_col="sequence_accession", year_col="resolved_year"
    )

    # Export tables
    for level in ("super", "backbone", "sub"):
        table = clusterer.get_cluster_table(level=level)
        table.to_csv(output_dir / f"clusters_{level}.tsv", sep="\t", index=False)

    # Integrate into records
    backbone_table = clusterer.get_cluster_table(level="backbone")
    if "sequence_accession" in records.columns:
        records = integrate_mash_backbones(records, backbone_table)
        records.to_csv(output_dir / "records_with_mash_backbones.tsv", sep="\t", index=False)

    return {
        "status": "ok",
        "n_super_clusters": len(set(clusterer._cluster_map["super"].values())),
        "n_backbone_clusters": len(set(clusterer._cluster_map["backbone"].values())),
        "n_sub_clusters": len(set(clusterer._cluster_map["sub"].values())),
    }


def run_embeddings(
    fasta_path: Path | None,
    output_dir: Path,
) -> dict[str, object]:
    """Run k-mer baseline sequence features."""
    _log.info("=== Sequence Embeddings ===")

    results: dict[str, object] = {"kmer": {}}

    if fasta_path is None or not fasta_path.exists():
        _log.warning("FASTA not found; skipping embeddings.")
        return {"status": "skipped", "reason": "fasta_missing"}

    # K-mer baseline (always runs, no heavy deps)
    kmer = KmerFeatureExtractor(k_values=(4, 6), normalize=True, top_kmers_only=256)
    kmer_df = kmer.fit_transform_fasta(fasta_path)
    kmer_df.to_csv(output_dir / "kmer_features.tsv", sep="\t", index=False)
    results["kmer"] = {
        "n_samples": len(kmer_df),
        "n_features": len([c for c in kmer_df.columns if c.startswith("kmer_")]),
    }
    _log.info(
        "K-mer features: %d samples x %d features",
        results["kmer"]["n_samples"],  # type: ignore
        results["kmer"]["n_features"],  # type: ignore
    )

    return results


def run_probabilistic_labels(
    records: pd.DataFrame,
    output_dir: Path,
    split_year: int = 2015,
) -> dict[str, object]:
    """Run probabilistic labels, co-teaching, and causal estimation."""
    _log.info("=== Probabilistic Labels ===")

    prob_labels = build_probabilistic_labels(records, split_year=split_year)
    prob_labels.to_csv(output_dir / "probabilistic_labels.tsv", sep="\t", index=False)
    _log.info(
        "Probabilistic labels: mean confidence=%.3f, noise=%.3f",
        prob_labels["label_confidence"].mean(),
        prob_labels["label_noise_estimate"].mean(),
    )

    # Co-teaching (if torch available)
    coteaching_results: dict[str, object] = {"status": "skipped"}
    try:
        import torch  # noqa: F401

        # Build a simple feature matrix for demonstration
        numeric = records.select_dtypes(include="number").dropna(axis=1)
        if not numeric.empty and "backbone_id" in records.columns:
            # Aggregate to backbone level
            bb_features = numeric.groupby(records["backbone_id"]).mean().fillna(0)
            bb_labels = (
                prob_labels.set_index("backbone_id")["fused_spread_label"]
                .reindex(bb_features.index)
                .fillna(0)
                .to_numpy()
            )

            if len(bb_features) >= 20 and len(np.unique(bb_labels)) > 1:
                trainer = CoTeachingTrainer(
                    input_dim=bb_features.shape[1],
                    hidden_dim=64,
                    forget_rate=0.2,
                    num_epochs=30,
                    device="cpu",
                )
                X = bb_features.to_numpy().astype(np.float32)
                y = bb_labels.astype(int)
                trainer.fit(X, y)
                proba = trainer.predict_proba(X)
                coteaching_results = {
                    "status": "ok",
                    "n_samples": len(X),
                    "mean_proba": float(proba.mean()),
                }
                _log.info("Co-teaching trained on %d backbones", len(X))
    except Exception as exc:
        _log.warning("Co-teaching failed: %s", exc)
        coteaching_results = {"status": "error", "reason": str(exc)}

    # Causal counterfactuals
    causal_results: dict[str, object] = {"status": "skipped"}
    try:
        if "backbone_id" in records.columns and {
            "country",
            "resolved_year",
        }.issubset(records.columns):
            # Simple treatment: observed in >3 countries pre-split
            records_copy = records.copy()
            country_counts = records_copy.groupby("backbone_id")["country"].nunique()
            records_copy["treatment"] = (
                records_copy["backbone_id"].map(country_counts > 3).astype(int)
            )
            records_copy["outcome"] = (
                records_copy["backbone_id"]
                .map(prob_labels.set_index("backbone_id")["fused_spread_label"])
                .fillna(0)
            )

            estimator = CausalLabelEstimator(
                treatment_col="treatment",
                outcome_col="outcome",
            )
            estimator.fit(records_copy)
            cf = estimator.estimate_counterfactual_labels(records_copy)
            cf.to_csv(output_dir / "counterfactual_labels.tsv", sep="\t", index=False)
            causal_results = {
                "status": "ok",
                "ate": float(cf["causal_effect"].iloc[0]),
            }
            _log.info("Causal counterfactuals computed, ATE=%.4f", causal_results["ate"])
    except Exception as exc:
        _log.warning("Causal estimation failed: %s", exc)
        causal_results = {"status": "error", "reason": str(exc)}

    return {
        "n_backbones": len(prob_labels),
        "mean_confidence": float(prob_labels["label_confidence"].mean()),
        "mean_noise": float(prob_labels["label_noise_estimate"].mean()),
        "coteaching": coteaching_results,
        "causal": causal_results,
    }


def main() -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(description="Phase 1 SOTA Pipeline")
    parser.add_argument(
        "--records", type=Path, required=True, help="Path to harmonized records TSV"
    )
    parser.add_argument("--fasta", type=Path, default=None, help="Path to plasmid FASTA")
    parser.add_argument("--output", type=Path, default=Path("data/phase1"), help="Output directory")
    parser.add_argument("--split-year", type=int, default=2015, help="Temporal split year")
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load records
    records = pd.read_csv(args.records, sep="\t", low_memory=False)
    _log.info("Loaded %d records from %s", len(records), args.records)

    # Run all four components
    results: dict[str, object] = {}

    results["survival"] = run_survival_analysis(records, output_dir, split_year=args.split_year)
    results["backbone_clustering"] = run_backbone_clustering(args.fasta, records, output_dir)
    results["embeddings"] = run_embeddings(args.fasta, output_dir)
    results["probabilistic_labels"] = run_probabilistic_labels(
        records, output_dir, split_year=args.split_year
    )

    # Write manifest
    manifest_path = output_dir / "phase1_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    _log.info("Phase 1 complete. Manifest written to %s", manifest_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
