from __future__ import annotations

from plasmid_priority.governance import compare_invariants, scientific_equivalence


def _sample_snapshot(*, roc_auc: float = 0.82, top_ids: list[str] | None = None) -> dict:
    ids = top_ids or [f"bb_{i}" for i in range(100)]
    return {
        "headline_metrics": {
            "published_primary": {
                "roc_auc": roc_auc,
                "average_precision": 0.70,
                "brier_score": 0.20,
                "ece": 0.04,
            }
        },
        "top_100_candidates": [
            {"backbone_id": backbone_id, "freeze_rank": index + 1}
            for index, backbone_id in enumerate(ids)
        ],
        "artifacts": {
            "reports/core_tables/headline_validation_summary.tsv": {
                "rows": 3,
                "columns": ["summary_label", "roc_auc"],
                "sha256": "abc",
            }
        },
    }


def test_invariants_pass_for_identical_snapshots() -> None:
    baseline = _sample_snapshot()
    candidate = _sample_snapshot()
    contract = {
        "invariants": {
            "ranking_drift": {"min_top100_overlap_ratio": 0.95},
            "metric_drift": {"max_abs_delta": 0.005},
            "row_count_drift": {"max_row_delta": 0},
            "schema_drift": {"allow_additive_columns": False},
        },
        "rollback_conditions": {
            "ranking_drift": {"rollback_on_fail": True},
            "metric_drift": {"rollback_on_fail": True},
            "row_count_drift": {"rollback_on_fail": True},
            "schema_drift": {"rollback_on_fail": True},
        },
    }

    results = compare_invariants(baseline=baseline, candidate=candidate, contract=contract)

    assert all(result.status == "pass" for result in results)
    assert all(not result.rollback for result in results)


def test_equivalence_fails_on_metric_and_ranking_drift() -> None:
    baseline = _sample_snapshot(roc_auc=0.82)
    candidate = _sample_snapshot(
        roc_auc=0.76,
        top_ids=[f"new_{i}" for i in range(100)],
    )
    contract = {
        "equivalence": {
            "metric_abs_tolerance": 0.0001,
            "ranking_min_overlap_ratio": 0.99,
        }
    }

    result = scientific_equivalence(baseline=baseline, candidate=candidate, contract=contract)

    assert result["status"] == "fail"
    assert result["metric_equality"] == "fail"
    assert result["ranking_stability"] == "fail"
    assert result["rollback_required"] is True
