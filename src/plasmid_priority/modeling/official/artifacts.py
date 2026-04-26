from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from plasmid_priority.decisions.candidate_policy import build_candidate_decision
from plasmid_priority.modeling.official_registry import (
    OFFICIAL_MODEL_FAMILY,
    OfficialModelFamily,
    OfficialModelRole,
    OfficialModelSpec,
    validate_official_model_family,
)

OFFICIAL_SCORE_COLUMNS: dict[str, str] = {
    "visibility_baseline": "visibility_baseline",
    "frozen_biological_prior": "frozen_biological_prior",
    "sparse_calibrated_logistic": "sparse_calibrated_logistic",
    "bounded_monotonic_tree": "bounded_monotonic_tree",
    "conservative_evidence_consensus": "conservative_consensus_score",
}


def _scalar_string(frame: pd.DataFrame, column: str, *, default: str = "") -> str:
    if column not in frame.columns or frame.empty:
        return default
    value = frame[column].iloc[0]
    if pd.isna(value):
        return default
    return str(value)


def _scalar_int(frame: pd.DataFrame, column: str, *, default: int = 0) -> int:
    if column not in frame.columns or frame.empty:
        return default
    value = pd.to_numeric(pd.Series([frame[column].iloc[0]]), errors="coerce").iloc[0]
    return default if pd.isna(value) else int(value)


def _numeric_series(frame: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype("float64")


def _string_series(frame: pd.DataFrame, column: str, *, default: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[column].fillna(default).astype(str)


def _boolean_series(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    normalized = values.fillna(str(default)).astype(str).str.lower().str.strip()
    return normalized.isin({"1", "true", "yes", "y", "pass"})


def _score_status(spec: OfficialModelSpec, scores: pd.DataFrame, score_column: str) -> str:
    if spec.role == OfficialModelRole.PRIMARY or spec.role == OfficialModelRole.CHALLENGER:
        status = str(scores.get("official_supervised_model_status", pd.Series(["missing"])).iloc[0])
        return "available" if status == "fit" else status
    if score_column not in scores.columns:
        return "missing_score_column"
    if not pd.to_numeric(scores[score_column], errors="coerce").notna().any():
        return "missing_scores"
    return "available"


def build_official_model_scorecard(
    scores: pd.DataFrame,
    *,
    family: OfficialModelFamily = OFFICIAL_MODEL_FAMILY,
) -> pd.DataFrame:
    """Build a release-facing scorecard for every official model role."""
    rows: list[dict[str, Any]] = []
    supervised_feature_count = _scalar_int(scores, "official_supervised_feature_count")
    supervised_requested_feature_count = _scalar_int(
        scores,
        "official_supervised_requested_feature_count",
    )
    supervised_missing_feature_count = _scalar_int(
        scores,
        "official_supervised_missing_feature_count",
    )
    supervised_missing_features = _scalar_string(scores, "official_supervised_missing_features")
    supervised_labeled_count = _scalar_int(scores, "official_supervised_labeled_count")
    supervised_positive_count = _scalar_int(scores, "official_supervised_positive_count")
    supervised_negative_count = _scalar_int(scores, "official_supervised_negative_count")
    supervised_min_labeled_rows = _scalar_int(scores, "official_supervised_min_labeled_rows")
    supervised_min_class_count = _scalar_int(scores, "official_supervised_min_class_count")
    for order, spec in enumerate(family.models):
        score_column = OFFICIAL_SCORE_COLUMNS.get(spec.name, spec.name)
        score_values = (
            pd.to_numeric(scores[score_column], errors="coerce")
            if score_column in scores.columns
            else pd.Series(dtype="float64")
        )
        has_scores = bool(score_values.notna().any())
        has_bounded_scores = bool(
            has_scores and score_values.dropna().between(0.0, 1.0).all(),
        )
        rows.append(
            {
                "official_model_order": order,
                "model_name": spec.name,
                "role": str(spec.role),
                "can_win_official": bool(spec.can_win_official),
                "experimental": bool(spec.experimental),
                "allowed_feature_groups": ",".join(spec.allowed_feature_groups),
                "score_column": score_column,
                "official_family_status": _score_status(spec, scores, score_column),
                "has_bounded_scores": has_bounded_scores,
                "score_min": float(score_values.min()) if has_scores else pd.NA,
                "score_max": float(score_values.max()) if has_scores else pd.NA,
                "supervised_feature_count": supervised_feature_count
                if spec.role in {OfficialModelRole.PRIMARY, OfficialModelRole.CHALLENGER}
                else 0,
                "supervised_requested_feature_count": supervised_requested_feature_count
                if spec.role in {OfficialModelRole.PRIMARY, OfficialModelRole.CHALLENGER}
                else 0,
                "supervised_missing_feature_count": supervised_missing_feature_count
                if spec.role in {OfficialModelRole.PRIMARY, OfficialModelRole.CHALLENGER}
                else 0,
                "supervised_missing_features": supervised_missing_features
                if spec.role in {OfficialModelRole.PRIMARY, OfficialModelRole.CHALLENGER}
                else "",
                "supervised_labeled_count": supervised_labeled_count
                if spec.role in {OfficialModelRole.PRIMARY, OfficialModelRole.CHALLENGER}
                else 0,
                "supervised_positive_count": supervised_positive_count
                if spec.role in {OfficialModelRole.PRIMARY, OfficialModelRole.CHALLENGER}
                else 0,
                "supervised_negative_count": supervised_negative_count
                if spec.role in {OfficialModelRole.PRIMARY, OfficialModelRole.CHALLENGER}
                else 0,
                "supervised_min_labeled_rows": supervised_min_labeled_rows
                if spec.role in {OfficialModelRole.PRIMARY, OfficialModelRole.CHALLENGER}
                else 0,
                "supervised_min_class_count": supervised_min_class_count
                if spec.role in {OfficialModelRole.PRIMARY, OfficialModelRole.CHALLENGER}
                else 0,
            },
        )
    return pd.DataFrame(rows)


def build_official_candidate_decisions(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    id_column: str,
) -> pd.DataFrame:
    """Convert official scores into claim-limited release candidate decisions."""
    if id_column not in frame.columns:
        raise ValueError(f"Missing candidate id column: {id_column}")
    if "conservative_consensus_score" not in scores.columns:
        raise ValueError("Missing conservative_consensus_score in official scores")

    priority_score = _numeric_series(scores, "conservative_consensus_score")
    model_agreement = _numeric_series(scores, "model_agreement", default=0.0)
    evidence_tier = _string_series(frame, "evidence_tier", default="insufficient")
    uncertainty_tier = _string_series(frame, "uncertainty_tier", default="high")
    knownness_score = _numeric_series(frame, "knownness_score", default=0.0)
    source_dominance_risk = _boolean_series(frame, "source_dominance_risk")
    annotation_conflict_risk = _boolean_series(frame, "annotation_conflict_risk")

    rows: list[dict[str, Any]] = []
    for row_position, index in enumerate(frame.index):
        decision = build_candidate_decision(
            backbone_id=str(frame.loc[index, id_column]),
            calibrated_priority_score=float(priority_score.loc[index]),
            evidence_tier=str(evidence_tier.loc[index]),
            uncertainty_tier=str(uncertainty_tier.loc[index]),
            model_agreement=float(model_agreement.loc[index]),
            knownness_score=float(knownness_score.loc[index]),
            source_dominance_risk=bool(source_dominance_risk.loc[index]),
            annotation_conflict_risk=bool(annotation_conflict_risk.loc[index]),
        )
        decision["_input_order"] = row_position
        decision["official_decision_surface"] = "conservative_evidence_consensus"
        rows.append(decision)

    decisions = pd.DataFrame(rows)
    if decisions.empty:
        return decisions
    decisions = decisions.sort_values(
        ["calibrated_priority_score", "backbone_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    decisions["official_rank"] = range(1, len(decisions) + 1)
    return decisions.drop(columns="_input_order")


def build_official_release_artifacts(
    frame: pd.DataFrame,
    *,
    id_column: str,
    label_column: str | None = None,
    family: OfficialModelFamily = OFFICIAL_MODEL_FAMILY,
) -> dict[str, Any]:
    from plasmid_priority.modeling.official.runner import score_official_model_family

    scores = score_official_model_family(frame, label_column=label_column)
    scores = pd.concat(
        [
            frame.loc[:, [id_column]].reset_index(drop=True),
            scores.reset_index(drop=True),
        ],
        axis=1,
    )
    scorecard = build_official_model_scorecard(scores, family=family)
    candidate_decisions = build_official_candidate_decisions(
        frame,
        scores,
        id_column=id_column,
    )
    summary = validate_official_model_family(family)
    summary.update(
        {
            "candidate_count": int(len(candidate_decisions)),
            "review_not_rank_count": int(
                candidate_decisions["recommended_monitoring_tier"].eq("review_not_rank").sum(),
            )
            if "recommended_monitoring_tier" in candidate_decisions.columns
            else 0,
            "official_model_family_status": cast(str, summary["status"]),
            "decision_surface": "conservative_evidence_consensus",
            "official_supervised_model_status": _scalar_string(
                scores,
                "official_supervised_model_status",
                default="missing",
            ),
            "official_supervised_feature_count": _scalar_int(
                scores,
                "official_supervised_feature_count",
            ),
            "official_supervised_requested_feature_count": _scalar_int(
                scores,
                "official_supervised_requested_feature_count",
            ),
            "official_supervised_missing_feature_count": _scalar_int(
                scores,
                "official_supervised_missing_feature_count",
            ),
            "official_supervised_missing_features": _scalar_string(
                scores,
                "official_supervised_missing_features",
            ),
            "official_supervised_labeled_count": _scalar_int(
                scores,
                "official_supervised_labeled_count",
            ),
            "official_supervised_positive_count": _scalar_int(
                scores,
                "official_supervised_positive_count",
            ),
            "official_supervised_negative_count": _scalar_int(
                scores,
                "official_supervised_negative_count",
            ),
            "official_supervised_models_used": _scalar_string(
                scores,
                "official_supervised_models_used",
            ),
            "official_consensus_score_columns": _scalar_string(
                scores,
                "official_consensus_score_columns",
            ),
        },
    )
    return {
        "scores": scores,
        "scorecard": scorecard,
        "candidate_decisions": candidate_decisions,
        "summary": summary,
    }


def write_official_release_artifacts(
    artifacts: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write official release artifacts as stable TSV/JSON files."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    scores_path = destination / "official_model_scores.tsv"
    scorecard_path = destination / "official_model_scorecard.tsv"
    decisions_path = destination / "official_candidate_decisions.tsv"
    summary_path = destination / "official_model_summary.json"

    cast(pd.DataFrame, artifacts["scores"]).to_csv(scores_path, sep="\t", index=False)
    cast(pd.DataFrame, artifacts["scorecard"]).to_csv(scorecard_path, sep="\t", index=False)
    cast(pd.DataFrame, artifacts["candidate_decisions"]).to_csv(
        decisions_path,
        sep="\t",
        index=False,
    )
    summary_path.write_text(
        json.dumps(artifacts["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "scores": scores_path,
        "scorecard": scorecard_path,
        "candidate_decisions": decisions_path,
        "summary": summary_path,
    }
