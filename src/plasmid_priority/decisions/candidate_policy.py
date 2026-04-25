from __future__ import annotations

from typing import Any

from plasmid_priority.decisions.abstention import abstention_reason
from plasmid_priority.decisions.claim_language import allowed_claim_language


def _monitoring_tier(
    *,
    calibrated_priority_score: float,
    claim_language: str,
    abstention: str,
) -> str:
    if abstention:
        return "review_not_rank"
    if claim_language == "strong_surveillance_candidate" and calibrated_priority_score >= 0.85:
        return "core_surveillance"
    if claim_language == "moderate_surveillance_candidate":
        return "extended_watchlist"
    if claim_language == "hypothesis_generating_only":
        return "low_confidence_backlog"
    return "review_not_rank"


def build_candidate_decision(
    *,
    backbone_id: str,
    calibrated_priority_score: float,
    evidence_tier: str,
    uncertainty_tier: str,
    model_agreement: float,
    knownness_score: float,
    source_dominance_risk: bool,
    annotation_conflict_risk: bool,
) -> dict[str, Any]:
    abstention = abstention_reason(
        uncertainty_tier=uncertainty_tier,
        model_agreement=float(model_agreement),
        knownness_score=float(knownness_score),
        source_dominance_risk=bool(source_dominance_risk),
        annotation_conflict_risk=bool(annotation_conflict_risk),
    )
    claim = allowed_claim_language(
        calibrated_priority_score=float(calibrated_priority_score),
        evidence_tier=str(evidence_tier),
        uncertainty_tier=str(uncertainty_tier),
        abstention_reason=abstention,
    )
    return {
        "backbone_id": str(backbone_id),
        "calibrated_priority_score": float(calibrated_priority_score),
        "evidence_tier": str(evidence_tier),
        "uncertainty_tier": str(uncertainty_tier),
        "model_agreement": float(model_agreement),
        "knownness_score": float(knownness_score),
        "source_dominance_risk": bool(source_dominance_risk),
        "annotation_conflict_risk": bool(annotation_conflict_risk),
        "abstention_reason": abstention,
        "allowed_claim_language": claim,
        "recommended_monitoring_tier": _monitoring_tier(
            calibrated_priority_score=float(calibrated_priority_score),
            claim_language=claim,
            abstention=abstention,
        ),
    }
