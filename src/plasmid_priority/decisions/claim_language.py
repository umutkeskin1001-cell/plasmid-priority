from __future__ import annotations


def allowed_claim_language(
    *,
    calibrated_priority_score: float,
    evidence_tier: str,
    uncertainty_tier: str,
    abstention_reason: str,
) -> str:
    if abstention_reason:
        return "insufficient_evidence"
    if evidence_tier == "high" and uncertainty_tier == "low" and calibrated_priority_score >= 0.85:
        return "strong_surveillance_candidate"
    if (
        evidence_tier in {"high", "moderate"}
        and uncertainty_tier != "high"
        and calibrated_priority_score >= 0.65
    ):
        return "moderate_surveillance_candidate"
    if calibrated_priority_score >= 0.50:
        return "hypothesis_generating_only"
    return "insufficient_evidence"
