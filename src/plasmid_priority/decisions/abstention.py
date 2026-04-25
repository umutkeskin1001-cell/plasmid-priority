from __future__ import annotations


def abstention_reason(
    *,
    uncertainty_tier: str,
    model_agreement: float,
    knownness_score: float,
    source_dominance_risk: bool,
    annotation_conflict_risk: bool,
) -> str:
    if knownness_score < 0.25 and model_agreement < 0.55:
        return "low_knownness_and_model_disagreement"
    if uncertainty_tier == "high" and model_agreement < 0.60:
        return "high_uncertainty_and_model_disagreement"
    if source_dominance_risk and knownness_score < 0.40:
        return "source_dominant_low_knownness"
    if annotation_conflict_risk and uncertainty_tier != "low":
        return "annotation_conflict_under_uncertainty"
    return ""
