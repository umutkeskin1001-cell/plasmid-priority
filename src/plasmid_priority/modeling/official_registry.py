from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OfficialModelRole(StrEnum):
    VISIBILITY_BASELINE = "visibility_baseline"
    BIOLOGICAL_PRIOR = "biological_prior"
    PRIMARY = "primary"
    CHALLENGER = "challenger"
    DECISION_SURFACE = "decision_surface"


@dataclass(frozen=True)
class OfficialModelSpec:
    name: str
    role: OfficialModelRole
    can_win_official: bool
    allowed_feature_groups: tuple[str, ...]
    claim_scope: tuple[str, ...] = ("surveillance_prioritization",)
    experimental: bool = False


@dataclass(frozen=True)
class OfficialModelFamily:
    models: tuple[OfficialModelSpec, ...]

    def by_role(self, role: OfficialModelRole) -> tuple[OfficialModelSpec, ...]:
        return tuple(model for model in self.models if model.role == role)

    def primary(self) -> OfficialModelSpec:
        matches = self.by_role(OfficialModelRole.PRIMARY)
        if len(matches) != 1:
            raise ValueError("Official model family must define exactly one primary model")
        return matches[0]

    def decision_surface(self) -> OfficialModelSpec:
        matches = self.by_role(OfficialModelRole.DECISION_SURFACE)
        if len(matches) != 1:
            raise ValueError("Official model family must define exactly one decision surface")
        return matches[0]


OFFICIAL_MODEL_FAMILY = OfficialModelFamily(
    models=(
        OfficialModelSpec(
            name="visibility_baseline",
            role=OfficialModelRole.VISIBILITY_BASELINE,
            can_win_official=False,
            allowed_feature_groups=("visibility",),
        ),
        OfficialModelSpec(
            name="frozen_biological_prior",
            role=OfficialModelRole.BIOLOGICAL_PRIOR,
            can_win_official=False,
            allowed_feature_groups=("mobility", "host_ecology", "amr_context", "evidence_depth"),
        ),
        OfficialModelSpec(
            name="sparse_calibrated_logistic",
            role=OfficialModelRole.PRIMARY,
            can_win_official=True,
            allowed_feature_groups=(
                "visibility",
                "mobility",
                "host_ecology",
                "amr_context",
                "backbone_structure",
                "evidence_depth",
            ),
        ),
        OfficialModelSpec(
            name="bounded_monotonic_tree",
            role=OfficialModelRole.CHALLENGER,
            can_win_official=True,
            allowed_feature_groups=(
                "visibility",
                "mobility",
                "host_ecology",
                "amr_context",
                "backbone_structure",
                "evidence_depth",
            ),
        ),
        OfficialModelSpec(
            name="conservative_evidence_consensus",
            role=OfficialModelRole.DECISION_SURFACE,
            can_win_official=False,
            allowed_feature_groups=("uncertainty", "evidence_depth"),
        ),
    ),
)


def validate_official_model_family(family: OfficialModelFamily) -> dict[str, object]:
    models = tuple(family.models)
    names = [model.name for model in models]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    roles = {model.role for model in models}
    required_roles = set(OfficialModelRole)
    missing_roles = sorted(str(role) for role in required_roles - roles)
    experimental = [model.name for model in models if model.experimental]

    reasons: list[str] = []
    if len(models) > 5:
        reasons.append("official_model_family_too_large")
    if duplicate_names:
        reasons.append("duplicate_model_names")
    if missing_roles:
        reasons.append("missing_required_roles")
    if len(family.by_role(OfficialModelRole.PRIMARY)) != 1:
        reasons.append("primary_model_count_not_one")
    if len(family.by_role(OfficialModelRole.DECISION_SURFACE)) != 1:
        reasons.append("decision_surface_count_not_one")
    if experimental:
        reasons.append("experimental_model_in_official_family")

    status = "pass" if not reasons else "fail"
    primary_model_name = (
        family.primary().name if "primary_model_count_not_one" not in reasons else None
    )
    decision_model_name = (
        family.decision_surface().name
        if "decision_surface_count_not_one" not in reasons
        else None
    )
    return {
        "status": status,
        "model_count": len(models),
        "primary_model_name": primary_model_name,
        "decision_model_name": decision_model_name,
        "experimental_model_count": len(experimental),
        "missing_roles": missing_roles,
        "duplicate_model_names": duplicate_names,
        "reasons": reasons,
    }


def official_model_names(family: OfficialModelFamily = OFFICIAL_MODEL_FAMILY) -> tuple[str, ...]:
    return tuple(model.name for model in family.models)


def official_winner_candidates(
    family: OfficialModelFamily = OFFICIAL_MODEL_FAMILY,
) -> tuple[str, ...]:
    return tuple(model.name for model in family.models if model.can_win_official)


def require_official_model(
    model_name: str,
    family: OfficialModelFamily = OFFICIAL_MODEL_FAMILY,
) -> OfficialModelSpec:
    for model in family.models:
        if model.name == model_name:
            return model
    raise KeyError(f"Model is not in the official family: {model_name}")
