"""Backbone assignment and coherence routines."""

from plasmid_priority.backbone.core import (
    assign_backbone_ids,
    assign_backbone_ids_training_only,
    compute_backbone_coherence,
    fallback_backbone_key,
)

__all__ = [
    "assign_backbone_ids",
    "assign_backbone_ids_training_only",
    "compute_backbone_coherence",
    "fallback_backbone_key",
]
