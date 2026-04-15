"""AMR / clinical hazard (A) feature computation.

Provides ``compute_feature_a`` which builds the A-component feature
matrix for backbone-level AMR and clinical hazard scoring.

Re-exported from ``features.core`` for backward compatibility.
"""

from __future__ import annotations

from plasmid_priority.features.core import (  # noqa: F401
    compute_feature_a,
)

__all__ = ["compute_feature_a"]
