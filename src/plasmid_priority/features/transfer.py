"""Transfer efficiency (T) feature computation.

Provides ``compute_feature_t`` which builds the T-component feature
matrix for backbone-level transfer efficiency scoring.

Re-exported from ``features.core`` for backward compatibility.
"""

from __future__ import annotations

from plasmid_priority.features.core import (  # noqa: F401
    compute_feature_t,
)

__all__ = ["compute_feature_t"]
