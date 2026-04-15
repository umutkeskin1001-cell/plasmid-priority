"""Host range (H) feature computation.

Provides ``compute_feature_h`` which builds the H-component feature
matrix for backbone-level host range scoring.

Re-exported from ``features.core`` for backward compatibility.
"""

from __future__ import annotations

from plasmid_priority.features.core import (  # noqa: F401
    compute_feature_h,
)

__all__ = ["compute_feature_h"]
