"""Sequence feature extraction for Plasmid Priority.

Provides k-mer frequency baselines as the primary sequence-derived feature
extraction method. DNA language models (DNABERT-2) were removed after
analysis showed limited practical benefit vs. computational cost for this
project. See docs/macos_compatibility.md for details.
"""

from __future__ import annotations

from plasmid_priority.embedding.kmer_baseline import KmerFeatureExtractor

__all__ = [
    "KmerFeatureExtractor",
]
