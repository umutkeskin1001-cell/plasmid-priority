"""K-mer frequency baseline features as lightweight sequence proxy.

When DNABERT-2 is unavailable or for rapid prototyping, k-mer frequencies
(4-mer, 6-mer) provide a strong biological signal: GC content, codon bias,
mobile element signatures, and repeat structure.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

try:
    from Bio import SeqIO

    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    SeqIO = None  # type: ignore[assignment]


class KmerFeatureExtractor:
    """Extract k-mer frequency features from DNA sequences.

    Parameters
    ----------
    k_values : tuple[int, ...]
        K-mer lengths to extract (default (4, 6)).
    normalize : bool
        Whether to L1-normalize k-mer counts (default True).
    include_gc : bool
        Whether to include GC content (default True).
    include_entropy : bool
        Whether to include sequence complexity (Shannon entropy, default True).
    top_kmers_only : int | None
        If set, only keep the top-N most frequent k-mers across the
        training corpus (dimensionality reduction). None = keep all.
    """

    def __init__(
        self,
        *,
        k_values: tuple[int, ...] = (4, 6),
        normalize: bool = True,
        include_gc: bool = True,
        include_entropy: bool = True,
        top_kmers_only: int | None = None,
    ) -> None:
        self.k_values = tuple(int(k) for k in k_values)
        self.normalize = bool(normalize)
        self.include_gc = bool(include_gc)
        self.include_entropy = bool(include_entropy)
        self.top_kmers_only = int(top_kmers_only) if top_kmers_only else None
        self._vocabulary: list[str] | None = None
        self._kmer_to_index: dict[str, int] = {}

    def _extract_kmers(self, sequence: str, k: int) -> Counter[str]:
        """Count all k-mers in a sequence."""
        seq = sequence.upper()
        counts: Counter[str] = Counter()
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            # Skip ambiguous kmers with N
            if "N" in kmer:
                continue
            counts[kmer] += 1
        return counts

    def _compute_entropy(self, sequence: str) -> float:
        """Shannon entropy of the sequence (complexity measure)."""
        seq = sequence.upper()
        if not seq:
            return 0.0
        counts = Counter(seq)
        length = len(seq)
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / length
                entropy -= p * np.log2(p)
        return entropy

    def fit_vocabulary(self, sequences: Iterable[str]) -> "KmerFeatureExtractor":
        """Learn the global k-mer vocabulary from a corpus.

        Required if ``top_kmers_only`` is set. Should be called on
        training sequences only to avoid leakage.
        """
        if self.top_kmers_only is None:
            # Build complete vocabulary from all possible kmers
            vocab: set[str] = set()
            for k in self.k_values:
                from itertools import product

                vocab.update("".join(p) for p in product("ACGT", repeat=k))
            self._vocabulary = sorted(vocab)
            self._kmer_to_index = {kmer: i for i, kmer in enumerate(self._vocabulary)}
            return self

        global_counts: Counter[str] = Counter()
        for seq in sequences:
            for k in self.k_values:
                global_counts.update(self._extract_kmers(seq, k))

        top = global_counts.most_common(self.top_kmers_only)
        self._vocabulary = [kmer for kmer, _ in top]
        self._kmer_to_index = {kmer: i for i, kmer in enumerate(self._vocabulary)}
        _log.info(
            "K-mer vocabulary: %d unique, top %d retained",
            len(global_counts),
            len(self._vocabulary),
        )
        return self

    def transform(self, sequence: str) -> dict[str, float]:
        """Extract k-mer feature vector for a single sequence.

        Returns a flat dictionary of feature_name -> value.
        """
        features: dict[str, float] = {}

        seq_upper = sequence.upper()

        if self.include_gc:
            gc_count = seq_upper.count("G") + seq_upper.count("C")
            features["gc_content"] = gc_count / len(seq_upper) if seq_upper else 0.0

        if self.include_entropy:
            features["sequence_entropy"] = self._compute_entropy(seq_upper)

        # K-mer frequencies
        all_counts: Counter[str] = Counter()
        for k in self.k_values:
            all_counts.update(self._extract_kmers(sequence, k))

        total = sum(all_counts.values())
        if self._vocabulary is None:
            # Fit on-the-fly with all observed kmers (not recommended for production)
            _log.warning(
                "KmerFeatureExtractor vocabulary not fitted; building on-the-fly. "
                "Call fit_vocabulary() before transform() for consistent dimensions."
            )
            self._vocabulary = sorted(all_counts.keys())
            self._kmer_to_index = {kmer: i for i, kmer in enumerate(self._vocabulary)}

        vec = np.zeros(len(self._vocabulary), dtype=float)
        for kmer, count in all_counts.items():
            if kmer in self._kmer_to_index:
                vec[self._kmer_to_index[kmer]] = float(count)

        if self.normalize and total > 0:
            vec = vec / total

        for kmer, idx in self._kmer_to_index.items():
            features[f"kmer_{kmer}"] = float(vec[idx])

        return features

    def transform_fasta(
        self,
        fasta_path: Path | str,
        id_col: str = "sequence_accession",
    ) -> pd.DataFrame:
        """Extract k-mer features from all sequences in a FASTA file."""
        if not BIOPYTHON_AVAILABLE or SeqIO is None:
            raise ImportError("biopython is required for FASTA parsing")

        fasta_path = Path(fasta_path)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")

        rows: list[dict[str, object]] = []
        for record in SeqIO.parse(fasta_path, "fasta"):  # type: ignore[no-untyped-call]
            sid = record.id.strip()
            features: dict[str, object] = {}
            features.update(self.transform(str(record.seq)))
            features[id_col] = sid
            rows.append(features)

        return pd.DataFrame(rows)

    def fit_transform_fasta(
        self,
        fasta_path: Path | str,
        id_col: str = "sequence_accession",
    ) -> pd.DataFrame:
        """Fit vocabulary on the FASTA and return feature table."""
        if not BIOPYTHON_AVAILABLE or SeqIO is None:
            raise ImportError("biopython is required for FASTA parsing")

        sequences = [
            str(record.seq)
            for record in SeqIO.parse(fasta_path, "fasta")  # type: ignore[no-untyped-call]
        ]
        self.fit_vocabulary(sequences)
        return self.transform_fasta(fasta_path, id_col=id_col)
