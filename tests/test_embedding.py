"""Tests for k-mer baseline features (DNA LM removed — see docs/macos_compatibility.md)."""

from __future__ import annotations

from plasmid_priority.embedding.kmer_baseline import KmerFeatureExtractor


class TestKmerFeatureExtractor:
    def test_basic_4mer(self) -> None:
        extractor = KmerFeatureExtractor(k_values=(4,), normalize=True)
        seq = "ACGTACGTACGT"
        features = extractor.transform(seq)
        assert "gc_content" in features
        assert "sequence_entropy" in features
        assert any(k.startswith("kmer_") for k in features)
        # GC content of ACGT = 50%
        assert abs(features["gc_content"] - 0.5) < 0.01

    def test_empty_sequence(self) -> None:
        extractor = KmerFeatureExtractor(k_values=(4,))
        features = extractor.transform("")
        assert features["gc_content"] == 0.0
        assert features["sequence_entropy"] == 0.0

    def test_vocabulary_fit(self) -> None:
        extractor = KmerFeatureExtractor(
            k_values=(2,),
            top_kmers_only=4,
            normalize=True,
        )
        sequences = ["ACACAC", "GTGTGT", "ACGTAC"]
        extractor.fit_vocabulary(sequences)
        assert extractor._vocabulary is not None
        assert len(extractor._vocabulary) == 4

        features = extractor.transform("ACACAC")
        for kmer in extractor._vocabulary:
            assert f"kmer_{kmer}" in features

    def test_all_kmers_without_topk(self) -> None:
        extractor = KmerFeatureExtractor(k_values=(2,), top_kmers_only=None)
        extractor.fit_vocabulary(["ACGT"])
        # 2-mers from ACGT = AC, CG, GT = 3 observed; but vocab = all 16 possible
        assert len(extractor._vocabulary) == 16
