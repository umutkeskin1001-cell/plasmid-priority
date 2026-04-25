from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from plasmid_priority.modeling.fold_cache import FoldCache, UniversalModelCache
from plasmid_priority.modeling.matrix_cache import MatrixCache
from plasmid_priority.modeling.oof_cache import OOFCache


def test_fold_cache_get_or_create_reuses_cached_folds() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = FoldCache(Path(tmp_dir))
        y = np.asarray([0, 1, 0, 1, 0, 1], dtype=int)
        folds = [
            (np.asarray([0, 1, 2, 3], dtype=int), np.asarray([4, 5], dtype=int)),
        ]
        key, loaded, status = cache.get_or_create(
            y=y,
            split_strategy="repeated_stratified",
            n_splits=2,
            n_repeats=1,
            seed=42,
            factory=lambda: folds,
        )
        assert status == "miss"
        assert len(loaded) == 1
        _, loaded2, status2 = cache.get_or_create(
            y=y,
            split_strategy="repeated_stratified",
            n_splits=2,
            n_repeats=1,
            seed=42,
            factory=lambda: [],
        )
        assert status2 == "hit"
        assert len(loaded2) == 1
        assert key


def test_matrix_cache_build_key_is_stable_for_same_inputs() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = MatrixCache(Path(tmp_dir))
        scored = pd.DataFrame(
            {
                "backbone_id": ["a", "b", "c"],
                "spread_label": [1, 0, 1],
                "x1": [0.1, 0.2, 0.3],
                "x2": [1.0, 2.0, 3.0],
            },
        )
        key1, payload1 = cache.build_key(
            scored=scored,
            columns=["x1", "x2"],
            preprocessing_config={"l2": 1.0},
        )
        key2, payload2 = cache.build_key(
            scored=scored,
            columns=["x1", "x2"],
            preprocessing_config={"l2": 1.0},
        )
        assert key1 == key2
        assert payload1["n_rows"] == payload2["n_rows"] == 3


def test_oof_cache_roundtrip_metadata_and_predictions() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = OOFCache(Path(tmp_dir))
        key, key_payload = cache.build_key(
            matrix_key="mx",
            fold_key="fd",
            model_name="demo",
            model_config={"l2": 1.0},
            model_code_hash="abc",
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": ["a", "b"],
                "oof_prediction": [0.7, 0.2],
                "spread_label": [1, 0],
            },
        )
        cache.save(
            key,
            key_payload=key_payload,
            model_name="demo",
            metrics={"roc_auc": 0.9},
            predictions=predictions,
            status="ok",
            error_message=None,
        )
        loaded = cache.load(key)
        assert loaded is not None
        metadata = loaded["metadata"]
        frame = loaded["predictions"]
        assert metadata["model_name"] == "demo"
        assert int(metadata["n_predictions"]) == 2
        assert list(frame["backbone_id"]) == ["a", "b"]


def test_universal_model_cache_key_changes_with_fit_config(tmp_path: Path) -> None:
    cache = UniversalModelCache(tmp_path)

    key_a = cache._get_key(
        "governance_linear",
        ("T_eff_norm",),
        0,
        "datahash",
        fit_config_hash="alpha1",
        protocol_hash="protocol",
        software_hash="software",
    )
    key_b = cache._get_key(
        "governance_linear",
        ("T_eff_norm",),
        0,
        "datahash",
        fit_config_hash="alpha2",
        protocol_hash="protocol",
        software_hash="software",
    )

    assert key_a != key_b


def test_universal_model_cache_key_preserves_feature_order(tmp_path: Path) -> None:
    cache = UniversalModelCache(tmp_path)

    key_a = cache._get_key(
        "governance_linear",
        ("a", "b"),
        0,
        "datahash",
        fit_config_hash="fit",
        protocol_hash="protocol",
        software_hash="software",
    )
    key_b = cache._get_key(
        "governance_linear",
        ("b", "a"),
        0,
        "datahash",
        fit_config_hash="fit",
        protocol_hash="protocol",
        software_hash="software",
    )

    assert key_a != key_b
