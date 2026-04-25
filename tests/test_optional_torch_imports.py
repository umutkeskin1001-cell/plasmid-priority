from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

_TARGET_MODULES = (
    "plasmid_priority.modeling.ft_transformer",
    "plasmid_priority.modeling.multi_task",
)


@contextmanager
def _block_torch_imports() -> Iterator[None]:
    original_import = builtins.__import__

    def _guarded_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return original_import(name, globals_, locals_, fromlist, level)

    builtins.__import__ = _guarded_import
    try:
        yield
    finally:
        builtins.__import__ = original_import


def _import_module_without_torch(module_name: str):
    sys.modules.pop(module_name, None)
    with _block_torch_imports():
        return importlib.import_module(module_name)


@pytest.fixture(autouse=True)
def _cleanup_target_modules() -> Iterator[None]:
    yield
    for module_name in _TARGET_MODULES:
        sys.modules.pop(module_name, None)


def test_ft_transformer_import_is_safe_without_torch() -> None:
    module = _import_module_without_torch("plasmid_priority.modeling.ft_transformer")

    assert module.TORCH_AVAILABLE is False
    assert module.nn is None
    assert module.torch is None
    assert hasattr(module, "FTTransformer")
    assert hasattr(module, "FTTransformerClassifier")


def test_multi_task_import_is_safe_without_torch() -> None:
    module = _import_module_without_torch("plasmid_priority.modeling.multi_task")

    assert module.TORCH_AVAILABLE is False
    assert module.nn is None
    assert module.torch is None
    assert hasattr(module, "MultiTaskPlasmidNet")
    assert hasattr(module, "MultiTaskTrainer")


def test_ft_transformer_usage_boundaries_raise_actionable_import_error() -> None:
    module = _import_module_without_torch("plasmid_priority.modeling.ft_transformer")

    with pytest.raises(ImportError, match="pip install torch"):
        module.FTTransformerClassifier()

    classifier = module.FTTransformerClassifier.__new__(module.FTTransformerClassifier)
    with pytest.raises(ImportError, match="pip install torch"):
        classifier.fit(np.zeros((2, 3), dtype=np.float32), np.zeros(2, dtype=np.float32))

    classifier._is_fitted = True
    classifier._model = object()
    with pytest.raises(ImportError, match="pip install torch"):
        classifier.predict_proba(np.zeros((2, 3), dtype=np.float32))


def test_multi_task_usage_boundaries_raise_actionable_import_error() -> None:
    module = _import_module_without_torch("plasmid_priority.modeling.multi_task")

    with pytest.raises(ImportError, match="pip install torch"):
        module.MultiTaskTrainer(input_dim=3)

    trainer = module.MultiTaskTrainer.__new__(module.MultiTaskTrainer)
    with pytest.raises(ImportError, match="pip install torch"):
        trainer.fit(
            np.zeros((2, 3), dtype=np.float32),
            {
                "geo_spread": np.zeros(2, dtype=np.float32),
                "bio_transfer": np.zeros(2, dtype=np.float32),
                "clinical_hazard": np.zeros(2, dtype=np.float32),
            },
        )

    trainer._is_fitted = True
    trainer._model = object()
    with pytest.raises(ImportError, match="pip install torch"):
        trainer.predict_proba(np.zeros((2, 3), dtype=np.float32))
