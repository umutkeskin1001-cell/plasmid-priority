"""Optimization utilities for PyTorch on CPU-only environments."""

from __future__ import annotations

import os
from typing import Any

torch: Any | None = None
try:
    import torch as _torch_mod
except ImportError:
    pass
else:
    torch = _torch_mod


def configure_torch_cpu() -> None:
    """Optimize PyTorch for CPU-only inference.

    Call this at the start of any script that uses Deep Learning models.
    Prevents oversubscription and enables hardware-specific optimizations.
    """
    torch_mod = torch
    if torch_mod is None:
        return

    # Use roughly half of the physical cores for DL inference
    # This prevents DL from starving other parallel pipeline steps
    n_threads = max(1, (os.cpu_count() or 4) // 2)
    torch_mod.set_num_threads(n_threads)
    torch_mod.set_num_interop_threads(1)

    # Enable Intel MKL-DNN (OneDNN) optimization if available
    mkldnn_backend = getattr(torch_mod.backends, "mkldnn", None)
    if mkldnn_backend is not None:
        setattr(mkldnn_backend, "enabled", True)

    # Optimization for repeated inference
    jit_mod = getattr(torch_mod, "jit", None)
    if jit_mod is not None and hasattr(jit_mod, "set_fusion_strategy"):
        jit_mod.set_fusion_strategy([("STATIC", 2)])


def export_to_torchscript(
    model: Any, example_input: Any, path: str | os.PathLike[str]
) -> None:
    """Export a model to TorchScript for faster CPU inference."""
    torch_mod = torch
    if torch_mod is None:
        return

    model.eval()
    with torch_mod.no_grad():
        scripted = torch_mod.jit.trace(model, example_input)
        scripted.save(str(path))


def quantize_model_dynamic(model: Any) -> Any:
    """Apply dynamic INT8 quantization to a model for 2-4x CPU speedup."""
    torch_mod = torch
    if torch_mod is None:
        return model

    quantize_dynamic = None
    ao_mod = getattr(torch_mod, "ao", None)
    if ao_mod is not None:
        ao_quantization = getattr(ao_mod, "quantization", None)
        quantize_dynamic = getattr(ao_quantization, "quantize_dynamic", None)
    if quantize_dynamic is None:
        legacy_quantization = getattr(torch_mod, "quantization", None)
        quantize_dynamic = getattr(legacy_quantization, "quantize_dynamic", None)

    if not callable(quantize_dynamic):
        return model

    # Quantize linear layers (typical bottleneck in transformers/MLPs)
    return quantize_dynamic(model, {torch_mod.nn.Linear}, dtype=torch_mod.qint8)
