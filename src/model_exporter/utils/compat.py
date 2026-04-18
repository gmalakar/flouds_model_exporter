# =============================================================================
# File: utils/compat.py
# Date: 2026-01-09
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

"""
Compatibility shim named `onnx_utils` to satisfy external optimizer imports.

This module exposes `extract_raw_data_from_model` (a minimal implementation
that returns an `onnx.ModelProto` when given a file path or model object) and
re-exports a few helper functions from the local `onnx_helpers.py`.

Why this exists: some third-party optimizer code imports `extract_raw_data_from_model`
from a top-level module named `onnx_utils`. The repo's internal helpers were
renamed to `onnx_helpers.py` to avoid collisions; this shim restores the
expected symbol names by delegating to the new helpers.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Optional
from typing import cast as _cast

import onnx

# Cast ONNX module to Any for attribute access that mypy stubs may not cover
_onnx = _cast(Any, onnx)

# Import helpers with multiple fallbacks to work whether this file is loaded as
# a package module (`onnx_exporter.onnx_utils`) or as a top-level module
# (`onnx_utils`) depending on how sys.path is arranged at runtime.
create_ort_session: Optional[Callable[..., Any]] = None
get_default_opset: Optional[Callable[[int], int]] = None
get_logger_func: Optional[Callable[..., Any]] = None
get_preferred_provider: Optional[Callable[[str], str]] = None


_onnx_helpers = None
for mod_name in (".helpers", "model_exporter.utils.helpers"):
    try:
        _mod = importlib.import_module(mod_name, package=__package__)
        _onnx_helpers = _mod
        break
    except Exception:
        continue

if _onnx_helpers is not None:
    create_ort_session = getattr(_onnx_helpers, "create_ort_session", None)
    get_default_opset = getattr(_onnx_helpers, "get_default_opset", None)
    get_logger_func = getattr(_onnx_helpers, "get_logger", None)
    get_preferred_provider = getattr(_onnx_helpers, "get_preferred_provider", None)


# Use safe get_logger function if available, otherwise create a minimal logger
if get_logger_func is not None:
    _logger = get_logger_func(__name__)
else:
    _logger = logging.getLogger(__name__)


def extract_raw_data_from_model(model_or_path: Any) -> Any:
    """Return an ONNX ModelProto for the given input.

    This is a minimal implementation that satisfies import and basic usage
    patterns in downstream optimizer code. If `model_or_path` is a string it
    will be interpreted as a filepath and loaded via `onnx.load`. If it is
    already an ONNX `ModelProto`, it will be returned as-is.
    """
    try:
        if isinstance(model_or_path, str):
            _logger.debug("Loading ONNX model from path: %s", model_or_path)
            return _onnx.load(model_or_path)
        # Assume it's already a ModelProto or similar object
        return model_or_path
    except Exception as e:
        _logger.exception("extract_raw_data_from_model failed: %s", e)
        raise


__all__ = [
    "extract_raw_data_from_model",
    "get_logger",
    "get_preferred_provider",
    "get_default_opset",
    "create_ort_session",
]


def has_external_data(model_or_path: Any) -> bool:
    """Return True if the given ONNX model (or filepath) references external data.

    This will first try to use `onnx.external_data_helper.has_external_data` when
    available, and fall back to a conservative manual inspection of initializers.
    """
    try:
        # Load model if a path was provided
        model = None
        if isinstance(model_or_path, str):
            import os

            if not os.path.exists(model_or_path):
                return False
            model = _onnx.load(model_or_path)
        else:
            model = model_or_path

        # Prefer onnx helper if present
        try:
            _ext = getattr(_onnx, "external_data_helper", None)
            if _ext and callable(getattr(_ext, "has_external_data", None)):
                return _ext.has_external_data(model)
        except Exception:
            pass

        # Fallback: inspect initializers for external data markers
        for init in getattr(model.graph, "initializer", []):
            # TensorProto.data_location == TensorProto.EXTERNAL indicates external data
            if getattr(init, "data_location", None) == getattr(_onnx.TensorProto, "EXTERNAL", 1):
                return True
            # external_data repeated field present
            if getattr(init, "external_data", None):
                return True

        return False
    except Exception:
        _logger.exception("has_external_data check failed")
        return False


__all__.append("has_external_data")
# The helper functions `get_logger`, `get_preferred_provider`, `get_default_opset`,
# and `create_ort_session` are imported above from `onnx_helpers` to avoid duplication.


# Provide a `get_logger` symbol that mirrors `onnx_helpers.get_logger` when
# available, or fall back to a minimal factory that returns stdlib loggers.
if get_logger_func is not None:
    get_logger = get_logger_func
else:

    def get_logger(name: str) -> Any:
        return logging.getLogger(name)
