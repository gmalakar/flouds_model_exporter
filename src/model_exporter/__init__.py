# =============================================================================
# File: __init__.py
# Date Created: 2026-04-19
# Date Updated: 2026-04-19
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Flouds model exporter - ONNX export, validation and optimization toolkit.

This module provides lazy re-exports for the public API so callers can do::

    from model_exporter import export
    from model_exporter import validate_onnx

The actual implementation is imported lazily via :func:`__getattr__` to avoid
import-time side effects that can interfere with subpackage imports during
test-time module resolution.
"""

from typing import Any

__all__ = ["ExportConfig", "export", "export_from_config", "validate_onnx"]


def __getattr__(name: str) -> Any:
    if name == "export":
        from .export.pipeline import export as _export

        return _export
    if name == "export_from_config":
        from .export.pipeline import export_from_config as _export_from_config

        return _export_from_config
    if name == "ExportConfig":
        from .export.options import ExportConfig as _export_config

        return _export_config
    if name == "validate_onnx":
        from .validation.numeric import validate_onnx as _validate

        return _validate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
