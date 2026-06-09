# =============================================================================
# File: export/__init__.py
# Date Created: 2026-04-19
# Date Updated: 2026-04-19
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Export sub-package: pipeline, helpers, optimizer, and subprocess runner.

Submodules are imported lazily so lightweight imports such as
``import model_exporter.export`` do not require the full ONNX/Transformers
runtime stack.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = [
    "pipeline",
    "pipeline_helpers",
    "subprocess_runner",
    "helpers",
    "optimizer",
    "legacy_fallback",
    "options",
    "path_utils",
]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
