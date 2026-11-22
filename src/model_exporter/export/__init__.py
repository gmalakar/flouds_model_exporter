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

import sys
from importlib import import_module
from types import ModuleType
from typing import Any

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


class _CallableExportPackage(ModuleType):
    """Lets ``from model_exporter import export`` be called.

    The public function and this sub-package share the name ``export``. Once the
    sub-package is imported, ``model_exporter.export`` is the package, so the documented
    ``from model_exporter import export; export(...)`` used to fail with "module object
    is not callable". Calling the package now runs the export function, and
    ``model_exporter.export.pipeline`` and the other submodules keep working.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        from model_exporter.export.pipeline import export as _export

        return _export(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableExportPackage
