# =============================================================================
# File: export/optimum_compat.py
# Date Created: 2026-11-26
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

"""Make optimum's ONNX export configs work on Python 3.14.

Python 3.14 made ``functools.partial`` a method descriptor: a partial stored as a
class attribute is now bound to the instance like a function. optimum-onnx defines
many export configs as ``NORMALIZED_CONFIG_CLASS = NormalizedXConfig.with_args(...)``
(a partial), so ``self.NORMALIZED_CONFIG_CLASS(config)`` also passes ``self`` and
fails with "got multiple values for argument 'allow_new'" (T5, GPT-2, BART, ...).

The fix upstream (huggingface/optimum PR #2496) makes ``with_args`` return a
``staticmethod``; this module applies the same change until a release ships it.
Remove it once the minimum optimum version includes that fix.
"""

from __future__ import annotations

import functools
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED_FLAG = "_flouds_py314_with_args"


def _iter_subclasses(base: type) -> list[type]:
    seen: list[type] = []
    stack: list[type] = [base]
    while stack:
        cls = stack.pop()
        subclasses: list[type] = cls.__subclasses__()
        for sub in subclasses:
            if sub not in seen:
                seen.append(sub)
                stack.append(sub)
    return seen


def patch_partial_config_attributes() -> int:
    """Apply the Python 3.14 fix to optimum. Returns how many class attributes were wrapped.

    Safe to call repeatedly and before or after optimum's export configs are imported:
    ``with_args`` returns a ``staticmethod`` for configs defined later, and partials
    already stored on loaded config classes are wrapped. Does nothing before Python 3.14
    or when optimum is not installed.
    """
    if sys.version_info < (3, 14):
        return 0
    try:
        from optimum.utils.normalized_config import NormalizedConfig
    except ImportError:
        return 0

    if not getattr(NormalizedConfig, _PATCHED_FLAG, False):

        def with_args(cls: type, allow_new: bool = False, **kwargs: Any) -> Any:
            # A staticmethod is callable directly too, so other uses of with_args keep working.
            return staticmethod(functools.partial(cls, allow_new=allow_new, **kwargs))

        NormalizedConfig.with_args = classmethod(with_args)
        setattr(NormalizedConfig, _PATCHED_FLAG, True)

    wrapped = 0
    try:
        from optimum.exporters.base import ExportConfig, ExporterConfig

        bases: tuple[type, ...] = (ExportConfig, ExporterConfig)
    except ImportError:
        bases = ()
    for base in bases:
        for cls in _iter_subclasses(base):
            for name, value in list(vars(cls).items()):
                if isinstance(value, functools.partial):
                    setattr(cls, name, staticmethod(value))
                    wrapped += 1
    if wrapped:
        logger.debug("Wrapped %d partial config attribute(s) in staticmethod for Python 3.14", wrapped)
    return wrapped
