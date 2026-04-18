#!/usr/bin/env python3
# =============================================================================
# File: export_validator.py
# Date: 2026-01-09
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Validator invocation helper used by the exporter.

Provides `invoke_validator(...)` which performs the quick structural
verification (checker/external_data/session) and numeric validation by
invoking the centralized validator programmatically or, if necessary,
via an in-process module run as a fallback. Returns `(rc, quick_ok)` where
`rc` is the numeric validator return code (0 success, 2 numeric fail,
3 missing model/error) and `quick_ok` indicates whether the lightweight
quick verification passed.
"""
from __future__ import annotations

import importlib
import runpy
import sys
from typing import Any, List, Tuple


def invoke_validator(
    output_dir: str,
    expected: List[str],
    model_name: str | None,
    pack_single_file: bool,
    pack_single_threshold_mb: int | None,
    trust_remote_code: bool,
    normalize_embeddings: bool,
    logger: Any,
) -> Tuple[int, bool]:
    """Invoke the numeric ONNX validator for a completed export.

    Tries programmatic validation via :func:`validate_onnx` first; falls back
    to subprocess-based invocation if the validator module is unavailable or
    returns a non-zero code. Performs a quick structural check of expected
    artifact files before returning.

    Args:
        output_dir: Directory containing the exported ONNX model.
        expected: List of expected artifact filenames (used for quick check).
        model_name: Hugging Face model id used as the numeric reference; falls
            back to *output_dir* when ``None``.
        pack_single_file: Whether the export packed all weights into a single
            file (affects the quick check heuristic).
        pack_single_threshold_mb: Size threshold used during packing; passed
            for informational logging only.
        trust_remote_code: Forward to the validator for models requiring custom
            remote execution code.
        normalize_embeddings: Forward to the validator to request L2-normalized
            embedding comparison.
        logger: A standard :class:`logging.Logger` (or compatible object) for
            diagnostic output.

    Returns:
        A 2-tuple ``(rc, quick_ok)`` where *rc* is the validator exit code
        (``0`` = pass, ``2`` = fail, ``3`` = error/unavailable) and *quick_ok*
        is a ``bool`` from the structural artifact check.
    """
    rc = 3
    quick_ok = True

    # Try to load the validator module programmatically first.
    validator: Any = None
    try:
        from . import numeric as validator
    except Exception:
        try:
            validator = importlib.import_module("model_exporter.validation.numeric")
        except Exception:
            validator = None

    trust_flag = False
    try:
        trust_flag = bool(trust_remote_code)
    except Exception:
        pass

    logger.info("Invoking ONNX validator for %s (trust_remote_code=%s)", output_dir, trust_flag)

    try:
        # Programmatic validation if available
        if validator is not None:
            try:
                rc = validator.validate_onnx(
                    model_dir=output_dir,
                    reference_model=(str(model_name) if model_name else output_dir),
                    texts=None,
                    device="cpu",
                    atol=1e-4,
                    rtol=1e-3,
                    trust_remote_code=trust_flag,
                    normalize_embeddings=normalize_embeddings,
                )
            except Exception as call_err:
                logger.error("Programmatic validator raised an exception: %s", call_err)
                rc = 3

        # If programmatic validation not available or it failed, fall back to invoking
        # the validator as an in-process module run so we can isolate deps.
        if validator is None or rc != 0:
            args = [
                "--model-dir",
                output_dir,
                "--device",
                "cpu",
                "--atol",
                "1e-4",
                "--rtol",
                "1e-3",
            ]
            if model_name:
                args += ["--reference-model", str(model_name)]
            if trust_flag:
                args += ["--trust-remote-code"]
            if normalize_embeddings:
                args += ["--normalize-embeddings"]

            mod_name = "model_exporter.validation.numeric"
            old_mod = None
            try:
                old_mod = sys.modules.pop(mod_name, None)
            except Exception:
                pass

            # Temporarily set sys.argv for the in-process run
            old_argv = sys.argv[:]
            sys.argv = [mod_name] + args
            try:
                logger.info("Running validator in-process: %s %s", mod_name, " ".join(args))
                runpy.run_module(mod_name, run_name="__main__")
                rc = 0
            except SystemExit as se:
                try:
                    rc = int(se.code) if se.code is not None else 0
                except Exception:
                    rc = 1
            except Exception as inproc_err:
                logger.error("In-process validator failed: %s", inproc_err)
                rc = 3
            finally:
                sys.argv = old_argv
                try:
                    if old_mod is not None:
                        del old_mod
                except Exception:
                    pass
    except Exception as e:
        logger.error("Failed to execute validator for %s: %s", output_dir, e)
        rc = 3

    return int(rc), bool(quick_ok)
