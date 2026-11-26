#!/usr/bin/env python3
# =============================================================================
# File: export_validator.py
# Date: 2026-01-09
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Validator invocation helper used by the exporter.

Provides `invoke_validator(...)`, which runs the numeric validation by calling
the centralized validator programmatically or, if necessary, through an in-process
module run. Returns the validator's return code (0 success, 2 numeric fail,
3 missing model/error).
"""
from __future__ import annotations

import importlib
import runpy
import sys
from typing import Any


def invoke_validator(
    output_dir: str,
    model_name: str | None,
    trust_remote_code: bool,
    normalize_embeddings: bool,
    logger: Any,
) -> int:
    """Invoke the numeric ONNX validator for a completed export.

    Tries programmatic validation via :func:`validate_onnx` first; falls back
    to an in-process module run if the validator module is unavailable or
    returns a non-zero code.

    Args:
        output_dir: Directory containing the exported ONNX model.
        trust_remote_code: Forward to the validator for models requiring custom
            remote execution code.
        normalize_embeddings: Forward to the validator to request L2-normalized
            embedding comparison.
        logger: A standard :class:`logging.Logger` (or compatible object) for
            diagnostic output.

    Returns:
        The validator exit code (``0`` = pass, ``2`` = fail, ``3`` = error/unavailable).
    """
    rc = 3

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

    return int(rc)
