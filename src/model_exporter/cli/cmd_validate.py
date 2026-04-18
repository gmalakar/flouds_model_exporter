# =============================================================================
# File: cli/cmd_validate.py
# Date: 2026-01-10
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Validate subcommand: argument definitions and _run_validate."""

from __future__ import annotations

import importlib


def _add_validate_arguments(parser):
    """Register all validate-specific CLI arguments onto *parser*.

    Adds arguments for model directory, reference model, example texts,
    device, trust-remote-code, tolerances, embedding normalization, and
    the skip-diagnostics flag.

    Args:
        parser: An :class:`argparse.ArgumentParser` instance to populate.
    """
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the exported ONNX model directory",
    )
    parser.add_argument(
        "--reference-model",
        required=True,
        help="Hugging Face model id to use as reference",
    )
    parser.add_argument(
        "--texts",
        nargs="*",
        default=None,
        help="Optional example texts to run through the validator",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device for the reference PyTorch model (default: cpu)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow executing custom model code from the HF repo (use with caution)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance threshold for max diff (default: 1e-4)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for diffs (default: 1e-3)",
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="L2-normalize sentence embeddings before comparison",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip diagnostic dump collection on validation failure",
    )


def _run_validate(args, parser):
    """Translate parsed *args* into a validate argv list and invoke the validator.

    Dynamically imports :mod:`model_exporter.validation.numeric` to
    avoid heavy import-time dependencies, builds an argv list from *args*,
    and calls :func:`numeric.main`.

    Args:
        args: Parsed :class:`argparse.Namespace` from the validate sub-command.
        parser: The :class:`argparse.ArgumentParser` (unused; kept for API
            symmetry with other sub-command runners).

    Returns:
        An integer exit code (``0`` = pass, ``2`` = fail, ``3`` = error).
    """
    validator = importlib.import_module("model_exporter.validation.numeric")
    validate_argv = [
        "--model-dir",
        args.model_dir,
        "--reference-model",
        args.reference_model,
        "--device",
        args.device,
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
    ]

    if args.texts:
        validate_argv.append("--texts")
        validate_argv.extend(args.texts)
    if args.trust_remote_code:
        validate_argv.append("--trust-remote-code")
    if args.normalize_embeddings:
        validate_argv.append("--normalize-embeddings")
    if args.skip_diagnostics:
        validate_argv.append("--skip-diagnostics")

    result = validator.main(validate_argv)
    if result is None:
        return 0
    return int(result)
