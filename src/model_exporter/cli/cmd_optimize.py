# =============================================================================
# File: cli/cmd_optimize.py
# Date: 2026-01-10
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Optimize subcommand: argument definitions and _run_optimize."""

from __future__ import annotations

import importlib
import logging


def _add_optimize_arguments(parser):
    """Register all optimize-specific CLI arguments onto *parser*.

    Adds arguments for model directory, model purpose, optimization level,
    and the portable flag.

    Args:
        parser: An :class:`argparse.ArgumentParser` instance to populate.
    """
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the exported ONNX model directory to optimize",
    )
    parser.add_argument(
        "--model-for",
        required=True,
        choices=["fe", "s2s", "sc", "llm", "ranker"],
        help="Model purpose used to decide which artifacts are eligible for optimization",
    )
    parser.add_argument(
        "--optimization-level",
        dest="optimization_level",
        type=int,
        default=99,
        help="ONNX optimization level (default: 99)",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Prefer conservative/portable ONNX optimizations (avoid hardware-specific passes)",
    )


def _run_optimize(args, parser):
    """Load the optimizer module and run encoder-model ONNX optimization.

    Sets up a logger, then calls :func:`optimize_if_encoder` with the
    parameters resolved from *args*.

    Args:
        args: Parsed :class:`argparse.Namespace` from the optimize sub-command.
        parser: The :class:`argparse.ArgumentParser` (unused; kept for API
            symmetry with other sub-command runners).

    Returns:
        An integer exit code (``0`` = success, non-zero = failure).
    """
    optimizer_module = importlib.import_module("model_exporter.export.optimizer")
    optimize_if_encoder = optimizer_module.optimize_if_encoder

    logger_name = "model_exporter.optimize"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    return int(
        optimize_if_encoder(
            args.model_dir,
            args.model_for,
            logger,
            args.optimization_level,
            portable=args.portable,
        )
    )
