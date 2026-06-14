# =============================================================================
# File: cli/main.py
# Date Created: 2026-01-10
# Date Updated: 2026-04-19
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

# HINTS:
# - `--model-for` supplies default task/library values: fe -> feature-extraction,
#   s2s -> seq2seq-lm, ranker -> sequence-classification,
#   llm -> text-generation-with-past.
# - Supply `--task` only when you need an explicit task override, such as
#   text2text-generation-with-past for seq2seq KV-cache behavior.
# - After exporting a seq2seq model, you should see encoder_model.onnx, decoder_model.onnx, and decoder_with_past_model.onnx in the output directory.
# - If decoder_with_past_model.onnx is missing, the model cannot be used for fast autoregressive generation (greedy decoding).
# - Always verify ONNX model inputs/outputs after export to ensure compatibility with your inference pipeline.
# - Optimization is optional but recommended for production; it can reduce inference latency.
# - If you encounter export errors, check that your optimum/transformers/onnxruntime versions are compatible.
# - For ONNX summarization inference, always start decoder_input_ids with eos_token_id for BART or pad_token_id for T5.
# - Use optimum.onnxruntime pipelines for easy ONNX inference testing after export.

import argparse
import warnings

from model_exporter.cli.cmd_batch import _add_batch_arguments, _run_batch
from model_exporter.cli.cmd_export import _add_export_arguments, _run_export
from model_exporter.cli.cmd_optimize import _add_optimize_arguments, _run_optimize
from model_exporter.cli.cmd_validate import _add_validate_arguments, _run_validate

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants.*")


def _build_root_parser():
    """Build and return the root argument parser with all sub-commands registered.

    Registers ``export``, ``validate``, ``optimize``, and ``batch``
    sub-parsers, each with their respective argument sets.

    Returns:
        A configured :class:`argparse.ArgumentParser` ready for parsing.
    """
    parser = argparse.ArgumentParser(description="Flouds model export CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export and optimize ONNX model")
    _add_export_arguments(export_parser)
    export_parser.set_defaults(func=lambda args: _run_export(args, export_parser))

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate exported ONNX models against a reference model",
    )
    _add_validate_arguments(validate_parser)
    validate_parser.set_defaults(func=lambda args: _run_validate(args, validate_parser))

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Run optimization on existing exported ONNX models",
    )
    _add_optimize_arguments(optimize_parser)
    optimize_parser.set_defaults(func=lambda args: _run_optimize(args, optimize_parser))

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run batch export workflows",
    )
    _add_batch_arguments(batch_parser)
    batch_parser.set_defaults(func=lambda args: _run_batch(args, batch_parser))
    return parser


def main(argv=None):
    """Entry point for the ``flouds-model-exporter`` CLI.

    Supports explicit sub-commands: ``export``, ``validate``, ``optimize``,
    and ``batch``.

    Args:
        argv: Optional sequence of command-line strings. Defaults to
            :data:`sys.argv[1:]` when ``None``.

    Returns:
        An integer exit code returned by the chosen sub-command handler.
    """
    root_parser = _build_root_parser()
    args = root_parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
