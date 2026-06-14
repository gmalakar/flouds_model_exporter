# =============================================================================
# File: cli/cmd_export.py
# Date Created: 2026-04-19
# Date Updated: 2026-04-19
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Export subcommand: argument definitions, helpers, and _run_export."""

from __future__ import annotations

import inspect
import os

from model_exporter.export.options import MODEL_FOR_DEFAULTS, ExportConfig
from model_exporter.export.pipeline import export as export_unified


def _add_export_arguments(parser):
    """Register all export-specific CLI arguments onto *parser*.

    Adds arguments for model name, model purpose, task, output path,
    optimization flags, quantization, validator settings, device, token,
    and a range of advanced flags.

    Args:
        parser: An :class:`argparse.ArgumentParser` instance to populate.
    """
    parser.add_argument(
        "--log-to-file",
        dest="log_to_file",
        action="store_true",
        default=False,
        help="Log to file and print log file path in terminal (default: False).",
    )
    parser.add_argument(
        "--model-name",
        dest="model_name",
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--model-for",
        dest="model_for",
        type=str,
        default="fe",
        help=(
            "Model purpose: 'fe' (feature-extraction), 's2s' (seq2seq-lm),"
            " 'ranker' (cross-encoder/ranking), or 'llm' (causal-lm). "
            "Defaults task/library based on this value. (default: fe)"
        ),
    )
    parser.add_argument("--optimize", action="store_true", help="Whether to optimize the ONNX model")
    parser.add_argument(
        "--optimization-level",
        dest="optimization_level",
        type=int,
        choices=[0, 1, 2, 99],
        default=None,
        help="ONNX optimization level. Choices: 0, 1, 2, 99. Only valid with --optimize (default when optimizing: 99).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Export task. Defaults from --model-for when omitted.",
    )
    parser.add_argument("--model-folder", dest="model_folder", help="HuggingFace model folder or path")
    parser.add_argument(
        "--onnx-path",
        dest="onnx_path",
        default=None,
        help=("Path to ONNX output directory. " "Defaults to the ONNX_PATH environment variable, or 'onnx' under the current working directory."),
    )
    parser.add_argument(
        "--framework",
        type=str,
        default=None,
        help="Framework to use for ONNX export (e.g., pt, tf).",
    )
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Allow executing custom code from model repos that require it (use with caution)",
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="Request the validator to L2-normalize sentence embeddings before comparison",
    )
    parser.add_argument(
        "--require-validator",
        action="store_true",
        help="Require the consolidated validator to pass; fail export if validation fails.",
    )
    parser.add_argument(
        "--skip-validator",
        action="store_true",
        help="Skip numeric ONNX validation (do not run validate_onnx_model).",
    )
    parser.add_argument("--force", action="store_true", help="Force re-export even if ONNX files exist")
    parser.add_argument(
        "--opset-version",
        dest="opset_version",
        type=int,
        default=None,
        help="ONNX opset version to use for export.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export (default: cpu). Use 'cuda' to force GPU export.",
    )
    parser.add_argument(
        "--quantize",
        dest="quantize",
        type=str,
        default=None,
        choices=["dynamic_int8", "fp16", "both"],
        help="Optional quantization to produce post-export variants. Choices: 'dynamic_int8', 'fp16', or 'both'.",
    )
    parser.add_argument(
        "--pack-single-file",
        dest="pack_single_file",
        action="store_true",
        help="If exported ONNX uses external_data, repack into a single-file model.",
    )
    parser.add_argument(
        "--use-external-data-format",
        dest="use_external_data_format",
        action="store_true",
        default=False,
        help="Enable external data format; prefer single-file ONNX when possible.",
    )
    parser.add_argument(
        "--no-local-prep",
        action="store_true",
        help="Skip creating a prepared local copy (temp_local) for LLMs before export",
    )
    parser.add_argument(
        "--huggingface_hub_token",
        dest="huggingface_hub_token",
        type=str,
        default=None,
        help="Hugging Face access token. Defaults to HUGGINGFACE_HUB_TOKEN when omitted.",
    )
    parser.add_argument(
        "--library",
        dest="library",
        type=str,
        default=None,
        required=False,
        help=(
            "Export library hint (e.g., 'sentence_transformers' or 'transformers'). "
            "Defaults from --model-for when omitted."
        ),
    )
    parser.add_argument(
        "--merge",
        dest="merge",
        action="store_true",
        help=(
            "Request model merging where applicable. Merge is only applicable to "
            "decoder-only causal LLMs that support text-generation-with-past (KV-cache)."
        ),
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="When set, remove extraneous ONNX files after optimization following prioritized cleanup rules",
    )
    parser.add_argument(
        "--prune-canonical",
        dest="prune_canonical",
        action="store_true",
        help="When set, remove canonical ONNX files (e.g., decoder_model.onnx) if merged artifacts exist",
    )
    parser.add_argument(
        "--no-post-process",
        dest="no_post_process",
        action="store_true",
        help="Skip optimum post-processing steps (deduplication). Useful to avoid MemoryError during large-model post-processing",
    )
    parser.add_argument(
        "--portable",
        dest="portable",
        action="store_true",
        help="Prefer conservative/portable ONNX optimizations (avoid hardware-specific passes)",
    )
    parser.add_argument(
        "--use-sub-process",
        dest="use_subprocess",
        action="store_true",
        help=("Force running exporter in a subprocess. " "By default the exporter will try in-process first; set this flag to force subprocess use."),
    )
    parser.add_argument(
        "--use-fallback-if-failed",
        dest="use_fallback_if_failed",
        action="store_true",
        help="Enable legacy fallback exporter only when the primary export path fails.",
    )
    parser.add_argument(
        "--min-free-memory-gb",
        dest="min_free_memory_gb",
        type=float,
        default=None,
        help="Minimum free RAM in GB required before export. Below this threshold, conservative low-memory export flags are applied.",
    )
    parser.add_argument(
        "--require-sufficient-memory",
        dest="require_sufficient_memory",
        action="store_true",
        help="Fail export when --min-free-memory-gb is not satisfied instead of continuing with low-memory flags.",
    )


def _build_export_parser(add_help=True, description="Export and optimize ONNX model."):
    """Build and return a standalone parser for the ``export`` sub-command.

    Args:
        add_help: Whether to add a ``-h``/``--help`` flag (default: ``True``).
        description: Description string shown in the parser's help text.

    Returns:
        A configured :class:`argparse.ArgumentParser` for the export command.
    """
    import argparse

    parser = argparse.ArgumentParser(description=description, add_help=add_help)
    _add_export_arguments(parser)
    return parser


def _validate_export_args(args, parser):
    """Reject option combinations that cannot affect the requested export."""
    if not args.optimize and args.optimization_level is not None:
        parser.error("--optimization-level requires --optimize.")
    model_for = (args.model_for or "").lower()
    if model_for not in MODEL_FOR_DEFAULTS:
        expected = ", ".join(MODEL_FOR_DEFAULTS)
        parser.error(f"Unknown --model-for value: {args.model_for!r}. Expected one of: {expected}.")
    if args.portable and not args.optimize:
        parser.error("--portable requires --optimize.")
    if args.prune_canonical and not args.cleanup:
        parser.error("--prune-canonical requires --cleanup.")
    if args.skip_validator and args.require_validator:
        parser.error("--skip-validator cannot be combined with --require-validator.")
    if args.skip_validator and args.normalize_embeddings:
        parser.error("--normalize-embeddings requires validator execution; remove --skip-validator.")
    if args.require_sufficient_memory and args.min_free_memory_gb is None:
        parser.error("--require-sufficient-memory requires --min-free-memory-gb.")
    if args.model_for != "llm" and args.no_local_prep:
        parser.error("--no-local-prep is only valid with --model-for llm.")
    if args.model_for != "llm" and args.merge:
        parser.error("--merge is only valid with --model-for llm.")


def _execute_export_kwargs(unified_kwargs, parser):
    """Validate *unified_kwargs* against the exporter signature and call it.

    Checks that every key in *unified_kwargs* is a known parameter of
    :func:`export_unified`. Normalises the ``quantize`` value, then invokes
    the exporter.

    Args:
        unified_kwargs: Keyword arguments to forward to :func:`export_unified`.
        parser: The active :class:`argparse.ArgumentParser`; used to surface
            invalid-parameter errors to the user.
    """
    try:
        sig = inspect.signature(export_unified)
        params = sig.parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not accepts_kwargs:
            allowed = [
                name
                for name, p in params.items()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ]
            invalid = [k for k in unified_kwargs.keys() if k not in allowed]
            if invalid:
                reason = "Possible typos or use of removed underscore-style aliases. Check flag names and use hyphenated forms."
                parser.error(
                    f"Invalid parameter name(s) passed to exporter: {', '.join(invalid)}. " f"Allowed parameters: {', '.join(allowed)}. {reason}"
                )
    except Exception:
        pass

    print("Using consolidated exporter")

    quantize = unified_kwargs.get("quantize")
    if quantize is None:
        unified_kwargs["quantize"] = False
    elif quantize == "both":
        unified_kwargs["quantize"] = True

    export_unified(**unified_kwargs)


def _run_export(args, parser):
    """Translate parsed *args* into exporter kwargs and run the export.

    Resolves ``onnx_path`` (from arg or env var; pipeline handles the default),
    assembles all export parameters into a dict, and delegates to
    :func:`_execute_export_kwargs`.

    Args:
        args: Parsed :class:`argparse.Namespace` from the export sub-command.
        parser: The :class:`argparse.ArgumentParser`; forwarded for error
            reporting inside :func:`_execute_export_kwargs`.
    """
    _validate_export_args(args, parser)

    # Pass onnx_path=None when not supplied so the pipeline applies its own
    # default (cwd-relative "onnx" or ONNX_PATH env var).
    onnx_path = args.onnx_path or os.getenv("ONNX_PATH") or None
    if onnx_path:
        print(f"Using ONNX path: {os.path.abspath(onnx_path)}")

    unified_kwargs = ExportConfig.from_namespace(args, onnx_path).to_kwargs()

    _execute_export_kwargs(unified_kwargs, parser)
