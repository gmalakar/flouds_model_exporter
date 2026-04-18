# =============================================================================
# File: cli/cmd_export.py
# Date: 2026-01-10
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Export subcommand: argument definitions, helpers, and _run_export."""

from __future__ import annotations

import inspect
import os

from model_exporter.export.pipeline import export as export_unified

# Default ONNX path: <project_root>/onnx, or ONNX_PATH env variable.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_default_onnx_path = os.path.join(_repo_root, "onnx")


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
        help="Log to file and print log file path in terminal (default: True if not set).",
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
        choices=["fe", "s2s", "sc", "llm", "ranker"],
        help=(
            "Model purpose: 'fe' (feature-extraction), 's2s' (seq2seq-lm),"
            " 'sc' (sequence-classification), 'ranker' (cross-encoder/ranking),"
            " or 'llm' (causal-lm). Allowed values: fe, s2s, sc, llm, ranker (default: fe)"
        ),
    )
    parser.add_argument("--optimize", action="store_true", help="Whether to optimize the ONNX model")
    parser.add_argument(
        "--optimization-level",
        dest="optimization_level",
        type=int,
        default=99,
        help="ONNX optimization level (default: 99)",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Export task (e.g., seq2seq-lm, sequence-classification, feature-extraction) (required)",
    )
    parser.add_argument("--model-folder", dest="model_folder", help="HuggingFace model folder or path")
    parser.add_argument(
        "--onnx-path",
        dest="onnx_path",
        help="Path to ONNX output directory (default: ../onnx or ONNX_PATH env var)",
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
        help=("Optional quantization to produce post-export variants. " "Choices: 'dynamic_int8', 'fp16', or 'both'."),
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
        "--pack-single-threshold-mb",
        dest="pack_single_threshold_mb",
        type=int,
        default=None,
        help=(
            "Size threshold in MB for single-file repack. If external weights exceed the"
            " threshold, repack is skipped. If omitted, exporter default (1536 MB) is used."
        ),
    )
    parser.add_argument(
        "--no-local-prep",
        action="store_true",
        help="Skip creating a prepared local copy (temp_local) for LLMs before export",
    )
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        type=str,
        default=None,
        help="HuggingFace access token (synonym for HUGGINGFACE_HUB_TOKEN/HF_TOKEN)",
    )
    parser.add_argument(
        "--library",
        dest="library",
        type=str,
        default=None,
        required=False,
        help=(
            "Export library hint (e.g., 'sentence_transformers' or 'transformers'). "
            "Optional — if omitted the exporter will attempt to infer the best library."
        ),
    )
    parser.add_argument(
        "--merge",
        dest="merge",
        action="store_true",
        help=(
            "Request model merging where applicable. Merge is only applicable to "
            "decoder-only causal LLMs that support text-generation-with-past (KV-cache). "
            "No merging is performed after optimization; the exporter will apply merging "
            "at the appropriate stage. Use this flag to request a merged decoder artifact."
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
        help=(
            "Prefer running exporter in a subprocess (inverse default). "
            "By default the exporter will try in-process first; set this flag "
            "to force subprocess use."
        ),
    )
    parser.add_argument(
        "--use-fallback-if-failed",
        dest="use_fallback_if_failed",
        action="store_true",
        help="Enable legacy fallback exporter only when the primary export path fails.",
    )
    parser.add_argument(
        "--low-memory-env",
        dest="low_memory_env",
        action="store_true",
        help="Treat the environment as low-memory and apply conservative export flags (use external_data, disable some post-processing).",
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


def _execute_export_kwargs(unified_kwargs, parser):
    """Validate *unified_kwargs* against the exporter signature and call it.

    Checks that every key in *unified_kwargs* is a known parameter of
    :func:`export_unified` (when the function does not accept ``**kwargs``).
    Normalises the ``quantize`` value, then invokes the exporter.

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
                reason = "Possible typos or use of removed underscore-style aliases. " "Check flag names and use hyphenated forms."
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

    Resolves ``onnx_path`` (from arg, env var, or default), assembles all
    export parameters into a dict, and delegates to
    :func:`_execute_export_kwargs`.

    Args:
        args: Parsed :class:`argparse.Namespace` from the export sub-command.
        parser: The :class:`argparse.ArgumentParser`; forwarded for error
            reporting inside :func:`_execute_export_kwargs`.
    """
    onnx_path = args.onnx_path or os.getenv("ONNX_PATH", _default_onnx_path)
    print(f"Using ONNX path: {os.path.abspath(onnx_path)}")

    unified_kwargs = {
        "model_name": args.model_name,
        "model_for": args.model_for,
        "optimize": args.optimize,
        "optimization_level": args.optimization_level,
        "portable": args.portable,
        "model_folder": args.model_folder,
        "onnx_path": onnx_path,
        "task": args.task,
        "force": args.force,
        "opset_version": (args.opset_version if hasattr(args, "opset_version") else None),
        "pack_single_file": args.pack_single_file,
        "use_external_data_format": args.use_external_data_format,
        "framework": args.framework,
        "pack_single_threshold_mb": args.pack_single_threshold_mb,
        "require_validator": args.require_validator,
        "trust_remote_code": args.trust_remote_code,
        "normalize_embeddings": args.normalize_embeddings,
        "skip_validator": args.skip_validator,
        "device": args.device,
        "hf_token": args.hf_token,
        "library": args.library,
        "merge": args.merge,
        "cleanup": args.cleanup,
        "prune_canonical": args.prune_canonical,
        "no_post_process": args.no_post_process,
        "no_local_prep": args.no_local_prep,
        "use_subprocess": args.use_subprocess,
        "use_fallback_if_failed": args.use_fallback_if_failed,
        "low_memory_env": args.low_memory_env,
        "quantize": None,
        "log_to_file": args.log_to_file,
    }

    _execute_export_kwargs(unified_kwargs, parser)
