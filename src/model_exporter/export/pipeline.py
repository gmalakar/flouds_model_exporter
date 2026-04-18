# =============================================================================
# File: pipeline.py
# Date: 2026-01-10
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Public export API for FloudsModelExporter.

This module provides the single public entry-point
:func:`export`.  All private helpers have been split
into :mod:`pipeline_helpers` (env/cache/token/quantize utilities) and
:mod:`pipeline_v2` (optimum main_export orchestration).
"""

import os
import shutil
from pathlib import Path
from typing import Any, Callable, List, Optional

# Ensure the pure-Python protobuf implementation is preferred by default on
# platforms where the C-extension may be unstable. This is a safe, low-risk
# mitigation for native crashes originating in `google.protobuf._message`.
try:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
except Exception:
    pass

from model_exporter.config.logging import setup_export_logging, teardown_export_logging
from model_exporter.export.helpers import configure_protobuf
from model_exporter.export.pipeline_helpers import (
    _auto_resolve_trust_remote_code,
    _build_expected_list,
    _check_optimized_artifacts,
    _cleanup_memory_caches,
    _is_seq2seq,
    _lift_temp_local_artifacts,
    _resolve_use_cache,
    _run_numeric_validator,
    _run_quantization_step,
    _setup_hf_token,
    _should_skip_validator,
    _with_export_lock,
)
from model_exporter.export.pipeline_v2 import (
    _run_export_with_fallback,
    _run_post_optimization_validator,
)
from model_exporter.utils.helpers import get_default_opset, get_logger
from model_exporter.validation.checker import verify_models

logger: Any = get_logger(__name__)


def export(
    model_name: str,
    model_for: str = "fe",
    optimize: bool = False,
    merge: bool = False,
    optimization_level: int = 99,
    portable: bool = False,
    model_folder: str | None = None,
    onnx_path: str | None = None,
    task: str | None = None,
    force: bool = False,
    opset_version: int | None = None,
    pack_single_file: bool = False,
    framework: str | None = None,
    pack_single_threshold_mb: int | None = 1536,
    require_validator: bool = False,
    trust_remote_code: bool = False,
    normalize_embeddings: bool = False,
    skip_validator: bool = False,
    device: str = "cpu",
    library: str | None = None,
    use_external_data_format: bool = False,
    no_local_prep: bool = False,
    use_subprocess: bool | None = None,
    use_fallback_if_failed: bool = False,
    quantize: Any = False,
    **kwargs: Any,
) -> str:
    """Export a HuggingFace model to ONNX and optionally optimize it.

    Orchestrates the full export pipeline: protobuf configuration, HuggingFace
    authentication, ONNX export via ``optimum``, structural verification, numeric
    validation, and post-export optimization.

    Args:
        model_name: HuggingFace model ID (e.g. ``"sentence-transformers/all-MiniLM-L6-v2"``)
            or a local directory path containing the model files.
        model_for: Model purpose. One of:
            - ``"fe"`` – feature extraction / sentence embeddings (default)
            - ``"s2s"`` – seq2seq (T5, BART, mT5, …)
            - ``"sc"`` – sequence classification
            - ``"ranker"`` – cross-encoder / ranking
            - ``"llm"`` – causal language model (GPT-2, LLaMA, …)
        optimize: Run ONNX Runtime graph optimizations after export.
        merge: Merge decoder-with-past artifacts into a single file (LLMs only).
        optimization_level: ORT optimization level when ``optimize=True``.
            Range 0–99; 99 enables all optimizations (default).
        portable: Use conservative optimizations that are safe across platforms
            and ORT versions. Implies a lower optimization level.
        model_folder: Override the output sub-folder name. Defaults to the last
            segment of ``model_name``.
        onnx_path: Root directory for ONNX output. The final model is written to
            ``<onnx_path>/models/<model_for>/<model_folder>/``.
            If omitted, the ``ONNX_PATH`` environment variable is used.
            Falls back to ``"onnx"`` relative to the pipeline module directory.
        task: Optimum export task string, e.g. ``"feature-extraction"``,
            ``"seq2seq-lm"``, ``"text-generation-with-past"``.
            Required for unambiguous export when a model supports multiple tasks.
        force: Overwrite an existing export in the output directory.
        opset_version: ONNX opset version. Defaults to the value returned by
            :func:`get_default_opset` (currently 17).
        pack_single_file: Repack a multi-file external-data export into one
            ``.onnx`` file after validation.
        framework: Deep-learning framework to use for tracing. ``"pt"`` (PyTorch)
            or ``"tf"`` (TensorFlow). Defaults to auto-detection.
        pack_single_threshold_mb: Only repack when the model is smaller than this
            size in MB (default 1536 MB). Set to ``None`` to always repack.
        require_validator: Raise an error if the numeric validator cannot run
            (e.g. because optional dependencies are missing).
        trust_remote_code: Allow execution of custom model code hosted in the
            model repository. Use with caution — only enable for repos you trust.
        normalize_embeddings: L2-normalize sentence embeddings before comparing
            reference and ONNX outputs during validation.
        skip_validator: Skip numeric validation entirely.
        device: Target inference device. ``"cpu"`` (default) or ``"cuda"``.
        library: Optimum model library hint (e.g. ``"transformers"``,
            ``"sentence_transformers"``). Auto-detected when omitted.
        use_external_data_format: Store weight tensors in separate ``.onnx_data``
            files. Required for models larger than 2 GB.
        no_local_prep: Skip local model preparation steps (e.g. config patching
            for LLMs). Useful when the model directory is already prepared.
        use_subprocess: Run the export in an isolated subprocess to protect the
            calling process from memory leaks or native crashes.
            ``None`` lets the pipeline decide based on model size and type.
        use_fallback_if_failed: Enable legacy fallback exporter only if the
            primary export path fails.
        quantize: Quantization configuration. Pass a quantization config object
            or ``True`` to enable default quantization. ``False`` disables it.
        **kwargs: Additional keyword arguments forwarded to the underlying
            exporter. Recognised extras:
            - ``hf_token`` / ``huggingface_hub_token`` – HuggingFace API token
              for accessing private or gated model repositories.

    Returns:
        Absolute path to the directory containing the exported ONNX model file(s).

    Raises:
        ValueError: If ``model_name`` is empty, ``model_for`` is not one of the
            accepted values, or ``onnx_path`` contains a path-traversal sequence.

    Environment variables:
        ONNX_PATH: Default root directory for ONNX output when ``onnx_path`` is
            not passed explicitly.
        HUGGINGFACE_TOKEN: HuggingFace API token used when ``hf_token`` is not
            supplied via kwargs.

    Example::

        from model_exporter.export.pipeline import export

        # ONNX_PATH env var sets the output root; no need to pass onnx_path
        output = export(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_for="fe",
            task="feature-extraction",
            optimize=True,
        )
        print(output)  # e.g. /models/onnx/models/fe/all-MiniLM-L6-v2
    """

    # Note: the `--suppress-warning` CLI option was removed; do not suppress global warnings here.

    # Configure protobuf limits early
    configure_protobuf()

    # Normalize Hugging Face token handling: accept token via kwargs or environment
    # and attempt to login for the session. If login fails (invalid token),
    # remove any env vars we set and continue anonymously.
    try:
        token: Optional[str] = kwargs.pop("hf_token", None) or kwargs.pop(
            "huggingface_hub_token", None
        )
    except Exception:
        token = None

    # Setup HF token and flags for cleanup
    token, hf_flags = _setup_hf_token(token, kwargs, logger)
    hf_flags = hf_flags or {}

    opset_version = opset_version or get_default_opset()

    if not model_name or not str(model_name).strip():
        raise ValueError("model_name cannot be empty")

    _model_for = (model_for or "").lower()
    if _model_for not in ["fe", "s2s", "sc", "llm", "ranker"]:
        raise ValueError(f"Invalid model_for: {model_for}")

    # an explicit `--task` (e.g., `feature-extraction`) when exporting T5 encoders.

    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    onnx_path = onnx_path or os.environ.get("ONNX_PATH") or "onnx"
    onnx_path = os.path.normpath(onnx_path)
    if ".." in onnx_path:
        raise ValueError("Path traversal detected in onnx_path")

    if not model_folder:
        model_folder = model_name.split("/")[-1] if "/" in str(model_name) else str(model_name)
    model_folder = os.path.basename(model_folder)
    _output_dir = os.path.join(BASE_DIR, onnx_path, "models", _model_for, model_folder)
    Path(_output_dir).mkdir(parents=True, exist_ok=True)

    use_cache: bool = _resolve_use_cache(model_name, _model_for, task, logger)
    trust_remote_code = _auto_resolve_trust_remote_code(model_name, token, trust_remote_code, logger)

    # finetune handling removed; no pre-export fine-tuning step

    # Create export lock to avoid concurrent exports (kept for backward compatibility)
    with _with_export_lock(_output_dir, model_name, logger) as (
        lock_path,
        created_lock,
    ):
        # Pre-export cleanup
        _cleanup_memory_caches(logger)

    # Setup per-run logging
    file_handler: Any | None = None
    logfile_fd: Any | None = None
    old_stdout: Any | None = None
    old_stderr: Any | None = None
    logfile_path: Optional[Path] = None
    safe_model = model_folder.replace("/", "_").replace("\\", "_")
    rev_tag = "local"
    try:
        from huggingface_hub import HfApi

        if "/" in str(model_name):
            try:
                info = HfApi().repo_info(str(model_name))
                rev_tag = getattr(info, "sha", None) or getattr(info, "revision", None) or "local"
            except Exception:
                rev_tag = "local"
    except Exception:
        rev_tag = "local"

    # Determine whether to log to file (opt-in; default False) and pass to setup_export_logging
    log_to_file = kwargs.pop("log_to_file", False)
    try:
        file_handler, logfile_fd, old_stdout, old_stderr, logfile_path = setup_export_logging(
            BASE_DIR, safe_model, rev_tag, logger, log_to_file
        )
    except Exception:
        logger.warning("Failed to initialize per-run logging; continuing without file capture")

    expected: List[str] = _build_expected_list(_model_for, use_cache, task, merge=bool(merge))

    try:
        # Skip export if outputs exist and no force requested
        all_exist: bool = all(
            os.path.exists(os.path.join(_output_dir, fname)) for fname in expected
        )
        if all_exist and not force:
            logger.info(
                "All expected ONNX files already exist in %s — skipping export (use --force to re-export)",
                _output_dir,
            )
            logger.info("Process log: export step skipped because outputs already present")
            return _output_dir

        # If force requested, remove existing output dir to ensure a clean export
        if force and os.path.exists(_output_dir):
            import pathlib

            try:
                shutil.rmtree(_output_dir)
                Path(_output_dir).mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Removed existing output directory because --force was given: %s",
                    _output_dir,
                )
            except Exception as e:
                logger.warning("Failed to remove existing output dir with --force: %s", e)
                # Best-effort cleanup: remove files inside to avoid stale artifacts
                try:
                    for p in pathlib.Path(_output_dir).glob("**/*"):
                        try:
                            if p.is_file():
                                p.unlink()
                            elif p.is_dir():
                                shutil.rmtree(p)
                        except Exception:
                            pass
                except Exception:
                    logger.debug("Best-effort cleanup of output dir failed")

        # Export phase: prepare local copy for certain HF models (LLMs) to
        # canonicalize tied weights and avoid duplicate initializer issues,
        # then delegate to helper that handles retry/clone fallbacks.
        prep_tmp_p: Optional[str] = None
        try:
            export_source: str = model_name
            # If source looks like a HF hub id (not a local path) and this is an
            # LLM export, prepare a local tied-weights copy to stabilize ONNX
            # initializer naming. Place it under onnx/tmp_export/<model_folder>-local.
            try:
                if (
                    _model_for == "llm"
                    and not os.path.exists(str(model_name))
                    and not bool(no_local_prep)
                ):
                    from model_exporter.export.helpers import prepare_local_model_dir

                    # Prepare the transient local copy inside the model folder
                    # so it's colocated with the eventual ONNX output. Use a
                    # `temp_local` subfolder to avoid colliding with the final
                    # output artifacts. This folder will be removed after export
                    # (success or failure).
                    tmp_p = os.path.join(
                        BASE_DIR,
                        onnx_path,
                        "models",
                        _model_for,
                        model_folder,
                        "temp_local",
                    )
                    # Ensure parent exists
                    try:
                        os.makedirs(os.path.dirname(tmp_p), exist_ok=True)
                    except Exception:
                        pass
                    # Remove stale temp_local before creating
                    try:
                        if os.path.exists(tmp_p):
                            shutil.rmtree(tmp_p)
                    except Exception:
                        pass
                    os.makedirs(tmp_p, exist_ok=True)
                    prep_ok = prepare_local_model_dir(model_name, tmp_p, trust_remote_code, logger)
                    if prep_ok:
                        export_source = tmp_p
                        prep_tmp_p = tmp_p
                        logger.info("Using prepared local model for export: %s", tmp_p)
                elif (
                    _model_for == "llm"
                    and not os.path.exists(str(model_name))
                    and bool(no_local_prep)
                ):
                    logger.info("Skipping local prep for LLM as requested by --no-local-prep")
            except Exception:
                logger.debug(
                    "Local model prep skipped or failed; continuing with original source",
                    exc_info=True,
                )
            # Always attempt export after any preparation step. Ensure
            # `export_succeeded` is assigned regardless of whether local
            # preparation succeeded or raised an exception.
            # Determine whether subprocess fallback is allowed. Only LLM
            # exports may use subprocesses; for those, honor the explicit
            # `use_subprocess` flag (convert to bool). For all other model
            # types, disallow subprocess fallback.
            # Allow subprocess for all model types if requested (was restricted to llm only)

            export_succeeded, used_trust_remote = _run_export_with_fallback(
                export_source,
                _output_dir,
                _model_for,
                opset_version,
                device,
                task,
                framework,
                library,
                logger,
                trust_remote_code,
                use_external_data_format=use_external_data_format,
                no_post_process=bool(kwargs.get("no_post_process", False)),
                merge=bool(merge),
                use_subprocess=bool(use_subprocess),
                use_fallback_if_failed=bool(use_fallback_if_failed),
            )
            if not export_succeeded:
                error_msg = f"All export attempts failed for model {model_name}"
                logger.error("%s. Check logs above for detailed error information.", error_msg)
                raise RuntimeError(error_msg)
        except Exception as e:
            # On export failure, attempt to remove the output model folder to
            # free disk space for retries or subsequent operations.
            try:
                logger.warning(
                    "Export failed; attempting to remove output dir to free space: %s",
                    _output_dir,
                )
                if os.path.exists(_output_dir):
                    try:
                        shutil.rmtree(_output_dir)
                        logger.info(
                            "Removed output directory after failed export: %s",
                            _output_dir,
                        )
                    except Exception as rm_e:
                        logger.warning(
                            "Failed to remove output directory %s: %s",
                            _output_dir,
                            rm_e,
                        )
            except Exception:
                logger.debug(
                    "Output-dir cleanup raised during export failure handling",
                    exc_info=True,
                )
            logger.exception("Export failed: %s", e)
            raise
        finally:
            # Clean up any prepared transient local copy used for export
            try:
                if prep_tmp_p and os.path.exists(prep_tmp_p):
                    try:
                        shutil.rmtree(prep_tmp_p)
                        logger.info("Removed temporary local model folder: %s", prep_tmp_p)
                    except Exception as rm_e:
                        logger.debug(
                            "Failed to remove temporary local model folder %s: %s",
                            prep_tmp_p,
                            rm_e,
                        )
            except Exception:
                logger.debug("Cleanup of prep_tmp_p failed", exc_info=True)

        # Quick structural verification
        quick_ok: bool = False
        try:
            quick_ok = bool(
                verify_models(
                    expected,
                    _output_dir,
                    pack_single=pack_single_file,
                    pack_single_threshold_mb=pack_single_threshold_mb,
                )
            )
        except Exception as v_err:
            logger.warning("Quick verification raised: %s", v_err)
            quick_ok = False

        # Seq2seq (encoder+decoder) merging is disallowed by policy.
        # Merge is only applicable to decoder-only causal LLMs that support
        # `text-generation-with-past` (KV-cache) and must run before
        # optimization. No encoder+decoder or post-optimization merging is
        # performed here.

        # Invoke numeric validator unless explicitly skipped.
        # Note: the centralized numeric validator expects a single `model.onnx` file.
        # For multi-file seq2seq exports (encoder/decoder[/decoder_with_past]) skip
        # numeric validation unless the user asked for `--pack_single_file`.
        validator_rc: int = 0
        validator_quick_ok: bool = quick_ok
        if not skip_validator:
            # Determine if validator should be skipped for this model type
            if _should_skip_validator(_model_for, pack_single_file, expected):
                if _is_seq2seq(_model_for) and not pack_single_file:
                    logger.info(
                        "Skipping numeric validator for multi-file seq2seq export; use "
                        "--pack_single_file to create model.onnx and enable numeric validation"
                    )
                else:
                    logger.info(
                        "--pack_single_file was requested but export produced "
                        "multi-file seq2seq artifacts; skipping numeric validator since "
                        "model.onnx is not present"
                    )
            else:
                try:
                    validator_rc, validator_quick_ok = _run_numeric_validator(
                        output_dir=_output_dir,
                        expected=expected,
                        model_name=model_name,
                        pack_single_file=pack_single_file,
                        pack_single_threshold_mb=pack_single_threshold_mb,
                        trust_remote_code=trust_remote_code,
                        used_trust_remote=("used_trust_remote" in locals() and used_trust_remote),
                        normalize_embeddings=normalize_embeddings,
                        logger=logger,
                        require_validator=require_validator,
                    )
                except Exception as e:
                    logger.exception("Validator invocation failed: %s", e)
                    if require_validator:
                        raise

        # Optimization step (optional)
        if optimize:
            try:
                # Optimization level selection is handled inside the
                # `run_optimization` helper. Pass the requested
                # `optimization_level` through and let the optimizer
                # decide conservative defaults for decoder/LLM artifacts.
                opt_level_for_run = optimization_level

                # Disk-space guard: avoid running optimizer when free space is low
                try:
                    try:
                        usage = shutil.disk_usage(_output_dir)
                    except Exception:
                        # Fallback to current working dir if output_dir not mounted yet
                        usage = shutil.disk_usage(os.getcwd())
                    free_bytes: int = int(getattr(usage, "free", 0))
                except Exception:
                    free_bytes = 2 << 30  # assume plenty if check fails

                MIN_FREE_BYTES_FOR_OPT = 1 << 30  # 1 GiB
                if free_bytes < MIN_FREE_BYTES_FOR_OPT:
                    logger.warning(
                        "Insufficient disk space (%.1f MB) to safely run optimizer; skipping optimization",
                        free_bytes / (1024.0 * 1024.0),
                    )
                    logger.info("Process log: optimization skipped due to low disk space")
                    rc_post = 0
                else:
                    try:
                        optimize_if_encoder: Optional[Callable[..., int]] = None
                        from model_exporter.export.optimizer import (
                            optimize_if_encoder as _optimize_if_encoder,
                        )

                        optimize_if_encoder = _optimize_if_encoder
                    except Exception:
                        optimize_if_encoder = None

                    if optimize_if_encoder is None:
                        logger.warning(
                            "optimize_if_encoder helper not available; skipping optimization"
                        )
                        rc_post = 0
                    else:
                        rc_post = int(
                            optimize_if_encoder(
                                _output_dir,
                                _model_for,
                                logger,
                                optimization_level,
                                portable=portable,
                            )
                        )
                    if rc_post != 0:
                        logger.warning("Post-optimization validator returned %s", rc_post)
                    # If optimization succeeded, detect whether the optimizer
                    # produced any optimized artifacts and, if so, run the
                    # numeric validator again to verify the optimized model.
                    if rc_post == 0:
                        try:
                            optimized_found: bool = _check_optimized_artifacts(_output_dir)
                            if optimized_found:
                                logger.info(
                                    "Optimized ONNX artifact detected; running quick structural verification and numeric validator post-optimization"
                                )
                                # Run post-optimization validator
                                rc_post_val: int = _run_post_optimization_validator(
                                    output_dir=_output_dir,
                                    expected=expected,
                                    model_name=model_name,
                                    pack_single_file=pack_single_file,
                                    pack_single_threshold_mb=pack_single_threshold_mb,
                                    trust_remote_code=trust_remote_code,
                                    used_trust_remote=(
                                        "used_trust_remote" in locals() and used_trust_remote
                                    ),
                                    normalize_embeddings=normalize_embeddings,
                                    logger=logger,
                                    skip_validator=skip_validator,
                                )
                        except Exception:
                            logger.debug(
                                "Post-optimization verification check failed",
                                exc_info=True,
                            )
            except SystemExit:
                raise
            except Exception as e:
                logger.exception("Optimization failed: %s", e)

        # invoke clean up here
        try:
            cleanup_extraneous_onnx_files: Optional[Callable[..., None]] = None
            try:
                from model_exporter.export.helpers import (
                    cleanup_extraneous_onnx_files as _cleanup_extraneous_onnx_files,
                )

                cleanup_extraneous_onnx_files = _cleanup_extraneous_onnx_files
            except Exception:
                cleanup_extraneous_onnx_files = None

            if cleanup_extraneous_onnx_files is not None:
                try:
                    cleanup_extraneous_onnx_files(
                        _output_dir,
                        logger,
                        bool(kwargs.get("cleanup", False)),
                        bool(kwargs.get("prune_canonical", False)),
                    )
                except Exception:
                    logger.debug("cleanup_extraneous_onnx_files failed", exc_info=True)
            else:
                logger.debug("cleanup_extraneous_onnx_files helper not available")
        except Exception:
            pass

        # Post-export quantization/precision-reduction step.
        _run_quantization_step(_output_dir, quantize, kwargs, logger)

        # If export wrote artifacts into a `temp_local` subfolder, lift them.
        _lift_temp_local_artifacts(_output_dir, logger)

        return _output_dir
    finally:
        # Cleanup any prepared local model snapshot created for this export
        try:
            if "prep_tmp_p" in locals() and prep_tmp_p:
                try:
                    if os.path.exists(prep_tmp_p):
                        shutil.rmtree(prep_tmp_p)
                        logger.info("Removed prepared local model folder: %s", prep_tmp_p)
                    # Do not remove parent directories; only remove the prepared
                    # `temp_local` folder to avoid accidentally deleting output
                    # directories that may contain export artifacts.
                except Exception:
                    logger.debug(
                        "Failed to cleanup prepared local model folder: %s",
                        prep_tmp_p,
                        exc_info=True,
                    )
        except Exception:
            pass
        # Best-effort: clean other temporary export artifacts (system temp and any
        # lingering temp_local folders). This helps when clone/copy attempts
        # failed due to disk pressure and left behind partially-copied folders.
        try:
            from model_exporter.export.helpers import cleanup_temporary_export_artifacts

            cleanup_temporary_export_artifacts(logger=logger, base_dir=BASE_DIR)
        except Exception:
            logger.debug("cleanup_temporary_export_artifacts raised", exc_info=True)

        if file_handler is not None:
            try:
                teardown_export_logging(file_handler, logfile_fd, old_stdout, old_stderr, logger)
            except Exception:
                pass


if __name__ == "__main__":
    print("pipeline.py is an importable orchestrator. Use export_model.py for CLI.")
