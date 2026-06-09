# =============================================================================
# File: pipeline.py
# Date Created: 2026-04-19
# Date Updated: 2026-04-19
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Public export API for flouds_model_exporter.

This module provides the single public entry-point :func:`export`.
All private helpers have been split into :mod:`pipeline_helpers`
(env/cache/token/quantize utilities) and :mod:`pipeline_v2`
(optimum main_export orchestration).
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
from model_exporter.export.options import ExportConfig
from model_exporter.export.path_utils import safe_remove_export_dir
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
    _teardown_hf_token,
    _with_export_lock,
)
from model_exporter.export.pipeline_v2 import _run_export_with_fallback, _run_post_optimization_validator
from model_exporter.utils.helpers import get_default_opset, get_logger
from model_exporter.validation.checker import verify_models

logger: Any = get_logger(__name__)


def export_from_config(config: ExportConfig) -> str:
    """Export a model using a typed :class:`ExportConfig` instance."""
    return export(**config.to_kwargs())


def _resolve_export_output_dir(
    onnx_path: str | None,
    model_for: str,
    model_name: str,
    model_folder: str | None,
    cwd: str | None = None,
) -> tuple[str, str, str]:
    """Resolve and validate the ONNX root, output directory, and folder name."""
    base_cwd = Path(cwd or os.getcwd()).resolve()
    root_value = onnx_path or os.environ.get("ONNX_PATH") or str(base_cwd / "onnx")
    root_path = Path(root_value)
    if not root_path.is_absolute():
        root_path = base_cwd / root_path
    root_path = root_path.resolve()

    if model_folder is None:
        model_folder = model_name.split("/")[-1] if "/" in str(model_name) else str(model_name)
    safe_folder = os.path.basename(str(model_folder).rstrip("/\\"))
    if safe_folder in {"", ".", ".."}:
        raise ValueError(f"Invalid model_folder: {model_folder!r}")

    model_type_root = (root_path / "models" / model_for).resolve()
    output_dir = (model_type_root / safe_folder).resolve()
    try:
        output_dir.relative_to(model_type_root)
    except ValueError as exc:
        raise ValueError(f"Resolved output directory escapes export root: {output_dir}") from exc

    return str(root_path), str(output_dir), safe_folder


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
    cleanup: bool = False,
    prune_canonical: bool = False,
    no_post_process: bool = False,
    low_memory_env: bool = False,
    log_to_file: bool = False,
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
            Falls back to ``"onnx"`` relative to the **current working directory**.
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
        cleanup: Remove extraneous ONNX files after optimization following
            prioritized cleanup rules.
        prune_canonical: Remove canonical ONNX files (e.g. ``decoder_model.onnx``)
            when merged artifacts exist.
        no_post_process: Skip ONNX post-processing (deduplication). Reduces peak
            memory usage during large-model export.
        low_memory_env: Treat the environment as low-memory and apply conservative
            export flags (use external_data, disable some post-processing).
        log_to_file: Write per-export log to a rotating file under
            ``logs/onnx_exports/`` (opt-in; default ``False``).
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

        from model_exporter import export

        output = export(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_for="fe",
            task="feature-extraction",
            optimize=True,
        )
        print(output)
    """
    # Configure protobuf limits early
    configure_protobuf()

    # Normalize Hugging Face token handling
    try:
        token: Optional[str] = kwargs.pop("hf_token", None) or kwargs.pop("huggingface_hub_token", None)
    except Exception:
        token = None

    token, hf_flags = _setup_hf_token(token, kwargs, logger)
    hf_flags = hf_flags or {}

    opset_version = opset_version or get_default_opset()

    if not model_name or not str(model_name).strip():
        raise ValueError("model_name cannot be empty")

    _model_for = (model_for or "").lower()
    if _model_for not in ["fe", "s2s", "sc", "llm", "ranker"]:
        raise ValueError(f"Invalid model_for: {model_for}")

    # Resolve paths once, before any cleanup/deletion can occur.
    _cwd = os.getcwd()
    onnx_path, _output_dir, model_folder = _resolve_export_output_dir(
        onnx_path,
        _model_for,
        str(model_name),
        model_folder,
        cwd=_cwd,
    )
    _output_parent = str(Path(_output_dir).parent)

    if low_memory_env:
        if not use_external_data_format:
            logger.info("Enabling external_data format for low-memory export")
        if not no_post_process:
            logger.info("Disabling ONNX post-processing for low-memory export")
        use_external_data_format = True
        no_post_process = True

    # BASE_DIR is still used for the logging helper (log path inside package src)
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

    Path(_output_dir).mkdir(parents=True, exist_ok=True)

    use_cache: bool = _resolve_use_cache(model_name, _model_for, task, logger)
    trust_remote_code = _auto_resolve_trust_remote_code(model_name, token, trust_remote_code, logger)

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

    try:
        file_handler, logfile_fd, old_stdout, old_stderr, logfile_path = setup_export_logging(BASE_DIR, safe_model, rev_tag, logger, log_to_file)
    except Exception:
        logger.warning("Failed to initialize per-run logging; continuing without file capture")

    expected: List[str] = _build_expected_list(_model_for, use_cache, task, merge=bool(merge))

    try:
        # The export lock now wraps the entire export body so that concurrent
        # runs for the same output directory are correctly serialized.
        with _with_export_lock(_output_dir, model_name, logger):
            # Pre-export memory cleanup
            _cleanup_memory_caches(logger)

            # Skip export if outputs exist and no force requested
            all_exist: bool = all(os.path.exists(os.path.join(_output_dir, fname)) for fname in expected)
            if all_exist and not force:
                logger.info(
                    "All expected ONNX files already exist in %s — skipping export (use --force to re-export)",
                    _output_dir,
                )
                return _output_dir

            # If force requested, remove existing output dir for a clean export
            if force and os.path.exists(_output_dir):
                try:
                    safe_remove_export_dir(
                        _output_dir,
                        _output_parent,
                        logger,
                        reason="force re-export",
                    )
                    Path(_output_dir).mkdir(parents=True, exist_ok=True)
                    logger.info(
                        "Removed existing output directory because --force was given: %s",
                        _output_dir,
                    )
                except Exception as e:
                    logger.warning("Failed to remove existing output dir with --force: %s", e)

            # Export phase
            prep_tmp_p: Optional[str] = None
            try:
                export_source: str = model_name
                try:
                    if _model_for == "llm" and not os.path.exists(str(model_name)) and not bool(no_local_prep):
                        from model_exporter.export.helpers import prepare_local_model_dir

                        tmp_p = os.path.join(_output_dir, "temp_local")
                        try:
                            os.makedirs(os.path.dirname(tmp_p), exist_ok=True)
                        except Exception:
                            pass
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
                    elif _model_for == "llm" and not os.path.exists(str(model_name)) and bool(no_local_prep):
                        logger.info("Skipping local prep for LLM as requested by --no-local-prep")
                except Exception:
                    logger.debug(
                        "Local model prep skipped or failed; continuing with original source",
                        exc_info=True,
                    )

                # Pass no_post_process and low_memory_env through kwargs for pipeline_v2
                _extra_kwargs: dict[str, Any] = {}
                if no_post_process:
                    _extra_kwargs["no_post_process"] = True
                if low_memory_env:
                    _extra_kwargs["low_memory_env"] = True

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
                    no_post_process=no_post_process,
                    merge=bool(merge),
                    use_subprocess=bool(use_subprocess),
                    use_fallback_if_failed=bool(use_fallback_if_failed),
                )
                if not export_succeeded:
                    error_msg = f"All export attempts failed for model {model_name}"
                    logger.error(
                        "%s. Check logs above for detailed error information.",
                        error_msg,
                    )
                    raise RuntimeError(error_msg)
            except Exception as e:
                try:
                    logger.warning(
                        "Export failed; attempting to remove output dir to free space: %s",
                        _output_dir,
                    )
                    if os.path.exists(_output_dir):
                        try:
                            safe_remove_export_dir(
                                _output_dir,
                                _output_parent,
                                logger,
                                reason="failed export cleanup",
                            )
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

            # Expose quick structural verification result for diagnostics
            logger.debug("Quick structural verification passed=%s", quick_ok)

            # Numeric validator
            validator_rc: int = 0
            if not skip_validator:
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
                        validator_rc, _ = _run_numeric_validator(
                            output_dir=_output_dir,
                            expected=expected,
                            model_name=model_name,
                            pack_single_file=pack_single_file,
                            pack_single_threshold_mb=pack_single_threshold_mb,
                            trust_remote_code=trust_remote_code,
                            used_trust_remote=used_trust_remote,
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
                    try:
                        usage = shutil.disk_usage(_output_dir)
                    except Exception:
                        usage = shutil.disk_usage(os.getcwd())
                    free_bytes: int = int(getattr(usage, "free", 0))
                except Exception:
                    free_bytes = 2 << 30

                MIN_FREE_BYTES_FOR_OPT = 1 << 30  # 1 GiB
                if free_bytes < MIN_FREE_BYTES_FOR_OPT:
                    logger.warning(
                        "Insufficient disk space (%.1f MB) to safely run optimizer; skipping optimization",
                        free_bytes / (1024.0 * 1024.0),
                    )
                else:
                    try:
                        optimize_if_encoder: Optional[Callable[..., int]] = None
                        from model_exporter.export.optimizer import optimize_if_encoder as _optimize_if_encoder

                        optimize_if_encoder = _optimize_if_encoder
                    except Exception:
                        optimize_if_encoder = None

                    if optimize_if_encoder is None:
                        logger.warning("optimize_if_encoder helper not available; skipping optimization")
                    else:
                        try:
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
                                logger.warning("Optimizer returned non-zero code: %s", rc_post)
                            elif _check_optimized_artifacts(_output_dir):
                                logger.info("Optimized ONNX artifact detected; running post-optimization validator")
                                try:
                                    _run_post_optimization_validator(
                                        output_dir=_output_dir,
                                        expected=expected,
                                        model_name=model_name,
                                        pack_single_file=pack_single_file,
                                        pack_single_threshold_mb=pack_single_threshold_mb,
                                        trust_remote_code=trust_remote_code,
                                        used_trust_remote=used_trust_remote,
                                        normalize_embeddings=normalize_embeddings,
                                        logger=logger,
                                        skip_validator=skip_validator,
                                    )
                                except Exception:
                                    logger.debug(
                                        "Post-optimization validation failed",
                                        exc_info=True,
                                    )
                        except (SystemExit, Exception) as e:
                            if isinstance(e, SystemExit):
                                raise
                            logger.exception("Optimization failed: %s", e)

            # Artifact cleanup
            try:
                from model_exporter.export.helpers import cleanup_extraneous_onnx_files as _cleanup_extraneous_onnx_files

                _cleanup_extraneous_onnx_files(_output_dir, logger, cleanup, prune_canonical)
            except ImportError:
                logger.debug("cleanup_extraneous_onnx_files helper not available")
            except Exception:
                logger.debug("cleanup_extraneous_onnx_files failed", exc_info=True)

            # Post-export quantization
            _run_quantization_step(_output_dir, quantize, kwargs, logger)

            # Lift any artifacts written under temp_local into output_dir
            _lift_temp_local_artifacts(_output_dir, logger)

            return _output_dir

    finally:
        # Clean up prepared local model snapshot
        try:
            if "prep_tmp_p" in dir() and prep_tmp_p and os.path.exists(prep_tmp_p):
                shutil.rmtree(prep_tmp_p, ignore_errors=True)
        except Exception:
            pass
        # Best-effort temp artifact cleanup
        try:
            from model_exporter.export.helpers import cleanup_temporary_export_artifacts

            cleanup_temporary_export_artifacts(logger=logger)
        except Exception:
            logger.debug("cleanup_temporary_export_artifacts raised", exc_info=True)
        # Tear down per-run file logging
        if file_handler is not None:
            try:
                teardown_export_logging(file_handler, logfile_fd, old_stdout, old_stderr, logger)
            except Exception:
                pass
        try:
            _teardown_hf_token(hf_flags, logger)
        except Exception:
            pass


if __name__ == "__main__":
    print(
        "pipeline.py is an importable orchestrator. "
        "Use the `flouds-export` CLI for command-line usage "
        "(or `python -m model_exporter.cli.main`)."
    )
