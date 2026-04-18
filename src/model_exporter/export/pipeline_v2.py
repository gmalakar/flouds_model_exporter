# =============================================================================
# File: pipeline_v2.py
# Date: 2026-04-17
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""V2 ONNX export implementation: optimum main_export orchestration,
fallback strategies, and post-optimization validation.

Extracted from pipeline.py to reduce file length.
"""
from __future__ import annotations

import contextlib as _contextlib
import io as _io
import os
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys
import tempfile as _tempfile
from typing import Any, Callable, List, Optional

from model_exporter.export.helpers import cleanup_temporary_export_artifacts
from model_exporter.export.legacy_fallback import run_legacy_v1_fallback
from model_exporter.export.pipeline_helpers import _remove_validation_marker, _write_validation_marker
from model_exporter.export.subprocess_runner import _run_main_export_subprocess
from model_exporter.utils.helpers import get_default_opset, get_logger, safe_log
from model_exporter.validation.checker import verify_models
from model_exporter.validation.invoker import invoke_validator

logger: Any = get_logger(__name__)


def _load_transformers_components(
    model_name: str,
    trust_remote_code: bool,
    token: Optional[str],
    logger: Any,
) -> tuple[Any, Any, Any]:
    """Load tokenizer, model, and config from transformers with trust_remote_code retry.

    Returns (tokenizer, model, config) tuple.
    """
    from transformers import AutoConfig as _AutoConfig  # noqa: F401
    from transformers import AutoModel as _AutoModel  # noqa: F401
    from transformers import AutoTokenizer as _AutoTokenizer  # noqa: F401

    def _attempt_load(load_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return load_fn(*args, **kwargs)
        except ValueError as ve:
            msg = str(ve) or ""
            if "trust_remote_code" in msg or "requires you to execute" in msg:
                safe_log(
                    logger,
                    "warning",
                    "Remote config requires `trust_remote_code=True` for %s; retrying with trust_remote_code=True",
                    model_name,
                )
                kwargs["trust_remote_code"] = True
                return load_fn(*args, **kwargs)
            raise

    # Load tokenizer
    safe_log(
        logger,
        "debug",
        "Fallback: loading tokenizer for %s (trust_remote_code=%s)",
        model_name,
        trust_remote_code,
    )
    try:
        tokenizer = _attempt_load(
            _AutoTokenizer.from_pretrained,
            model_name,
            trust_remote_code=trust_remote_code,
            use_auth_token=token,
        )
    except Exception:
        tokenizer = None

    # Load model (generic fallback)
    safe_log(
        logger,
        "debug",
        "Fallback: loading model for %s (trust_remote_code=%s)",
        model_name,
        trust_remote_code,
    )
    try:
        model = _attempt_load(
            _AutoModel.from_pretrained,
            model_name,
            trust_remote_code=trust_remote_code,
            use_auth_token=token,
        )
    except Exception:
        model = None

    if model is None and tokenizer is None:
        raise RuntimeError(f"Fallback ONNX export failed: could not load model or tokenizer for {model_name}")

    # Load config optionally (with the same retry-on-trust behavior)
    safe_log(
        logger,
        "debug",
        "Fallback: loading config for %s (trust_remote_code=%s)",
        model_name,
        trust_remote_code,
    )
    try:
        config = _attempt_load(
            _AutoConfig.from_pretrained,
            model_name,
            trust_remote_code=trust_remote_code,
            use_auth_token=token,
        )
    except Exception:
        config = None

    return tokenizer, model, config


def export_onnx_fallback(me_kwargs: dict[str, Any]) -> None:
    """
    Generic fallback ONNX exporter that accepts the prepared `me_kwargs`
    dictionary from the v2 orchestration, extracts the relevant values,
    loads model/tokenizer/config using `transformers`, and calls
    `optimum.exporters.onnx.export` with a dynamically filtered kwargs set.
    """

    from pathlib import Path as _Path

    from transformers import AutoConfig as _AutoConfig  # noqa: F401
    from transformers import AutoModel as _AutoModel  # noqa: F401
    from transformers import AutoTokenizer as _AutoTokenizer  # noqa: F401

    _onnx_export: Optional[Callable[..., Any]] = None
    try:
        import importlib

        _mod = importlib.import_module("optimum.exporters.onnx")
        _onnx_export = getattr(_mod, "export", None)
    except Exception:
        _onnx_export = None

    # Pull through all relevant parameters from me_kwargs
    model_name = me_kwargs.get("model_name_or_path")
    if not model_name or not isinstance(model_name, str):
        raise ValueError("model_name_or_path must be a non-empty string in me_kwargs")

    output_dir = me_kwargs.get("output") or me_kwargs.get("output_dir")
    task = me_kwargs.get("task")
    opset = int(me_kwargs.get("opset", 17))
    trust_remote_code = bool(me_kwargs.get("trust_remote_code", True))

    # Optional parameters that may or may not exist
    token = (
        me_kwargs.get("use_auth_token")
        if isinstance(me_kwargs.get("use_auth_token"), str)
        else me_kwargs.get("token") if isinstance(me_kwargs.get("token"), str) else None
    )

    # Additional optional parameters to pass through if supported
    # (These will be filtered dynamically later)
    additional_params = {
        "use_external_data_format": me_kwargs.get("use_external_data_format"),
        "framework": me_kwargs.get("framework"),
        "device": me_kwargs.get("device"),
        "pad_token_id": me_kwargs.get("pad_token_id"),
        "sequence_length": me_kwargs.get("sequence_length"),
    }

    # Choose a logger: prefer one passed in me_kwargs, otherwise use module logger
    import logging as _logging

    lg: Any = me_kwargs.get("logger")
    if lg is None or not hasattr(lg, "info"):
        lg = _logging.getLogger(__name__)

    # Emit a concise invocation log (redact tokens)
    safe_log(
        lg,
        "info",
        "export_onnx_fallback invoked: model=%s output=%s task=%s opset=%s trust_remote_code=%s",
        model_name,
        output_dir,
        task,
        opset,
        bool(trust_remote_code),
    )

    if not output_dir:
        raise ValueError("output_dir cannot be None or empty")
    safe_output = _Path(output_dir)

    # Load transformers components (tokenizer, model, config)
    tokenizer: Any
    model: Any
    config: Any
    tokenizer, model, config = _load_transformers_components(model_name, trust_remote_code, token, lg)

    # Patch config with expected attributes for optimum
    _patch_config_for_optimum(config, model_name, lg)

    if _onnx_export is None:
        raise RuntimeError("optimum.exporters.onnx.export not available in this environment")

    # Build export kwargs dynamically
    import inspect as _inspect

    sig = _inspect.signature(_onnx_export)

    candidate: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer,
        "output": safe_output,
        "task": task,
        "opset": opset,
        "config": config,
        **additional_params,  # include optional params
    }

    # Filter only supported parameters and non-None values
    filtered: dict[str, Any] = {k: v for k, v in candidate.items() if k in sig.parameters and v is not None}

    # Try full export, then minimal fallback
    safe_log(
        lg,
        "info",
        "Fallback: calling optimum.exporters.onnx.export (filtered args: %s)",
        list(filtered.keys()),
    )
    try:
        _onnx_export(**filtered)
    except Exception:
        safe_log(
            lg,
            "warning",
            "Fallback: full export raised exception, trying minimal export",
        )
        _onnx_export(
            model=model,
            output=safe_output,
            opset=opset,
            config=config,
        )

    safe_log(lg, "info", "export_onnx_fallback completed for %s", model_name)


def _prepare_strategy(
    base_kwargs: dict,
    fb_name: str,
    fb_kwargs: dict,
    export_source: str,
    logger: Any,
) -> tuple[Optional[dict], Optional[Callable[[], None]], Optional[str]]:
    """Prepare `me_try` dict and a cleanup callable for a given fallback strategy.

    Returns (me_try, cleanup_callable, error_msg). If the strategy should be
    skipped, returns (None, None, "reason").
    """
    try:
        me_try: dict = base_kwargs.copy()
        cleanup: Optional[Callable[[], None]] = None

        # Local path strategy
        if fb_kwargs.get("__use_local__"):
            repo = str(export_source)
            if os.path.exists(repo) and os.path.isdir(repo):
                me_try["model_name_or_path"] = repo
                me_try["trust_remote_code"] = True
                return me_try, lambda: None, None
            return None, None, "local path not available"

        # Snapshot strategy
        if fb_kwargs.get("__snapshot__"):
            tmp = _tempfile.mkdtemp(prefix="onnx_opt_clone_")
            try:
                try:
                    from model_exporter.export.helpers import prepare_local_model_dir
                except Exception:
                    return None, None, "prepare_local_model_dir unavailable"

                ok = prepare_local_model_dir(str(export_source), tmp, True, logger)
                if not ok:
                    try:
                        _shutil.rmtree(tmp)
                    except Exception:
                        pass
                    return None, None, "prepare_local_model_dir failed"

                me_try["model_name_or_path"] = tmp
                me_try["trust_remote_code"] = True

                def _cleanup() -> None:
                    try:
                        _shutil.rmtree(tmp)
                    except Exception:
                        pass

                return me_try, _cleanup, None
            except Exception:
                try:
                    _shutil.rmtree(tmp)
                except Exception:
                    pass
                return None, None, "prepare_local_model_dir failed"
        # Clone strategy
        if fb_kwargs.get("__clone__"):
            tmp = _tempfile.mkdtemp(prefix="onnx_export_")
            repo = str(export_source)
            try:
                # If source is local dir, copy; otherwise clone from HF
                if os.path.exists(repo) and os.path.isdir(repo):
                    try:
                        _shutil.copytree(repo, tmp, dirs_exist_ok=True)
                    except TypeError:
                        _shutil.copytree(repo, tmp)
                else:
                    repo_to_clone = repo
                    try:
                        if ("/" in repo) and (not repo.startswith("http://")) and (not repo.startswith("https://")) and (not os.path.exists(repo)):
                            repo_to_clone = f"https://huggingface.co/{repo}"
                    except Exception:
                        repo_to_clone = repo

                    _subprocess.check_call(["git", "clone", "--depth", "1", repo_to_clone, tmp])

                me_try["model_name_or_path"] = tmp
                me_try["trust_remote_code"] = True

                def _cleanup() -> None:
                    try:
                        _shutil.rmtree(tmp)
                    except Exception:
                        pass

                cleanup = _cleanup
                return me_try, cleanup, None
            except Exception as e:
                try:
                    _shutil.rmtree(tmp)
                except Exception:
                    pass
                return None, None, str(e)

        # Normal strategy: just update kwargs
        me_try.update({k: v for k, v in fb_kwargs.items() if not k.startswith("__")})
        return me_try, lambda: None, None
    except Exception as e:
        return None, None, str(e)


def _cleanup_child_tmp(child_tmp: Optional[str], logger: Any) -> Optional[str]:
    """Best-effort removal of a transient short working dir.

    Returns None (so callers can assign the result back to `child_tmp`).
    """
    if not child_tmp:
        return None
    try:
        if os.path.exists(child_tmp):
            try:
                _shutil.rmtree(child_tmp)
            except Exception:
                safe_log(
                    logger,
                    "warning",
                    "Failed to remove short working dir: %s",
                    child_tmp,
                )
        safe_log(logger, "debug", "Removed short working dir: %s", child_tmp)
    except Exception:
        safe_log(logger, "warning", "Error while cleaning short working dir: %s", child_tmp)
    return None


def _move_working_to_output(working_output: str, output_dir: str, logger: Any) -> bool:
    """Move artifacts from working_output to output_dir. Returns True on success."""
    if not working_output or working_output == output_dir:
        return True

    try:
        # Diagnostic listing before move
        if not os.path.exists(working_output):
            safe_log(
                logger,
                "warning",
                "Expected working_output does not exist: %s",
                working_output,
            )
            return False

        try:
            sample = os.listdir(working_output)
            safe_log(
                logger,
                "debug",
                "Working output dir %s contents (sample): %s",
                working_output,
                sample[:20],
            )
        except Exception:
            safe_log(
                logger,
                "debug",
                "Could not list working_output contents: %s",
                working_output,
            )

        if os.path.exists(output_dir):
            _shutil.rmtree(output_dir)
        _shutil.copytree(working_output, output_dir)
        _shutil.rmtree(working_output)
        return True
    except Exception as move_exc:
        safe_log(
            logger,
            "warning",
            "Failed to move v2 export from working_output %s -> %s: %s",
            working_output,
            output_dir,
            move_exc,
        )
        return False


def _sanitize_kwargs_for_logging(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive tokens/auth values from kwargs dict for logging."""
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        try:
            k_lower = k.lower()
            if "token" in k_lower or "auth" in k_lower:
                out[k] = "<redacted>"
            else:
                out[k] = v
        except Exception:
            out[k] = "<error>"
    return out


def _setup_export_environment() -> dict[str, Optional[str]]:
    """Set thread limits for export and return old environment dict."""
    old_env: dict[str, Optional[str]] = {}
    for k in (
        "TMP",
        "TEMP",
        "TMPDIR",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        old_env[k] = os.environ.get(k)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    return old_env


def _restore_export_environment(old_env: dict[str, Optional[str]]) -> None:
    """Restore environment from saved dict."""
    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _patch_config_for_optimum(config: Any, model_name: str, logger: Any) -> None:
    """Patch config with expected attributes for optimum export."""
    if config is None:
        return
    if not hasattr(config, "is_transformers_support_available"):
        config.is_transformers_support_available = True
    if not hasattr(config, "is_torch_support_available"):
        config.is_torch_support_available = True
    if not hasattr(config, "values_override"):
        config.values_override = None
    safe_log(
        logger,
        "debug",
        "Patched config support flags for %s: transformers=%s torch=%s",
        model_name,
        getattr(config, "is_transformers_support_available", None),
        getattr(config, "is_torch_support_available", None),
    )


def _setup_working_directory(output_dir: str, use_external_data_format: bool, logger: Any) -> tuple[str, Optional[str]]:
    """Setup working directory for Windows long-path scenarios.

    Returns (working_output, child_tmp).
    """
    working_output = output_dir
    child_tmp = None
    try:
        is_windows = (os.name == "nt") or (_sys.platform and _sys.platform.startswith("win"))
        if is_windows and bool(use_external_data_format):
            try:
                abs_out = os.path.abspath(output_dir)
                threshold = 190
                tempdir = _tempfile.gettempdir()
                different_device = False
                try:
                    out_drive = os.path.splitdrive(abs_out)[0].lower()
                    temp_drive = os.path.splitdrive(os.path.abspath(tempdir))[0].lower()
                    different_device = out_drive != temp_drive
                except Exception:
                    different_device = False

                if (len(abs_out) > threshold) or different_device:
                    child_tmp = _tempfile.mkdtemp(prefix="onnx_working_", dir=_tempfile.gettempdir())
                    working_output = child_tmp
                    safe_log(
                        logger,
                        "info",
                        "Using short system temp working_output: %s",
                        working_output,
                    )
            except Exception:
                pass
    except Exception:
        child_tmp = None
    return working_output, child_tmp


def _build_fallback_strategies(err: str, trust_remote_code: bool) -> list[tuple[str, dict[str, Any]]]:
    """Build prioritized fallback strategies based on error patterns.

    Returns list of (name, kwargs) tuples.
    """
    fallbacks: list[tuple[str, dict[str, Any]]] = []

    def has_error(*patterns: str) -> bool:
        return any(p in err for p in patterns)

    # 1. Remote-code requirement (highest priority)
    if not trust_remote_code and has_error(
        "requires you to execute",
        "trust_remote_code",
        "execute the configuration",
    ):
        fallbacks.append(("trust_remote_code", {"trust_remote_code": True}))

    # 2. Proto/encode errors -> opset fallback
    if has_error("failed to serialize proto", "encodeerror"):
        fb_opset = 14 if "scaled_dot_product_attention" in err else 11
        fallbacks.append((f"opset_{fb_opset}", {"opset": fb_opset}))

    # 3. Post-processing failures -> skip post-process & reduce memory
    if has_error(
        "post-processing of the onnx export failed",
        "remove_duplicate_weights_from_tied_info",
        "numpy_helper.to_array",
        "deduplicate_gather_matmul",
    ):
        fallbacks.append(
            (
                "no_post_process_reduce",
                {
                    "no_post_process": True,
                    "dtype": "float16",
                    "use_external_data_format": True,
                },
            )
        )

    # 4. Memory errors -> lower precision AND reduce batch size
    if has_error("memoryerror", "out of memory", "memory error", "not enough memory", "alloc_cpu"):
        fallbacks.append(
            (
                "dtype_float16_batch1",
                {
                    "dtype": "float16",
                    "use_external_data_format": True,
                },
            )
        )

    # 5. Conservative combined fallback
    if not any(name.startswith("opset_") for name, _ in fallbacks):
        fallbacks.append(
            (
                "final_opset11_float16_noext",
                {
                    "opset": 11,
                    "dtype": "float16",
                    "use_external_data_format": True,
                },
            )
        )

    # 6. Local/clone strategies
    if trust_remote_code:
        fallbacks.append(("local_path", {"__use_local__": True}))
        fallbacks.append(("clone_repo", {"__clone__": True}))
        if has_error("model type", "does not recognize", "not recognize", "model_type"):
            fallbacks.append(("snapshot_retry", {"__snapshot__": True}))

    return fallbacks


def _attempt_inprocess_export(
    me_kwargs: dict,
    working_output: str,
    output_dir: str,
    child_tmp: Optional[str],
    logger: Any,
) -> tuple[bool, str, Optional[str]]:
    """Attempt in-process ONNX export with environment setup.

    Returns (success, stderr, child_tmp) tuple.
    """
    import gc

    gc.collect()

    ok: bool = False
    stderr: str = ""
    captured_exception: Exception | None = None

    try:
        old_env = _setup_export_environment()
        try:
            buf_err = _io.StringIO()
            try:
                try:
                    logger.info(
                        "main_export final_kwargs: %s",
                        _sanitize_kwargs_for_logging(me_kwargs),
                    )
                except Exception:
                    logger.debug("Failed to log sanitized kwargs", exc_info=True)

                # reuse the outer `captured_exception` variable
                with _contextlib.redirect_stderr(buf_err):
                    try:
                        from optimum.exporters.onnx import main_export as _main_export
                    except ImportError as import_err:
                        captured_exception = import_err
                        _main_export = None

                    if _main_export is not None:
                        try:
                            me_kwargs["force_download"] = True
                            # me_kwargs["use_subprocess"] = None

                            # For diagnostic runs we invoke the exporter synchronously
                            # so that stdout/stderr are propagated directly into the
                            # surrounding stderr redirect (this makes native crashes
                            # and tracebacks visible in our captured buffer).
                            safe_log(
                                logger,
                                "info",
                                "Starting in-process main_export synchronously for diagnostics",
                            )
                            try:
                                # If the caller passed the CLI flag `-low-memory-env` (becomes
                                # `low_memory_env` in `me_kwargs`), apply conservative flags
                                # unconditionally. Otherwise fall back to a local-directory
                                # size check using `large_model_threshold_gb` (default 3.0 GB).
                                try:
                                    low_mem_flag = me_kwargs.get("low_memory_env", False)
                                    if isinstance(low_mem_flag, str):
                                        low_mem_flag = low_mem_flag.strip().lower() in (
                                            "1",
                                            "true",
                                            "yes",
                                        )
                                    if low_mem_flag:
                                        safe_log(
                                            logger,
                                            "info",
                                            "low-memory flag detected via CLI; applying large-export flags",
                                        )
                                        me_kwargs.update(
                                            {
                                                "use_external_data_format": True,
                                                "external_data_size_threshold": 1024,
                                                "optimize": False,
                                                "use_subprocess": False,
                                                "no_subprocess": True,
                                                "disable_onnx_constant_folding": True,
                                                "safe_serialization": True,
                                            }
                                        )
                                    else:
                                        threshold_gb = float(me_kwargs.get("large_model_threshold_gb", 3.0))
                                        threshold_bytes = int(threshold_gb * 1024**3)

                                        total_size = 0
                                        try:
                                            model_path = me_kwargs.get("model_name_or_path")
                                            if isinstance(model_path, str) and os.path.exists(model_path) and os.path.isdir(model_path):
                                                for _root, _dirs, files in os.walk(model_path):
                                                    for fname in files:
                                                        try:
                                                            fp = os.path.join(_root, fname)
                                                            total_size += os.path.getsize(fp)
                                                        except Exception:
                                                            pass
                                        except Exception as e:
                                            safe_log(
                                                logger,
                                                "warning",
                                                "Could not compute model directory size: %s",
                                                e,
                                            )

                                        if total_size >= threshold_bytes:
                                            safe_log(
                                                logger,
                                                "info",
                                                "Detected large model (%.2f GB); applying large-export flags",
                                                total_size / (1024**3),
                                            )
                                            me_kwargs.update(
                                                {
                                                    "use_external_data_format": True,
                                                    "external_data_size_threshold": 1024,
                                                    "optimize": False,
                                                    "use_subprocess": False,
                                                    "no_subprocess": True,
                                                    "disable_onnx_constant_folding": True,
                                                    "safe_serialization": True,
                                                }
                                            )
                                        else:
                                            safe_log(
                                                logger,
                                                "debug",
                                                "Model size %.2f GB < %.2f GB; skipping large-export flags",
                                                (total_size / (1024**3) if total_size else 0.0),
                                                threshold_gb,
                                            )
                                except Exception:
                                    safe_log(
                                        logger,
                                        "warning",
                                        "Error while evaluating large-model flags; skipping.",
                                    )
                                _main_export(**me_kwargs)
                                try:
                                    safe_log(
                                        logger,
                                        "info",
                                        "In-process main_export completed for model=%s output=%s",
                                        me_kwargs.get("model_name_or_path"),
                                        me_kwargs.get("output"),
                                    )
                                except Exception:
                                    pass
                                ok = True
                            except Exception as e_main:
                                import traceback

                                tb_main = traceback.format_exc()

                                try:
                                    safe_log(
                                        logger,
                                        "error",
                                        "In-process main_export raised exception: %s",
                                        repr(e_main),
                                    )
                                except Exception:
                                    logger.error("In-process main_export raised an exception (repr failed)")

                                try:
                                    safe_log(
                                        logger,
                                        "debug",
                                        "In-process main_export traceback:\n%s",
                                        tb_main,
                                    )
                                except Exception:
                                    pass

                                captured_exception = e_main
                                stderr = (
                                    (stderr + "\n\nIn-process main_export traceback:\n" + tb_main)
                                    if stderr
                                    else ("In-process main_export traceback:\n" + tb_main)
                                )

                                # Attempt fallback exporter if available
                                try:
                                    export_onnx_fallback(me_kwargs)
                                    ok = True
                                except Exception as fb_exc:
                                    captured_exception = fb_exc
                                    ok = False
                                else:
                                    ok = True
                        except Exception as main_exc:
                            captured_exception = main_exc
                            try:
                                export_onnx_fallback(me_kwargs)
                                ok = True
                            except Exception:
                                ok = False
                    else:
                        try:
                            export_onnx_fallback(me_kwargs)
                            ok = True
                        except Exception as fb_exc:
                            captured_exception = fb_exc
                            ok = False

            except Exception as outer_exc:
                if not captured_exception:
                    captured_exception = outer_exc
                ok = False
                pass

            stderr = buf_err.getvalue()

            if not ok and captured_exception:
                logger.error("main_export raised exception during in-process export:")
                logger.error("%s: %s", type(captured_exception).__name__, str(captured_exception))
                logger.error("Full traceback available in debug logs")

            if not ok and captured_exception:
                stderr = (
                    f"{stderr}\n\nCaptured exception:\n{type(captured_exception).__name__}: {str(captured_exception)}"
                    if stderr
                    else f"{type(captured_exception).__name__}: {str(captured_exception)}"
                )

            if ok and child_tmp:
                _move_working_to_output(working_output, output_dir, logger)
                child_tmp = _cleanup_child_tmp(child_tmp, logger)
        finally:
            _restore_export_environment(old_env)
    except Exception as e:
        tb = None
        try:
            import traceback

            tb = traceback.format_exc()
        except Exception:
            pass

        exception_msg = str(e) if e is not None else "<exception during in-process export>"
        stderr = f"{stderr}\n\n{exception_msg}" if stderr else exception_msg
        safe_log(
            logger,
            "warning",
            "In-process main_export raised exception: %s",
            exception_msg,
        )
        if tb:
            safe_log(logger, "debug", "Traceback: %s", tb)

    return ok, stderr, child_tmp


def _execute_fallback_loop(
    me_kwargs: dict,
    stderr: str,
    export_source: str,
    working_output: str,
    output_dir: str,
    trust_remote_code: bool,
    logger: Any,
) -> tuple[bool, bool]:
    """Execute fallback strategies based on error patterns.

    Returns (success, used_trust_remote_code) tuple.
    """
    err: str = (stderr or "main_export subprocess failed").lower()
    safe_log(logger, "warning", "main_export failed (subprocess): %s", stderr)

    try:
        fallbacks = _build_fallback_strategies(err, trust_remote_code)

        for fb_name, fb_kwargs in fallbacks:
            safe_log(
                logger,
                "info",
                "Attempting fallback '%s' with kwargs %s",
                fb_name,
                fb_kwargs,
            )

            me_try, cleanup, prep_err = _prepare_strategy(me_kwargs, fb_name, fb_kwargs, export_source, logger)
            if me_try is None:
                safe_log(logger, "info", "Skipping fallback '%s': %s", fb_name, prep_err)
                continue

            try:
                safe_log(
                    logger,
                    "info",
                    "Subprocess fallback '%s' trust_remote_code=%s",
                    fb_name,
                    bool(me_try.get("trust_remote_code", False)),
                )
                ok_fb, stderr_fb = _run_main_export_subprocess(me_try, logger)
            finally:
                if cleanup:
                    try:
                        cleanup()
                    except Exception:
                        pass

            if ok_fb:
                _move_working_to_output(working_output, output_dir, logger)
                safe_log(
                    logger,
                    "info",
                    "v2 main_export (fallback: %s) saved artifacts to %s",
                    fb_name,
                    output_dir,
                )
                return True, bool(me_try.get("trust_remote_code", trust_remote_code))

            safe_log(logger, "warning", "Fallback '%s' failed: %s", fb_name, stderr_fb)

        safe_log(logger, "info", "All configured fallbacks exhausted")
    except Exception:
        safe_log(logger, "debug", "Fallback loop failed", exc_info=True)

    return False, False


def export_v2_main_export(
    export_source: str,
    output_dir: str,
    model_for: str,
    opset_version: Optional[int],
    device: str,
    task: Optional[str],
    framework: Optional[str],
    library: Optional[str],
    trust_remote_code: bool,
    logger: Any,
    auth_token: Optional[str] = None,
    use_auth_token: Optional[bool] = None,
    use_external_data_format: bool = False,
    no_post_process: bool = False,
    merge: bool = False,
    use_subprocess: bool = False,
) -> tuple[bool, bool]:
    """Run optimum.exporters.onnx.main_export with sensible retries.

    Returns (success: bool, used_trust_remote_code: bool).
    """
    safe_log(
        logger,
        "info",
        "export_v2_main_export start: export_source=%s output_dir=%s model_for=%s opset=%s device=%s task=%s "
        "framework=%s library=%s trust_remote_code=%s",
        export_source,
        output_dir,
        model_for,
        opset_version,
        device,
        task,
        framework,
        library,
        trust_remote_code,
    )

    try:
        cleanup_temporary_export_artifacts(logger=logger, max_age_seconds=3600)
    except Exception:
        safe_log(logger, "debug", "Pre-export temp cleanup failed", exc_info=True)
    logger.info(
        "Using v2 main_export for %s (opset=%s device=%s)",
        export_source,
        opset_version,
        device,
    )

    export_task = task
    export_library = library

    opset = opset_version or get_default_opset()
    safe_log(logger, "debug", "Selected opset for v2: %s", opset)

    me_kwargs: dict[str, Any] = {
        "model_name_or_path": export_source,
        "output": output_dir,
        "task": export_task,
        "opset": opset,
        "device": device or "cpu",
        "use_external_data_format": bool(use_external_data_format),
        "use_subprocess": use_subprocess,
    }
    try:
        auto_cache = model_for in ["llm"]
        try:
            if auto_cache and str(task or "").lower().startswith("text-generation"):
                me_kwargs["use_cache"] = True
                if merge and str(task or "").lower().endswith("-with-past"):
                    me_kwargs["use_merged"] = True
                    me_kwargs["file_name"] = "model.onnx"
        except Exception:
            pass

    except Exception:
        pass
    if no_post_process:
        me_kwargs["no_post_process"] = True

    working_output, child_tmp = _setup_working_directory(output_dir, use_external_data_format, logger)
    if working_output != output_dir:
        me_kwargs["output"] = working_output

    if use_auth_token:
        me_kwargs["use_auth_token"] = True
    if auth_token:
        me_kwargs["use_auth_token"] = auth_token

    safe_log(
        logger,
        "debug",
        "Prepared main_export kwargs (pre-trust/library): %s",
        dict(me_kwargs),
    )

    if not framework:
        framework = "pt"
        safe_log(logger, "debug", "Framework not specified, defaulting to 'pt'")

    if framework:
        me_kwargs["framework"] = framework
    if export_library:
        me_kwargs["library"] = export_library

    if trust_remote_code:
        me_kwargs["trust_remote_code"] = True

    try:
        ok, stderr, child_tmp = _attempt_inprocess_export(me_kwargs, working_output, output_dir, child_tmp, logger)

        if not use_subprocess:
            if ok:
                child_tmp = _cleanup_child_tmp(child_tmp, logger)
                return True, bool(trust_remote_code)
            else:
                safe_log(
                    logger,
                    "error",
                    "In-process export failed for %s (subprocess fallback disabled). Error: %s",
                    export_source,
                    stderr[:500] if stderr else "<no error details captured>",
                )
            child_tmp = _cleanup_child_tmp(child_tmp, logger)
            return False, False

        if not ok:
            if use_subprocess:
                ok, stderr = _run_main_export_subprocess(me_kwargs, logger)
            else:
                safe_log(
                    logger,
                    "info",
                    "Subprocess fallback not allowed; skipping subprocess run",
                )
        if ok:
            _move_working_to_output(working_output, output_dir, logger)
            safe_log(logger, "info", "v2 main_export saved artifacts to %s", output_dir)
            return True, bool(trust_remote_code)

        ok_fb, used_trust = _execute_fallback_loop(
            me_kwargs,
            stderr,
            export_source,
            working_output,
            output_dir,
            trust_remote_code,
            logger,
        )
        if ok_fb:
            return True, used_trust

        safe_log(
            logger,
            "error",
            "v2 main_export failed completely for %s. Stderr: %s",
            export_source,
            stderr[:1000] if stderr else "<no stderr captured>",
        )

        child_tmp = _cleanup_child_tmp(child_tmp, logger)
        return False, False
    except Exception:
        safe_log(logger, "exception", "Unexpected exception in export_v2_main_export")
        return False, False
    finally:
        try:
            prefixes = ("onnx_out_", "onnx_export_", "onnx_opt_clone_", "onnx_working_")
            cleanup_temporary_export_artifacts(logger=logger, prefixes=prefixes, max_age_seconds=300)
        except Exception:
            safe_log(logger, "debug", "Fallback export temp cleanup failed", exc_info=True)


def _run_post_optimization_validator(
    output_dir: str,
    expected: List[str],
    model_name: str,
    pack_single_file: bool,
    pack_single_threshold_mb: int | None,
    trust_remote_code: bool,
    used_trust_remote: bool,
    normalize_embeddings: bool,
    logger: Any,
    skip_validator: bool,
) -> int:
    """Run validator after optimization and handle results."""
    if skip_validator:
        return 0

    try:
        post_quick_ok = bool(
            verify_models(
                expected,
                output_dir,
                pack_single=pack_single_file,
                pack_single_threshold_mb=pack_single_threshold_mb,
            )
        )
        if post_quick_ok:
            logger.info(
                "Post-optimization quick verification passed for %s",
                output_dir,
            )
        else:
            logger.warning(
                "Post-optimization quick verification failed for %s",
                output_dir,
            )
    except Exception as v_err:
        post_quick_ok = False
        logger.warning(
            "Post-optimization quick verification raised: %s",
            v_err,
        )

    if not post_quick_ok:
        return 1

    try:
        post_rc, _ = invoke_validator(
            output_dir=output_dir,
            expected=expected,
            model_name=model_name,
            pack_single_file=pack_single_file,
            pack_single_threshold_mb=pack_single_threshold_mb,
            trust_remote_code=trust_remote_code or used_trust_remote,
            normalize_embeddings=normalize_embeddings,
            logger=logger,
        )

        if post_rc != 0:
            logger.warning(
                "Post-optimization numeric validator returned non-zero code: %s",
                post_rc,
            )
            _write_validation_marker(output_dir, post_rc, model_name, is_post_opt=True)
        else:
            _remove_validation_marker(output_dir)

        return post_rc
    except Exception as post_e:
        logger.exception(
            "Post-optimization validator invocation failed: %s",
            post_e,
        )
        return 1


def _run_export_with_fallback(
    export_source: str,
    output_dir: str,
    model_for: str,
    opset_version: int | None,
    device: str,
    task: str | None,
    framework: str | None,
    library: str | None,
    logger: Any,
    trust_remote_code: bool,
    use_external_data_format: bool = False,
    no_post_process: bool = False,
    merge: bool = False,
    use_subprocess: bool = False,
    use_fallback_if_failed: bool = False,
) -> tuple[bool, bool]:
    """Try v2 export with opset retry and trust_remote_code fallback.

    Returns a tuple (export_succeeded, used_trust_remote).
    """
    export_succeeded: bool = False
    used_trust_remote: bool = False

    try:
        logger.info(
            "_run_export_with_fallback start: source=%s output_dir=%s model_for=%s opset=%s device=%s task=%s "
            "framework=%s library=%s trust_remote_code=%s use_fallback_if_failed=%s",
            export_source,
            output_dir,
            model_for,
            opset_version,
            device,
            task,
            framework,
            library,
            trust_remote_code,
            use_fallback_if_failed,
        )
    except Exception:
        try:
            logger.info("_run_export_with_fallback start")
        except Exception:
            pass

    # Attempt v2 export via helper (returns (success, used_trust_remote)).
    try:
        logger.info("Attempting v2 main_export via export_v2_main_export")
        v2_ok, v2_used_trust = export_v2_main_export(
            export_source,
            output_dir,
            model_for,
            opset_version,
            device,
            task,
            framework,
            library,
            trust_remote_code,
            logger,
            use_external_data_format=use_external_data_format,
            no_post_process=no_post_process,
            merge=merge,
            use_subprocess=use_subprocess,
        )
        if v2_ok:
            export_succeeded = True
            logger.info("v2 main_export succeeded")
        if v2_used_trust:
            used_trust_remote = True
        if not v2_ok:
            logger.info("v2 main_export did not succeed")
    except Exception as import_err:
        logger.warning("v2 exporter wrapper failed to import or execute: %s", import_err)

    # Final compatibility fallback: legacy v1-style exporter path.
    if not export_succeeded and use_fallback_if_failed:
        try:
            logger.info("Attempting legacy fallback exporter after v2 failures")
            legacy_ok, legacy_used_trust = run_legacy_v1_fallback(
                export_source,
                output_dir,
                model_for,
                opset_version,
                device,
                task,
                framework,
                library,
                logger,
                trust_remote_code,
                use_external_data_format=use_external_data_format,
                no_post_process=no_post_process,
                merge=merge,
                use_subprocess=use_subprocess,
            )
            if legacy_ok:
                export_succeeded = True
                logger.info("Legacy fallback exporter succeeded")
            if legacy_used_trust:
                used_trust_remote = True
            if not legacy_ok:
                logger.info("Legacy fallback exporter did not succeed")
        except Exception as legacy_err:
            logger.warning("Legacy fallback exporter failed to execute: %s", legacy_err)
    elif not export_succeeded:
        logger.info("Legacy fallback exporter disabled by --use-fallback-if-failed")

    # After successful export, run ONNX sanitizer to deduplicate tied initializers
    try:
        if export_succeeded:
            sanitize_onnx_initializers: Optional[Callable[..., int]] = None
            try:
                from model_exporter.export.helpers import sanitize_onnx_initializers as _sanitize_onnx_initializers

                sanitize_onnx_initializers = _sanitize_onnx_initializers
            except Exception:
                sanitize_onnx_initializers = None
            if sanitize_onnx_initializers is not None:
                try:
                    modified = sanitize_onnx_initializers(output_dir, logger)
                    if modified:
                        logger.info("Sanitized %d ONNX files after export", modified)
                except Exception:
                    logger.debug("sanitize_onnx_initializers failed", exc_info=True)
    except Exception:
        logger.debug("Post-export sanitizer handling failed", exc_info=True)

    try:
        logger.info(
            "_run_export_with_fallback result: export_succeeded=%s used_trust_remote=%s",
            export_succeeded,
            used_trust_remote,
        )
    except Exception:
        pass

    return export_succeeded, used_trust_remote
