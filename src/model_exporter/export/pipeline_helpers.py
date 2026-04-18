# =============================================================================
# File: pipeline_helpers.py
# Date: 2026-04-17
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Model-type predicates, export lock management, HF token setup, validation
markers, memory cleanup, quantization, and per-call orchestration helpers.

Extracted from pipeline.py to reduce file length.
"""
from __future__ import annotations

import gc
import os
import shutil
import time
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, cast

from model_exporter.export.helpers import (
    cleanup_validator_logging_handlers,
    is_pid_running,
)
from model_exporter.utils.helpers import get_logger, safe_log
from model_exporter.validation.invoker import invoke_validator

logger: Any = get_logger(__name__)


def _build_expected_list(
    model_for: str, use_cache: bool, task: str | None = None, merge: bool = False
) -> List[str]:
    """Compute the list of expected ONNX artifact filenames for an export.

    For seq2seq models the encoder and decoder files are always included;
    the ``decoder_with_past_model.onnx`` is added when *use_cache* is ``True``
    or the *task* contains ``"with-past"``.
    For other model types ``model.onnx`` is always expected; a
    ``model_with_past.onnx`` companion is added only when *use_cache* and
    *merge* is ``False``.

    Args:
        model_for: Model purpose string (``"s2s"``, ``"fe"``, ``"llm"``, etc.).
        use_cache: Whether KV-cache exports are active.
        task: Optional Optimum export task string; ``"text2text-generation-with-past"``
            and ``"text-generation-with-past"`` trigger additional artifacts.
        merge: When ``True``, the with-past artifact is omitted since it was
            merged into the base model.

    Returns:
        A list of expected filename strings.
    """
    mf = (model_for or "").lower()
    t = (task or "").lower()
    # For seq2seq models we normally expect encoder+decoder files. If KV-cache
    # is requested via `use_cache` or the `task` indicates a "with-past" export,
    # include the `decoder_with_past_model.onnx` artifact in the expected list.
    if mf in ["s2s"]:
        names = ["encoder_model.onnx", "decoder_model.onnx"]
        if use_cache or (t == "text2text-generation-with-past"):
            names.append("decoder_with_past_model.onnx")
        return names
    names = ["model.onnx"]
    # When a merged export was requested (`merge=True`), the merged artifact
    # contains with-past semantics consolidated into the merged model. Do not
    # require a separate `model_with_past.onnx` file in that case.
    if not merge and (use_cache or (t == "text-generation-with-past")):
        names.append("model_with_past.onnx")
    return names


def _auto_enable_use_cache(model_name: str, model_for: str, task: str | None) -> bool:
    """
    Decide whether to enable `use_cache` automatically.

    Priority:
    1. Caller flag
    2. model_for classification (fe, s2s, llm, sc, ranker)
    3. Task name (e.g., text-generation-with-past)
    4. HF config inspection
    5. Fallback
    """

    # 2. model_for classification
    mf = model_for.lower()

    if mf in {"sc", "ranker"}:
        logger.info(
            "Model %s is used for %s; disabling use_cache",
            model_name,
            "ranking (cross-encoder)" if mf == "ranker" else "sequence classification",
        )
        return False

    if mf in {"fe"}:
        logger.info(
            "Model %s is used for feature extraction (%s); disabling use_cache",
            model_name,
            model_for,
        )
        return False

    if mf in {"s2s"}:
        logger.info(
            "Model %s is seq2seq (%s); enabling use_cache",
            model_name,
            model_for,
        )
        return True

    if mf in {"llm"}:
        logger.info(
            "Model %s is an LLM (%s); enabling use_cache",
            model_name,
            model_for,
        )
        return True

    # 3. Task-based detection
    if task:
        task_l = task.lower()
        if "with-past" in task_l:
            logger.info(
                "Task %s indicates past-key-values support; enabling use_cache",
                task,
            )
            return True

    # 4. HF config inspection (fallback)
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(str(model_name))

        is_enc_dec = getattr(cfg, "is_encoder_decoder", False)
        supports_cache = getattr(cfg, "use_cache", False)
        model_type = getattr(cfg, "model_type", "").lower()

        # Known decoder-only architectures
        decoder_only_types = {
            "gpt2",
            "gptj",
            "gpt_neo",
            "gpt_neox",
            "llama",
            "mistral",
            "falcon",
            "bloom",
            "qwen",
            "qwen2",
            "phi",
            "phi3",
            "opt",
        }

        # Decoder-only models
        if model_type in decoder_only_types and supports_cache:
            logger.info(
                "Model %s is decoder-only (%s) and supports caching; enabling use_cache",
                model_name,
                model_type,
            )
            return True

        # Encoder-decoder models (BART, T5, Pegasus)
        if is_enc_dec and supports_cache:
            logger.info(
                "Model %s is encoder-decoder and supports caching; enabling use_cache",
                model_name,
            )
            return True

        # Encoder-only models (BERT, RoBERTa, BGE-M3, etc.)
        logger.info(
            "Model %s (%s) does not support caching; disabling use_cache",
            model_name,
            model_type,
        )
        return False

    except Exception:
        return False


def _create_export_lock(output_dir: str, model_name: str, logger: Any) -> tuple[str, bool]:
    """Create a lock file for `output_dir`. Returns (lock_path, created_lock)."""
    lock_path = os.path.join(output_dir, ".export.lock")
    created_lock = False
    try:
        if os.path.exists(lock_path):
            try:
                with open(lock_path, "r", encoding="utf-8") as fh:
                    lines = [line.strip() for line in fh.readlines() if line.strip()]
                pid = int(lines[0]) if lines else None
                ts = float(lines[1]) if len(lines) > 1 else None
            except Exception:
                pid = None
                ts = None

            now = time.time()
            stale_threshold = 24 * 60 * 60
            if pid is not None and is_pid_running(pid):
                logger.info(
                    "An export for this model is already running (pid=%s). Exiting.",
                    pid,
                )
                raise SystemExit(0)
            else:
                age = (now - ts) if ts else None
                removal_needed = False

                if age is None or age > stale_threshold:
                    removal_needed = True
                else:
                    removal_needed = True

                if removal_needed:
                    try:
                        os.remove(lock_path)
                        if age is None or age > stale_threshold:
                            logger.info(
                                "Removed stale lock file (age=%.0fs): %s",
                                age if age else 0,
                                lock_path,
                            )
                        else:
                            logger.info(
                                "Found recent lock file but owner not running; removing and continuing: %s",
                                lock_path,
                            )
                    except Exception:
                        if age is not None and age <= stale_threshold:
                            logger.warning(
                                "Could not remove recent lock file; exiting to avoid race: %s",
                                lock_path,
                            )
                            raise SystemExit(0)
                        else:
                            logger.warning("Could not remove stale lock file: %s", lock_path)

        try:
            with open(lock_path, "x", encoding="utf-8") as fh:
                fh.write(f"{os.getpid()}\n{time.time()}\n{model_name}\n")
            created_lock = True
            logger.info("Acquired export lock: %s", lock_path)
        except FileExistsError:
            logger.info("Lock file created by concurrent process; exiting")
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        logger.warning("Could not create or check export lock; proceeding without exclusive lock")

    return lock_path, created_lock


@contextmanager
def _with_export_lock(output_dir: str, model_name: str, logger: Any) -> Iterator[tuple[str, bool]]:
    """Context manager that creates an export lock and removes it on exit

    Yields `(lock_path, created_lock)`.
    """
    lock_path: Optional[str] = None
    created_lock: bool = False
    try:
        lock_path, created_lock = _create_export_lock(output_dir, model_name, logger)
        yield lock_path, created_lock
    finally:
        if created_lock and lock_path and os.path.exists(lock_path):
            try:
                os.remove(lock_path)
                logger.info("Removed lock file: %s", lock_path)
            except Exception:
                logger.debug("Could not remove lock file: %s", lock_path)


def _requires_trust_remote_code_fast(model_name: str, token: str | None = None) -> bool:
    """Quickly determine whether a model requires ``trust_remote_code``.

    Attempts to load the model config without ``trust_remote_code=True`` and
    checks whether the resulting exception message contains known trigger
    strings indicating that remote code execution is required.

    Args:
        model_name: HuggingFace model ID or local path.
        token: Optional HuggingFace API token for gated models.

    Returns:
        ``True`` if the model config raises an error indicating that
        ``trust_remote_code`` must be set; ``False`` otherwise.
    """
    try:
        from transformers import AutoConfig

        _ = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=False,
            use_auth_token=token,
        )
        return False
    except Exception as e:
        msg = str(e).lower()
        triggers = [
            "trust_remote_code",
            "please set trust_remote_code",
            "set trust_remote_code",
            "requires you to execute the configuration file",
            "execute the configuration file",
            "execute the model",
            "requires executing",
            "requires running",
            "resolve_trust_remote_code",
            "custom code",
            "remote code",
            "requires you to run",
            "requires executing the configuration",
            "requires executing the model",
            "requires running the configuration",
        ]
        return any(t in msg for t in triggers)


def _write_validation_marker(
    output_dir: str, rc: int, model_name: str, is_post_opt: bool = False
) -> bool:
    """Write validation failure marker file. Returns True if successful."""
    try:
        prefix = "post_optimization_" if is_post_opt else ""
        marker = os.path.join(output_dir, ".validation_failed")
        with open(marker, "w", encoding="utf-8") as mf:
            mf.write(
                f"{prefix}validation_failed: rc={rc} ts={int(time.time())} model={model_name}\n"
            )
        marker_desc = (
            "post-optimization validation failure" if is_post_opt else "validation failure"
        )
        logger.info(f"Wrote {marker_desc} marker: %s", marker)
        return True
    except Exception:
        logger.debug("Could not write validation marker", exc_info=True)
        return False


def _remove_validation_marker(output_dir: str) -> bool:
    """Remove validation failure marker file. Returns True if successful."""
    try:
        marker = os.path.join(output_dir, ".validation_failed")
        if os.path.exists(marker):
            os.remove(marker)
            logger.info("Removed stale validation failure marker: %s", marker)
            return True
    except Exception:
        logger.debug("Could not remove validation failure marker", exc_info=True)
    return False


def _cleanup_memory_caches(logger: Any) -> None:
    """Perform pre-export memory cleanup: GC, CUDA cache, check memory."""
    try:
        gc.collect()
        try:
            import torch

            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache before export to free GPU memory")
        except Exception:
            logger.debug("No torch/CUDA available for pre-export cache clear")

        try:
            import psutil

            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            logger.info("Available memory before export: %.1f MB", available_mb)
        except Exception:
            logger.debug("psutil not available for memory check before export")
    except Exception:
        logger.debug("Pre-export memory cleanup failed", exc_info=True)


def _is_seq2seq(model_for: str) -> bool:
    """Check if model type is seq2seq."""
    return (model_for or "").lower() in ["s2s", "seq2seq-lm"]


def _is_ranker(model_for: str) -> bool:
    """Check if model type is ranker (cross-encoder)."""
    return (model_for or "").lower() in ["ranker", "ce", "cross-encoder"]


def _should_skip_validator(model_for: str, pack_single_file: bool, expected: List[str]) -> bool:
    """Determine if numeric validator should be skipped."""
    if _is_seq2seq(model_for) and not pack_single_file:
        return True
    if pack_single_file and _is_seq2seq(model_for) and "model.onnx" not in expected:
        return True
    return False


def _setup_hf_token(token: str | None, kwargs: dict, logger: Any) -> tuple[str | None, dict]:
    """Setup Hugging Face token: extract from kwargs/env, set env vars, login.

    Returns (token, flags_dict) where flags_dict tracks which env vars were set.
    """
    flags = {"set_hf": False, "set_hub": False, "login_ok": False}

    if not token:
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )

    if token:
        try:
            prev_hf = os.environ.get("HF_TOKEN")
            prev_hub = os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if prev_hf is None:
                os.environ["HF_TOKEN"] = token
                flags["set_hf"] = True
            if prev_hub is None:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = token
                flags["set_hub"] = True
        except Exception:
            logger.debug("Could not set HF token environment variables")

        # Try to login
        try:
            from huggingface_hub import login as _hf_login

            _hf_login(token=token, add_to_git_credential=False)
            flags["login_ok"] = True
            logger.info("Logged into Hugging Face hub for this session via provided token")
        except Exception:
            try:
                import huggingface_hub as _hfh

                _hfh.login(token=token, add_to_git_credential=False)
                flags["login_ok"] = True
                logger.info("Logged into Hugging Face hub for this session via provided token")
            except Exception as e:
                logger.debug("huggingface_hub.login failed: %s", e)
                logger.warning("Hugging Face login failed; will continue without token")

    return token, flags


def _teardown_hf_token(flags: dict, logger: Any) -> None:
    """Cleanup HF token env vars if login failed."""
    try:
        if not flags.get("login_ok", False):
            for env_var in ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]:
                if (env_var == "HF_TOKEN" and flags.get("set_hf")) or (
                    env_var == "HUGGINGFACE_HUB_TOKEN" and flags.get("set_hub")
                ):
                    try:
                        if env_var in os.environ:
                            del os.environ[env_var]
                    except Exception:
                        pass
    except Exception:
        logger.debug("HF token teardown failed", exc_info=True)


def _setup_tokenizer_pad_token(output_dir: str, trust_remote_code: bool, logger: Any) -> None:
    """Pre-validator tokenizer setup: ensure pad_token is set for validation.

    Some tokenizers lack pad_token which causes validator to fail.
    Prefer eos_token when available; otherwise add '[PAD]' token.
    """
    try:
        from transformers import AutoTokenizer
    except Exception:
        return

    try:
        tok = AutoTokenizer.from_pretrained(
            output_dir,
            use_fast=False,
            trust_remote_code=trust_remote_code,
        )
        pad_ok = getattr(tok, "pad_token", None)
        if pad_ok is None:
            eos = getattr(tok, "eos_token", None)
            pad_token_to_set = None
            pad_msg = ""

            if eos:
                pad_token_to_set = eos
                pad_msg = f"Set pad_token=eos_token ({eos})"
            else:
                try:
                    tok.add_special_tokens({"pad_token": "[PAD]"})
                    pad_token_to_set = "[PAD]"
                    pad_msg = "Added '[PAD]' token"
                except Exception:
                    logger.debug("Failed to add pad_token", exc_info=True)

            if pad_token_to_set:
                try:
                    if eos:
                        tok.pad_token = eos
                    logger.info("Saving tokenizer to %s", output_dir)
                    tok.save_pretrained(output_dir)
                    logger.info(
                        "Tokenizer pad_token fallback: %s and saved tokenizer",
                        pad_msg,
                    )
                except Exception:
                    logger.debug("Failed to set/save pad_token", exc_info=True)
    except Exception:
        logger.debug(
            "Could not load tokenizer for pad_token fallback",
            exc_info=True,
        )


def _run_numeric_validator(
    output_dir: str,
    expected: List[str],
    model_name: str,
    pack_single_file: bool,
    pack_single_threshold_mb: int | None,
    trust_remote_code: bool,
    used_trust_remote: bool,
    normalize_embeddings: bool,
    logger: Any,
    require_validator: bool,
) -> tuple[int, bool]:
    """Run numeric validator with all setup and result handling."""
    try:
        cleanup_validator_logging_handlers()
    except Exception:
        pass

    _setup_tokenizer_pad_token(output_dir, trust_remote_code, logger)

    logger.info(
        "Validator flags: request_trust=%s used_trust_remote=%s",
        trust_remote_code,
        used_trust_remote,
    )

    validator_rc, _ = invoke_validator(
        output_dir=output_dir,
        expected=expected,
        model_name=model_name,
        pack_single_file=pack_single_file,
        pack_single_threshold_mb=pack_single_threshold_mb,
        trust_remote_code=trust_remote_code or used_trust_remote,
        normalize_embeddings=normalize_embeddings,
        logger=logger,
    )

    if validator_rc != 0:
        logger.warning("Numeric validator returned non-zero code: %s", validator_rc)
        _write_validation_marker(output_dir, validator_rc, model_name)
    else:
        # Validator passed — cleanup
        _remove_validation_marker(output_dir)
        try:
            dumps_dir = os.path.join(output_dir, "validation_dumps")
            if os.path.exists(dumps_dir):
                shutil.rmtree(dumps_dir)
                logger.info("Removed validation_dumps directory: %s", dumps_dir)
        except Exception:
            logger.debug("Could not remove validation_dumps directory")

    if require_validator and validator_rc != 0:
        raise SystemExit(validator_rc)

    return validator_rc, False


def _check_optimized_artifacts(output_dir: str) -> bool:
    """Check if optimizer produced any optimized artifacts."""
    marker_path = os.path.join(output_dir, ".optimizations_applied")
    if os.path.exists(marker_path):
        return True

    for _root, _dirs, files in os.walk(output_dir):
        for fname in files:
            if fname == "ort_config.json":
                return True
    return False


def _resolve_use_cache(model_name: str, model_for: str, task: str | None, logger: Any) -> bool:
    """Determine whether KV-cache exports should be enabled for this model.

    Checks model_for classification, runs HuggingFace config auto-detection,
    then applies explicit task overrides (with-past enables, seq2seq disables).

    Args:
        model_name: HuggingFace model ID or local path.
        model_for: Model purpose string (e.g. ``"llm"``, ``"s2s"``).
        task: Optimum export task string; ``"with-past"`` suffix enables
            KV-cache, ``"seq2seq"`` disables it for s2s models.
        logger: Logger instance for informational messages.

    Returns:
        True if KV-cache exports should be enabled, False otherwise.
    """
    use_cache = False
    try:
        auto_flag = _auto_enable_use_cache(model_name, model_for, task)
        if auto_flag:
            logger.info("Automatically enabling use_cache for model %s", model_name)
        use_cache = bool(auto_flag)
    except Exception:
        logger.debug("Auto-detection for use_cache failed; defaulting to False")

    try:
        if task and "with-past" in str(task).lower():
            if not use_cache:
                logger.info("Enabling use_cache because task requests KV-cache: %s", task)
            use_cache = True
    except Exception:
        pass

    try:
        if task and "seq2seq" in str(task).lower() and model_for in ["s2s"]:
            if use_cache:
                logger.info("Disabling use_cache for seq2seq task (no KV-cache): %s", task)
            use_cache = False
    except Exception:
        pass

    return use_cache


def _auto_resolve_trust_remote_code(
    model_name: str, token: Optional[str], trust_remote_code: bool, logger: Any
) -> bool:
    """Auto-enable trust_remote_code when the model config requires it.

    If ``trust_remote_code`` is already True, returns True immediately.
    Otherwise performs a fast config check and auto-enables when needed.

    Args:
        model_name: HuggingFace model ID or local path.
        token: Optional HuggingFace auth token for config fetch.
        trust_remote_code: Caller-supplied trust flag.
        logger: Logger instance for informational messages.

    Returns:
        True if trust_remote_code should be enabled, False otherwise.
    """
    if trust_remote_code:
        return True
    try:
        if _requires_trust_remote_code_fast(model_name, token):
            logger.info(
                "Auto-enabling trust_remote_code for %s based on quick check", model_name
            )
            return True
    except Exception:
        logger.debug("requires_trust_remote_code_fast check failed", exc_info=True)
    return False


def _run_quantization_step(
    output_dir: str,
    quantize: Any,
    kwargs: dict,
    logger: Any,
) -> None:
    """Quantize exported ONNX files when ``quantize`` is requested.

    Walks ``output_dir`` and produces dynamic-int8 and/or FP16 variants for
    every ``.onnx`` file found. Missing optional converters (onnxruntime,
    onnxconverter-common) are tolerated and logged as warnings.

    Args:
        output_dir: Directory containing exported ``.onnx`` files.
        quantize: ``True`` enables both ``dynamic_int8`` and ``fp16``;
            a string or list selects specific type(s). ``False`` is a no-op.
        kwargs: Export kwargs dict (checked for a ``"quantize"`` override).
        logger: Logger instance for progress and warning messages.
    """
    try:
        qparam = quantize if quantize is not None else kwargs.get("quantize", False)
        qtypes: list[str] = []
        if qparam is True:
            qtypes = ["dynamic_int8", "fp16"]
        elif isinstance(qparam, str):
            qtypes = [qparam]
        elif isinstance(qparam, (list, tuple, set)):
            qtypes = list(qparam)
        qtypes = [str(x).lower() for x in qtypes]

        if not qtypes:
            return

        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except Exception:
            quantize_dynamic = None
            QuantType = None

        try:
            import importlib

            _oc = importlib.import_module("onnxconverter_common")
            convert_float_to_float16 = getattr(_oc, "convert_float_to_float16", None)
        except Exception:
            convert_float_to_float16 = None

        for root, _dirs, files in os.walk(output_dir):
            for fname in files:
                if not fname.endswith(".onnx"):
                    continue
                src = os.path.join(root, fname)

                if "dynamic_int8" in qtypes:
                    if quantize_dynamic is not None and QuantType is not None:
                        try:
                            dst = os.path.join(root, fname.replace(".onnx", ".dynamic-int8.onnx"))
                            safe_log(logger, "info", "Quantizing (dynamic int8): %s -> %s", src, dst)
                            quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
                            safe_log(logger, "info", "Dynamic int8 artifact written: %s", dst)
                        except Exception as q_e:
                            safe_log(logger, "warning", "Dynamic quantization failed for %s: %s", src, q_e)
                    else:
                        safe_log(logger, "warning", "quantize_dynamic not available; skipping dynamic_int8 for %s", src)

                if "fp16" in qtypes:
                    if convert_float_to_float16 is not None:
                        try:
                            dst16 = os.path.join(root, fname.replace(".onnx", ".fp16.onnx"))
                            safe_log(logger, "info", "Converting to FP16: %s -> %s", src, dst16)
                            try:
                                convert_float_to_float16(src, dst16)
                            except TypeError:
                                model_proto = convert_float_to_float16(src)
                                import onnx as _onnx_mod

                                cast(_onnx_mod, Any).save(model_proto, dst16)
                            safe_log(logger, "info", "FP16 artifact written: %s", dst16)
                        except Exception as fp_e:
                            safe_log(logger, "warning", "FP16 conversion failed for %s: %s", src, fp_e)
                    else:
                        safe_log(logger, "warning", "FP16 converter not available; skipping FP16 for %s", src)
    except Exception:
        safe_log(logger, "debug", "Quantization step failed", exc_info=True)


def _lift_temp_local_artifacts(output_dir: str, logger: Any) -> None:
    """Move artifacts from ``output_dir/temp_local/`` into ``output_dir``.

    Called after an LLM export that used a local snapshot sub-folder to
    promote the artifacts to their expected location and remove the now-empty
    staging directory.

    Args:
        output_dir: Root export output directory.
        logger: Logger instance for progress and warning messages.
    """
    try:
        temp_local_dir = os.path.join(output_dir, "temp_local")
        if not (os.path.exists(temp_local_dir) and os.path.isdir(temp_local_dir)):
            return

        moved_any = False
        for name in os.listdir(temp_local_dir):
            src = os.path.join(temp_local_dir, name)
            dst = os.path.join(output_dir, name)
            try:
                if os.path.exists(dst):
                    safe_log(logger, "info", "Skipping move since target exists: %s", dst)
                    continue
                shutil.move(src, dst)
                moved_any = True
                safe_log(logger, "info", "Moved artifact from temp_local: %s -> %s", src, dst)
            except Exception as mv_e:
                safe_log(logger, "warning", "Failed to move %s -> %s: %s", src, dst, mv_e)

        try:
            if moved_any and os.path.exists(temp_local_dir):
                shutil.rmtree(temp_local_dir)
                safe_log(logger, "info", "Removed temp_local folder: %s", temp_local_dir)
        except Exception:
            safe_log(logger, "debug", "Failed to remove temp_local folder", exc_info=True)
    except Exception:
        safe_log(logger, "debug", "Lifting artifacts from temp_local failed", exc_info=True)
