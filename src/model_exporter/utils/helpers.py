# =============================================================================
# File: onnx_helpers.py
# Date: 2026-01-09
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def get_logger(name: str) -> logging.Logger:
    """Return a named :class:`logging.Logger` pre-configured with a stream handler.

    The log level defaults to ``INFO`` and can be overridden with ``LOG_LEVEL``.
    Calling this function multiple times with the same *name* is safe — the
    handler is only attached once.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(fmt)
        handler.setLevel(level)
        logger.addHandler(handler)
        logger.setLevel(level)
    else:
        # Ensure existing handlers and logger respect the desired level
        try:
            logger.setLevel(level)
            for h in logger.handlers:
                try:
                    h.setLevel(level)
                except Exception:
                    pass
        except Exception:
            pass
    return logger


def get_preferred_provider(default: str = "CPUExecutionProvider") -> str:
    """Get the default ONNX Runtime provider."""
    return default


def get_default_opset(default: int = 17) -> int:
    """Get the default ONNX opset version."""
    return default


def create_ort_session(path: str, provider: Optional[str] = None, retries: int = 2, backoff: float = 1.0) -> Any:
    """Create an ONNX Runtime InferenceSession with simple retry/backoff.

    Args:
        path: Path to ONNX model file
        provider: Execution provider (e.g., 'CPUExecutionProvider')
        retries: Number of retry attempts
        backoff: Backoff multiplier for retries

    Returns:
        ort.InferenceSession: Configured ONNX Runtime session

    Raises:
        RuntimeError: If onnxruntime is not available
        FileNotFoundError: If model file doesn't exist
        Exception: Last exception if all attempts fail
    """
    logger = get_logger("onnx_helpers")

    if ort is None:
        raise RuntimeError("onnxruntime is not available in the environment")

    # Validate model file exists
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {path}")

    if not model_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    providers = ort.get_available_providers()
    if provider is None or provider not in providers:
        provider = get_preferred_provider()
        if provider not in providers:
            provider = "CPUExecutionProvider"

    last_exc = None
    for attempt in range(retries + 1):
        try:
            logger.debug(
                "Creating ORT session for %s with provider=%s (attempt=%d)",
                model_path.name,  # Log only filename for security
                provider,
                attempt + 1,
            )
            sess = ort.InferenceSession(str(model_path), providers=[provider])
            return sess
        except Exception as e:
            last_exc = e
            logger.warning("ORT session creation failed (attempt %d): %s", attempt + 1, str(e))
            if attempt < retries:
                time.sleep(backoff * (1 + attempt))

    # If we reach here, re-raise the last exception
    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to create ORT session after all attempts")


def safe_log(logger: Any, level: str, message: str, *args: Any, **kwargs: Any) -> bool:
    """Safely call a logger method with try/except protection.

    Consolidates the common pattern of:
        try:
            logger.info("message", args)
        except Exception:
            pass

    Args:
        logger: Logger instance (or None for no-op)
        level: Log level ("debug", "info", "warning", "error")
        message: Log message format string
        *args: Positional arguments for message formatting
        **kwargs: Additional kwargs (e.g., exc_info=True)

    Returns:
        True if logging succeeded, False otherwise
    """
    if logger is None:
        return False

    try:
        method = getattr(logger, level.lower(), None)
        if method and callable(method):
            method(message, *args, **kwargs)
            return True
    except Exception:
        pass
    return False
