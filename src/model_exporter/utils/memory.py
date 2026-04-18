# =============================================================================
# File: memory_utils.py
# Date: 2026-01-16
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

"""Memory monitoring and cleanup utilities for ONNX export operations."""

import gc
import logging
from typing import Any

import psutil

logger = logging.getLogger(__name__)


def get_memory_info() -> dict:
    """Get current memory usage information.

    Returns:
        dict with total_gb, used_gb, free_gb, percent_used
    """
    try:
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "free_gb": round(mem.available / (1024**3), 2),
            "percent_used": round(mem.percent, 1),
        }
    except Exception as e:
        logger.warning(f"Failed to get memory info: {e}")
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent_used": 0}


def log_memory_status(context: str = "Memory", level: int = logging.INFO) -> None:
    """Log current memory status with context."""
    mem = get_memory_info()
    logger.log(
        level,
        f"[{context}] RAM: {mem['used_gb']}GB used / {mem['total_gb']}GB total "
        f"({mem['free_gb']}GB free, {mem['percent_used']}% used)",
    )


def check_memory_available(min_free_gb: float = 2.0) -> bool:
    """Check if sufficient memory is available.

    Args:
        min_free_gb: Minimum free memory in GB required

    Returns:
        True if sufficient memory available, False otherwise
    """
    mem = get_memory_info()
    if mem["free_gb"] < min_free_gb:
        logger.warning(f"Low memory: Only {mem['free_gb']}GB free (threshold: {min_free_gb}GB)")
        return False
    return True


def aggressive_cleanup() -> None:
    """Perform aggressive memory cleanup.

    This includes:
    - Multiple garbage collection passes
    - PyTorch cache clearing (if available)
    - Transformer cache clearing (if available)
    """
    logger.debug("Performing aggressive memory cleanup...")

    # Multiple GC passes
    for _ in range(3):
        gc.collect()

    # Clear PyTorch cache if available
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared PyTorch CUDA cache")
    except Exception:
        pass

    # Clear transformers cache if available
    try:
        import transformers

        # Clear any cached models in memory
        if hasattr(transformers, "model") and hasattr(transformers.model, "cache_clear"):
            transformers.model.cache_clear()
    except Exception:
        pass

    logger.debug("Memory cleanup completed")


def memory_guard(min_free_gb: float = 2.0, auto_cleanup: bool = True) -> bool:
    """Check memory and optionally perform cleanup if low.

    Args:
        min_free_gb: Minimum free memory in GB
        auto_cleanup: Automatically run cleanup if memory is low

    Returns:
        True if memory is sufficient (after cleanup if needed), False otherwise
    """
    if check_memory_available(min_free_gb):
        return True

    if auto_cleanup:
        logger.info("Low memory detected, performing automatic cleanup...")
        aggressive_cleanup()

        # Check again after cleanup
        if check_memory_available(min_free_gb):
            logger.info("Memory cleanup successful, continuing operation")
            return True
        else:
            logger.error(
                f"Insufficient memory even after cleanup. "
                f"Available: {get_memory_info()['free_gb']}GB, Required: {min_free_gb}GB"
            )
            return False

    return False


class MemoryMonitor:
    """Context manager for monitoring and guarding memory around operations.

    Logs memory usage at entry and exit, optionally runs cleanup if memory is
    low before the operation starts, and always runs cleanup on exit.

    Example::

        with MemoryMonitor("ExportModel", min_free_gb=2.0):
            run_export(...)
    """

    def __init__(self, operation_name: str = "Operation", min_free_gb: float = 2.0) -> None:
        """Initialise the monitor.

        Args:
            operation_name: Human-readable label used in log messages.
            min_free_gb: Minimum free RAM in GB required before the operation;
                triggers cleanup if not met.
        """
        self.operation_name = operation_name
        self.min_free_gb = min_free_gb
        self.start_memory: dict[str, Any] | None = None

    def __enter__(self) -> "MemoryMonitor":
        """Log start memory and run cleanup if RAM is below the threshold.

        Returns:
            The :class:`MemoryMonitor` instance itself.
        """
        self.start_memory = get_memory_info()
        log_memory_status(f"{self.operation_name} Start")

        if not check_memory_available(self.min_free_gb):
            logger.warning(f"Starting {self.operation_name} with low memory")
            aggressive_cleanup()

        return self

    def __exit__(
        self, exc_type: object | None, exc_val: object | None, exc_tb: object | None
    ) -> None:
        """Log end memory, report the delta, and run aggressive cleanup.

        Args:
            exc_type: Exception type if an error occurred, else ``None``.
            exc_val: Exception value if an error occurred, else ``None``.
            exc_tb: Traceback if an error occurred, else ``None``.
        """
        end_memory = get_memory_info()
        log_memory_status(f"{self.operation_name} End")

        # Log memory delta
        start_used = self.start_memory["used_gb"] if self.start_memory is not None else 0
        delta = end_memory["used_gb"] - start_used
        if abs(delta) > 0.1:  # Only log if significant change
            logger.info(
                f"{self.operation_name} memory delta: {delta:+.2f}GB "
                f"({start_used:.2f}GB → {end_memory['used_gb']:.2f}GB)"
            )

        # Cleanup after operation
        aggressive_cleanup()
