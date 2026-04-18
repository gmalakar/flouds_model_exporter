#!/usr/bin/env python3
# =============================================================================
# File: logging.py
# Date: 2026-01-09
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Export logging helpers extracted from `onnx_exporter.py`.

Provides setup/teardown functions that configure a per-export rotating
file logger and tee stdout/stderr to the logfile for easier debugging.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Tuple, cast


class Tee:
    """File-like object that duplicates writes to multiple underlying streams.

    Used to tee ``sys.stdout`` and ``sys.stderr`` to a log file while keeping
    the original console output intact.
    """

    def __init__(self, *streams: Any) -> None:
        """Initialise Tee with one or more output streams.

        Args:
            *streams: Writable file-like objects that will each receive a copy
                of every :meth:`write` call.
        """
        self.streams = streams

    def write(self, data: str) -> None:
        """Write *data* to all underlying streams, ignoring per-stream errors.

        Args:
            data: The string to write.
        """
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self) -> None:
        """Flush all underlying streams, ignoring per-stream errors."""
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        """Return ``True`` if any underlying stream reports itself as a TTY."""
        for s in self.streams:
            try:
                if s and getattr(s, "isatty", lambda: False)():
                    return True
            except Exception:
                pass
        return False


def setup_export_logging(
    base_dir: str,
    safe_model: str,
    rev_tag: str,
    logger: logging.Logger,
    log_to_file: bool = False,
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    If `log_to_file` is True, log to file and print the log file path in terminal.
    If `log_to_file` is False, log only to terminal (stdout/stderr), no file is created.
    Returns: (file_handler, logfile_fd, old_stdout, old_stderr, logfile_path or None)
    """
    import os
    import time

    file_handler = None
    logfile_fd = None
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    logfile = None

    if log_to_file:
        logs_dir = Path(base_dir).parent / "logs" / "onnx_exports"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        logfile = logs_dir / f"{safe_model}_{rev_tag}_{ts}.log"
        file_handler = logging.handlers.RotatingFileHandler(str(logfile), maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler.setFormatter(fmt)
        env = os.getenv("FLOUDS_API_ENV", "").strip()
        log_level = logging.DEBUG if env.lower() == "development" else logging.INFO
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        logger.propagate = False
        logger.info("Logging to file: %s", logfile)
        print(f"[INFO] Logging to file: {logfile}")
        try:
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
        except Exception:
            pass
        logfile_fd = open(logfile, "a", encoding="utf-8")
        sys.stdout = cast(Any, Tee(sys.stdout, logfile_fd))
        sys.stderr = cast(Any, Tee(sys.stderr, logfile_fd))
    else:
        # Remove all file handlers, ensure only console logging
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
        logger.propagate = True
        # Set up a stream handler if not present
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            stream_handler = logging.StreamHandler(sys.stdout)
            fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            stream_handler.setFormatter(fmt)
            logger.addHandler(stream_handler)
        print("[INFO] Logging to terminal only (no log file)")

    return file_handler, logfile_fd, old_stdout, old_stderr, logfile


def teardown_export_logging(
    file_handler: logging.Handler,
    logfile_fd: Any,
    old_stdout: Any,
    old_stderr: Any,
    logger: logging.Logger,
) -> None:
    """Restore stdout/stderr and remove the file handler."""
    # Flush and close logfile descriptor
    if logfile_fd:
        for op in (lambda: logfile_fd.flush(), lambda: logfile_fd.close()):
            try:
                op()
            except Exception:
                pass

    # Restore stdout/stderr
    if old_stdout is not None:
        try:
            sys.stdout = old_stdout
        except Exception:
            pass
    if old_stderr is not None:
        try:
            sys.stderr = old_stderr
        except Exception:
            pass

    # Remove and close file handler
    if file_handler is not None:
        for op in (
            lambda: logger.removeHandler(file_handler),
            lambda: logging.getLogger().removeHandler(file_handler),
            lambda: file_handler.close(),
        ):
            try:
                op()
            except Exception:
                pass
