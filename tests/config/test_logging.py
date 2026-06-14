# =============================================================================
# File: test_logging.py
# Date Created: 2026-06-08
# Date Updated: 2026-06-08
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

import logging

from model_exporter.config.logging import setup_export_logging, teardown_export_logging


def test_setup_export_logging_uses_configured_log_dir(monkeypatch, tmp_path):
    logger = logging.getLogger("test-configured-log-dir")
    logger.handlers.clear()
    log_dir = tmp_path / "custom-logs"
    monkeypatch.setenv("LOG_DIR", str(log_dir))

    file_handler, logfile_fd, old_stdout, old_stderr, logfile_path = setup_export_logging(str(tmp_path), "model", "rev", logger, log_to_file=True)

    try:
        assert logfile_path is not None
        assert logfile_path.parent == log_dir
    finally:
        teardown_export_logging(file_handler, logfile_fd, old_stdout, old_stderr, logger)


def test_setup_export_logging_defaults_to_cwd_logs(monkeypatch, tmp_path):
    logger = logging.getLogger("test-default-log-dir")
    logger.handlers.clear()
    monkeypatch.delenv("LOG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    file_handler, logfile_fd, old_stdout, old_stderr, logfile_path = setup_export_logging(str(tmp_path), "model", "rev", logger, log_to_file=True)

    try:
        assert logfile_path is not None
        assert logfile_path.parent == tmp_path / "logs" / "onnx_exports"
    finally:
        teardown_export_logging(file_handler, logfile_fd, old_stdout, old_stderr, logger)
