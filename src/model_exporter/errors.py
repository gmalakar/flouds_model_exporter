# =============================================================================
# File: errors.py
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

"""Exceptions raised by the exporter.

Library code raises these instead of ending the process, so a caller that embeds the
exporter (a service, a notebook, a batch job) decides what a failure means. The
``flouds-export`` command line turns them into exit codes.
"""

from __future__ import annotations


class ExporterError(Exception):
    """Base class for every error the exporter raises on purpose."""


class InvalidModelReferenceError(ExporterError, ValueError):
    """The model name, path or repository URL is not one the exporter will use."""


class RemoteCodeNotAllowedError(ExporterError):
    """The model needs ``trust_remote_code`` and the caller did not allow it.

    Loading such a model runs Python from the model repository on this machine. The
    exporter never turns that on by itself: pass ``trust_remote_code=True`` after
    reviewing the repository.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(
            f"Model {model_name!r} requires trust_remote_code, which was not allowed. "
            "Loading it would run code from the model repository; pass trust_remote_code=True "
            "(--trust-remote-code) only after reviewing that code."
        )


class ExportInProgressError(ExporterError):
    """Another process is already exporting to the same output directory."""

    def __init__(self, output_dir: str, pid: int | None = None) -> None:
        self.output_dir = output_dir
        self.pid = pid
        owner = f" (pid {pid})" if pid is not None else ""
        super().__init__(f"An export to {output_dir} is already running{owner}.")


class ExportFailedError(ExporterError, RuntimeError):
    """Every export attempt failed. ``detail`` holds the last error reported."""

    def __init__(self, model_name: str, detail: str = "") -> None:
        self.model_name = model_name
        self.detail = detail
        lines = [line.strip() for line in (detail or "").splitlines() if line.strip()]
        reason = f": {lines[-1][:500]}" if lines else ""
        super().__init__(f"All export attempts failed for model {model_name}{reason}")


class ExportValidationError(ExporterError):
    """The exported model failed numeric validation and ``require_validator`` was set."""

    def __init__(self, output_dir: str, returncode: int) -> None:
        self.output_dir = output_dir
        self.returncode = returncode
        super().__init__(f"Validation of the export in {output_dir} failed (validator code {returncode}).")
