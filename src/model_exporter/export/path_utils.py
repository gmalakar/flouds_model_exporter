# =============================================================================
# File: export/path_utils.py
# Date Created: 2026-06-08
# Date Updated: 2026-06-08
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Filesystem safety helpers for export output directories."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def _validate_child_dir(path: str | Path, allowed_parent: str | Path) -> Path:
    """Return resolved *path* after proving it is below *allowed_parent*."""
    target = Path(path).resolve()
    parent = Path(allowed_parent).resolve()
    if target == parent:
        raise ValueError(f"Refusing to operate on export parent directory: {target}")
    try:
        target.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"Refusing to operate outside export parent: {target}") from exc
    if target.name in {"", ".", ".."}:
        raise ValueError(f"Refusing to operate on invalid export directory: {target}")
    return target


def safe_remove_export_dir(
    output_dir: str | Path,
    allowed_parent: str | Path,
    logger: Any | None = None,
    reason: str = "cleanup",
) -> bool:
    """Remove an export output directory after validating its parent boundary."""
    target = _validate_child_dir(output_dir, allowed_parent)
    if not target.exists():
        return False
    if not target.is_dir():
        raise ValueError(f"Export output path is not a directory: {target}")
    shutil.rmtree(target)
    if logger is not None:
        try:
            logger.info("Removed export directory for %s: %s", reason, target)
        except Exception:
            pass
    return True


def safe_replace_export_dir(
    source_dir: str | Path,
    output_dir: str | Path,
    allowed_parent: str | Path,
    logger: Any | None = None,
) -> None:
    """Replace an export output directory with a prepared source directory."""
    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise ValueError(f"Replacement source is not a directory: {source}")
    target = _validate_child_dir(output_dir, allowed_parent)
    if target.exists():
        safe_remove_export_dir(target, allowed_parent, logger, reason="replace")
    shutil.copytree(source, target)
    shutil.rmtree(source)
