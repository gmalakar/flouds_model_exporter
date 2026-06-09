# =============================================================================
# File: tests/export/test_path_utils.py
# Date Created: 2026-06-08
# Date Updated: 2026-06-08
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Tests for guarded export filesystem operations."""

from __future__ import annotations

import pytest

from model_exporter.export.path_utils import safe_remove_export_dir, safe_replace_export_dir


def test_safe_remove_export_dir_removes_child_only(tmp_path):
    parent = tmp_path / "onnx" / "models" / "fe"
    child = parent / "model-a"
    child.mkdir(parents=True)
    (child / "model.onnx").write_text("placeholder", encoding="utf-8")

    removed = safe_remove_export_dir(child, parent)

    assert removed is True
    assert not child.exists()
    assert parent.exists()


def test_safe_remove_export_dir_refuses_parent(tmp_path):
    parent = tmp_path / "onnx" / "models" / "fe"
    parent.mkdir(parents=True)

    with pytest.raises(ValueError, match="parent directory"):
        safe_remove_export_dir(parent, parent)

    assert parent.exists()


def test_safe_remove_export_dir_refuses_outside_parent(tmp_path):
    parent = tmp_path / "onnx" / "models" / "fe"
    outside = tmp_path / "outside"
    parent.mkdir(parents=True)
    outside.mkdir()

    with pytest.raises(ValueError, match="outside export parent"):
        safe_remove_export_dir(outside, parent)

    assert outside.exists()


def test_safe_replace_export_dir_replaces_child_output(tmp_path):
    parent = tmp_path / "onnx" / "models" / "fe"
    target = parent / "model-a"
    source = tmp_path / "working"
    target.mkdir(parents=True)
    source.mkdir()
    (target / "old.onnx").write_text("old", encoding="utf-8")
    (source / "model.onnx").write_text("new", encoding="utf-8")

    safe_replace_export_dir(source, target, parent)

    assert not source.exists()
    assert not (target / "old.onnx").exists()
    assert (target / "model.onnx").read_text(encoding="utf-8") == "new"
