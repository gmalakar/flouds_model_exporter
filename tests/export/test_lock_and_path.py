# =============================================================================
# File: tests/export/test_lock_and_path.py
# Date: 2026-04-19
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Tests for export lock correctness and onnx_path anchoring."""

from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest


def _make_fake_modules(monkeypatch):
    """Register minimal fake modules so pipeline.py can be imported without ML deps."""
    fake_config_logging = cast(Any, types.ModuleType("model_exporter.config.logging"))
    fake_export_helpers = cast(Any, types.ModuleType("model_exporter.export.helpers"))
    fake_subprocess_runner = cast(Any, types.ModuleType("model_exporter.export.subprocess_runner"))
    fake_utils_helpers = cast(Any, types.ModuleType("model_exporter.utils.helpers"))
    fake_validation_invoker = cast(Any, types.ModuleType("model_exporter.validation.invoker"))
    fake_validation_checker = cast(Any, types.ModuleType("model_exporter.validation.checker"))
    fake_pipeline_v2 = cast(Any, types.ModuleType("model_exporter.export.pipeline_v2"))
    fake_pipeline_helpers = cast(Any, types.ModuleType("model_exporter.export.pipeline_helpers"))

    fake_config_logging.setup_export_logging = lambda *a, **kw: (
        None,
        None,
        None,
        None,
        None,
    )
    fake_config_logging.teardown_export_logging = lambda *a, **kw: None

    fake_export_helpers.cleanup_temporary_export_artifacts = lambda *a, **kw: None
    fake_export_helpers.cleanup_validator_logging_handlers = lambda *a, **kw: None
    fake_export_helpers.configure_protobuf = lambda: None
    fake_export_helpers.is_pid_running = lambda pid: False
    fake_export_helpers.cleanup_extraneous_onnx_files = lambda *a, **kw: None

    fake_utils_helpers.get_default_opset = lambda default=17: default
    fake_utils_helpers.get_logger = logging.getLogger
    fake_utils_helpers.safe_log = lambda *a, **kw: None

    fake_validation_invoker.invoke_validator = lambda *a, **kw: (0, True)
    fake_validation_checker.verify_models = lambda *a, **kw: True

    fake_pipeline_v2._run_export_with_fallback = lambda *a, **kw: (True, False)
    fake_pipeline_v2._run_post_optimization_validator = lambda *a, **kw: 0

    # pipeline_helpers: expose only what pipeline.py imports from it
    import contextlib

    @contextlib.contextmanager
    def _fake_lock(output_dir, model_name, logger):
        yield output_dir, True

    fake_pipeline_helpers._auto_resolve_trust_remote_code = lambda *a, **kw: False
    fake_pipeline_helpers._build_expected_list = lambda *a, **kw: ["model.onnx"]
    fake_pipeline_helpers._check_optimized_artifacts = lambda *a, **kw: False
    fake_pipeline_helpers._cleanup_memory_caches = lambda *a, **kw: (None, True)
    fake_pipeline_helpers._is_seq2seq = lambda *a, **kw: False
    fake_pipeline_helpers._lift_temp_local_artifacts = lambda *a, **kw: None
    fake_pipeline_helpers._resolve_use_cache = lambda *a, **kw: False
    fake_pipeline_helpers._run_numeric_validator = lambda *a, **kw: (0, True)
    fake_pipeline_helpers._run_quantization_step = lambda *a, **kw: None
    fake_pipeline_helpers._setup_huggingface_hub_token = lambda *a, **kw: (None, {})
    fake_pipeline_helpers._teardown_huggingface_hub_token = lambda *a, **kw: None
    fake_pipeline_helpers._should_skip_validator = lambda *a, **kw: True
    fake_pipeline_helpers._with_export_lock = _fake_lock

    for mod_name, mod in [
        ("model_exporter.config.logging", fake_config_logging),
        ("model_exporter.export.helpers", fake_export_helpers),
        ("model_exporter.export.subprocess_runner", fake_subprocess_runner),
        ("model_exporter.utils.helpers", fake_utils_helpers),
        ("model_exporter.validation.invoker", fake_validation_invoker),
        ("model_exporter.validation.checker", fake_validation_checker),
        ("model_exporter.export.pipeline_v2", fake_pipeline_v2),
        ("model_exporter.export.pipeline_helpers", fake_pipeline_helpers),
    ]:
        monkeypatch.setitem(sys.modules, mod_name, mod)

    sys.modules.pop("model_exporter.export.pipeline", None)
    return importlib.import_module("model_exporter.export.pipeline")


def test_export_lock_wraps_full_export_body(monkeypatch, tmp_path):
    """The export lock context manager must be entered before any export work
    and exited only after the export completes (not before)."""
    lock_events: list[str] = []

    import contextlib

    @contextlib.contextmanager
    def _tracking_lock(output_dir, model_name, log):
        lock_events.append("enter")
        try:
            yield output_dir, True
        finally:
            lock_events.append("exit")

    pipeline = _make_fake_modules(monkeypatch)
    # Override the fake lock with a tracking version
    pipeline._with_export_lock = _tracking_lock
    # Also patch the module-level helper reference used inside export()
    import model_exporter.export.pipeline_helpers as _ph

    original = _ph._with_export_lock
    monkeypatch.setattr(_ph, "_with_export_lock", _tracking_lock)

    # Reload so the patched symbol is used
    sys.modules.pop("model_exporter.export.pipeline", None)
    pipeline = importlib.import_module("model_exporter.export.pipeline")
    monkeypatch.setattr(pipeline, "_with_export_lock", _tracking_lock)

    # Create an existing model.onnx so the skip-if-exists path is taken
    out = tmp_path / "onnx" / "models" / "fe" / "dummy-model"
    out.mkdir(parents=True)
    (out / "model.onnx").write_text("placeholder")

    monkeypatch.chdir(tmp_path)
    pipeline.export(
        model_name="dummy/dummy-model",
        model_for="fe",
        task="feature-extraction",
        onnx_path=str(tmp_path / "onnx"),
    )

    assert lock_events == [
        "enter",
        "exit",
    ], "Lock must be entered and exited exactly once wrapping the export body"

    monkeypatch.setattr(_ph, "_with_export_lock", original)


def test_onnx_path_defaults_to_cwd_not_package_dir(monkeypatch, tmp_path):
    """When onnx_path is not supplied and ONNX_PATH is unset, the output
    directory must be anchored to the current working directory, not to the
    pipeline.py package directory."""
    pipeline = _make_fake_modules(monkeypatch)

    monkeypatch.delenv("ONNX_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    # Pre-create model.onnx so the skip-if-exists path is taken
    expected_out = tmp_path / "onnx" / "models" / "fe" / "test-model"
    expected_out.mkdir(parents=True)
    (expected_out / "model.onnx").write_text("placeholder")

    result = pipeline.export(
        model_name="org/test-model",
        model_for="fe",
        task="feature-extraction",
    )

    result_path = Path(result)
    # Output must be under the tmp cwd, not inside the Python package tree
    assert result_path.is_relative_to(tmp_path), f"Expected output under cwd ({tmp_path}), got: {result_path}"
    # Must NOT be inside the Python package installation directory
    pkg_dir = Path(__file__).resolve().parents[3] / "src"
    assert not result_path.is_relative_to(pkg_dir), f"Output should not be inside the package directory: {result_path}"


def test_onnx_path_from_env_var_is_respected(monkeypatch, tmp_path):
    """ONNX_PATH env var must be honoured when no explicit onnx_path is passed."""
    pipeline = _make_fake_modules(monkeypatch)

    custom_dir = tmp_path / "custom_onnx"
    monkeypatch.setenv("ONNX_PATH", str(custom_dir))
    monkeypatch.chdir(tmp_path)

    expected_out = custom_dir / "models" / "fe" / "mymodel"
    expected_out.mkdir(parents=True)
    (expected_out / "model.onnx").write_text("placeholder")

    result = pipeline.export(
        model_name="org/mymodel",
        model_for="fe",
        task="feature-extraction",
    )

    assert Path(result).is_relative_to(custom_dir)


@pytest.mark.parametrize("bad_folder", ["..", ".", ""])
def test_model_folder_rejects_reserved_path_segments(monkeypatch, tmp_path, bad_folder):
    """Reserved model_folder values must not resolve to parent export dirs."""
    pipeline = _make_fake_modules(monkeypatch)

    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Invalid model_folder"):
        pipeline.export(
            model_name="org/mymodel",
            model_for="fe",
            task="feature-extraction",
            model_folder=bad_folder,
        )


def test_resolved_output_dir_stays_under_model_type_root(monkeypatch, tmp_path):
    """Path resolution should anchor relative ONNX paths to cwd safely."""
    pipeline = _make_fake_modules(monkeypatch)

    monkeypatch.chdir(tmp_path)
    onnx_root, output_dir, model_folder = pipeline._resolve_export_output_dir(
        "relative-onnx",
        "fe",
        "org/mymodel",
        None,
        cwd=str(tmp_path),
    )

    assert Path(onnx_root) == (tmp_path / "relative-onnx").resolve()
    assert Path(output_dir) == (tmp_path / "relative-onnx" / "models" / "fe" / "mymodel").resolve()
    assert model_folder == "mymodel"


def test_memory_threshold_applies_conservative_export_flags(monkeypatch, tmp_path):
    """Low available RAM should force conservative export flags when a threshold is set."""
    pipeline = _make_fake_modules(monkeypatch)
    captured: dict[str, Any] = {}

    def _capture_export(*args, **kwargs):
        captured.update(kwargs)
        return True, False

    monkeypatch.setattr(pipeline, "_cleanup_memory_caches", lambda *a, **kw: (1.0, False))
    monkeypatch.setattr(pipeline, "_run_export_with_fallback", _capture_export)
    monkeypatch.chdir(tmp_path)

    pipeline.export(
        model_name="org/mymodel",
        model_for="fe",
        task="feature-extraction",
        min_free_memory_gb=4.0,
    )

    assert captured["use_external_data_format"] is True
    assert captured["no_post_process"] is True


def test_memory_threshold_can_fail_fast(monkeypatch, tmp_path):
    """require_sufficient_memory should fail before export when threshold is unmet."""
    pipeline = _make_fake_modules(monkeypatch)

    def _raise_low_memory(*args, **kwargs):
        raise RuntimeError("Available memory 1.00 GB is below threshold 4.00 GB")

    monkeypatch.setattr(pipeline, "_cleanup_memory_caches", _raise_low_memory)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="below threshold"):
        pipeline.export(
            model_name="org/mymodel",
            model_for="fe",
            task="feature-extraction",
            min_free_memory_gb=4.0,
            require_sufficient_memory=True,
        )


def test_public_api_export_importable_from_package():
    """``from model_exporter import export`` must not raise ImportError."""
    # This import will fail if __init__.py is missing the re-export
    try:
        from model_exporter import export  # noqa: F401
    except ImportError as exc:
        raise AssertionError(f"Public API 'from model_exporter import export' failed: {exc}") from exc


def test_public_api_export_config_importable_from_package():
    """``from model_exporter import ExportConfig, export_from_config`` must work."""
    try:
        from model_exporter import ExportConfig, export_from_config  # noqa: F401
    except ImportError as exc:
        raise AssertionError(f"Public config API import failed: {exc}") from exc


def test_public_api_validate_importable_from_package():
    """``from model_exporter import validate_onnx`` must not raise ImportError."""
    try:
        from model_exporter import validate_onnx  # noqa: F401
    except ImportError as exc:
        raise AssertionError(f"Public API 'from model_exporter import validate_onnx' failed: {exc}") from exc


def test_export_subpackage_import_is_lazy():
    """Importing the export package should not eagerly import heavy submodules."""
    for module_name in (
        "model_exporter.export",
        "model_exporter.export.pipeline",
        "model_exporter.export.pipeline_v2",
        "model_exporter.export.helpers",
    ):
        sys.modules.pop(module_name, None)

    import model_exporter.export as export_package

    assert export_package is not None
    assert "model_exporter.export.pipeline" not in sys.modules
