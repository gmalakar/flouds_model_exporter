# =============================================================================
# File: test_pipeline.py
# Date: 2026-04-18
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

import importlib
import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest


def import_pipeline_module(monkeypatch) -> Any:
    fake_config_logging = cast(Any, types.ModuleType("model_exporter.config.logging"))
    fake_export_helpers = cast(Any, types.ModuleType("model_exporter.export.helpers"))
    fake_subprocess_runner = cast(Any, types.ModuleType("model_exporter.export.subprocess_runner"))
    fake_utils_helpers = cast(Any, types.ModuleType("model_exporter.utils.helpers"))
    fake_validation_invoker = cast(Any, types.ModuleType("model_exporter.validation.invoker"))
    fake_validation_checker = cast(Any, types.ModuleType("model_exporter.validation.checker"))

    fake_config_logging.setup_export_logging = lambda *args, **kwargs: (
        None,
        None,
        None,
        None,
        None,
    )
    fake_config_logging.teardown_export_logging = lambda *args, **kwargs: None

    fake_export_helpers.cleanup_temporary_export_artifacts = lambda *args, **kwargs: None
    fake_export_helpers.cleanup_validator_logging_handlers = lambda *args, **kwargs: None
    fake_export_helpers.configure_protobuf = lambda: None
    fake_export_helpers.is_pid_running = lambda pid: False

    fake_subprocess_runner._run_main_export_subprocess = lambda *args, **kwargs: (
        True,
        "",
    )

    fake_utils_helpers.get_default_opset = lambda default=17: default
    fake_utils_helpers.get_logger = logging.getLogger
    fake_utils_helpers.safe_log = lambda *args, **kwargs: None

    fake_validation_invoker.invoke_validator = lambda *args, **kwargs: (0, True)
    fake_validation_checker.verify_models = lambda *args, **kwargs: True

    monkeypatch.setitem(sys.modules, "model_exporter.config.logging", fake_config_logging)
    monkeypatch.setitem(sys.modules, "model_exporter.export.helpers", fake_export_helpers)
    monkeypatch.setitem(sys.modules, "model_exporter.export.subprocess_runner", fake_subprocess_runner)
    monkeypatch.setitem(sys.modules, "model_exporter.utils.helpers", fake_utils_helpers)
    monkeypatch.setitem(sys.modules, "model_exporter.validation.invoker", fake_validation_invoker)
    monkeypatch.setitem(sys.modules, "model_exporter.validation.checker", fake_validation_checker)

    sys.modules.pop("model_exporter.export.pipeline", None)
    return importlib.import_module("model_exporter.export.pipeline")


def test_build_expected_list_handles_seq2seq_with_past(monkeypatch):
    pipeline = import_pipeline_module(monkeypatch)

    expected = pipeline._build_expected_list("s2s", use_cache=True, task="text2text-generation-with-past")

    assert expected == [
        "encoder_model.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx",
    ]


def test_should_skip_validator_for_multi_file_seq2seq(monkeypatch):
    pipeline = import_pipeline_module(monkeypatch)

    should_skip = pipeline._should_skip_validator(
        model_for="s2s",
        pack_single_file=False,
        expected=["encoder_model.onnx", "decoder_model.onnx"],
    )

    assert should_skip is True


def test_auto_enable_use_cache_for_multimodal_generators(monkeypatch):
    import_pipeline_module(monkeypatch)
    from model_exporter.export import pipeline_helpers

    enabled = pipeline_helpers._auto_enable_use_cache(
        model_name="llava-hf/llava-1.5-7b-hf",
        model_for="vlm",
        task="text-generation-with-past",
    )

    assert enabled is True


def test_optimize_if_encoder_skips_decoder_only_models():
    from model_exporter.export.optimizer import optimize_if_encoder

    rc = optimize_if_encoder(
        model_dir=Path("ignored"),
        model_type="llm",
        logger=logging.getLogger("test"),
        optimization_level=99,
    )

    assert rc == 0


def test_run_main_export_subprocess_returns_success_and_stderr(monkeypatch, tmp_path):
    from model_exporter.export.subprocess_runner import _run_main_export_subprocess

    run_calls: dict[str, Any] = {}

    def fake_run(cmd, stdout, stderr, text, env, timeout):
        run_calls["cmd"] = cmd
        run_calls["env"] = env
        run_calls["timeout"] = timeout
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("model_exporter.export.subprocess_runner.subprocess.run", fake_run)

    output_dir = tmp_path / "export-output"
    result = _run_main_export_subprocess(
        {"output": str(output_dir), "task": "feature-extraction"},
        logging.getLogger("test"),
    )

    assert result == (True, "")
    assert run_calls["timeout"] == 3600
    assert run_calls["env"]["OMP_NUM_THREADS"] == "1"
    assert run_calls["cmd"][0] == sys.executable


def _run_with_fallback(optimum_export, tmp_path, use_fallback_if_failed):
    return optimum_export._run_export_with_fallback(
        export_source="some/model",
        output_dir=str(tmp_path),
        model_for="llm",
        opset_version=17,
        device="cpu",
        task="feature-extraction",
        framework="pt",
        library=None,
        logger=logging.getLogger("test-fallback-flag"),
        trust_remote_code=False,
        use_subprocess=False,
        use_fallback_if_failed=use_fallback_if_failed,
        revision="abc123",
    )


@pytest.mark.parametrize("flag", [True, False])
def test_run_export_with_fallback_passes_the_flag_to_v2(monkeypatch, tmp_path, flag):
    from model_exporter.export import optimum_export

    seen: dict[str, Any] = {}

    def _fake_v2(*args, **kwargs):
        seen.update(kwargs)
        return False, False

    monkeypatch.setattr(optimum_export, "export_v2_main_export", _fake_v2)
    assert _run_with_fallback(optimum_export, tmp_path, flag) == (False, False)
    assert seen["retry_after_inprocess_failure"] is flag
    assert seen["allow_precision_fallback"] is flag
    assert seen["revision"] == "abc123"


def _v2_after_inprocess_failure(monkeypatch, tmp_path, retry):
    from model_exporter.export import optimum_export

    loop_calls: list[dict[str, Any]] = []

    def _fake_loop(me_kwargs, stderr, *args, **kwargs):
        loop_calls.append(dict(me_kwargs))
        return True, False

    monkeypatch.setattr(optimum_export, "_attempt_inprocess_export", lambda me, wo, od, ct, lg: (False, "boom", None))
    monkeypatch.setattr(optimum_export, "_execute_fallback_loop", _fake_loop)
    monkeypatch.setattr(optimum_export, "cleanup_temporary_export_artifacts", lambda **kw: None)
    result = optimum_export.export_v2_main_export(
        "org/model",
        str(tmp_path),
        "fe",
        17,
        "cpu",
        "feature-extraction",
        "pt",
        "transformers",
        False,
        logging.getLogger("test-inprocess-retry"),
        revision="abc123",
        retry_after_inprocess_failure=retry,
    )
    return result, loop_calls


def test_inprocess_failure_runs_fallback_strategies_when_allowed(monkeypatch, tmp_path):
    result, loop_calls = _v2_after_inprocess_failure(monkeypatch, tmp_path, retry=True)
    assert result == (True, False)
    assert len(loop_calls) == 1 and loop_calls[0].get("revision") == "abc123"


def test_inprocess_failure_does_not_retry_by_default(monkeypatch, tmp_path):
    result, loop_calls = _v2_after_inprocess_failure(monkeypatch, tmp_path, retry=False)
    assert result == (False, False)
    assert loop_calls == []


def test_fallback_strategies_use_optimum_dtype_values_and_full_precision_opset_retries():
    from model_exporter.export import optimum_export

    generic = dict(optimum_export._build_fallback_strategies("something unexpected", False))
    assert generic == {"opset_14": {"opset": 14}, "opset_11_external": {"opset": 11, "use_external_data_format": True}}

    post = dict(optimum_export._build_fallback_strategies("post-processing of the onnx export failed", False))
    assert post["no_post_process_external"] == {"no_post_process": True, "use_external_data_format": True}

    memory = dict(optimum_export._build_fallback_strategies("out of memory", False))
    assert memory["dtype_fp16_external"]["dtype"] == "fp16"
    dtypes = {
        kw["dtype"]
        for name, kw in optimum_export._build_fallback_strategies("out of memory post-processing of the onnx export failed", True)
        if "dtype" in kw
    }
    assert dtypes <= {"fp32", "fp16", "bf16"}, "optimum ignores any other dtype value"


def test_auto_resolve_trust_remote_code_does_not_auto_enable(monkeypatch):
    from model_exporter.export import pipeline_helpers

    warnings: list[str] = []
    logger = SimpleNamespace(
        warning=lambda message, *args: warnings.append(message % args),
        debug=lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(pipeline_helpers, "_requires_trust_remote_code_fast", lambda *args: True)

    result = pipeline_helpers._auto_resolve_trust_remote_code("some/model", None, False, logger)

    assert result is False
    assert "trust_remote_code" in warnings[0]
    assert "not explicitly allowed" in warnings[0]


def _hub_that_must_not_be_logged_into(monkeypatch):
    fake_hub = types.ModuleType("huggingface_hub")

    def _no_login(*args, **kwargs):
        raise AssertionError("the exporter must not log in: a login saves the token on disk")

    fake_hub.login = _no_login
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)


def test_hub_token_is_returned_without_login_or_environment_change(monkeypatch):
    from model_exporter.export import pipeline_helpers

    _hub_that_must_not_be_logged_into(monkeypatch)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    token = pipeline_helpers._resolve_hub_token("hf_test")

    assert token == "hf_test"
    assert "HUGGINGFACE_HUB_TOKEN" not in os.environ
    assert "HF_TOKEN" not in os.environ


def test_hub_token_falls_back_to_the_environment(monkeypatch):
    from model_exporter.export import pipeline_helpers

    _hub_that_must_not_be_logged_into(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_env")
    monkeypatch.setenv("HF_TOKEN", "hf_other")
    assert pipeline_helpers._resolve_hub_token(None) == "hf_env"
    assert os.environ["HUGGINGFACE_HUB_TOKEN"] == "hf_env"

    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN")
    assert pipeline_helpers._resolve_hub_token(None) == "hf_other"
    assert pipeline_helpers._resolve_hub_token("hf_given") == "hf_given"


def test_export_from_config_forwards_normalized_kwargs(monkeypatch):
    from model_exporter.export import pipeline
    from model_exporter.export.options import ExportConfig

    captured: dict[str, Any] = {}

    def _fake_export(**kwargs):
        captured.update(kwargs)
        return "exported"

    monkeypatch.setattr(pipeline, "export", _fake_export)

    result = pipeline.export_from_config(
        ExportConfig(
            model_name="org/model",
            task="feature-extraction",
            quantize="both",
        )
    )

    assert result == "exported"
    assert captured["model_name"] == "org/model"
    assert captured["quantize"] is True
