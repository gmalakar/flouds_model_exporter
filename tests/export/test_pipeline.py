import importlib
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast



def import_pipeline_module(monkeypatch) -> Any:
    fake_config_logging = cast(Any, types.ModuleType("model_exporter.config.logging"))
    fake_export_helpers = cast(Any, types.ModuleType("model_exporter.export.helpers"))
    fake_subprocess_runner = cast(Any, types.ModuleType("model_exporter.export.subprocess_runner"))
    fake_utils_helpers = cast(Any, types.ModuleType("model_exporter.utils.helpers"))
    fake_validation_invoker = cast(Any, types.ModuleType("model_exporter.validation.invoker"))
    fake_validation_checker = cast(Any, types.ModuleType("model_exporter.validation.checker"))

    fake_config_logging.setup_export_logging = lambda *args, **kwargs: (None, None, None, None, None)
    fake_config_logging.teardown_export_logging = lambda *args, **kwargs: None

    fake_export_helpers.cleanup_temporary_export_artifacts = lambda *args, **kwargs: None
    fake_export_helpers.cleanup_validator_logging_handlers = lambda *args, **kwargs: None
    fake_export_helpers.configure_protobuf = lambda: None
    fake_export_helpers.is_pid_running = lambda pid: False

    fake_subprocess_runner._run_main_export_subprocess = lambda *args, **kwargs: (True, "")

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

    assert expected == ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]



def test_should_skip_validator_for_multi_file_seq2seq(monkeypatch):
    pipeline = import_pipeline_module(monkeypatch)

    should_skip = pipeline._should_skip_validator(
        model_for="s2s",
        pack_single_file=False,
        expected=["encoder_model.onnx", "decoder_model.onnx"],
    )

    assert should_skip is True



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
    result = _run_main_export_subprocess({"output": str(output_dir), "task": "feature-extraction"}, logging.getLogger("test"))

    assert result == (True, "")
    assert run_calls["timeout"] == 3600
    assert run_calls["env"]["OMP_NUM_THREADS"] == "1"
    assert run_calls["cmd"][0] == sys.executable


def test_run_export_with_fallback_uses_legacy_after_v2_failure(monkeypatch, tmp_path):
    from model_exporter.export import pipeline_v2

    called: dict[str, bool] = {"legacy": False}

    monkeypatch.setattr(pipeline_v2, "export_v2_main_export", lambda *args, **kwargs: (False, False))

    def _fake_legacy(*args, **kwargs):
        called["legacy"] = True
        return True, True

    monkeypatch.setattr(pipeline_v2, "run_legacy_v1_fallback", _fake_legacy)

    ok, used_trust = pipeline_v2._run_export_with_fallback(
        export_source="some/model",
        output_dir=str(tmp_path),
        model_for="llm",
        opset_version=17,
        device="cpu",
        task="feature-extraction",
        framework="pt",
        library=None,
        logger=logging.getLogger("test-legacy-fallback"),
        trust_remote_code=False,
        use_external_data_format=False,
        no_post_process=False,
        merge=False,
        use_subprocess=False,
        use_fallback_if_failed=True,
    )

    assert called["legacy"] is True
    assert ok is True
    assert used_trust is True


def test_run_export_with_fallback_skips_legacy_when_flag_disabled(monkeypatch, tmp_path):
    from model_exporter.export import pipeline_v2

    called: dict[str, bool] = {"legacy": False}

    monkeypatch.setattr(pipeline_v2, "export_v2_main_export", lambda *args, **kwargs: (False, False))

    def _fake_legacy(*args, **kwargs):
        called["legacy"] = True
        return True, True

    monkeypatch.setattr(pipeline_v2, "run_legacy_v1_fallback", _fake_legacy)

    ok, used_trust = pipeline_v2._run_export_with_fallback(
        export_source="some/model",
        output_dir=str(tmp_path),
        model_for="llm",
        opset_version=17,
        device="cpu",
        task="feature-extraction",
        framework="pt",
        library=None,
        logger=logging.getLogger("test-legacy-fallback-disabled"),
        trust_remote_code=False,
        use_external_data_format=False,
        no_post_process=False,
        merge=False,
        use_subprocess=False,
        use_fallback_if_failed=False,
    )

    assert called["legacy"] is False
    assert ok is False
    assert used_trust is False
