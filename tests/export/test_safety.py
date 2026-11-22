# =============================================================================
# File: tests/export/test_safety.py
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

"""The exporter as a library: it does not run remote code unasked, does not keep a
caller's token, does not end the caller's process, and passes values on unchanged."""

import json
import logging
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model_exporter.errors import ExporterError, ExportInProgressError, ExportValidationError, InvalidModelReferenceError, RemoteCodeNotAllowedError
from model_exporter.export import legacy_fallback, pipeline_helpers, pipeline_v2, source_ref, subprocess_runner
from model_exporter.export.source_ref import hub_clone_url, validate_model_reference

log = logging.getLogger("test_safety")
NEEDS_REMOTE_CODE = "Loading x requires you to execute the configuration file in that repo. Please pass trust_remote_code=True"


# -- remote code ----------------------------------------------------------------


def _loader(seen):
    def from_pretrained(name, **kwargs):
        seen.append(kwargs.get("trust_remote_code"))
        if not kwargs.get("trust_remote_code"):
            raise ValueError(NEEDS_REMOTE_CODE)
        return MagicMock()

    return from_pretrained


def test_trust_remote_code_is_never_raised_on_the_callers_behalf():
    import transformers

    seen: list = []
    with (
        patch.object(transformers.AutoTokenizer, "from_pretrained", side_effect=_loader(seen)),
        patch.object(transformers.AutoModel, "from_pretrained", side_effect=_loader(seen)),
        patch.object(transformers.AutoConfig, "from_pretrained", side_effect=_loader(seen)),
    ):
        with pytest.raises(RemoteCodeNotAllowedError) as raised:
            pipeline_v2._load_transformers_components("org/custom-code-model", False, None, log)

    assert seen == [False], "the load must not be retried with trust_remote_code=True"
    assert "org/custom-code-model" in str(raised.value)
    assert isinstance(raised.value, ExporterError)


def test_trust_remote_code_is_used_when_the_caller_allows_it():
    import transformers

    seen: list = []
    with (
        patch.object(transformers.AutoTokenizer, "from_pretrained", side_effect=_loader(seen)),
        patch.object(transformers.AutoModel, "from_pretrained", side_effect=_loader(seen)),
        patch.object(transformers.AutoConfig, "from_pretrained", side_effect=_loader(seen)),
    ):
        tokenizer, model, config = pipeline_v2._load_transformers_components("org/custom-code-model", True, None, log)

    assert seen == [True, True, True]
    assert tokenizer is not None and model is not None and config is not None


def test_fallback_exporter_does_not_default_to_trusting_remote_code():
    source = Path(pipeline_v2.__file__).read_text(encoding="utf-8")
    assert 'me_kwargs.get("trust_remote_code", True)' not in source


@pytest.mark.parametrize("strategy", ["__use_local__", "__snapshot__", "__clone__"])
def test_remote_code_strategies_are_refused_without_consent(strategy, tmp_path):
    with patch.object(source_ref.subprocess, "check_call") as git:
        me_try, cleanup, reason = pipeline_v2._prepare_strategy({"trust_remote_code": False}, "s", {strategy: True}, str(tmp_path), log)
    assert me_try is None and cleanup is None
    assert "trust_remote_code" in reason
    git.assert_not_called()


def test_fallback_list_offers_remote_code_strategies_only_with_consent():
    without = [name for name, _ in pipeline_v2._build_fallback_strategies("does not recognize model type", False)]
    allowed = [name for name, _ in pipeline_v2._build_fallback_strategies("does not recognize model type", True)]
    for name in ("local_path", "clone_repo", "snapshot_retry"):
        assert name not in without
        assert name in allowed


# -- model references and git ---------------------------------------------------


@pytest.mark.parametrize("value", ["", "   ", None, "--upload-pack=calc.exe", "-c", "org/model\nname", "a\x00b"])
def test_invalid_model_references_are_rejected(value):
    with pytest.raises(InvalidModelReferenceError):
        validate_model_reference(value)


@pytest.mark.parametrize(
    "value", ["bert-base-uncased", "sentence-transformers/all-MiniLM-L6-v2", "123", "none", r"C:\models\local", "/srv/models/local"]
)
def test_ordinary_model_references_pass_unchanged(value):
    assert validate_model_reference(f"  {value} ") == value


def test_hub_clone_url_accepts_ids_and_https_only():
    assert hub_clone_url("org/model") == "https://huggingface.co/org/model"
    assert hub_clone_url("https://huggingface.co/org/model") == "https://huggingface.co/org/model"
    for bad in ("--upload-pack=x", "ext::sh -c id", "file:///etc", "http://example.com/x", "git@host:org/repo", "org/model;id", "a/b/c"):
        with pytest.raises(InvalidModelReferenceError):
            hub_clone_url(bad)


def test_clone_strategy_gives_git_a_checked_url_after_a_separator():
    calls = []
    with patch.object(source_ref.subprocess, "check_call", side_effect=calls.append):
        me_try, cleanup, _ = pipeline_v2._prepare_strategy({"trust_remote_code": True}, "clone_repo", {"__clone__": True}, "org/model", log)
    assert me_try is not None
    cleanup()
    argv = calls[0]
    assert argv[:5] == ["git", "clone", "--depth", "1", "--"]
    assert argv[5] == "https://huggingface.co/org/model"

    with patch.object(source_ref.subprocess, "check_call") as git:
        me_try, _, reason = pipeline_v2._prepare_strategy({"trust_remote_code": True}, "clone_repo", {"__clone__": True}, "--upload-pack=x", log)
    assert me_try is None and reason
    git.assert_not_called()


def test_legacy_clone_refuses_option_like_sources():
    with patch.object(source_ref.subprocess, "check_call") as git:
        path, error = legacy_fallback._prepare_clone_source("--upload-pack=x", log)
    assert path is None and error
    git.assert_not_called()


# -- token ----------------------------------------------------------------------


def test_token_reaches_the_exporter_as_an_argument(tmp_path):
    captured = {}

    def fake_inprocess(me_kwargs, working_output, output_dir, child_tmp, logger):
        captured.update(me_kwargs)
        return True, "", child_tmp

    with (
        patch.object(pipeline_v2, "_attempt_inprocess_export", side_effect=fake_inprocess),
        patch.object(pipeline_v2, "cleanup_temporary_export_artifacts"),
    ):
        ok, _ = pipeline_v2.export_v2_main_export(
            "org/model", str(tmp_path), "fe", 17, "cpu", "feature-extraction", "pt", "transformers", False, log, auth_token="hf_secret"
        )
    assert ok
    assert captured["token"] == "hf_secret"
    assert "use_auth_token" not in captured


def test_token_is_not_written_to_the_subprocess_argument_file(monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["args_file"] = Path(argv[-1]).read_text(encoding="utf-8")
        seen["env"] = kwargs["env"]
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    with patch.object(subprocess_runner.subprocess, "run", side_effect=fake_run):
        subprocess_runner._run_main_export_subprocess({"model_name_or_path": "org/model", "output": "out", "token": "hf_secret"}, log)

    assert "hf_secret" not in seen["args_file"]
    assert "token" not in json.loads(seen["args_file"])
    assert seen["env"]["HF_TOKEN"] == "hf_secret"
    assert "HF_TOKEN" not in os.environ, "the parent environment must not keep the token"


def test_tokens_are_redacted_from_logged_arguments():
    logged = pipeline_v2._sanitize_kwargs_for_logging({"token": "hf_secret", "model_name_or_path": "org/model"})
    assert "hf_secret" not in json.dumps(logged, default=str)
    assert logged["model_name_or_path"] == "org/model"


# -- process exit ---------------------------------------------------------------


def test_running_export_raises_an_error_instead_of_exiting(tmp_path):
    (tmp_path / ".export.lock").write_text(f"{os.getpid()}\n0\nmodel\n", encoding="utf-8")
    with pytest.raises(ExportInProgressError) as raised:
        pipeline_helpers._create_export_lock(str(tmp_path), "model", log)
    assert raised.value.pid == os.getpid()
    assert raised.value.output_dir == str(tmp_path)
    assert not isinstance(raised.value, SystemExit)


def test_lock_left_by_a_dead_process_is_taken_over(tmp_path):
    (tmp_path / ".export.lock").write_text("999999999\n0\nmodel\n", encoding="utf-8")
    with patch.object(pipeline_helpers, "is_pid_running", return_value=False):
        with pipeline_helpers._with_export_lock(str(tmp_path), "model", log) as (lock_path, created):
            assert created and Path(lock_path).read_text(encoding="utf-8").startswith(str(os.getpid()))
    assert not (tmp_path / ".export.lock").exists()


def test_library_source_never_ends_the_process():
    root = Path(pipeline_v2.__file__).parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        if "cli" in path.parts or path.name == "numeric.py":  # command-line entry points may exit; the checker's child script uses sys.exit
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\braise SystemExit\b|\bos\._exit\(", line) and not line.lstrip().startswith(("#", '"', "'")):
                offenders.append(f"{path.relative_to(root)}:{number}")
    assert offenders == []


def test_cli_maps_exporter_errors_to_exit_codes(capsys):
    from model_exporter.cli import cmd_export

    with patch.object(cmd_export, "export_unified", side_effect=ExportInProgressError("out", 42)):
        cmd_export._execute_export_kwargs({"model_name": "m"}, MagicMock())
    assert "already running" in capsys.readouterr().out

    with patch.object(cmd_export, "export_unified", side_effect=ExportValidationError("out", 3)):
        with pytest.raises(SystemExit) as exit_info:
            cmd_export._execute_export_kwargs({"model_name": "m"}, MagicMock())
    assert exit_info.value.code == 3

    with patch.object(cmd_export, "export_unified", side_effect=RemoteCodeNotAllowedError("m")):
        with pytest.raises(SystemExit) as exit_info:
            cmd_export._execute_export_kwargs({"model_name": "m"}, MagicMock())
    assert exit_info.value.code == 2

    from model_exporter.errors import ExportFailedError

    for error in (InvalidModelReferenceError("model_name must not start with '-'"), ExportFailedError("m", "boom")):
        with patch.object(cmd_export, "export_unified", side_effect=error):
            with pytest.raises(SystemExit) as exit_info:
                cmd_export._execute_export_kwargs({"model_name": "m"}, MagicMock())
        assert exit_info.value.code == 1
        assert "[ERROR]" in capsys.readouterr().out


# -- values are passed on unchanged ---------------------------------------------


def _runner_norm():
    source = Path(subprocess_runner.__file__).read_text(encoding="utf-8")
    body = re.search(r"(def _norm\(d\):.*?)\n(?=\S)", source, flags=re.S).group(1)
    namespace: dict = {}
    exec(body, namespace)  # the helper exactly as the child process receives it
    return namespace["_norm"]


def test_subprocess_arguments_keep_their_types():
    values = {"model_name_or_path": "123", "output": "none", "task": "true", "revision": "1.0", "nested": ["false", "7"]}
    given = json.loads(json.dumps(values))
    _runner_norm()(given)
    assert given == values


def test_subprocess_arguments_are_still_renamed():
    given = {"model_name": "org/model", "opset_version": 17, "onnx_path": "out", "torch_dtype": "float16"}
    _runner_norm()(given)
    assert given == {"model_name_or_path": "org/model", "opset": 17, "output": "out", "dtype": "float16"}


# -- fallbacks ------------------------------------------------------------------


def _run_fallbacks(tmp_path, allow):
    attempted = []

    def fake_subprocess(me_try, logger):
        attempted.append(me_try.get("dtype"))
        return True, ""

    with (
        patch.object(pipeline_v2, "_build_fallback_strategies", return_value=[("half", {"dtype": "float16"}), ("opset_14", {"opset": 14})]),
        patch.object(pipeline_v2, "_run_main_export_subprocess", side_effect=fake_subprocess),
        patch.object(pipeline_v2, "_move_working_to_output"),
    ):
        ok, _ = pipeline_v2._execute_fallback_loop(
            {"opset": 17}, "boom", "org/model", str(tmp_path), str(tmp_path), False, log, allow_precision_fallback=allow
        )
    return ok, attempted


def test_precision_changing_fallback_needs_permission(tmp_path):
    ok, attempted = _run_fallbacks(tmp_path, allow=False)
    assert ok and attempted == [None], "the float16 strategy must be skipped"
    record = json.loads((tmp_path / "export_fallback.json").read_text(encoding="utf-8"))
    assert record == {"fallback": "opset_14", "changed": {"opset": 14}}


def test_precision_changing_fallback_runs_when_allowed_and_is_recorded(tmp_path):
    ok, attempted = _run_fallbacks(tmp_path, allow=True)
    assert ok and attempted == ["float16"]
    record = json.loads((tmp_path / "export_fallback.json").read_text(encoding="utf-8"))
    assert record == {"fallback": "half", "changed": {"dtype": "float16"}}


# -- other exports' temporary files ---------------------------------------------


def test_temp_cleanup_leaves_recent_directories_alone(tmp_path, monkeypatch):
    from model_exporter.export import helpers
    from model_exporter.export.path_utils import STALE_TEMP_SECONDS

    recent = tmp_path / "onnx_export_running"
    stale = tmp_path / "onnx_export_abandoned"
    for d in (recent, stale):
        d.mkdir()
    old = os.path.getmtime(stale) - STALE_TEMP_SECONDS - 60
    os.utime(stale, (old, old))
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    helpers.cleanup_temporary_export_artifacts(logger=log, max_age_seconds=STALE_TEMP_SECONDS)

    assert recent.exists(), "a directory another export may be using was removed"
    assert not stale.exists()
    assert STALE_TEMP_SECONDS >= 3600


def test_export_passes_a_stale_age_to_every_cleanup_call():
    root = Path(pipeline_v2.__file__).parent
    for name in ("pipeline.py", "pipeline_v2.py"):
        for call in re.findall(r"cleanup_temporary_export_artifacts\((.*?)\)", (root / name).read_text(encoding="utf-8")):
            assert "max_age_seconds=STALE_TEMP_SECONDS" in call, f"{name}: {call}"


def test_python_version_guard():
    assert sys.version_info >= (3, 11)


# -- public API ----------------------------------------------------------------


def test_documented_import_of_export_is_callable():
    """``from model_exporter import export`` once returned the sub-package, which could
    not be called. Checked in a fresh interpreter, where import order is what a user gets."""
    import subprocess

    code = (
        "from model_exporter import export, export_model\n"
        "import model_exporter.export.pipeline as pipeline\n"
        "assert callable(export) and callable(export_model)\n"
        "assert export_model is pipeline.export\n"
        "try:\n    export('')\nexcept ValueError as exc:\n    print('called:', exc)\n"
    )
    src = str(Path(pipeline_v2.__file__).parents[2])
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": src}, timeout=300)
    assert result.returncode == 0, result.stderr[-800:]
    assert "called: model_name cannot be empty" in result.stdout


# -- a failed export says why ---------------------------------------------------


def test_failed_export_error_carries_the_last_reason():
    from model_exporter.errors import ExportFailedError

    pipeline_v2.reset_export_failure()
    pipeline_v2._note_failure("line one\nOSError: org/missing is not a valid model identifier\n")
    detail, cause = pipeline_v2.last_export_failure()
    error = ExportFailedError("org/missing", detail)

    assert cause is None
    assert str(error) == "All export attempts failed for model org/missing: OSError: org/missing is not a valid model identifier"
    assert isinstance(error, RuntimeError) and isinstance(error, ExporterError)
    assert error.detail.startswith("line one")

    pipeline_v2.reset_export_failure()
    assert pipeline_v2.last_export_failure() == ("", None)


def test_failure_notes_are_per_thread():
    import threading

    pipeline_v2.reset_export_failure()
    pipeline_v2._note_failure("main thread failure")
    seen = []
    worker = threading.Thread(target=lambda: seen.append(pipeline_v2.last_export_failure()))
    worker.start()
    worker.join()
    assert seen == [("", None)]
    assert pipeline_v2.last_export_failure()[0] == "main thread failure"
    pipeline_v2.reset_export_failure()


# -- revision pinning -----------------------------------------------------------


def test_revision_reaches_the_exporter(tmp_path):
    captured = {}

    def fake_inprocess(me_kwargs, working_output, output_dir, child_tmp, logger):
        captured.update(me_kwargs)
        return True, "", child_tmp

    with (
        patch.object(pipeline_v2, "_attempt_inprocess_export", side_effect=fake_inprocess),
        patch.object(pipeline_v2, "cleanup_temporary_export_artifacts"),
    ):
        pipeline_v2.export_v2_main_export(
            "org/model", str(tmp_path), "fe", 17, "cpu", "feature-extraction", "pt", "transformers", False, log, revision="abc123"
        )
        assert captured["revision"] == "abc123"
        captured.clear()
        pipeline_v2.export_v2_main_export("org/model", str(tmp_path), "fe", 17, "cpu", "feature-extraction", "pt", "transformers", False, log)
        assert "revision" not in captured


def test_revision_is_an_export_option_and_a_cli_flag():
    import argparse

    from model_exporter.cli.cmd_export import _add_export_arguments
    from model_exporter.export.options import ExportConfig

    parser = argparse.ArgumentParser()
    _add_export_arguments(parser)
    args = parser.parse_args(["--model-name", "org/model", "--revision", "abc123"])
    assert ExportConfig.from_namespace(args, None).to_kwargs()["revision"] == "abc123"
    assert ExportConfig(model_name="org/model").to_kwargs()["revision"] is None
