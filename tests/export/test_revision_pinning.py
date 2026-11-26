# =============================================================================
# File: test_revision_pinning.py
# Date: 2026-11-26
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Every Hugging Face Hub download honours the caller's ``revision``.

The hub functions are replaced by fakes that record their calls, so nothing is
downloaded.
"""

import logging
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

REVISION = "0123456789abcdef0123456789abcdef01234567"
LOG = logging.getLogger("test_revision_pinning")


class _Recorder:
    """A ``from_pretrained``-like callable that records (source, kwargs)."""

    def __init__(self, result: Any = None, fail_for: tuple = (), error: Exception | None = None) -> None:
        self.calls: list[tuple[Any, dict]] = []
        self._result = result if result is not None else MagicMock()
        self._fail_for = fail_for
        self._error = error or OSError("not here")

    def __call__(self, source: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((source, kwargs))
        if source in self._fail_for:
            raise self._error
        return self._result


def _fake_transformers(monkeypatch, **classes: _Recorder) -> None:
    module = types.ModuleType("transformers")
    for name, loader in classes.items():
        setattr(module, name, types.SimpleNamespace(from_pretrained=loader))
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))


def _hub_calls(recorder: _Recorder, source: str) -> list[dict]:
    return [kwargs for called_with, kwargs in recorder.calls if called_with == source]


def test_prepare_local_model_dir_pins_model_and_tokenizer(monkeypatch, tmp_path):
    from model_exporter.export import helpers

    dest = str(tmp_path / "prep")
    model_loader = _Recorder()
    # The tokenizer is first read from the prepared folder; make that fail so the
    # hub download runs.
    tokenizer_loader = _Recorder(fail_for=(dest,))
    _fake_transformers(monkeypatch, AutoModelForCausalLM=model_loader, AutoTokenizer=tokenizer_loader)

    assert helpers.prepare_local_model_dir("org/model", dest, False, LOG, revision=REVISION) is True

    assert [kw["revision"] for kw in _hub_calls(model_loader, "org/model")] == [REVISION]
    assert [kw["revision"] for kw in _hub_calls(tokenizer_loader, "org/model")] == [REVISION]


def test_prepare_local_model_dir_pins_the_snapshot_download(monkeypatch, tmp_path):
    from model_exporter.export import helpers

    dest = str(tmp_path / "prep")
    snap_path = str(tmp_path / "prep" / "_hf_snapshot" / "snapshots" / REVISION)
    model_loader = _Recorder(fail_for=("org/model",), error=ValueError("Transformers does not recognize this model type"))
    _fake_transformers(monkeypatch, AutoModelForCausalLM=model_loader, AutoTokenizer=_Recorder())

    snapshot_calls: list[dict] = []

    def snapshot_download(**kwargs: Any) -> str:
        snapshot_calls.append(kwargs)
        return snap_path

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert helpers.prepare_local_model_dir("org/model", dest, True, LOG, revision=REVISION) is True

    assert snapshot_calls[0]["repo_id"] == "org/model"
    assert snapshot_calls[0]["revision"] == REVISION
    # The model is then loaded from the folder snapshot_download returned.
    assert _hub_calls(model_loader, snap_path)


def test_use_cache_detection_reads_the_pinned_config(monkeypatch):
    from model_exporter.export import pipeline_helpers

    config_loader = _Recorder(result=types.SimpleNamespace(is_encoder_decoder=False, use_cache=False, model_type="bert"))
    _fake_transformers(monkeypatch, AutoConfig=config_loader)

    pipeline_helpers._resolve_use_cache("org/model", "sc", None, LOG, revision=REVISION)

    assert [kw["revision"] for kw in _hub_calls(config_loader, "org/model")] == [REVISION]


def test_remote_code_check_reads_the_pinned_config(monkeypatch):
    from model_exporter.export import pipeline_helpers

    config_loader = _Recorder()
    _fake_transformers(monkeypatch, AutoConfig=config_loader)

    pipeline_helpers._auto_resolve_trust_remote_code("org/model", "tok", False, LOG, revision=REVISION)

    (kwargs,) = _hub_calls(config_loader, "org/model")
    assert kwargs["revision"] == REVISION
    assert kwargs["trust_remote_code"] is False


def test_fallback_exporter_loads_every_component_at_the_revision(monkeypatch):
    from model_exporter.export import optimum_export

    loaders = {name: _Recorder() for name in ("AutoTokenizer", "AutoModel", "AutoConfig")}
    _fake_transformers(monkeypatch, **loaders)

    optimum_export._load_transformers_components("org/model", False, None, LOG, REVISION)

    for name, loader in loaders.items():
        assert [kw["revision"] for kw in _hub_calls(loader, "org/model")] == [REVISION], name


def test_snapshot_strategy_prepares_the_pinned_revision(monkeypatch):
    from model_exporter.export import helpers, optimum_export

    seen: dict[str, Any] = {}

    def prepare(model_name: str, dest_dir: str, trust_remote_code: bool, logger: Any, revision: Any = None) -> bool:
        seen["revision"] = revision
        return True

    monkeypatch.setattr(helpers, "prepare_local_model_dir", prepare)
    base = {"model_name_or_path": "org/model", "trust_remote_code": True, "revision": REVISION}

    me_try, cleanup, err = optimum_export._prepare_strategy(base, {"__snapshot__": True}, "org/model", LOG)
    try:
        assert err is None
        assert seen["revision"] == REVISION
    finally:
        if cleanup:
            cleanup()


def test_pooling_config_download_is_pinned(monkeypatch, tmp_path):
    from model_exporter.validation import numeric

    cfg = tmp_path / "config.json"
    cfg.write_text('{"pooling_mode_cls_token": true}', encoding="utf-8")
    calls: list[tuple[tuple, dict]] = []

    def hf_hub_download(*args: Any, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return str(cfg)

    monkeypatch.setattr(numeric, "hf_hub_download", hf_hub_download)

    assert numeric._load_pooling_config("org/model", REVISION) == {"pooling_mode_cls_token": True}
    assert calls == [(("org/model", "1_Pooling/config.json"), {"revision": REVISION})]


@pytest.mark.parametrize("revision, expected", [(REVISION, ["--revision", REVISION]), (None, [])])
def test_validate_onnx_forwards_the_revision(monkeypatch, revision, expected):
    from model_exporter.validation import numeric

    seen: dict[str, list[str]] = {}

    def main(argv: list[str]) -> int:
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(numeric, "main", main)

    assert numeric.validate_onnx("out", "org/model", revision=revision) == 0
    argv = seen["argv"]
    if expected:
        i = argv.index("--revision")
        assert argv[i : i + 2] == expected
    else:
        assert "--revision" not in argv
    assert numeric._parse_validation_args(argv).revision == revision


def test_invoke_validator_forwards_the_revision(monkeypatch):
    from model_exporter.validation import invoker, numeric

    seen: dict[str, Any] = {}

    def validate_onnx(**kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(numeric, "validate_onnx", validate_onnx)

    assert invoker.invoke_validator("out", "org/model", False, False, LOG, revision=REVISION) == 0
    assert seen["revision"] == REVISION
