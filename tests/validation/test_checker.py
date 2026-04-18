import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast



def import_checker_module(monkeypatch) -> Any:
    fake_onnx = cast(Any, types.ModuleType("onnx"))
    fake_onnx.checker = SimpleNamespace(check_model=lambda *args, **kwargs: None)
    fake_onnx.load = lambda path: {"loaded_from": path}
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    sys.modules.pop("model_exporter.validation.checker", None)
    return importlib.import_module("model_exporter.validation.checker")



def test_safe_check_model_prefers_path_checker_for_external_data(monkeypatch, tmp_path):
    checker = import_checker_module(monkeypatch)

    model_path = tmp_path / "model.onnx"
    model_path.write_text("placeholder", encoding="utf-8")
    external_data_path = tmp_path / "model.onnx.onnx_data"
    external_data_path.write_text("external", encoding="utf-8")

    def fake_run(cmd, capture_output, text, timeout):
        script = Path(cmd[1]).read_text(encoding="utf-8")
        assert "onnx.checker.check_model(sys.argv[1])" in script
        assert "onnx.load(sys.argv[1])" not in script
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "ok"}), stderr="")

    monkeypatch.setattr(checker.subprocess, "run", fake_run)

    ok, info = checker._safe_check_model(str(model_path))

    assert ok is True
    assert info == "ok"



def test_verify_models_discovers_onnx_files_when_expected_are_missing(monkeypatch, tmp_path):
    checker = import_checker_module(monkeypatch)

    model_path = tmp_path / "model.onnx"
    model_path.write_text("placeholder", encoding="utf-8")

    class FakeMeta:
        def __init__(self, name: str):
            self.name = name

    class FakeSession:
        def get_inputs(self):
            return [FakeMeta("input_ids")]

        def get_outputs(self):
            return [FakeMeta("sentence_embedding")]

    monkeypatch.setattr(checker, "_safe_check_model", lambda path: (True, "ok"))
    monkeypatch.setattr(checker, "load", lambda path: {"loaded_from": path})
    monkeypatch.setattr(checker, "has_external_data", lambda model: False)
    monkeypatch.setattr(checker, "create_ort_session", lambda path, provider=None: FakeSession())
    monkeypatch.setattr(checker.glob, "glob", lambda pattern: [str(model_path)])

    result = checker.verify_models(["missing-model.onnx"], str(tmp_path))

    assert result is True
