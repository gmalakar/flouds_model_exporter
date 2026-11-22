# =============================================================================
# File: tests/integration/test_real_export.py
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

"""Real exports: download a tiny model, export it, and run the result.

The unit suite replaces the exporter, the hub and the file helpers with fakes, so it
cannot show that an export works. These tests can. They need network access and the
full runtime stack, so they only run when asked for::

    pytest -m integration tests/integration
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

TINY_MODEL = "hf-internal-testing/tiny-random-BertModel"


@pytest.fixture()
def isolated_hub(tmp_path, monkeypatch):
    """A private Hugging Face home, so the run neither uses nor leaves credentials."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "ONNX_PATH"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.mark.parametrize("use_subprocess", [False, True], ids=["in-process", "subprocess"])
def test_tiny_model_exports_and_runs(isolated_hub, use_subprocess):
    import numpy
    import onnxruntime

    from model_exporter import export

    output = Path(export(model_name=TINY_MODEL, model_for="fe", onnx_path=str(isolated_hub / "onnx"), use_subprocess=use_subprocess, force=True))

    assert output == isolated_hub / "onnx" / "models" / "fe" / "tiny-random-BertModel"
    assert (output / "model.onnx").is_file()
    assert (output / "config.json").is_file()
    assert not (output / ".export.lock").exists(), "the lock must be released"
    assert not (output / "export_fallback.json").exists(), "a plain export must not need a fallback"

    session = onnxruntime.InferenceSession(str(output / "model.onnx"), providers=["CPUExecutionProvider"])
    feed = {item.name: numpy.ones((1, 4), dtype=numpy.int64) for item in session.get_inputs()}
    hidden = session.run(None, feed)[0]
    assert hidden.shape[:2] == (1, 4)
    assert numpy.isfinite(hidden).all()

    # Nothing about the caller's credentials or environment changed.
    assert "HF_TOKEN" not in os.environ and "HUGGINGFACE_HUB_TOKEN" not in os.environ
    assert not list((isolated_hub / "hf_home").glob("token")) and not list((isolated_hub / "hf_home").glob("stored_tokens"))


def test_existing_export_is_reused_without_force(isolated_hub):
    from model_exporter import export

    first = export(model_name=TINY_MODEL, model_for="fe", onnx_path=str(isolated_hub / "onnx"))
    stamp = (Path(first) / "model.onnx").stat().st_mtime_ns
    second = export(model_name=TINY_MODEL, model_for="fe", onnx_path=str(isolated_hub / "onnx"))
    assert second == first
    assert (Path(second) / "model.onnx").stat().st_mtime_ns == stamp


def test_missing_model_fails_with_a_reason(isolated_hub):
    from model_exporter import export
    from model_exporter.errors import ExporterError, ExportFailedError

    with pytest.raises(ExporterError) as raised:
        export(model_name="hf-internal-testing/this-model-does-not-exist-flouds", model_for="fe", onnx_path=str(isolated_hub / "onnx"))

    assert isinstance(raised.value, ExportFailedError)
    message = str(raised.value)
    assert message.startswith("All export attempts failed for model hf-internal-testing/this-model-does-not-exist-flouds")
    assert len(message) > len("All export attempts failed for model hf-internal-testing/this-model-does-not-exist-flouds"), "the reason is missing"
    # A failed export leaves no half-written model directory behind.
    assert not (isolated_hub / "onnx" / "models" / "fe" / "this-model-does-not-exist-flouds" / "model.onnx").exists()
