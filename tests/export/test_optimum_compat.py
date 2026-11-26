# =============================================================================
# File: tests/export/test_optimum_compat.py
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

"""optimum's ONNX export configs work on Python 3.14 (functools.partial binding)."""

import functools
import sys

import pytest

from model_exporter.export.optimum_compat import patch_partial_config_attributes

py314 = pytest.mark.skipif(sys.version_info < (3, 14), reason="the partial binding change is Python 3.14+")


def test_does_nothing_before_python_314(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 13, 9))
    assert patch_partial_config_attributes() == 0


@py314
def test_t5_onnx_config_builds_after_patch():
    transformers = pytest.importorskip("transformers")
    model_configs = pytest.importorskip("optimum.exporters.onnx.model_configs")
    patch_partial_config_attributes()
    onnx_config = model_configs.T5OnnxConfig(transformers.T5Config(), task="text2text-generation")
    assert onnx_config._normalized_config.config.model_type == "t5"


@py314
def test_configs_defined_after_patch_and_repeat_calls():
    normalized = pytest.importorskip("optimum.utils.normalized_config")
    base = pytest.importorskip("optimum.exporters.base")
    patch_partial_config_attributes()

    class Probe(base.ExporterConfig):  # defined after the patch: with_args gives a staticmethod
        NORMALIZED_CONFIG_CLASS = normalized.NormalizedTextConfig.with_args(num_layers="n_layer")

    Probe.__abstractmethods__ = frozenset()
    probe = object.__new__(Probe)
    config = probe.NORMALIZED_CONFIG_CLASS({"n_layer": 2})
    assert isinstance(config, normalized.NormalizedTextConfig)

    patch_partial_config_attributes()  # idempotent
    assert isinstance(vars(Probe)["NORMALIZED_CONFIG_CLASS"], staticmethod)


@py314
def test_partials_on_already_loaded_classes_are_wrapped():
    normalized = pytest.importorskip("optimum.utils.normalized_config")
    base = pytest.importorskip("optimum.exporters.base")

    class Loaded(base.ExporterConfig):
        pass

    Loaded.NORMALIZED_CONFIG_CLASS = functools.partial(normalized.NormalizedTextConfig, allow_new=True)
    assert patch_partial_config_attributes() >= 1
    assert isinstance(vars(Loaded)["NORMALIZED_CONFIG_CLASS"], staticmethod)
