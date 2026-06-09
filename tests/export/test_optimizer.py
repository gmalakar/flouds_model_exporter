# =============================================================================
# File: test_optimizer.py
# Date: 2026-06-08
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

import logging


def test_optimize_if_encoder_reports_missing_optimizer(monkeypatch, tmp_path):
    from model_exporter.export import optimizer

    (tmp_path / "model.onnx").write_bytes(b"fake")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".optimizations_applied").write_text("stale", encoding="utf-8")

    def _raise_import_error():
        raise ImportError("missing optimum")

    monkeypatch.setattr(optimizer, "_load_ort_optimizer_classes", _raise_import_error)

    rc = optimizer.optimize_if_encoder(
        model_dir=tmp_path,
        model_type="ranker",
        logger=logging.getLogger("test-missing-optimizer"),
    )

    assert rc == 1
    assert not (tmp_path / ".optimizations_applied").exists()


def test_optimize_if_encoder_fails_when_no_optimized_artifacts(monkeypatch, tmp_path):
    from model_exporter.export import optimizer

    class FakeOptimizationConfig:
        def __init__(self, optimization_level):
            self.optimization_level = optimization_level

    class FakeORTOptimizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def optimize(self, *args, **kwargs):
            return None

    (tmp_path / "model.onnx").write_bytes(b"fake")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        optimizer,
        "_load_ort_optimizer_classes",
        lambda: (FakeORTOptimizer, FakeOptimizationConfig),
    )

    rc = optimizer.optimize_if_encoder(
        model_dir=tmp_path,
        model_type="ranker",
        logger=logging.getLogger("test-empty-optimizer-output"),
    )

    assert rc == 1
    assert not (tmp_path / ".optimizations_applied").exists()
