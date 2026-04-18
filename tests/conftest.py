# =============================================================================
# File: conftest.py
# Date: 2026-04-18
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def add_src_to_path(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    monkeypatch.syspath_prepend(str(repo_root / "src"))


@pytest.fixture
def cli_module(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> tuple[Any, dict[str, Any]]:
    captured: dict[str, Any] = {"calls": []}

    fake_pipeline = cast(Any, types.ModuleType("model_exporter.export.pipeline"))
    fake_yaml = cast(Any, types.ModuleType("yaml"))
    fake_validator = cast(Any, types.ModuleType("model_exporter.validation.numeric"))
    fake_optimizer = cast(Any, types.ModuleType("model_exporter.export.optimizer"))

    def fake_export_unified(**kwargs: Any) -> None:
        captured["calls"].append(dict(kwargs))
        captured.update(kwargs)

    def fake_validate_main(argv: list[str]) -> int:
        captured["validate_argv"] = list(argv)
        return 0

    def fake_optimize_if_encoder(
        model_dir: str,
        model_for: str,
        logger: Any,
        optimization_level: int,
        portable: bool = False,
    ) -> int:
        captured["optimize_call"] = {
            "model_dir": model_dir,
            "model_for": model_for,
            "logger_name": getattr(logger, "name", None),
            "optimization_level": optimization_level,
            "portable": portable,
        }
        return 0

    def fake_safe_load(handle: Any) -> dict[str, Any]:
        text = handle.read()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "exporter_preferences": {},
                "batch_presets": {
                    "recommended": [
                        {
                            "model_name": "BAAI/bge-base-en-v1.5",
                            "model_for": "fe",
                            "task": "feature-extraction",
                            "library": "transformers",
                            "normalize_embeddings": True,
                        },
                        {
                            "model_name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                            "model_for": "ranker",
                            "task": "sequence-classification",
                            "library": "transformers",
                            "optimize": True,
                        },
                    ]
                },
            }

    fake_pipeline.export = fake_export_unified
    fake_yaml.safe_load = fake_safe_load
    fake_validator.main = fake_validate_main
    fake_optimizer.optimize_if_encoder = fake_optimize_if_encoder

    monkeypatch.setitem(sys.modules, "model_exporter.export.pipeline", fake_pipeline)
    monkeypatch.setitem(sys.modules, "model_exporter.validation.numeric", fake_validator)
    monkeypatch.setitem(sys.modules, "model_exporter.export.optimizer", fake_optimizer)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    for module_name in (
        "model_exporter.cli.main",
        "model_exporter.cli.cmd_export",
        "model_exporter.cli.cmd_validate",
        "model_exporter.cli.cmd_optimize",
        "model_exporter.cli.cmd_batch",
    ):
        sys.modules.pop(module_name, None)

    module = importlib.import_module("model_exporter.cli.main")
    return module, captured
