# =============================================================================
# File: test_memory.py
# Date: 2026-04-18
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any, cast


def import_memory_module(monkeypatch, virtual_memory_impl) -> Any:
    fake_psutil = cast(Any, types.ModuleType("psutil"))
    fake_psutil.virtual_memory = virtual_memory_impl
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    sys.modules.pop("model_exporter.utils.memory", None)
    return importlib.import_module("model_exporter.utils.memory")


def test_get_memory_info_uses_psutil_values(monkeypatch):
    fake_mem = SimpleNamespace(
        total=16 * 1024**3,
        used=6 * 1024**3,
        available=10 * 1024**3,
        percent=37.5,
    )
    memory = import_memory_module(monkeypatch, lambda: fake_mem)

    info = memory.get_memory_info()

    assert info == {
        "total_gb": 16.0,
        "used_gb": 6.0,
        "free_gb": 10.0,
        "percent_used": 37.5,
    }


def test_memory_guard_triggers_cleanup_and_recovers(monkeypatch):
    memory = import_memory_module(monkeypatch, lambda: None)
    calls = {"cleanup": 0}
    states = iter([False, True])

    monkeypatch.setattr(memory, "check_memory_available", lambda min_free_gb: next(states))
    monkeypatch.setattr(
        memory,
        "aggressive_cleanup",
        lambda: calls.__setitem__("cleanup", calls["cleanup"] + 1),
    )

    assert memory.memory_guard(min_free_gb=4.0, auto_cleanup=True) is True
    assert calls["cleanup"] == 1


def test_memory_monitor_cleans_up_on_exit(monkeypatch):
    memory = import_memory_module(monkeypatch, lambda: None)
    cleanup_calls = {"count": 0}
    samples = iter(
        [
            {"used_gb": 2.0, "free_gb": 6.0, "total_gb": 8.0, "percent_used": 25.0},
            {"used_gb": 2.4, "free_gb": 5.6, "total_gb": 8.0, "percent_used": 30.0},
        ]
    )

    monkeypatch.setattr(memory, "get_memory_info", lambda: next(samples))
    monkeypatch.setattr(memory, "log_memory_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(memory, "check_memory_available", lambda min_free_gb: True)
    monkeypatch.setattr(
        memory,
        "aggressive_cleanup",
        lambda: cleanup_calls.__setitem__("count", cleanup_calls["count"] + 1),
    )

    with memory.MemoryMonitor("Export", min_free_gb=2.0) as monitor:
        assert monitor.operation_name == "Export"

    assert cleanup_calls["count"] == 1
