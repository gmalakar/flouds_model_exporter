# =============================================================================
# File: legacy_fallback.py
# Date: 2026-04-17
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Legacy exporter fallback used only after v2 fallback exhaustion."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Optional

from model_exporter.export.subprocess_runner import _run_main_export_subprocess
from model_exporter.utils.helpers import get_default_opset, safe_log


def _run_inprocess_main_export(me_kwargs: dict[str, Any]) -> tuple[bool, str]:
    """Run optimum main_export in-process and return (success, stderr_text)."""
    try:
        from optimum.exporters.onnx import main_export as _main_export

        _main_export(**me_kwargs)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _prepare_clone_source(export_source: str, logger: Any) -> tuple[Optional[str], Optional[str]]:
    """Prepare a temporary local source by copy/clone. Returns (path, error)."""
    tmp_dir = tempfile.mkdtemp(prefix="onnx_v1_clone_")
    try:
        if os.path.isdir(export_source):
            try:
                shutil.copytree(export_source, tmp_dir, dirs_exist_ok=True)
            except TypeError:
                shutil.copytree(export_source, tmp_dir)
            return tmp_dir, None

        repo_to_clone = export_source
        if "/" in export_source and not export_source.startswith("http://") and not export_source.startswith("https://"):
            repo_to_clone = f"https://huggingface.co/{export_source}"

        subprocess.check_call(["git", "clone", "--depth", "1", repo_to_clone, tmp_dir])
        return tmp_dir, None
    except Exception as exc:
        safe_log(logger, "warning", "Legacy clone/source prep failed: %s", exc)
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass
        return None, str(exc)


def run_legacy_v1_fallback(
    export_source: str,
    output_dir: str,
    model_for: str,
    opset_version: int | None,
    device: str,
    task: str | None,
    framework: str | None,
    library: str | None,
    logger: Any,
    trust_remote_code: bool,
    use_external_data_format: bool = False,
    no_post_process: bool = False,
    merge: bool = False,
    use_subprocess: bool = False,
) -> tuple[bool, bool]:
    """Run legacy-style ONNX export fallback.

    Returns (success, used_trust_remote_code).
    """
    del model_for, merge  # Reserved for compatibility with caller signature.

    base_kwargs: dict[str, Any] = {
        "model_name_or_path": export_source,
        "output": output_dir,
        "task": task,
        "opset": int(opset_version or get_default_opset()),
        "device": device or "cpu",
        "use_external_data_format": bool(use_external_data_format),
    }
    if no_post_process:
        base_kwargs["no_post_process"] = True
    if framework:
        base_kwargs["framework"] = framework
    if library:
        base_kwargs["library"] = library
    if trust_remote_code:
        base_kwargs["trust_remote_code"] = True

    attempts: list[tuple[str, dict[str, Any]]] = [
        ("legacy_base", dict(base_kwargs)),
        (
            "legacy_opset14_trust",
            {
                **dict(base_kwargs),
                "opset": 14,
                "trust_remote_code": True,
            },
        ),
        (
            "legacy_opset11_trust",
            {
                **dict(base_kwargs),
                "opset": 11,
                "trust_remote_code": True,
                "use_external_data_format": True,
            },
        ),
    ]

    # Final attempt: clone/copy model to local temp and retry conservatively.
    clone_path, _ = _prepare_clone_source(export_source, logger)
    if clone_path:
        attempts.append(
            (
                "legacy_local_clone",
                {
                    **dict(base_kwargs),
                    "model_name_or_path": clone_path,
                    "trust_remote_code": True,
                    "opset": 11,
                    "use_external_data_format": True,
                },
            )
        )

    try:
        for attempt_name, kwargs_try in attempts:
            safe_log(
                logger,
                "info",
                "Attempting legacy fallback '%s' with trust_remote_code=%s opset=%s",
                attempt_name,
                bool(kwargs_try.get("trust_remote_code", False)),
                kwargs_try.get("opset"),
            )

            if use_subprocess:
                ok, err = _run_main_export_subprocess(kwargs_try, logger)
            else:
                ok, err = _run_inprocess_main_export(kwargs_try)

            if ok:
                safe_log(logger, "info", "Legacy fallback '%s' succeeded", attempt_name)
                return True, bool(kwargs_try.get("trust_remote_code", False))

            safe_log(logger, "warning", "Legacy fallback '%s' failed: %s", attempt_name, err)

        return False, False
    finally:
        if clone_path:
            try:
                shutil.rmtree(clone_path)
            except Exception:
                pass
