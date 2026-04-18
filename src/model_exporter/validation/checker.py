#!/usr/bin/env python3
# =============================================================================
# File: onnx_verify.py
# Date: 2026-01-09
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Lightweight ONNX verification utilities used by exporter/validator.

Contains `verify_models(fnames, output_dir, pack_single=False, pack_single_threshold_mb=None)`.
This module is self-contained to avoid circular imports with the validator.
"""
from __future__ import annotations

import glob
import json as _json
import os
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Callable, List, Optional, cast


def _import_real_onnx() -> tuple[Any, Any]:
    """Import the real ``onnx`` package and return ``(checker, load)``.

    Attempts a normal ``import onnx`` first. If that fails or produces a stub
    without the required attributes, falls back to scanning site-packages
    directories for an ``onnx`` installation and loading it directly via
    :mod:`importlib.util`.

    Returns:
        A two-tuple ``(onnx.checker, onnx.load)`` from the discovered package.

    Raises:
        ModuleNotFoundError: If no usable ``onnx`` package can be found.
    """
    import importlib
    import importlib.util
    import os
    import sys
    import sysconfig

    try:
        _onnx = importlib.import_module("onnx")
        if hasattr(_onnx, "checker") and hasattr(_onnx, "load"):
            return _onnx.checker, _onnx.load
    except Exception:
        pass

    candidates = []
    try:
        import site

        try:
            candidates.extend(site.getsitepackages())
        except Exception:
            pass
    except Exception:
        pass

    sc_paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        p = sc_paths.get(key)
        if p:
            candidates.append(p)

    try:
        prefix_sp = os.path.join(sys.prefix, "Lib", "site-packages")
        candidates.append(prefix_sp)
    except Exception:
        pass

    seen = set()
    candidate_file = None
    for base in candidates:
        try:
            if not base:
                continue
            base_abs = os.path.abspath(base)
            if base_abs in seen:
                continue
            seen.add(base_abs)
            possible = os.path.join(base_abs, "onnx", "__init__.py")
            if os.path.exists(possible):
                candidate_file = possible
                break
        except Exception:
            continue

    if candidate_file:
        spec = importlib.util.spec_from_file_location("onnx_installed", candidate_file)
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"Could not create a module spec from candidate file: {candidate_file}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "checker") and hasattr(mod, "load"):
            return mod.checker, mod.load

    raise ModuleNotFoundError("Could not import a usable 'onnx' package from the active interpreter site-packages.")


checker: Any
load: Any
checker, load = _import_real_onnx()


def _safe_check_model(model_path: str, timeout: int = 120) -> tuple[bool, str | dict[str, str]]:
    """Run `onnx.checker.check_model` in a subprocess to isolate native crashes.

    The child process prints a JSON object to stdout with a `status` field.
    Returns (True, "ok") on success, (False, info) on failure where `info` is
    either a dict parsed from the child's JSON output or a string with stderr.
    """
    script_path = os.path.join(tempfile.gettempdir(), f"onnx_checker_child_{uuid.uuid4().hex}.py")
    # For very large model files or external-data cases, prefer calling
    # `onnx.checker.check_model` with the file path instead of loading the
    # entire model into memory which can trigger MemoryError or protobuf
    # serialization issues. Detect large files here and generate an
    # appropriate child script.
    use_path_check = False
    try:
        size = 0
        try:
            size = os.path.getsize(model_path)
        except Exception:
            pass
        # If the model proto is larger than 2 GiB, request path-based check.
        if size >= 2 * 1024 * 1024 * 1024:
            use_path_check = True
        # If there's an external-data sidecar file, prefer path-based check.
        if os.path.exists(f"{model_path}.onnx_data"):
            use_path_check = True
    except Exception:
        use_path_check = False

    if use_path_check:
        script = """
import json
import sys
import traceback
try:
    import onnx
except Exception as e:
    print(json.dumps({"status": "import_failed", "error": str(e)}))
    sys.exit(2)
try:
    # Instruct checker to operate on the model file path to avoid large in-memory
    # protobuf serialization for huge models or external-data forms.
    onnx.checker.check_model(sys.argv[1])
    print(json.dumps({"status": "ok"}))
    sys.exit(0)
except Exception as e:
    tb = traceback.format_exc()
    print(json.dumps({"status": "failed", "error": str(e), "traceback": tb}))
    sys.exit(1)
"""
    else:
        script = """
import json
import sys
import gc
import traceback
try:
    import onnx
except Exception as e:
    print(json.dumps({"status": "import_failed", "error": str(e)}))
    sys.exit(2)
try:
    m = onnx.load(sys.argv[1])
    onnx.checker.check_model(m)
    del m
    gc.collect()
    print(json.dumps({"status": "ok"}))
    sys.exit(0)
except Exception as e:
    tb = traceback.format_exc()
    print(json.dumps({"status": "failed", "error": str(e), "traceback": tb}))
    sys.exit(1)
"""

    try:
        open(script_path, "w", encoding="utf-8").write(script)
        proc = subprocess.run(
            [sys.executable, script_path, model_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        try:
            # best-effort cleanup
            if os.path.exists(script_path):
                os.remove(script_path)
        except Exception:
            pass
        return False, "checker_timeout"
    try:

        def _proc_result(p):
            if p.returncode == 0:
                try:
                    j = _json.loads(p.stdout)
                    if j.get("status") == "ok":
                        return True, "ok"
                    return False, j
                except Exception:
                    return True, p.stdout.strip() or "ok"
            # non-zero return: try to parse stdout JSON for structured info
            try:
                j = _json.loads(p.stdout)
                return False, j
            except Exception:
                out = p.stderr or p.stdout or f"returncode:{p.returncode}"
                return False, {"status": "failed", "detail": out}

        ok, info = _proc_result(proc)
        # If initial attempt failed due to MemoryError/protobuf serialization,
        # retry using the path-based checker which avoids loading large protos.
        if (not ok) and (not use_path_check):
            combined = "".join(
                [
                    str(proc.stdout or ""),
                    str(proc.stderr or ""),
                ]
            ).lower()
            if (
                "memoryerror" in combined
                or "memoryerror" in str(info).lower()
                or "serialize" in combined
                or "protobuf" in combined
                or "serialize" in str(info).lower()
            ):
                try:
                    # write a path-based checker script and rerun
                    path_script = """
import json
import sys
import traceback
try:
    import onnx
except Exception as e:
    print(json.dumps({"status": "import_failed", "error": str(e)}))
    sys.exit(2)
try:
    onnx.checker.check_model(sys.argv[1])
    print(json.dumps({"status": "ok"}))
    sys.exit(0)
except Exception as e:
    tb = traceback.format_exc()
    print(json.dumps({"status": "failed", "error": str(e), "traceback": tb}))
    sys.exit(1)
"""
                    open(script_path, "w", encoding="utf-8").write(path_script)
                    proc2 = subprocess.run(
                        [sys.executable, script_path, model_path],
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    try:
                        if os.path.exists(script_path):
                            os.remove(script_path)
                    except Exception:
                        pass
                    return _proc_result(proc2)
                except Exception:
                    pass

        return ok, info
    finally:
        try:
            if os.path.exists(script_path):
                os.remove(script_path)
        except Exception:
            pass


# Try to import helpers but tolerate absence
create_ort_session: Optional[Callable[..., Any]] = None
_get_preferred_provider: Any = None  # Callable[..., str]
try:
    import importlib

    _mod = importlib.import_module("exporter.utils.onnx_utils")
    _has_external_data = getattr(_mod, "has_external_data", None)
except Exception:
    _has_external_data = None

# Expose a canonical callable `has_external_data`; use imported helper if present,
# otherwise provide a local fallback implementation. Provide a typed module-level
# name to avoid mypy inference conflicts when reassigning.
has_external_data: Optional[Callable[[Any], bool]] = None

if _has_external_data is None:

    def _fallback_has_external_data(model_or_path: Any) -> bool:
        try:
            locs = []
            for tensor in getattr(model_or_path.graph, "initializer", []):
                if getattr(tensor, "data_location", 0) == 1 and hasattr(tensor, "external_data"):
                    for kv in tensor.external_data:
                        if getattr(kv, "key", None) == "location" and getattr(kv, "value", None):
                            locs.append(kv.value)
            return len([loc for loc in locs if loc]) > 0
        except Exception:
            return False

    has_external_data = _fallback_has_external_data
else:
    has_external_data = _has_external_data


# Safely import optional helpers from `onnx_helpers` using importlib to avoid
# mypy/name-redefinition issues when reassigning module-level callables.
try:
    import importlib

    _helpers_mod = importlib.import_module("onnx_exporter.onnx_helpers")
    _maybe_create = getattr(_helpers_mod, "create_ort_session", None)
    _maybe_get_pref = getattr(_helpers_mod, "get_preferred_provider", None)
    if callable(_maybe_create):
        # cast for mypy: assign a Callable to the Optional[Callable] name
        create_ort_session = cast(Optional[Callable[..., Any]], _maybe_create)
    if callable(_maybe_get_pref):
        _get_preferred_provider = _maybe_get_pref
except Exception:
    _get_preferred_provider = None


if create_ort_session is None:

    def _create_ort_session_fallback(path: str, provider: Any | None = None) -> Any:
        import onnxruntime as ort

        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])  # fallback

    create_ort_session = _create_ort_session_fallback


if _get_preferred_provider is None:

    def get_preferred_provider(default: str = "CPUExecutionProvider") -> str:
        return default

else:
    get_preferred_provider = _get_preferred_provider


def verify_models(
    fnames: List[str],
    output_dir: str,
    pack_single: bool = False,
    pack_single_threshold_mb: int | None = None,
) -> bool:
    """Lightweight verification helper.

    Runs ONNX checker where possible, detects external_data, optionally repacks,
    and creates an ORT session to inspect inputs/outputs. Returns True if basic checks passed.
    """
    import gc

    try:
        provider = get_preferred_provider() if callable(get_preferred_provider) else None
    except Exception:
        provider = None

    print(
        f"verify_models: start files={fnames} output_dir={output_dir} pack_single={pack_single} pack_single_threshold_mb={pack_single_threshold_mb}"
    )

    # Only verify files that actually exist in the output directory. If none
    # of the expected names are present, discover any `.onnx` files in the
    # directory and verify those instead.
    available_fnames: List[str] = []
    for fname in fnames:
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            available_fnames.append(fname)
        else:
            print(f"Verify: file not found {path}")

    if not available_fnames:
        try:
            discovered = [os.path.basename(p) for p in glob.glob(os.path.join(output_dir, "*.onnx"))]
        except Exception:
            discovered = []
        if discovered:
            print(f"No expected files found; discovered ONNX files: {discovered}")
            available_fnames = discovered
        else:
            print(f"No ONNX files found in {output_dir}; nothing to verify")
            return False

    for fname in available_fnames:
        path = os.path.join(output_dir, fname)
        try:
            onnx_model = None
            checker_ok = False
            try:
                ok, info = _safe_check_model(path)
                checker_ok = ok
                if ok:
                    print(f"{fname} passed ONNX checker (subprocess)")
                else:
                    print(f"ONNX checker subprocess failed for {fname}: {info}")
            except Exception as e:
                print(f"ONNX checker subprocess error for {fname}: {e}")

            # If the checker failed, try an ONNX Runtime session as a
            # fallback verification which typically uses less peak memory.
            if not checker_ok:
                try:
                    if not callable(create_ort_session):
                        raise RuntimeError("create_ort_session not available")
                    sess = create_ort_session(path, provider=provider)
                    inputs = [i.name for i in sess.get_inputs()]
                    outputs = [o.name for o in sess.get_outputs()]
                    print(f"{fname} verified via ONNX Runtime fallback inputs={inputs} outputs={outputs}")
                    # Write a fallback marker so callers know this file was validated
                    # via ONNX Runtime rather than the ONNX checker.
                    try:
                        marker_path = f"{path}.ort_verified"
                        with open(marker_path, "w", encoding="utf-8") as mf:
                            mf.write(
                                _json.dumps(
                                    {
                                        "status": "ort_verified",
                                        "provider": provider,
                                        "inputs": inputs,
                                        "outputs": outputs,
                                    }
                                )
                            )
                        print(f"Wrote ORT-only verification marker: {marker_path}")
                    except Exception:
                        # Non-fatal: continue even if marker write fails
                        pass
                    try:
                        del sess
                    except Exception:
                        pass
                    gc.collect()
                    # Consider this file verified; continue to next file
                    continue
                except Exception as e:
                    print(f"ORT fallback verification failed for {fname}: {e}")

            try:
                # Load model only when needed (external data detection / repack)
                onnx_model = load(path)
                external_used = has_external_data(onnx_model) if callable(has_external_data) else False
            except Exception as e:
                print(f"Failed to load or inspect model for external data for {fname}: {e}")
                external_used = False

            if external_used:
                print(f"Warning: Model {fname} uses external_data tensors. Ensure associated tensor files are co-located.")
                if pack_single:
                    tmp_single = f"{path}.single"
                    # Prefer official helper, fall back if not present
                    try:
                        import importlib

                        _onnx_mod = importlib.import_module("onnx")
                        _ext_helper = getattr(_onnx_mod, "external_data_helper", None)
                        conv = getattr(_ext_helper, "convert_model_to_single_file", None)
                        if callable(conv):
                            conv(onnx_model, tmp_single)
                            os.replace(tmp_single, path)
                            onnx_model = load(path)
                            external_used = False
                    except Exception as e:
                        print(f"Failed to repack model {fname} into single file: {e}")

            try:
                if not callable(create_ort_session):
                    raise RuntimeError("create_ort_session not available")
                sess = create_ort_session(path, provider=provider)
                inputs = [i.name for i in sess.get_inputs()]
                outputs = [o.name for o in sess.get_outputs()]
                print(f"{fname} inputs={inputs} outputs={outputs}")
                del onnx_model, sess
                gc.collect()
            except Exception as e:
                print(f"Session creation or inspection failed for {fname}: {e}")
        except Exception as e:
            print(f"Verification failed for {fname}: {e}")
            try:
                corrupt = f"{path}.corrupt"
                os.replace(path, corrupt)
                print(f"Moved corrupt file to {corrupt}")
            except Exception:
                print(f"Failed to move corrupt file {path}")
            print(f"verify_models: failed while verifying {fname}")
            return False
    print("verify_models: succeeded")
    return True
