# =============================================================================
# File: export_subprocess.py
# Date: 2026-01-09
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

"""
Subprocess helper that runs `optimum.exporters.onnx.main_export` in a
separate Python process. Extracted from `export_v2.py` to allow reuse and
testing of the in-process path independent of subprocess plumbing.
"""
import json
import os
import shutil as _shutil
import subprocess
import sys
import tempfile as _tempfile

# time not needed at module level here
from typing import Any


def _run_main_export_subprocess(me_kwargs: dict, logger: Any) -> tuple[bool, str]:
    """Run `optimum.exporters.onnx.main_export(**me_kwargs)` in a child
    Python process using a temporary JSON file for arguments.

    Returns (success, stderr_output).
    """
    tmpf = None
    runner_fh = None
    try:
        tmpf = _tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        json.dump(me_kwargs, tmpf)
        tmpf.close()

        runner_fh = _tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py", encoding="utf-8")
        runner_code = """import json,sys
from optimum.exporters.onnx import main_export

def _norm(d):
    if isinstance(d, dict):
        if 'torch_dtype' in d:
            d['dtype'] = d.pop('torch_dtype')
        if 'model_name' in d and 'model_name_or_path' not in d:
            d['model_name_or_path'] = d.pop('model_name')
        if 'opset_version' in d and 'opset' not in d:
            d['opset'] = d.pop('opset_version')
        if 'onnx_path' in d and 'output' not in d:
            d['output'] = d.pop('onnx_path')
        for k, v in list(d.items()):
            if isinstance(v, str):
                lv = v.lower()
                if lv == 'true':
                    d[k] = True
                    continue
                if lv == 'false':
                    d[k] = False
                    continue
                if lv == 'none':
                    d[k] = None
                    continue
                try:
                    if '.' in v:
                        d[k] = float(v)
                    else:
                        d[k] = int(v)
                except Exception:
                    pass
            _norm(d[k])
    elif isinstance(d, list):
        for i, item in enumerate(d):
            if isinstance(item, str):
                li = item.lower()
                if li == 'true':
                    d[i] = True
                    continue
                if li == 'false':
                    d[i] = False
                    continue
                if li == 'none':
                    d[i] = None
                    continue
                try:
                    if '.' in item:
                        d[i] = float(item)
                    else:
                        d[i] = int(item)
                except Exception:
                    pass
            _norm(d[i])

kwargs = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
_norm(kwargs)
import json as _json
import sys as _sys
# Debug logging commented out due to string literal issue
# try:
#     _sys.stderr.write("RUNNER_KWARGS:" + _json.dumps(kwargs)[:2000] + "\\n")
#     try:
#         _sys.stderr.flush()
#     except Exception:
#         pass
# except Exception:
#     try:
#         print('RUNNER_KWARGS:' + _json.dumps(kwargs)[:2000])
#     except Exception:
#         pass
main_export(**kwargs)
"""
        runner_fh.write(runner_code)
        runner_fh.close()
        runner = runner_fh.name

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
        env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
        env["OPENBLAS_NUM_THREADS"] = env.get("OPENBLAS_NUM_THREADS", "1")
        env["NUMEXPR_NUM_THREADS"] = env.get("NUMEXPR_NUM_THREADS", "1")

        out_dir = None
        if isinstance(me_kwargs, dict):
            out_dir = me_kwargs.get("output") or me_kwargs.get("output_dir")
        if out_dir:
            out_dir = os.path.abspath(out_dir)
            child_tmp = os.path.join(out_dir, "onnx_subproc_tmp")
            try:
                os.makedirs(child_tmp, exist_ok=True)
            except Exception:
                child_tmp = out_dir
            env["TMP"] = child_tmp
            env["TEMP"] = child_tmp
            env["TMPDIR"] = child_tmp
            # Do not emit non-essential logging about temp dirs here.

        if runner:
            proc = subprocess.run(
                [sys.executable, "-u", runner, tmpf.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=3600,  # 1 hour timeout to prevent hanging on Windows
            )
        else:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import json,sys; from optimum.exporters.onnx import main_export; main_export(**json.load(open(sys.argv[1])))",
                    tmpf.name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=3600,  # 1 hour timeout to prevent hanging on Windows
            )

        # Suppress verbose subprocess stdout/stderr logging here.
        try:
            if runner and ("proc" in locals()) and proc.returncode != 0:
                logs_root = None
                try:
                    logs_root = os.environ.get("FLOUDS_LOG_DIR")
                except Exception:
                    logs_root = None

                if not logs_root:
                    try:
                        tmpdir = _tempfile.gettempdir()
                        out_dir_dbg = None
                        if isinstance(me_kwargs, dict):
                            out_dir_dbg = me_kwargs.get("output") or me_kwargs.get("output_dir")
                        for h in getattr(logger, "handlers", []) or []:
                            fname = getattr(h, "baseFilename", None)
                            if not (isinstance(fname, str) and os.path.isabs(fname)):
                                continue
                            try:
                                candidate = os.path.dirname(fname)
                                if candidate and (
                                    (
                                        tmpdir
                                        and os.path.commonpath(
                                            [
                                                os.path.abspath(candidate),
                                                os.path.abspath(tmpdir),
                                            ]
                                        )
                                        == os.path.abspath(tmpdir)
                                    )
                                    or (
                                        out_dir_dbg
                                        and os.path.commonpath(
                                            [
                                                os.path.abspath(candidate),
                                                os.path.abspath(out_dir_dbg),
                                            ]
                                        )
                                        == os.path.abspath(out_dir_dbg)
                                    )
                                ):
                                    continue
                                logs_root = candidate
                                break
                            except Exception:
                                continue
                    except Exception:
                        logs_root = None

                if not logs_root:
                    try:
                        logs_root = os.path.join(os.getcwd(), "logs", "onnx_exports")
                    except Exception:
                        logs_root = None

                # Do not persist the temporary runner script to disk. It may
                # contain transient data and can pollute logs/repo. Instead
                # log the subprocess stderr for debugging purposes.
                try:
                    if runner and ("proc" in locals()) and proc.returncode != 0:
                        logger.error(
                            "Export subprocess failed (returncode=%s). Stderr: %s",
                            proc.returncode,
                            (proc.stderr[:2000] if proc.stderr else "<no stderr>"),
                        )
                except Exception:
                    logger.debug("Failed while logging subprocess failure", exc_info=True)
        except Exception:
            logger.debug(
                "Failed while attempting to handle subprocess runner logging",
                exc_info=True,
            )

        try:
            err_text = proc.stderr or ""
            if any(k in err_text for k in ("No space left", "ENOSPC", "Errno 28")):
                tmpdir = env.get("TMP") or env.get("TEMP") or env.get("TMPDIR") or _tempfile.gettempdir()
                tmp_free = None
                try:
                    tmp_free = _shutil.disk_usage(tmpdir).free // (1024 * 1024)
                except Exception:
                    pass

                out_dir = None
                if isinstance(me_kwargs, dict):
                    out_dir = me_kwargs.get("output") or me_kwargs.get("output_dir")
                out_free = None
                if out_dir:
                    try:
                        out_free = _shutil.disk_usage(out_dir).free // (1024 * 1024)
                    except Exception:
                        pass

                logger.error(
                    "Export subprocess hit disk-space error. subprocess_tmp=%s free_mb=%s output=%s free_mb=%s stderr=%s",
                    tmpdir,
                    str(tmp_free) if tmp_free is not None else "?",
                    out_dir,
                    str(out_free) if out_free is not None else "?",
                    (err_text[:1000] if err_text else "<no stderr>"),
                )
        except Exception:
            logger.debug("Failed to emit subprocess disk-space diagnostics", exc_info=True)

        return (proc.returncode == 0, proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        try:
            logger.error(
                "Export subprocess timed out after 3600s (1 hour). Model export may be too large or slow for this system. "
                "Try running with increased timeout or check disk space/memory."
            )
        except Exception:
            pass
        return False, f"Subprocess timeout after 3600s: {e}"
    except Exception as e:
        try:
            # Capture any partial output before the exception
            if "proc" in locals():
                stderr_output = getattr(proc, "stderr", "") or ""
                stdout_output = getattr(proc, "stdout", "") or ""
                returncode = getattr(proc, "returncode", None)

                if returncode == -1073741819 or returncode == 0xC0000005:
                    logger.error(
                        "Export subprocess crashed with access violation (0xC0000005). "
                        "This typically indicates:\n"
                        "  1. Insufficient memory for large model export\n"
                        "  2. Incompatible ONNX Runtime version\n"
                        "  3. Model-specific export issues\n"
                        "Recommendations:\n"
                        "  - Ensure at least 8GB free RAM\n"
                        "  - Try without --optimize flag\n"
                        "  - Update: pip install --upgrade onnxruntime onnx optimum\n"
                        "  - Some models may not support ONNX export on Windows"
                    )
                    return (
                        False,
                        f"Access violation crash (0xC0000005): {stderr_output[:500]}",
                    )

            logger.warning("Failed to launch main_export subprocess: %s", e)
        except Exception:
            pass
        return False, str(e)
    finally:
        # Cleanup temp files
        for f_handle, var_name in [(tmpf, "tmpf"), (runner_fh, "runner_fh")]:
            try:
                if var_name == "tmpf" and f_handle is not None:
                    os.unlink(f_handle.name)
                elif var_name == "runner_fh" and f_handle is not None and f_handle.name and os.path.exists(f_handle.name):
                    os.unlink(f_handle.name)
            except Exception:
                pass
