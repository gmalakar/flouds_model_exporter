#!/usr/bin/env python3
# =============================================================================
# File: pre_push_checks.py
# Date: 2026-04-18
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

"""Run project sanity checks before push: pytest and mypy.
Exits non-zero if any step fails.
"""
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _reexec_with_venv_python() -> None:
    root = _repo_root()
    venv_python = _venv_python(root)
    current_python = Path(sys.executable).resolve()

    if venv_python.exists() and current_python != venv_python.resolve():
        # A child process, not os.execv: on Windows execv starts a new process and
        # ends this one at once, so pre-commit would read exit code 0 before the
        # checks finish. PYTHONHOME is dropped (the hook runs `python -E`).
        env = {key: value for key, value in os.environ.items() if key != "PYTHONHOME"}
        script = str(Path(__file__).resolve())
        completed = subprocess.run([str(venv_python), script, *sys.argv[1:]], env=env, check=False)
        sys.exit(completed.returncode)


_reexec_with_venv_python()

commands = [
    [sys.executable, "-m", "pytest", "-q"],
    [sys.executable, "-m", "pre_commit", "run", "-a", "mypy"],
]

for cmd in commands:
    print("Running: ", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed (exit {e.returncode}): {' '.join(cmd)}")
        sys.exit(e.returncode)

print("All pre-push checks passed.")
sys.exit(0)
