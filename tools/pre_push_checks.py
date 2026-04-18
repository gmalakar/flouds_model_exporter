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
import subprocess
import sys

commands = [
    [sys.executable, "-m", "pytest", "-q"],
    ["pre-commit", "run", "-a", "mypy"],
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
