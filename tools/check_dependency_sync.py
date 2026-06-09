# =============================================================================
# File: tools/check_dependency_sync.py
# Date: 2026-06-08
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Check that runtime requirements mirror pyproject metadata."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_requirements(path: Path) -> list[str]:
    requirements = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        requirements.append(line)
    return requirements


def main() -> int:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pyproject_runtime = list(pyproject["project"]["dependencies"])
    requirements_runtime = _read_requirements(REPO_ROOT / "requirements-prod.txt")

    if pyproject_runtime == requirements_runtime:
        print("Runtime dependency mirrors are in sync.")
        return 0

    print("Runtime dependency mirrors are out of sync.", file=sys.stderr)
    print("\npyproject.toml [project.dependencies]:", file=sys.stderr)
    for dependency in pyproject_runtime:
        print(f"  {dependency}", file=sys.stderr)
    print("\nrequirements-prod.txt:", file=sys.stderr)
    for dependency in requirements_runtime:
        print(f"  {dependency}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
