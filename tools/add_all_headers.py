# =============================================================================
# File: add_all_headers.py
# Date: 2025-08-01
# Copyright (c) 2025 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

import glob
import subprocess
from pathlib import Path


def has_header(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
        return "Copyright (c)" in text
    except Exception:
        return False


py_files = glob.glob("**/*.py", recursive=True)
for f in py_files:
    # Skip virtualenvs and caches
    if ".venv" in f or "__pycache__" in f:
        continue
    p = Path(f)
    # Skip if header already present to avoid duplicates
    if has_header(p):
        continue
    # Call add_header.py with an absolute path to avoid cwd issues
    try:
        subprocess.run(["python", "add_header.py", str(p)], check=False)
    except Exception:
        # best-effort; continue with other files
        pass
