# =============================================================================
# File: add_all_headers.py
# Date Created: 2025-08-01
# Date Updated: 2026-06-09
# Copyright (c) 2025 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

import glob
import sys
from datetime import datetime
from pathlib import Path

HEADER = """# =============================================================================
# File: {filename}
# Date Created: {date_created}
# Date Updated: {date_updated}
# Copyright (c) {year} Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""


def has_header(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
        # Consider SPDX or File markers as existing headers too
        return "Copyright (c)" in text or "SPDX-License-Identifier" in text or text.lstrip().startswith("# File:")
    except Exception:
        return False


EXCLUDES = {
    ".venv",
    ".venv311",
    "__pycache__",
    ".git",
    "build",
    "dist",
    ".egg-info",
    ".vs",
    ".vscode",
    ".pytest_cache",
}


def add_header_to_file(path: Path) -> None:
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    header = HEADER.format(
        filename=path.name,
        date_created=date_str,
        date_updated=date_str,
        year=now.year,
    )
    content = path.read_text(encoding="utf-8")
    path.write_text(header + "\n" + content, encoding="utf-8")


def iter_python_files(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(path) for path in paths if Path(path).suffix == ".py"]
    return [Path(path) for path in glob.glob("**/*.py", recursive=True)]


def main(paths: list[str]) -> int:
    errors = 0
    for p in iter_python_files(paths):
        # Skip files in excluded directories
        if any(part in EXCLUDES for part in p.parts):
            continue
        # Skip if header already present to avoid duplicates
        if has_header(p):
            continue
        try:
            add_header_to_file(p)
        except Exception as e:
            errors += 1
            print(f"Error adding header for {p}: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
