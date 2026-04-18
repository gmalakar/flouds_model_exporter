import sys
from datetime import datetime
from pathlib import Path

HEADER = """# =============================================================================
# File: {filename}
# Date: {date}
# Copyright (c) {year} Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""


def add_header_to_file(filepath: str) -> None:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # If any copyright marker is present, assume header exists
    if "Copyright (c)" in content:
        return  # Already has header
    today = datetime.now().strftime("%Y-%m-%d")
    year = datetime.now().year
    header = HEADER.format(filename=Path(filepath).name, date=today, year=year)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + "\n" + content)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if arg.endswith(".py"):
            try:
                add_header_to_file(arg)
            except Exception as e:
                print(f"Error processing {arg}: {e}", file=sys.stderr)
                sys.exit(1)
