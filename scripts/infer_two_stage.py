#!/usr/bin/env python3
"""
Backward-compatible entry point: delegates to ``infer.py`` (same CLI).

Prefer: ``python scripts/infer.py --refinement-dir <run_dir> -i <input> -o <output>``
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parent / "infer.py"
    if not target.is_file():
        print(f"infer.py not found next to this file: {target}", file=sys.stderr)
        sys.exit(1)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
