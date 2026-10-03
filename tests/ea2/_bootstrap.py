"""Resolve local source and test helpers for direct script execution."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "tests" / "ea2"):
    sys.path.insert(0, str(path))
if not __debug__:
    raise RuntimeError("Verification requires assertions; do not use python -O")
