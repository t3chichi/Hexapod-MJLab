"""Exclusive artifact directory shared by standalone checks and the runner."""

from datetime import datetime
import os
from pathlib import Path


def artifact_dir(script):
    configured = os.environ.get("HEXAPOD_TEST_OUTPUT")
    if configured:
        output = Path(configured)
    else:
        root = Path(__file__).resolve().parents[1]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = root / "logs/verification/standalone" / (Path(script).stem + "-" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    return output
