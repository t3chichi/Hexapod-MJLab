"""Flat Baseline play; starts paused, or runs finite headless inference."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play_cli import (
    PlayOwner, discover_acquisition_options, main, parse_args, run_play,
)

if __name__ == "__main__":
    main()
