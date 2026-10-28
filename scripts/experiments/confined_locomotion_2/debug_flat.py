"""Flat Baseline event-driven browser diagnostics beside native Play."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play_cli import main

if __name__ == "__main__":
    main(debug=True)
