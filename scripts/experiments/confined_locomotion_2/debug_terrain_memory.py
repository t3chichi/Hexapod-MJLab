"""Terrain Memory browser diagnostics: heights, envelopes and source rays."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from play_terrain_memory import TerrainMemoryPlayOwner
from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play_cli import main
from hexapod_mjlab.tasks.confined_locomotion_2.visualization.terrain_debug import TerrainMemoryDebugViewer

if __name__ == "__main__":
    main(owner_type=TerrainMemoryPlayOwner, viewer_type=TerrainMemoryDebugViewer, debug=True)
