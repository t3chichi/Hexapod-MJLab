"""Terrain Memory native Play on the actual compiled Heightfield."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from hexapod_mjlab.tasks.confined_locomotion_2.config.terrain_memory_config import TASK_ID, make_play_env_cfg
from hexapod_mjlab.tasks.confined_locomotion_2.play_terrain_memory import TerrainMemoryPlaySession
from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play_cli import PlayOwner, main


class TerrainMemoryPlayOwner(PlayOwner):
    task_id = TASK_ID
    env_factory = staticmethod(make_play_env_cfg)
    session_class = TerrainMemoryPlaySession


if __name__ == "__main__":
    main(owner_type=TerrainMemoryPlayOwner)
