"""Terrain Memory inference over the shared play lifecycle; no optimization."""

from hexapod_mjlab.rl.confined_locomotion_2.network import TerrainMemoryNetwork
from .config.terrain_memory_config import TASK_ID
from .play import PlaySession, PlayConfigError
from .rl.checkpoint import read_checkpoint
from .components.terrain_query import HeightGridCfg
from . import display_terrain_memory


def checkpoint_grid(checkpoint, *, far_sampling=None):
    payload = read_checkpoint(checkpoint, expected_task_id=TASK_ID, expected_far_sampling=far_sampling)
    saved = payload["identity"]["recipe"]["height_grid"]
    grid = HeightGridCfg(tuple(saved["x_range_m"]), tuple(saved["y_range_m"]))
    if grid.identity() != saved:
        raise PlayConfigError("checkpoint height grid semantics differ")
    return grid


class TerrainMemoryPlaySession(PlaySession):
    task_id = TASK_ID
    network_type = TerrainMemoryNetwork
    display = display_terrain_memory

    def __init__(self, *, checkpoint, **kwargs):
        self.height_grid = checkpoint_grid(checkpoint)
        super().__init__(checkpoint=checkpoint, **kwargs)
        if self.runtime.height_grid != self.height_grid or self.runtime.terrain_query is None:
            self.close()
            raise PlayConfigError("Terrain Memory play requires its real terrain and checkpoint grid")
