"""Terrain Memory's independent real Heightfield and learning configuration.

The engineering grid is explicit; a positive height-loss coefficient must be
chosen before a learning recipe can run. Flat defaults are never modified.
"""

from dataclasses import dataclass, field
import numpy as np

from .flat_baseline import ConfinedLocomotionRlCfg

TASK_ID = "El-4090-Confined-Locomotion-Terrain-Memory_2"
IMPLEMENTED = True


def make_collection_env_cfg(*, terrain, num_envs: int, seed: int):
    """Real Heightfield MDP for collection only; deliberately unregistered."""
    from .terrain_geometry import make_geometry_env_cfg
    from hexapod_mjlab.common.mdp.rewards import build_training_reward_cfgs
    from hexapod_mjlab.common.mdp.terminations import build_training_termination_cfgs

    cfg = make_geometry_env_cfg(terrain=terrain, num_envs=num_envs, seed=seed)
    cfg.observations["policy"].nan_policy = "error"
    cfg.rewards = build_training_reward_cfgs()
    cfg.terminations = build_training_termination_cfgs()
    return cfg


def make_default_terrain():
    from .terrain_geometry import HeightfieldTerrainCfg

    # Non-square samples expose accidental axis transposition. The flat spawn
    # platform is part of the SAME genuinely non-flat physical Heightfield.
    x, y = np.linspace(-5, 5, 65), np.linspace(-4, 4, 49)
    xx, yy = np.meshgrid(x, y)
    z = 0.12 + 0.015*xx + 0.01*yy + 0.08*np.sin(0.7*xx)*np.sin(0.8*yy)
    # A raised surface outside the second test pose's robot collision footprint
    # supplies positive labels without placing the root inside terrain.
    z += 0.9*np.maximum(1-abs(xx-1.25)/.35, 0)*np.maximum(1-abs(yy-1.8)/.55, 0)
    z[(abs(xx+2) < 1.35) & (abs(yy) < 1.4)] = 0.12
    return HeightfieldTerrainCfg(
        size=(10., 8.), elevations_m=tuple(map(tuple, z)), spawn_origin=(-2., 0., 0.12))


def make_training_env_cfg(*, num_envs=2, seed=20261025):
    return make_collection_env_cfg(terrain=make_default_terrain(), num_envs=num_envs, seed=seed)


def make_play_env_cfg(*, num_envs=1, seed=20261025):
    cfg = make_training_env_cfg(num_envs=num_envs, seed=seed)
    cfg.terminations.pop("episode_timeout")
    return cfg


@dataclass
class TerrainMemoryRlCfg(ConfinedLocomotionRlCfg):
    experiment_name: str = "confined_locomotion_terrain_memory_2"
    terrain_memory: dict = field(default_factory=lambda: {
        "x_range_m": (-0.75, 0.75), "y_range_m": (-0.65, 0.65), "height_loss_weight": None,
    })


def terrain_memory_rl_cfg():
    """Registered defaults stay guarded until explicit budget and weight exist."""
    return TerrainMemoryRlCfg()


def terrain_memory_formal_rl_cfg(*, seed, num_envs, max_iterations, height_loss_weight):
    from .flat_baseline import confined_locomotion_formal_rl_cfg
    from dataclasses import fields as dataclass_fields
    from hexapod_mjlab.rl.confined_locomotion_2.auxiliary import HeightReconstructionAuxiliary
    HeightReconstructionAuxiliary(height_loss_weight)
    base = confined_locomotion_formal_rl_cfg(seed=seed, num_envs=num_envs, max_iterations=max_iterations)
    fields = {item.name: getattr(base, item.name) for item in dataclass_fields(base)}
    fields["experiment_name"] = "confined_locomotion_terrain_memory_2"
    cfg = TerrainMemoryRlCfg(**fields)
    cfg.terrain_memory["height_loss_weight"] = float(height_loss_weight)
    return cfg

