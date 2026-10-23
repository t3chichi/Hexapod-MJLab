"""Small, explicit construction helpers for confined-locomotion tests.

This module owns test-only recipes and constructors. In particular, the M12
grid below is an engineering fixture, not a production default.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch


DEFAULT_SEED = 20261008
DEFAULT_NUM_ENVS = 2
DEFAULT_ROLLOUT_STEPS = 24


def m12_training_grid():
    """Return the explicit 4×3 engineering grid used by finite fixtures."""
    from hexapod_mjlab.tasks.confined_locomotion.ground import (
        VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
        GroundGridCfg,
    )

    return GroundGridCfg.from_ranges(
        (0.1, 0.8),
        (-0.3, 0.3),
        4,
        3,
        validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
    )


def build_network(*, seed: int = DEFAULT_SEED, ground_samples: int | None = 12,
                  device: str | torch.device = "cpu"):
    """Build one seeded policy network with the requested grid head width."""
    from hexapod_mjlab.rl.confined_locomotion.network import ConfinedLocomotionNetwork

    torch.manual_seed(seed)
    network = ConfinedLocomotionNetwork(
        mlp_activation=torch.nn.ELU(),
        init_fn=None,
        envelope_mode="raw",
        ground_samples=ground_samples,
    )
    return network.to(device)


def build_provider(*, seed: int = DEFAULT_SEED,
                   device: str | torch.device = "cpu"):
    """Build the production action-distribution provider with a seeded stream."""
    from hexapod_mjlab.rl.confined_locomotion.distribution import (
        IndependentDiagonalNormalProvider,
        derive_action_generator_seed,
    )

    return IndependentDiagonalNormalProvider(
        device=device,
        generator_seed=derive_action_generator_seed(seed),
    )


def build_adapter(*, network, provider=None, seed: int = DEFAULT_SEED,
                  device: str | torch.device = "cpu"):
    """Pair a network with an optional production distribution provider."""
    from hexapod_mjlab.rl.confined_locomotion.policy import ConfinedPolicyAdapter

    network = network.to(device)
    if provider is None:
        return ConfinedPolicyAdapter(network=network)
    return ConfinedPolicyAdapter(
        network=network,
        distribution_provider=provider,
        distribution_identity=provider.distribution_identity(),
    )


def build_collector(runtime, *, device: str | torch.device,
                    steps: int = DEFAULT_ROLLOUT_STEPS,
                    seed: int = DEFAULT_SEED):
    """Build a TrainingCollector and its policy adapter for an existing runtime."""
    from hexapod_mjlab.tasks.confined_locomotion.collector import TrainingCollector

    grid_samples = int(runtime.grid_cfg.num_points)
    network = build_network(seed=seed, ground_samples=grid_samples, device=device)
    provider = build_provider(seed=seed, device=device)
    adapter = build_adapter(network=network, provider=provider, seed=seed, device=device)
    collector = TrainingCollector(runtime, adapter, steps=steps)
    return collector, adapter


def build_env_cfg(*, num_envs: int = DEFAULT_NUM_ENVS, seed: int = DEFAULT_SEED):
    """Build an independent task training config without constructing a simulator."""
    from hexapod_mjlab.tasks.confined_locomotion.flat_env_cfg import make_training_env_cfg

    return make_training_env_cfg(num_envs=num_envs, seed=seed)


def build_runtime(output: Path, *, mode: str, device: str, ground_grid,
                  seed: int = DEFAULT_SEED, num_envs: int = DEFAULT_NUM_ENVS,
                  perturbation_cfg=None):
    """Build the real manager environment, official wrapper, and task runtime.

    Callers select this explicitly for an actual backend recipe. CPU contract
    tests should use typed config objects and must not call this function.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

    from hexapod_mjlab.tasks.confined_locomotion.runtime import TrainingRuntime

    cfg = build_env_cfg(num_envs=num_envs, seed=seed)
    env = ManagerBasedRlEnv(cfg, device=device)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
    runtime = TrainingRuntime(
        wrapper,
        grid_cfg=ground_grid,
        seed=seed,
        output_dir=Path(output) / "runtime",
        mode=mode,
        perturbation_cfg=perturbation_cfg,
    )
    return env, wrapper, runtime


def make_plane_model(*, pos=(0.3, -0.2, 0.5), tilt_deg: float = 0.0,
                     second_plane: bool = False, attach: str = "world"):
    """Compile a tiny CPU MuJoCo model for real ground-geometry queries."""
    import mujoco

    spec = mujoco.MjSpec()
    spec.modelname = "confined_ground_fixture"

    def add_plane(body, plane_pos):
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50.0, 50.0, 0.1])
        geom.pos = [float(v) for v in plane_pos]
        geom.friction = [1.0, 0.005, 0.0001]
        if tilt_deg:
            half = math.radians(tilt_deg) / 2.0
            geom.quat = [math.cos(half), math.sin(half), 0.0, 0.0]
        return geom

    if attach == "world":
        add_plane(spec.worldbody, pos)
        if second_plane:
            add_plane(spec.worldbody, (5.0, 5.0, 0.2))
    elif attach == "dynamic":
        body = spec.worldbody.add_body(name="carrier", pos=[0.0, 0.0, 0.0])
        body.add_joint(type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0.0, 1.0, 0.0])
        sphere = body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0.05, 0.05])
        sphere.pos = [0.5, 0.0, 0.0]
        add_plane(body, pos)
    elif attach == "mocap":
        body = spec.worldbody.add_body(name="mocap_plane", mocap=True, pos=[0.0, 0.0, 0.0])
        add_plane(body, pos)
    else:
        raise ValueError(f"unsupported plane fixture attachment {attach!r}")
    return spec.compile()

