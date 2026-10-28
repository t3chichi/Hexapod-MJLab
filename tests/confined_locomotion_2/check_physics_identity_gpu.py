"""Finite CUDA rejection/relocation witnesses using a new Flat hit10 checkpoint."""

import argparse
import copy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import torch

from check_gpu import differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    from hexapod_mjlab.tasks.confined_locomotion_2.rl import checkpoint as cp
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.entry import build_formal_train_config
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import ConfinedLocomotionRunner
    from hexapod_mjlab.common import flat_env

    payload = cp.read_checkpoint(args.checkpoint, expected_far_sampling="hit_only_10m")
    seed, count = payload["identity"]["seed"], payload["identity"]["num_envs"]
    if count > 2:
        raise ValueError("finite witness supports at most two environments")
    selection = Path(__file__).resolve().parents[2] / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_hit10/near_x6_far_x10.npz"
    moved = args.output / "relocated_flat_env.py"
    moved.write_text(Path(flat_env.__file__).read_text())
    spec = importlib.util.spec_from_file_location("relocated_flat_env", moved)
    relocated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(relocated)
    legacy = copy.deepcopy(payload)
    legacy["identity"]["compiled_model"].pop("physics")
    legacy_path = args.output / "legacy_without_physics.pt"
    torch.save(legacy, legacy_path)
    results = {}

    def snapshot(runner):
        env = runner.runtime.env
        return cp.cpu_tree({
            **{name: getattr(runner, name).state_dict() for name in
               ("network", "provider", "trainer", "runtime", "collector", "learning_metrics")},
            "physics": cp._physics_arrays(env)[0], "phase": runner.phase,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()},
        })

    for case in ("same_model", "relocated_source", "changed_mass", "legacy_missing_physics"):
        cfg = build_formal_train_config(
            seed=seed, num_envs=count, max_iterations=1, engineering_validation=True,
            raycast_mode="sparse", sparse_selection_path=str(selection),
            far_sampling="hit_only_10m")
        if case == "relocated_source":
            cfg = replace(cfg, env=relocated.make_training_env_cfg(num_envs=count, seed=seed))
        if case == "changed_mass":
            original = cfg.env.scene.entities["robot"].spec_fn
            def changed_spec():
                robot = original()
                body = next(body for body in robot.bodies if body.name == "BASE")
                assert body.mass == 10
                body.mass = 20
                return robot
            cfg.env.scene.entities["robot"].spec_fn = changed_spec
        env = ManagerBasedRlEnv(cfg.env, device="cuda:0")
        runner = None
        try:
            runner = ConfinedLocomotionRunner(RslRlVecEnvWrapper(env, clip_actions=None),
                asdict(cfg.agent), args.output / case, "cuda:0")
            if case in ("changed_mass", "legacy_missing_physics"):
                before = snapshot(runner)
                try:
                    runner.load(legacy_path if case == "legacy_missing_physics" else args.checkpoint)
                except ValueError as error:
                    assert "physical model" in str(error)
                    if case == "changed_mass":
                        assert "body_mass" in str(error)
                    results[case] = {"rejected": True, "reason": str(error)}
                else:
                    raise AssertionError(f"{case} was accepted")
                delta = differences(before, snapshot(runner))
                assert not delta, delta
                results[case]["training_state_and_rng_unchanged"] = True
                if case == "legacy_missing_physics":
                    cp.load_weights(copy.deepcopy(runner.network), legacy_path,
                                    expected_far_sampling="hit_only_10m")
                    results[case]["weights_only_still_loads"] = True
            else:
                runner.load(args.checkpoint)
                assert runner.phase == "ready"
                for name in ("network", "provider", "trainer", "runtime", "collector", "learning_metrics"):
                    assert not differences(payload[name], cp.cpu_tree(getattr(runner, name).state_dict())), name
                assert not differences(payload["physics"], cp._physics_arrays(env)[0])
                results[case] = {"loaded": True, "restored_boundary_equal": True}
        finally:
            if runner is not None:
                runner.close()
            else:
                env.close()
    results["passed"] = True
    (args.output / "checks.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
