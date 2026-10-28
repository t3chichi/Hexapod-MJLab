"""Explicit finite GPU integration check; exits 2 when CUDA is unavailable."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="perform two-env one-update training and strict restore")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({"status": "unavailable", "gpu_acceptance": False,
                          "reason": "CUDA/driver unavailable; no training attempted"}))
        raise SystemExit(2)
    if not args.run:
        print(json.dumps({"status": "available", "gpu": torch.cuda.get_device_name(args.gpu),
                          "training_attempted": False, "next": "pass --run for finite acceptance"}))
        return
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    from dataclasses import asdict
    from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low.env import make_env_cfg
    from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low.runner import TripodRunner, TripodRlCfg
    device = f"cuda:{args.gpu}"
    root = Path(tempfile.mkdtemp(prefix="tripod-gpu-check-"))
    cfg = asdict(TripodRlCfg(seed=1, max_iterations=1, formal_num_envs=2, engineering_validation=True))

    def create(directory):
        env = ManagerBasedRlEnv(make_env_cfg(num_envs=2, seed=1), device=device)
        try:
            return TripodRunner(RslRlVecEnvWrapper(env), cfg, log_dir=directory, device=device)
        except BaseException:
            env.close()
            raise

    first = create(root / "first")
    try:
        first.learn(1)
        checkpoint = first.final_checkpoint
        # Save the causal state, advance one action, then compare with a new
        # environment restored from the completed-update checkpoint.
        before = first.runtime.prepare()
        action = torch.zeros(2, 18, device=device)
        expected = first.runtime.step(action)
        expected_qpos = expected.context_record["qpos"].clone()
        expected_reward = expected.reward.clone()
    finally:
        first.env.unwrapped.close()
    second = create(root / "restored")
    try:
        from hexapod_mjlab.common.checkpoint import model_physics_identity
        from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low.checkpoint import _model_tensors, read_checkpoint
        nominal = model_physics_identity(second.runtime.env.sim.mj_model)
        saved_model = read_checkpoint(checkpoint)["model"]
        with torch.no_grad():
            for value in _model_tensors(second.runtime.env).values():
                value.mul_(.5)  # Different live DR state, same nominal model.
        second.load(checkpoint)
        assert nominal == model_physics_identity(second.runtime.env.sim.mj_model)
        for name, value in _model_tensors(second.runtime.env).items():
            assert torch.equal(value.cpu(), saved_model[name]), name
        after = second.runtime.prepare()
        if not torch.equal(before.near_points, after.near_points) or not torch.equal(
                before.observation.proprio_vector(), after.observation.proprio_vector()):
            raise AssertionError("strict restore changed cached policy inputs")
        actual = second.runtime.step(action)
        torch.testing.assert_close(actual.context_record["qpos"], expected_qpos, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual.reward, expected_reward, atol=1e-6, rtol=1e-5)
        print(json.dumps({"status": "passed", "finite_updates": 1,
                          "randomized_model_restored": True,
                          "strict_next_transition": True, "artifacts": str(root),
                          "learning_effect_verified": False}))
    finally:
        second.env.unwrapped.close()


if __name__ == "__main__":
    main()
