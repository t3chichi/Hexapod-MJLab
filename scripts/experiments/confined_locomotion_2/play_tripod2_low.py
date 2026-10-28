"""Finite deterministic policy diagnostic; no learning or implicit resets."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--command", type=float, nargs=3, default=(.5, 0., 0.))
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    import torch
    from torch import nn
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    from hexapod_mjlab.rl.confined_locomotion_2.network import ConfinedLocomotionNetwork
    from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.scanner import ConfinedLidarCfg
    from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low.checkpoint import read_checkpoint
    from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low.env import make_env_cfg
    from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low.runtime import LegacyRuntime
    payload = read_checkpoint(args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("tripod policy playback requires CUDA; CPU checks are not GPU verification")
    device = f"cuda:{args.gpu}"
    torch.manual_seed(args.seed)
    net = ConfinedLocomotionNetwork(mlp_activation=nn.ELU(), init_fn=None,
                                    envelope_mode="raw", ground_samples=None).to(device).eval()
    net.load_state_dict(payload["network"], strict=True)
    directory = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="tripod-play-"))
    directory.mkdir(parents=True, exist_ok=True)
    # Use the exact acquisition settings saved by the training scanner.
    saved_lidar = payload["identity"]["recipe"]["lidar_config"]
    lidar = ConfinedLidarCfg(**saved_lidar)
    env = ManagerBasedRlEnv(make_env_cfg(num_envs=1, seed=args.seed, play=True), device=device)
    runtime = None
    try:
        runtime = LegacyRuntime(RslRlVecEnvWrapper(env), seed=args.seed,
                                output_dir=directory / "runtime", mode="play", lidar_cfg=lidar)
        if runtime.scan_identity != payload["identity"]["scan_identity"]:
            raise ValueError("play scanner differs from checkpoint")
        command = torch.tensor(args.command, device=device).reshape(1, 3)
        if not bool(torch.isfinite(command).all()) or bool((command.abs() > command.new_tensor([2.5, 1., 1.2])).any()):
            raise ValueError("command outside legacy curriculum limits")
        runtime.set_manual_command([0], command)
        hidden = net.initial_hidden(1, device)
        with (directory / "play_metrics.jsonl").open("w") as stream, torch.inference_mode():
            for step in range(args.steps):
                prepared = runtime.prepare()
                output = net(prepared.observation, prepared.near_points, prepared.near_valid,
                             prepared.far_points, prepared.far_valid, hidden,
                             near_angle_keys=prepared.near_angle_keys, far_angle_keys=prepared.far_angle_keys)
                hidden = output.hidden
                transition = runtime.step(output.action_mean)
                stream.write(json.dumps({"step": step + 1, "reward": float(transition.reward[0]),
                    "root_height_m": float(transition.context_record["root_pos_w"][0, 2]),
                    "flipped": bool(transition.terminated[0])}) + "\n")
                if bool(transition.terminated.any()):
                    break
        print(f"Finite play diagnostics: {directory}")
    finally:
        if runtime is not None:
            runtime.close()
        env.close()


if __name__ == "__main__":
    main()
