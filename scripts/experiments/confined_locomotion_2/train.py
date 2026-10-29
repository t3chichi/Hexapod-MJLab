"""Explicit local launch; no seed, environment count or budget is inferred."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log-root", default="logs/confined_locomotion_2")
    parser.add_argument("--execution-mode", choices=("low_memory", "speed"), default="low_memory",
                        help="low_memory recomputes point-GRU activations; speed retains them")
    parser.add_argument("--point-replay-chunk", type=int, default=1024,
                        help="positive point-GRU replay scan batch cap (default 1024; try 256/128 for less VRAM)")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="save a full checkpoint every N cumulative completed PPO iterations")
    parser.add_argument("--engineering-validation", action="store_true")
    parser.add_argument("--raycast-mode", choices=("full", "sparse"), default="sparse")
    parser.add_argument("--far-sampling", choices=("all_endpoints", "hit_only_10m"),
                        default="hit_only_10m")
    parser.add_argument("--sparse-selection", "--sparse-selection-path",
                        dest="sparse_selection_path")
    args = parser.parse_args()
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.entry import launch_formal_training
    result = launch_formal_training(
        seed=args.seed, num_envs=args.num_envs, max_iterations=args.iterations,
        gpu_ids=(args.gpu,), log_root=args.log_root,
        engineering_validation=args.engineering_validation,
        resume_checkpoint=args.resume_checkpoint,
        raycast_mode=args.raycast_mode,
        sparse_selection_path=args.sparse_selection_path, far_sampling=args.far_sampling,
        save_interval=args.save_interval, execution_mode=args.execution_mode,
        point_replay_chunk=args.point_replay_chunk)
    print(f"Checkpoint: {result.checkpoint_path}")
    print("Engineering completion does not establish walking or terrain-memory performance.")


if __name__ == "__main__":
    main()
