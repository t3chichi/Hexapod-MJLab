"""Launch the migrated recipe only with an explicit iteration budget."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log-root", default="logs/confined_locomotion_2")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--execution-mode", choices=("low_memory", "speed"), default="low_memory")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--engineering-validation", action="store_true")
    parser.add_argument("--raycast-mode", choices=("full", "sparse"), default="full")
    parser.add_argument("--sparse-selection-path")
    parser.add_argument("--far-sampling", choices=("all_endpoints", "hit_only_10m"), default="all_endpoints")
    args = parser.parse_args()
    from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low.entry import launch
    result = launch(max_iterations=args.iterations, **{k: v for k, v in vars(args).items() if k != "iterations"})
    print(f"Checkpoint: {result.checkpoint_path}")
    print("Finite engineering completion does not establish locomotion performance.")


if __name__ == "__main__":
    main()
