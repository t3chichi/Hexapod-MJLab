#!/usr/bin/env bash
# M5 baseline harness — OLD stack (Isaac Gym / legged_gym, li_leggym conda
# env) training launcher.
#
# Task: `el4090_lidar_head` (registered in
# Lidar_legged_gym/legged_gym/legged_gym/envs/__init__.py from
# envs/el_4090/pd_gru_lidar_head/el_4090_lidar_head_config.py::
# El4090LidarHeadCfg / El4090LidarHeadCfgPPO).
#
# The old train.py writes logs under the old repo's own logs/ tree, which is
# READ-ONLY for this migration.  This launcher therefore reproduces
# train.py's exact flow (make_env -> make_alg_runner -> learn) but passes
# `log_root=<this run dir>` to task_registry.make_alg_runner, so nothing is
# written into /home/t3chichi/Lidar_legged_gym.  The cfg snapshots that
# train.py dumps are reproduced as well.
#
# Plumbing required by the old stack (kept here for the record):
#   * `import isaacgym` MUST precede torch (asserted in launcher);
#   * isaacgym 1.0rc4 / legged_gym / rsl-rl 2.3.3 are pip-editable in
#     li_leggym, so no PYTHONPATH/ISAACGYM_PATH setup is needed;
#   * `--headless` (no viewer), sim + RL on cuda:0 (defaults);
#   * CLI overrides: --num_envs / --max_iterations / --seed (get_args).
#
# Output layout — single top-level logs/ root (same convention as the
# new-stack harness; the old runner has no --log-root default of its own,
# its log_root is passed explicitly):
#   logs/m5/baseline_old/<timestamp>/
#     run_meta.json        num_envs/iters/seed, task, wall time
#     train_stdout.log     full stdout (per-iteration rsl_rl tables)
#     <date>_<run_name>/   TensorBoard events + cfg snapshots + checkpoints
#     per_iteration.csv    extracted from TB events (baseline_compare.py --extract)
#
# Recommended baseline budget for THIS GPU (RTX 4050 6 GB laptop):
#   * Isaac Gym's PhysX GPU pipeline allocates ~4.8 GB of FIXED GPU pools
#     (independent of num_envs), leaving < 1 GB for torch on this card:
#     128 envs OOMs in the LiDAR buffers, 64 envs OOMs in the PPO update
#     (proximal-GRU minibatch); 32 envs trains end-to-end (dry run
#     logs/m5/baseline_old/20260929_182158: collection 0.91 s + learning
#     3.63 s per iteration) and 16 envs matches the old G5 bench config.
#   * => 32 envs x 200 iters x seed 1 ~= 15 min, 153,600 env steps — the
#     SAME env count / iterations / seed as the new-stack harness for a
#     fair baseline_compare.py comparison.
#
# Usage:
#   scripts/bench/run_baseline_isaacgym.sh [NUM_ENVS=32] [ITERS=200] [SEED=1]
#
# 2-minute dry run: scripts/bench/run_baseline_isaacgym.sh 32 2 1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_ENVS="${1:-32}"
ITERS="${2:-200}"
SEED="${3:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$REPO_ROOT/logs/m5/baseline_old"
RUN_DIR="$RUN_ROOT/$TS"
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/run_meta.json" <<EOF
{
  "stack": "old",
  "task": "el4090_lidar_head",
  "num_envs": $NUM_ENVS,
  "iters": $ITERS,
  "seed": $SEED,
  "num_steps_per_env": 24,
  "started": "$(date -Is)",
  "gpu": "RTX 4050 6GB (laptop)",
  "engine": "isaacgym 1.0rc4 (PhysX) + rsl_rl 2.3.3 (vendored)",
  "cmd": "conda run -n li_leggym python launch_train.py (inlined below)"
}
EOF

# Launcher mirrors legged_gym/legged_gym/scripts/train.py with a redirected
# log_root; config passed via env vars (old get_args owns sys.argv).
cat > "$RUN_DIR/launch_train.py" <<'PYEOF'
import isaacgym  # noqa: F401  must precede torch
import json
import os
import time

import torch

import legged_gym.envs  # noqa: F401  task registration
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict

NUM_ENVS = os.environ["M5_NUM_ENVS"]
ITERS = os.environ["M5_ITERS"]
SEED = os.environ["M5_SEED"]
RUN_DIR = os.environ["M5_RUN_DIR"]

argv_backup = __import__("sys").argv
__import__("sys").argv = [
    "old_train",
    "--task=el4090_lidar_head",
    f"--num_envs={NUM_ENVS}",
    "--headless",
    f"--max_iterations={ITERS}",
    f"--seed={SEED}",
]
args = get_args()
__import__("sys").argv = argv_backup

t0 = time.time()
env, env_cfg = task_registry.make_env(name=args.task, args=args)
ppo_runner, train_cfg = task_registry.make_alg_runner(
    env=env, name=args.task, args=args, log_root=RUN_DIR
)
os.makedirs(ppo_runner.log_dir, exist_ok=True)
with open(os.path.join(ppo_runner.log_dir, "env_cfg_snapshot.json"), "w",
          encoding="utf-8") as f:
    json.dump(class_to_dict(env_cfg), f, indent=2, ensure_ascii=False)
with open(os.path.join(ppo_runner.log_dir, "train_cfg_snapshot.json"), "w",
          encoding="utf-8") as f:
    json.dump(class_to_dict(train_cfg), f, indent=2, ensure_ascii=False)
print(f"[m5-old] log_dir={ppo_runner.log_dir}")
ppo_runner.learn(
    num_learning_iterations=train_cfg.runner.max_iterations,
    init_at_random_ep_len=True,
)
print(f"[m5-old] WALL_TIME_S {time.time() - t0:.1f}")
PYEOF

export M5_NUM_ENVS="$NUM_ENVS" M5_ITERS="$ITERS" M5_SEED="$SEED" M5_RUN_DIR="$RUN_DIR"

# Wait for a mostly-idle GPU (a previous training process can take ~30 s to
# release its context; launching into it OOMs on this 6 GB card).
for _ in $(seq 1 60); do
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$FREE_MB" -ge 4500 ] && break
  echo "[m5-old] waiting for GPU (free ${FREE_MB} MiB)..."
  sleep 5
done

START=$(date +%s)
conda run --no-capture-output -n li_leggym python "$RUN_DIR/launch_train.py" \
  2>&1 | tee "$RUN_DIR/train_stdout.log" || true
END=$(date +%s)

python3 - "$RUN_DIR" "$START" "$END" <<'EOF'
import json, sys
from pathlib import Path
run_dir, start, end = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
meta_path = run_dir / "run_meta.json"
meta = json.loads(meta_path.read_text())
meta["wall_time_s"] = end - start
meta_path.write_text(json.dumps(meta, indent=2))
print(f"[m5-old] wall time {end - start} s -> {run_dir}")
EOF

# Machine-readable per-iteration log from the TB events.
conda run --no-capture-output -n hp_mjlab python "$REPO_ROOT/scripts/tools/baseline_compare.py" \
  --extract "$RUN_DIR" || echo "[m5-old] WARN: TB extraction failed"
