#!/usr/bin/env bash
# M5 baseline harness — NEW stack (mjlab) training launcher.
#
# Launches the registered task El-4090-Lidar via the
# official train CLI with TensorBoard logging, tees stdout, and converts the
# TB event files into a per-iteration CSV (machine-readable log) at the end.
#
# Output layout — single top-level logs/ root, per mjlab's official
# convention (mjlab src/mjlab/scripts/train.py: log_root default
# "logs/rsl_rl", runs under {log_root}/{experiment_name}/{timestamp}/;
# docs/source/training/rsl_rl.rst "Checkpoints and logging"):
#   logs/rsl_rl/<experiment_name>/<mjlab-ts>/   training artifacts
#       (TensorBoard events + params dumps + model_*.pt checkpoints)
#       — written by mjlab's own logger at its default root, pointed
#       there explicitly via --log-root so the location does not depend
#       on the caller's CWD;
#   logs/m5/baseline_new/<timestamp>/            harness metadata (this dir,
#       passed to baseline_compare.py --new):
#       run_meta.json        num_envs/iters/seed, exact command, wall time,
#                           mjlab_log_dir (the training-artifact dir above)
#       train_stdout.log     full stdout (per-iteration rsl_rl tables)
#       per_iteration.csv    extracted from TB events (baseline_compare.py --extract)
#
# Recommended baseline budget for THIS GPU (RTX 4050 6 GB laptop), chosen
# for a FAIR two-stack comparison (same env count + iterations + seed) and
# measured end-to-end (rollout + LidarPPO5x update):
#   * new stack (this script):  32 envs -> 6.6 s/iter (collection 1.62 s +
#     learning 4.99 s; dry run logs/m5/baseline_new/20260929_182239)
#     => 32 envs x 200 iters ~= 22 min, 153,600 env steps.
#   * old stack: only <= 32 envs fits its VRAM (Isaac Gym PhysX GPU fixed
#     pools ~4.8 GB regardless of env count; 64 envs OOMs in the PPO
#     update, 128 in the LiDAR buffers on this 6 GB card) -> the shared
#     env count for the comparison is 32 (see run_baseline_isaacgym.sh).
#   * new stack alone scales further (bench: 128 envs trains at ~21 s/iter,
#     256 envs ~2.3 GB env+LiDAR; 512+ OOMs) but per-iteration update cost
#     (grad-checkpointed distal GRU) pushes 256 x 150 past 60 min.
#   Budget: 32 envs x 200 iters x seed 1 on BOTH stacks (~22 min new,
#   ~15 min old; both inside the 15-40 min target).
#
# Usage:
#   scripts/bench/run_baseline_mjlab.sh [NUM_ENVS=32] [ITERS=200] [SEED=1]
#
# 2-minute dry run: scripts/bench/run_baseline_mjlab.sh 32 2 1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_ENVS="${1:-32}"
ITERS="${2:-200}"
SEED="${3:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$REPO_ROOT/logs/m5/baseline_new"
RUN_DIR="$RUN_ROOT/$TS"
mkdir -p "$RUN_DIR"
# mjlab's default training-log root (train.py TrainConfig.log_root), pinned
# absolutely so artifacts land in <repo>/logs/rsl_rl regardless of CWD.
LOG_ROOT="$REPO_ROOT/logs/rsl_rl"

CMD=(conda run --no-capture-output -n hp_mjlab python -m mjlab.scripts.train
  El-4090-Lidar
  --env.scene.num-envs "$NUM_ENVS"
  --agent.max-iterations "$ITERS"
  --agent.seed "$SEED"
  --agent.logger tensorboard
  --log-root "$LOG_ROOT")

cat > "$RUN_DIR/run_meta.json" <<EOF
{
  "stack": "new",
  "task": "El-4090-Lidar",
  "num_envs": $NUM_ENVS,
  "iters": $ITERS,
  "seed": $SEED,
  "num_steps_per_env": 24,
  "started": "$(date -Is)",
  "gpu": "RTX 4050 6GB (laptop)",
  "engine": "mujoco 3.x + mujoco-warp (mjlab 1.6.0)",
  "cmd": "${CMD[*]}"
}
EOF

START=$(date +%s)
# Wait for a mostly-idle GPU (a previous training process can take ~30 s to
# release its context; launching into it OOMs on this 6 GB card).
for _ in $(seq 1 60); do
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$FREE_MB" -ge 4500 ] && break
  echo "[m5-new] waiting for GPU (free ${FREE_MB} MiB)..."
  sleep 5
done
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/train_stdout.log" || true
END=$(date +%s)

# Locate the run dir mjlab's logger created for this launch (newest
# {log_root}/{experiment_name}/{timestamp} dir created since START) and
# record it in run_meta.json so baseline_compare.py can find the TB events.
MJLAB_RUN_DIR="$(find "$LOG_ROOT" -mindepth 2 -maxdepth 2 -type d \
  -newermt "@$START" 2>/dev/null | sort | tail -1 || true)"

python3 - "$RUN_DIR" "$START" "$END" "$MJLAB_RUN_DIR" <<'EOF'
import json, sys
from pathlib import Path
run_dir, start, end = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
mjlab_run_dir = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
meta_path = run_dir / "run_meta.json"
meta = json.loads(meta_path.read_text())
meta["wall_time_s"] = end - start
if mjlab_run_dir:
    meta["mjlab_log_dir"] = mjlab_run_dir
meta_path.write_text(json.dumps(meta, indent=2))
print(f"[m5-new] wall time {end - start} s -> {run_dir}")
if mjlab_run_dir:
    print(f"[m5-new] mjlab training artifacts -> {mjlab_run_dir}")
else:
    print("[m5-new] WARN: no mjlab run dir created under the log root")
EOF

# Machine-readable per-iteration log from the TB events (written next to the
# events in the mjlab run dir, then copied into the harness metadata dir).
if [ -n "$MJLAB_RUN_DIR" ]; then
  conda run --no-capture-output -n hp_mjlab python "$REPO_ROOT/scripts/tools/baseline_compare.py" \
    --extract "$MJLAB_RUN_DIR" \
    && cp "$MJLAB_RUN_DIR/per_iteration.csv" "$RUN_DIR/per_iteration.csv" \
    || echo "[m5-new] WARN: TB extraction failed"
else
  conda run --no-capture-output -n hp_mjlab python "$REPO_ROOT/scripts/tools/baseline_compare.py" \
    --extract "$RUN_DIR" || echo "[m5-new] WARN: TB extraction failed"
fi
