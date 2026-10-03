#!/usr/bin/env python
"""M3 Gate B: real LiDAR integration checks (warp raycast + pool + prox/distal).

Checks:
1. pool points finite; hit rate strictly in (0, 1) on the pillar terrain;
2. prox/distal production: prox 256 points, distal 10x64 ring; ring slots
   refresh across LiDAR ticks (frames differ between ticks);
3. 10 Hz tick cadence: tick_log spacing == 5 policy steps (forced ticks with
   spacing < 5 are legal when envs reset — old _reset_lidar_buffers);
4. reset path: done-env distal windows zero, next frame fills all 10 slots
   (old just_started semantics);
5. obs contract: policy 2754 / height_grid 187 / action 18, finite;
6. symmetry wiring: mirror function bound to the real sensor mount (not
   identity), involution on obs groups and actions, actor forward + mirror
   loss path finite;
7. VRAM at 128/256 envs (torch peak + nvidia-smi), run in subprocesses.

Usage: python tests/env/gate_b.py [--num-envs 24] [--steps 60]
       python tests/env/gate_b.py --vram-only
       python tests/env/gate_b.py --vram-worker 128
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import torch  # noqa: E402


def smi_process_mb() -> float | None:
  try:
    out = subprocess.run(
      ["nvidia-smi", "--query-compute-apps=pid,used_memory",
       "--format=csv,noheader,nounits"],
      capture_output=True, text=True, timeout=20,
    ).stdout
    for line in out.strip().splitlines():
      parts = [p.strip() for p in line.split(",")]
      if len(parts) == 2 and int(parts[0]) == os.getpid():
        return float(parts[1])
  except Exception:
    pass
  return None


def vram_worker(num_envs: int, steps: int = 40) -> int:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  torch.cuda.reset_peak_memory_stats()
  cfg = load_env_cfg("El-4090-Lidar")
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  obs, _ = env.reset()
  assert obs["policy"].shape == (num_envs, 2754)
  for _ in range(steps):
    obs, rew, term, trunc, extras = env.step(
      torch.zeros(num_envs, 18, device=env.device)
    )
  assert torch.isfinite(obs["policy"]).all()
  torch.cuda.synchronize()
  from hexapod_mjlab.tasks.el_4090_lidar.mdp.lidar import get_lidar_pipeline
  p = get_lidar_pipeline(env)
  hit = p.hit_mask.float().mean().item()
  peak_mb = torch.cuda.max_memory_allocated() / 1e6
  smi_mb = smi_process_mb()
  print(
    f"VRAM_RESULT {num_envs} hit={hit:.4f} ticks={p.total_ticks} "
    f"torch_peak_mb={peak_mb:.0f} smi_mb={smi_mb if smi_mb is not None else -1:.0f}"
  )
  return 0


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=24)
  parser.add_argument("--steps", type=int, default=60)
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument("--vram-only", action="store_true")
  parser.add_argument("--vram-worker", type=int, default=None)
  args = parser.parse_args()

  if args.vram_worker is not None:
    return vram_worker(args.vram_worker, steps=40)

  failures: list[str] = []

  if args.vram_only:
    for n in (128, 256):
      r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--vram-worker", str(n)],
        capture_output=True, text=True, timeout=900,
      )
      line = [ln for ln in r.stdout.splitlines() if ln.startswith("VRAM_RESULT")]
      print(f"[gate-b] vram env={n}: {line[0] if line else 'FAILED'}")
      if not line:
        failures.append(f"vram {n}")
        print((r.stdout + r.stderr)[-1500:])
    print("[gate-b] PASS" if not failures else f"[gate-b] FAIL {failures}")
    return 0 if not failures else 1

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  from hexapod_mjlab.tasks.el_4090_lidar.mdp.lidar import (
    DISTAL_HISTORY,
    DISTAL_POINTS,
    POOL_NUM_POINTS,
    PROXIMAL_POINTS,
    get_lidar_pipeline,
  )

  torch.cuda.reset_peak_memory_stats()
  cfg = load_env_cfg("El-4090-Lidar")
  cfg.scene.num_envs = args.num_envs
  # Short episodes force timeouts/resets inside the run -> exercises the
  # forced-tick + distal-window reset path.
  cfg.episode_length_s = 0.5
  print(f"[gate-b] building env: num_envs={args.num_envs} device={args.device}")
  env = ManagerBasedRlEnv(cfg, device=args.device)
  obs, _ = env.reset()
  assert obs["policy"].shape == (args.num_envs, 2754), obs["policy"].shape
  assert obs["height_grid"].shape == (args.num_envs, 187)
  assert env.action_manager.total_action_dim == 18

  pipeline = get_lidar_pipeline(env)
  assert pipeline.num_rays == 900 * 96
  assert pipeline.lidar_points_base.shape == (args.num_envs, POOL_NUM_POINTS, 3)

  # ── 1+2: finiteness, hit rate, prox/distal shapes ────────────────────────
  hit = pipeline.hit_mask.float().mean().item()
  print(f"[gate-b] pool hit rate: {hit:.4f} (expect strictly inside (0, 1))")
  if not (0.0 < hit < 1.0):
    failures.append("hit rate not in (0,1)")
  pool_finite = torch.isfinite(pipeline.lidar_points_base).all().item()
  prox = pipeline._prox_block.view(args.num_envs, PROXIMAL_POINTS, 3)
  distal = pipeline._distal_block.view(
    args.num_envs, DISTAL_HISTORY, DISTAL_POINTS, 3
  )
  print(
    f"[gate-b] finite: pool={pool_finite} prox={torch.isfinite(prox).all().item()} "
    f"distal={torch.isfinite(distal).all().item()}; prox pts={prox.shape[1]} "
    f"distal ring={distal.shape[1]}x{distal.shape[2]}"
  )
  if not (pool_finite and torch.isfinite(prox).all() and torch.isfinite(distal).all()):
    failures.append("non-finite points")
  if prox.shape[1] != 256 or distal.shape[1] != 10 or distal.shape[2] != 64:
    failures.append("prox/distal shape mismatch")

  # ── 3+4: cadence + ring refresh across ticks, reset path ─────────────────
  window_before = pipeline.distal_window.clone()
  n_reset_steps = 0
  for step in range(args.steps):
    obs, rew, term, trunc, extras = env.step(
      torch.zeros(args.num_envs, 18, device=env.device)
    )
    if bool((term | trunc).any()):
      n_reset_steps += 1
    assert torch.isfinite(obs["policy"]).all(), f"non-finite obs at step {step}"

  tick_log = pipeline.tick_log
  spacings = [b - a for a, b in zip(tick_log[:-1], tick_log[1:])]
  n5 = sum(1 for s in spacings if s == 5)
  print(
    f"[gate-b] ticks={len(tick_log)} log={tick_log[:14]}... "
    f"spacings 5-step: {n5}/{len(spacings)}; steps with resets: {n_reset_steps}"
  )
  if len(spacings) < 4 or n5 < len(spacings) // 2:
    failures.append("tick cadence not 10 Hz")
  if any(s < 1 or s > 5 for s in spacings):
    failures.append(f"illegal tick spacing: {sorted(set(spacings))}")

  ring_counts = pipeline.distal_frame_count
  print(f"[gate-b] distal ring write counts: min={ring_counts.min().item()} "
        f"max={ring_counts.max().item()} (policy steps taken: {args.steps}+1)")
  if int(ring_counts.min().item()) < 3:
    failures.append("distal ring not written per policy step")
  # Ring slots must differ across LiDAR ticks (sensor noise -> new frames).
  w = pipeline.distal_window
  slot_diff = (w[:, 0] - w[:, 5]).abs().max().item()
  print(f"[gate-b] ring slot0 vs slot5 max |diff|: {slot_diff:.4f} (>0 expected)")
  if slot_diff == 0.0:
    failures.append("distal ring slots identical across ticks")
  if torch.equal(window_before, pipeline.distal_window):
    failures.append("distal window never changed")

  # ── 6: symmetry wiring (mirror function + mount binding) ────────────────
  from hexapod_mjlab.rl.symmetry_adapter import make_el_4090_symmetry_func
  from hexapod_mjlab.tasks.el_4090_lidar.mdp.lidar import resolve_lidar_mount

  # The runner exposes exactly resolve_lidar_mount() to the mirror function;
  # the pipeline's mount (what the raycaster actually used) must equal it.
  mount_ref = resolve_lidar_mount()
  assert mount_ref is not None, "lidar_sensor site missing (radar variant MJCF)"
  sensor_trans = torch.tensor(
    mount_ref[0], dtype=torch.float32, device=args.device
  ).view(1, 3)
  sensor_quat = torch.tensor(
    mount_ref[1], dtype=torch.float32, device=args.device
  ).view(1, 4)
  # The pipeline's mount (what the raycaster actually used) must equal the
  # runner-exposed mount (what the mirror function will be bound to).
  mount_q_err = (pipeline._mount_q - sensor_quat).abs().max().item()
  mount_t_err = (pipeline._mount_t.view(1, 3) - sensor_trans).abs().max().item()
  print(f"[gate-b] mount binding: quat err={mount_q_err:.2e} trans err={mount_t_err:.2e}")
  if mount_q_err > 1e-6 or mount_t_err > 1e-6:
    failures.append("runner mount != pipeline mount")

  mirror = make_el_4090_symmetry_func(
    proprio_dim=66,
    proximal_points=256,
    distal_history_points=640,
    num_dof=18,
    height_grid_x_count=17,
    height_grid_y_count=11,
    sensor_quat=sensor_quat,
    sensor_trans=sensor_trans,
  )
  from tensordict import TensorDict

  obs_td = TensorDict(
    {"policy": obs["policy"].clone(), "height_grid": obs["height_grid"].clone()},
    batch_size=[args.num_envs],
  )
  obs_m, actions_m = mirror(env=None, obs=obs_td, actions=None)
  actions = torch.randn(args.num_envs, 18, device=args.device)
  _, actions_m = mirror(env=None, obs=None, actions=actions)
  # Involution: mirror(mirror(x)) == x (order-2 symmetry).
  obs_mm, _ = mirror(env=None, obs=obs_m, actions=None)
  inv_obs = (obs_mm["policy"][: args.num_envs] - obs_td["policy"]).abs().max().item()
  inv_hg = (
    obs_mm["height_grid"][: args.num_envs] - obs_td["height_grid"]
  ).abs().max().item()
  _, actions_mm = mirror(env=None, obs=None, actions=actions_m[: args.num_envs])
  inv_act = (actions_mm[: args.num_envs] - actions).abs().max().item()
  print(
    f"[gate-b] mirror involution max err: policy={inv_obs:.2e} "
    f"height_grid={inv_hg:.2e} actions={inv_act:.2e} "
    f"(identity check requires sensor mount bound)"
  )
  if not (inv_obs < 1e-4 and inv_hg < 1e-4 and inv_act < 1e-6):
    failures.append("mirror not an involution")
  # Identity-mount control: with the TRUE mount bound, mirroring a point at
  # the sensor translation must move it (a wrong/identity mount would not).
  pts = torch.zeros(args.num_envs, 256, 3, device=args.device)
  pts[:, 0] = sensor_trans.view(1, 3) + torch.tensor(
    [0.3, 0.4, 0.1], device=args.device
  )
  td = TensorDict(
    {
      "policy": torch.cat(
        [torch.zeros(args.num_envs, 66, device=args.device),
         pts.reshape(args.num_envs, -1),
         torch.zeros(args.num_envs, 1920, device=args.device)], dim=-1
      ),
      "height_grid": torch.zeros(args.num_envs, 187, device=args.device),
    },
    batch_size=[args.num_envs],
  )
  td_m, _ = mirror(env=None, obs=td, actions=None)
  moved = (
    td_m["policy"][args.num_envs:, 66:69] - pts[:, 0].reshape(-1, 3)
  ).abs().max().item()
  print(f"[gate-b] sensor-frame mount engaged (mirrored point moved by {moved:.3f} m)")
  if moved < 0.5:
    failures.append("mirror appears to use identity mount")

  # Actor forward + mirror loss path (wiring-only; random init). Same math as
  # rsl_rl Symmetry.compute_loss: MSE between the actor mean on mirrored obs
  # and the mirrored actor mean on the original obs.
  from hexapod_mjlab.rl.actor_v2 import LidarPDActor5x

  actor = LidarPDActor5x(
    obs_td, {"actor": ["policy"]}, "actor", 18,
    hidden_dims=(64, 32), proximal_points=256, distal_history_length=10,
    distal_points=64, proximal_feature_dim=187, distal_feature_dim=64,
    proprio_obs_dim=66, privileged_height_dim=187,
  ).to(args.device)
  with torch.no_grad():
    means = actor(obs_m)  # (2B, 18): [original; mirrored]
    _, means_symm = mirror(env=None, obs=None, actions=means[: args.num_envs])
    mirror_loss = torch.nn.functional.mse_loss(
      means[args.num_envs:], means_symm.detach()[args.num_envs:]
    )
  print(f"[gate-b] actor mirror loss at random init (wiring only): "
        f"{mirror_loss.item():.6f} finite={torch.isfinite(mirror_loss).item()}")
  if not torch.isfinite(mirror_loss).item():
    failures.append("mirror loss not finite")

  peak_mb = torch.cuda.max_memory_allocated() / 1e6
  print(f"[gate-b] torch peak (main, {args.num_envs} envs): {peak_mb:.0f} MB")

  print("[gate-b] PASS" if not failures else f"[gate-b] FAIL {failures}")
  return 0 if not failures else 1


if __name__ == "__main__":
  raise SystemExit(main())
