#!/usr/bin/env python
"""Task-local checkpoint play CLI (explicit checkpoint; CUDA; mean-only).

The real interactive entry of ``El-4090-Confined-Locomotion-Flat``: loads an
explicit checkpoint WEIGHTS-ONLY into the accepted play seam (env → wrapper
→ play-mode ``TrainingRuntime`` → eval network) and runs the confirmed play
lifecycle with the live local Viser viewer:

* manual body commands vx∈[−1.5,1.5] m/s, vy∈[−1.0,1.0] m/s, ωz∈[−1.0,1.0]
  rad/s (validated FIRST — an illegal value is rejected with ZERO state
  change, never a physics fault); no 4 s random resampling, ever;
* pause / continue / single-step / explicit full-episode manual reset /
  exit; the run STARTS PAUSED by default so acceptance can inspect the
  initial state before any stepping (use --start-running to change);
* a fall decided ONLY by the confirmed formal gates saves the scene and
  PAUSES — continue cannot bypass it, only the manual reset continues; a
  fault saves and freezes permanently (step AND reset refused); collisions
  never pause anything; the 20 s training truncation never fires;
* default clean observation path; the observation presets (gap/gap_fixed0/range/
  proprio/all) are explicit and recorded; --allow-source-change is OFF by
  default and only relaxes the weights-only source-hash rule (never the
  full-resume rules);
* the viewer binds 127.0.0.1 only (no public tunnel, no upload); GUI
  callbacks submit intents, the single loop below owns every
  physics/scan/network advance; layer toggles / picking / pausing re-render
  the retained snapshot only.

Exit code: 0 on operator exit / step budget; 1 on any fault (the terminal
record and the saved scene remain under --output).

Usage:
  python scripts/experiments/confined_locomotion/play.py \
      --checkpoint /path/to/confined_checkpoint.pt \
      --output /path/to/new_run_dir \
      [--seed 20261009] [--preset clean] [--num-envs 1] \
      [--vx 0.0 --vy 0.0 --wz 0.0] [--port 8093] [--start-running] \
      [--no-viewer] [--max-steps 0] [--manifest-every 0]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_SEED = 20261009
PRESET_CHOICES = ("clean", "gap", "gap_fixed0", "range", "proprio", "all")
CONTROL_PERIOD_S = 0.02


def _parse_args(argv=None) -> argparse.Namespace:
  from hexapod_mjlab.tasks.confined_locomotion.ground import DEFAULT_PARALLEL_TOL
  from hexapod_mjlab.tasks.confined_locomotion.ground_presets import GROUND_GRID_PRESET_NAMES

  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--checkpoint", type=Path, required=True,
                      help="REQUIRED: explicit checkpoint file (weights-only or "
                           "current full schema used weights-only; mean_dev/fault scenes "
                           "are rejected)")
  parser.add_argument("--output", type=Path, required=True,
                      help="REQUIRED: NEW run directory for recipe/evidence "
                           "(must not exist)")
  parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                      help=f"play seed (default {DEFAULT_SEED}; a NEW play "
                      "seed — never a training RNG continuation)")
  parser.add_argument("--preset", choices=PRESET_CHOICES, default="clean",
                      help="observation preset (default clean)")
  parser.add_argument("--num-envs", type=int, default=1,
                      help="env count of the play batch (default 1)")
  parser.add_argument("--grid-x", type=float, nargs="+",
                      help="explicit ordered body-frame x coordinates (requires --grid-y)")
  parser.add_argument("--grid-y", type=float, nargs="+",
                      help="explicit ordered body-frame y coordinates (requires --grid-x)")
  parser.add_argument("--grid-preset", choices=GROUND_GRID_PRESET_NAMES,
                      help="explicit grid override; if omitted, read exact grid coordinates from checkpoint")
  parser.add_argument(
    "--parallel-tol", type=float, default=None,
    help=f"tolerance only for explicit --grid-preset/--grid-x/y overrides (default {DEFAULT_PARALLEL_TOL:g}); absent grid args use checkpoint identity",
  )
  parser.add_argument("--vx", type=float, default=0.0,
                      help="initial manual vx m/s within [-1.5, 1.5]")
  parser.add_argument("--vy", type=float, default=0.0,
                      help="initial manual vy m/s within [-1.0, 1.0]")
  parser.add_argument("--wz", type=float, default=0.0,
                      help="initial manual omega z rad/s within [-1.0, 1.0]")
  parser.add_argument("--allow-source-change", action="store_true",
                      help="explicitly accept the weights-only load under "
                      "changed source hashes (default OFF; never relaxes "
                      "full-resume rules)")
  parser.add_argument("--host", default="127.0.0.1",
                      help="viewer bind host (local only; default 127.0.0.1)")
  parser.add_argument("--port", type=int, default=8093,
                      help="viewer port (default 8093)")
  parser.add_argument("--start-running", action="store_true",
                      help="start UNpaused (default: start PAUSED so "
                      "acceptance can inspect before stepping)")
  parser.add_argument("--no-viewer", action="store_true",
                      help="run without the viewer (scripted smoke; the "
                      "lifecycle rules are identical)")
  parser.add_argument("--max-steps", type=int, default=0,
                      help="safety step budget; 0 = unlimited (default 0)")
  parser.add_argument("--manifest-every", type=int, default=0,
                      help="export a snapshot manifest every N steps under "
                      "OUTPUT/manifests (0 = off, default 0)")
  parser.add_argument("--export-exit-manifest", action="store_true",
                      help="export one snapshot manifest at exit")
  args = parser.parse_args(argv)
  if args.grid_preset is None and ((args.grid_x is None) != (args.grid_y is None)):
    parser.error("custom grid requires BOTH --grid-x and --grid-y")
  if args.grid_preset is not None and (args.grid_x is not None or args.grid_y is not None):
    parser.error("--grid-preset must not be mixed with --grid-x/--grid-y")
  if (args.grid_preset is None and args.grid_x is None and args.grid_y is None
      and args.parallel_tol is not None):
    parser.error("--parallel-tol may be set only with an explicit grid override")
  return args


def _pick_result(snapshot: dict, kind: str, index: int) -> dict:
  """Resolve a picker intent purely from the retained snapshot (no recompute)."""
  if kind == "near":
    valid = bool(snapshot["input"]["near"]["valid"][index])
    xyz = snapshot["input"]["near"]["points_base"][index]
    raw = int(snapshot["provenance"]["near_raw_ids"][index])
    no_hit = bool(snapshot["provenance"]["near_no_hit"][index])
    scan_id = int(snapshot["input"]["near"]["scan_id"])
    scan_time = float(snapshot["input"]["near"]["time_s"])
  else:
    valid = bool(snapshot["input"]["far"]["valid"][index])
    xyz = snapshot["input"]["far"]["points_base"][index]
    raw = int(snapshot["provenance"]["far_raw_ids"][index])
    no_hit = bool(snapshot["provenance"]["far_no_hit"][index])
    scan_id = int(snapshot["provenance"]["far_source_scan_ids"][index])
    scan_time = float(snapshot["input"]["far"]["times"][index])
  return {
    "kind": kind,
    "index": int(index),
    "valid": valid,
    "env": snapshot["env"],
    "episode_id": snapshot["identity"]["episode_id"],
    "scan_id": scan_id,
    "scan_time_s": scan_time,
    "raw_ray_id": raw,
    "no_hit": no_hit,
    "xyz_base": xyz,
    "note": (
      "source by exact stable angle-key permutation (XYZ equality never "
      "used); raw_ray_id=-1 means the slot is invalid or its source scan "
      "metadata was pruned from the bounded ring"
    ),
  }


def main(argv=None) -> int:
  args = _parse_args(argv)
  output = args.output.resolve()
  if output.exists():
    raise SystemExit(f"output directory must not exist: {output}")
  if args.host != "127.0.0.1":
    raise SystemExit("the task viewer bind host is fixed to 127.0.0.1")
  if args.no_viewer and not args.start_running:
    raise SystemExit(
      "--no-viewer requires --start-running (headless runs have no pause "
      "control; the interactive default starts paused)"
    )

  from hexapod_mjlab.tasks.confined_locomotion.ground_presets import (
    resolve_explicit_ground_grid,
  )
  from hexapod_mjlab.tasks.confined_locomotion.play import (
    ground_grid_from_checkpoint_header, peek_checkpoint_header,
  )
  if args.grid_preset is None and args.grid_x is None and args.grid_y is None:
    # Default Play must follow the checkpoint's actual target positions, not
    # the editable future training preset. Explicit overrides stay strict.
    ground_grid = ground_grid_from_checkpoint_header(
      peek_checkpoint_header(args.checkpoint)
    )
  else:
    ground_grid = resolve_explicit_ground_grid(
      grid_preset=args.grid_preset,
      grid_x=args.grid_x,
      grid_y=args.grid_y,
      parallel_tol=(
        args.parallel_tol if args.parallel_tol is not None else DEFAULT_PARALLEL_TOL
      ),
    )

  import torch

  if not torch.cuda.is_available():
    raise SystemExit("task play requires actual CUDA (CPU claims nothing)")

  from hexapod_mjlab.tasks.confined_locomotion.play import (
    PlayConfigError,
    PlaySession,
  )
  from hexapod_mjlab.tasks.confined_locomotion.visualization.play import (
    PlayViewer,
    export_snapshot_manifest,
  )

  session = PlaySession(
    checkpoint=args.checkpoint,
    output_dir=output,
    seed=args.seed,
    ground_grid=ground_grid,
    device="cuda:0",
    num_envs=args.num_envs,
    perturbation_preset=args.preset,
    allow_source_change=args.allow_source_change,
    initial_command=(args.vx, args.vy, args.wz),
  )
  viewer = None
  if not args.no_viewer:
    try:
      viewer = PlayViewer(
        num_envs=session.num_envs,
        mj_model=session.env.sim.mj_model,
        pipeline=session.runtime.pipeline,
        ground_grid=session.grid_cfg,
        identity_header={
          "path": session.header["path"],
          "kind": session.header["kind"],
          "ground_samples": session.header["ground_samples"],
          "grid_identity_hash": (
            session.header["grid_identity_hash"]
            or session.grid_cfg.identity_hash()
          ),
          "preset": args.preset,
          "weights_loaded": True,
        },
        host=args.host,
        port=args.port,
      )
    except BaseException as viewer_error:
      try:
        session.write_terminal_record(
          "viewer construction failed",
          extra={"viewer_error": repr(viewer_error)},
        )
      except BaseException as record_error:
        print(f"[play] startup terminal record failed: {record_error!r}")
      session.close()
      raise

  paused = not args.start_running
  display_row = 0
  steps_done = 0
  rejection_log: list[str] = []
  exit_reason = "operator exit"
  exit_code = 0
  manifests_dir = output / "manifests"
  manifest_sequence = 0

  def _render() -> None:
    if viewer is None:
      return
    viewer.render(session.build_display_snapshot(display_row, paused=paused))

  def _export_manifest(tag: str) -> None:
    nonlocal manifest_sequence
    manifests_dir.mkdir(exist_ok=True)
    manifest_sequence += 1
    export_snapshot_manifest(
      session.build_display_snapshot(display_row, paused=paused),
      viewer,
      manifests_dir / f"manifest_{manifest_sequence:06d}_{tag}.json",
    )

  try:
    _render()
    while True:
      pending = (
        viewer.pending.take() if viewer is not None
        else {"exit": False, "pause_toggle": False, "step": False, "reset": False,
              "command": None, "zero_command": False, "rerender": False,
              "pick": None, "env_row": None}
      )
      if pending["exit"]:
        exit_reason = (
          "operator exit after permanent fault: "
          f"{session.runtime.last_fault}"
          if session.runtime._faulted else "viewer exit button"
        )
        break
      if (
        pending["env_row"] is not None
        and 0 <= int(pending["env_row"]) < session.num_envs
      ):
        display_row = int(pending["env_row"])
        if viewer is not None:
          viewer.set_pick_result(None)
        _render()
      if pending["pause_toggle"] and not (
        session.runtime._stopped or session.runtime._faulted
      ):
        paused = not paused
        _render()
      if pending["command"] is not None:
        if session.runtime._stopped or session.runtime._faulted:
          rejection_log.append("command refused: play is read-only until reset/exit")
          print("[play] command REFUSED: fallen/faulted scene is read-only")
        else:
          try:
            record = session.set_command(*pending["command"])
            print(f"[play] manual command applied: {record['command']}")
          except PlayConfigError as error:
            rejection_log.append(repr(error))
            print(f"[play] manual command REJECTED (zero state change): {error}")
        _render()
      if pending["zero_command"]:
        if session.runtime._stopped or session.runtime._faulted:
          rejection_log.append("zero-command refused: play is read-only until reset/exit")
          print("[play] command zero REFUSED: fallen/faulted scene is read-only")
        else:
          record = session.zero_command()
          print(f"[play] manual command zeroed: {record['command']}")
        _render()
      if pending["reset"]:
        if session.runtime._faulted:
          rejection_log.append("reset refused: runtime faulted (permanent)")
          print("[play] reset REFUSED: the runtime is faulted (permanent freeze)")
        else:
          try:
            record = session.reset()
            paused = True  # the fresh episode waits for explicit input
            steps_done = 0
            if viewer is not None:
              viewer.set_pick_result(None)
            print(f"[play] manual full-episode reset: {record['reason']}")
          except BaseException as error:
            if session.runtime._faulted:
              exit_reason = f"fault during manual reset: {error!r}"
              exit_code = 1
              paused = True
              print(f"[play] reset faulted; scene remains read-only: {error!r}")
            else:
              raise
        _render()
      if pending["pick"] is not None and viewer is not None:
        kind, index = pending["pick"]
        bounds = 256 if kind == "near" else 640
        if 0 <= int(index) < bounds:
          viewer.set_pick_result(
            _pick_result(
              session.build_display_snapshot(display_row, paused=paused),
              kind,
              int(index),
            )
          )
        _render()
      wants_step = pending["step"] or (
        not paused
        and not session.runtime._stopped and not session.runtime._faulted
      )
      if pending["step"] and (session.runtime._stopped or session.runtime._faulted):
        rejection_log.append("single-step refused: fall/fault scene is read-only")
        print("[play] single-step REFUSED: fall/fault scene is read-only")
        wants_step = False
      if wants_step:
        step_start = time.monotonic()
        try:
          session.step_once()
        except Exception as error:  # noqa: BLE001 — honest fault exit below
          if session.runtime._faulted:
            exit_reason = f"fault: {error!r}"
            exit_code = 1
            paused = True
            print("[play] fault saved; viewer remains open in read-only mode")
            _render()
            if viewer is None:
              break
            continue
          raise
        steps_done += 1
        if session.runtime._stopped:
          paused = True
          print(
            "[play] FORMAL FALL: scene saved and the run is PAUSED; only the "
            "manual reset continues (see the runtime stop record)"
          )
        if viewer is not None:
          _render()
          if args.manifest_every and steps_done % args.manifest_every == 0:
            _export_manifest(f"step{steps_done:06d}")
        if args.max_steps and steps_done >= args.max_steps:
          exit_reason = f"step budget reached ({args.max_steps})"
          break
        elapsed = time.monotonic() - step_start
        if elapsed < CONTROL_PERIOD_S:
          time.sleep(CONTROL_PERIOD_S - elapsed)
        continue
      if pending["rerender"] and viewer is not None:
        _render()
        if args.manifest_every:
          _export_manifest("rerender")
      time.sleep(0.005 if (paused or session.runtime._stopped or session.runtime._faulted) else 0.0)
  except KeyboardInterrupt:
    exit_reason = "keyboard interrupt"
  except BaseException as error:
    exit_reason = f"error: {error!r}"
    exit_code = 1
    if not session.runtime._faulted:
      session.runtime._save_fault(error, phase="play/driver_or_render")
    try:
      session._persist_terminal_event("fault", exit_reason)
    except BaseException as scene_error:
      print(f"[play] immediate fault scene save failed: {scene_error!r}")
    raise
  finally:
    try:
      if args.export_exit_manifest or args.manifest_every:
        _export_manifest("exit")
    except BaseException as manifest_error:
      print(f"[play] exit manifest export failed: {manifest_error!r}")
    try:
      session.write_terminal_record(
        exit_reason,
        extra={
          "steps_this_episode": steps_done,
          "input_rejections": rejection_log[-64:],
          "viewer": "none" if viewer is None else f"{args.host}:{args.port}",
        },
      )
    except BaseException as record_error:
      print(f"[play] terminal record write failed: {record_error!r}")
      exit_code = 1
    if viewer is not None:
      viewer.close()
    session.close()
  print(f"[play] exit: {exit_reason}")
  return exit_code


if __name__ == "__main__":
  raise SystemExit(main())
