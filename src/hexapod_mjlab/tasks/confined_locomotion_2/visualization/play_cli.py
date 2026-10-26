"""Independent recurrent-task play: paused browser by default, finite headless option."""

import argparse
import json
from pathlib import Path
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", "--max-steps", type=int, default=None,
                        help="total control-step budget; browser default 0 (unlimited), headless default 250")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--command", nargs=3, type=float, default=(0, 0, 0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--no-viewer", action="store_true", help="run finite inference without a browser server")
    parser.add_argument("--start-running", action="store_true", help="browser starts running instead of paused")
    parser.add_argument("--port", type=int, default=8094, help="local browser port (old task defaults to 8093)")
    parser.add_argument("--preset", choices=("clean", "gap", "gap_fixed0", "range", "proprio", "all"), default="clean")
    parser.add_argument("--raycast-mode", choices=("full", "sparse"), default="full")
    parser.add_argument("--far-sampling", choices=("all_endpoints", "hit_only_10m"),
                        default="all_endpoints", help="must match checkpoint sampling semantics")
    parser.add_argument("--sparse-selection", "--sparse-selection-path",
                        dest="sparse_selection_path")
    parser.add_argument("--sparse-selection-dir", action="append", default=[],
                        help="additional offline selection directory; repeat to add directories")
    parser.add_argument("--allow-scan-override", action="store_true",
                        help="evaluate these weights with another mode/selection in the same fixed T1 catalog")
    args = parser.parse_args(argv)
    if args.steps is None:
        args.steps = 250 if args.no_viewer else 0
    if args.steps < 0 or (args.no_viewer and args.steps == 0):
        parser.error("--steps must be positive in headless mode, or nonnegative with a browser")
    if args.num_envs < 1 or args.gpu < 0 or not 0 <= args.seed < 2**32:
        parser.error("num-envs must be positive, gpu nonnegative, seed in [0,2**32)")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    import math
    limits = (1.5, 1.0, 1.0)
    if any(not math.isfinite(x) or abs(x) > limit for x, limit in zip(args.command, limits)):
        parser.error("--command must be finite and inside vx ±1.5, vy/wz ±1.0")
    if not Path(args.checkpoint).is_file():
        parser.error("--checkpoint must name an existing file")
    if Path(args.output).exists():
        parser.error("--output must be a new directory")
    if args.raycast_mode == "full" and args.sparse_selection_path is not None:
        parser.error("--sparse-selection is only valid with --raycast-mode sparse")
    if args.raycast_mode == "sparse" and not args.sparse_selection_path:
        parser.error("--sparse-selection is required with --raycast-mode sparse")
    for directory in args.sparse_selection_dir:
        if not Path(directory).expanduser().is_dir():
            parser.error("--sparse-selection-dir must name an existing directory")
    return args


def discover_acquisition_options(directories, explicit_selection=None):
    """Expose only validated offline files, keyed by independent near/far budgets."""
    from hexapod_mjlab.tasks.confined_locomotion_2.channels import build_channel_catalog, load_sparse_selection
    catalog = build_channel_catalog()
    options, identities = {}, {}

    def read(path):
        path = Path(path).expanduser().resolve()
        ids, identity = load_sparse_selection(path, catalog)
        metadata = identity.get("selection", {})
        near, far = metadata.get("near_multiplier"), metadata.get("far_multiplier")
        if all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (near, far)):
            actual_near = int(catalog.near_mask[ids].sum())
            if actual_near != 256 * near or len(ids) - actual_near != 128 * far:
                raise ValueError(f"selection counts disagree with multiplier metadata: {path}")
            return f"near x{near} / far x{far}", str(path), identity
        return "CLI selection", str(path), identity

    for directory in directories:
        folder = Path(directory).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"selection directory does not exist: {folder}")
        for path in sorted(folder.glob("near_x*_far_x*.npz")):
            label, filename, identity = read(path)
            if label == "CLI selection":
                raise ValueError(f"directory selection lacks positive integer multipliers: {filename}")
            if label in identities and identity != identities[label]:
                raise ValueError(f"conflicting offline selections for {label}; choose one directory")
            if label not in options:
                options[label], identities[label] = filename, identity
    if explicit_selection is not None:
        label, filename, identity = read(explicit_selection)
        # An explicitly named CLI file has precedence for its own budget pair.
        options[label], identities[label] = filename, identity
    return options


class PlayOwner:
    """One active GPU session; GUI acquisition changes rebuild it at reset."""

    task_id = "El-4090-Confined-Locomotion-Flat_2"
    env_factory = None
    session_class = None

    def __init__(self, args):
        import hashlib
        self.args = args
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.checkpoint = Path(args.checkpoint).resolve()
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.viewer = self.session = self.env = None
        self.segments = []
        self.segment_index = 0
        self.weights_sha256 = None
        self._closed = False
        asset_root = Path(__file__).resolve().parents[1] / "assets/fixed_channels"
        directories = [asset_root / "v1", asset_root / "v1_independent", asset_root / "v1_hit10"]
        directories.extend(getattr(args, "sparse_selection_dir", []))
        custom = args.sparse_selection_path
        self.acquisition_options = discover_acquisition_options(directories, custom)
        self._open(args.raycast_mode, custom,
                   allow_override=getattr(args, "allow_scan_override", False), command=args.command)

    @staticmethod
    def weight_digest(network):
        import hashlib
        h = hashlib.sha256()
        for name, tensor in sorted(network.state_dict().items()):
            h.update(name.encode())
            h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def _open(self, mode, selection, *, allow_override, command=(0, 0, 0)):
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        from hexapod_mjlab.tasks.confined_locomotion_2.play import PlaySession, make_play_env_cfg
        from hexapod_mjlab.tasks.confined_locomotion_2.scanner import ConfinedLidarCfg
        from hexapod_mjlab.tasks.confined_locomotion_2.perturbations import preset_by_name
        env_factory = self.env_factory or make_play_env_cfg
        session_class = self.session_class or PlaySession
        env = ManagerBasedRlEnv(env_factory(num_envs=self.args.num_envs, seed=self.args.seed),
                                device=f"cuda:{self.args.gpu}")
        session = None
        try:
            session = session_class(
                wrapper=RslRlVecEnvWrapper(env, clip_actions=None),
                checkpoint=self.checkpoint,
                output_dir=self.output / f"segment_{self.segment_index:03d}_{mode}",
                seed=self.args.seed, own_environment=True,
                perturbation_cfg=preset_by_name(self.args.preset),
                lidar_cfg=ConfinedLidarCfg(raycast_mode=mode, sparse_selection_path=selection,
                                         far_sampling=getattr(self.args, "far_sampling", "all_endpoints")),
                allow_scan_override=allow_override,
            )
            digest = self.weight_digest(session.network)
            if self.weights_sha256 is not None and digest != self.weights_sha256:
                raise RuntimeError("checkpoint weights changed while switching acquisition")
            self.weights_sha256 = digest
            session.set_command(command)
        except BaseException:
            if session is not None:
                session.close()
            else:
                env.close()
            raise
        self.env, self.session = env, session

    def _finish_segment(self, reason):
        session = self.session
        if session is None:
            return
        if session._display_rows is not None:
            from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play import export_snapshot_manifest
            row = 0 if self.viewer is None else self.viewer.selected_env_row
            snapshot = session.build_display_snapshot(row, paused=True)
            export_snapshot_manifest(snapshot, self.viewer, session.output_dir / "play_snapshot_manifest.json")
        entry = {
            "segment": self.segment_index, "reason": reason,
            "output": str(session.output_dir),
            "control_steps": session.forward_count,
            "frame_encodings": session.frame_encoding_count,
            "preview_count": session.preview_count,
            "scan_evaluation": session.scan_evaluation_report,
            "weights_sha256": self.weights_sha256,
        }
        (session.output_dir / "segment_summary.json").write_text(json.dumps(entry, indent=2))
        self.segments.append(entry)

    def switch_acquisition(self, request):
        """Preflight first; release old GPU objects before creating the next env."""
        import gc
        import hashlib
        import torch
        from hexapod_mjlab.tasks.confined_locomotion_2.channels import build_channel_catalog, load_sparse_selection
        from hexapod_mjlab.tasks.confined_locomotion_2.scanner import ConfinedLidarCfg
        mode, path = request["mode"], request.get("selection_path")
        if self.viewer is not None:
            self.viewer.set_acquisition_status("正在检查扫描设置…")
        try:
            cfg = ConfinedLidarCfg(raycast_mode=mode, sparse_selection_path=path,
                                  far_sampling=getattr(self.args, "far_sampling", "all_endpoints"))
            catalog = build_channel_catalog()
            if catalog.identity != self.session.runtime.scan_identity["catalog"]:
                raise ValueError("fixed T1 catalog changed; open a new play process")
            if cfg.raycast_mode == "sparse":
                load_sparse_selection(cfg.sparse_selection_path, catalog)
            if hashlib.sha256(self.checkpoint.read_bytes()).hexdigest() != self.checkpoint_sha256:
                raise ValueError("checkpoint file changed; open a new play process")
        except (ValueError, TypeError, OSError) as error:
            if self.viewer is not None:
                self.viewer.set_acquisition_status(f"设置未更改：{error}")
            return self.session
        self._finish_segment("acquisition_switch")
        if self.viewer is not None:
            self.viewer.set_acquisition_status("正在重建扫描并重置回合…")
            self.viewer.detach_session()
        old = self.session
        self.session = self.env = None
        old.close()
        del old
        gc.collect()
        torch.cuda.empty_cache()
        self.segment_index += 1
        self._open(mode, path, allow_override=True)
        if self.viewer is not None:
            self.viewer.rebind_session(
                self.env.sim.mj_model, self.session.runtime.pipeline,
                identity_header={"task": self.task_id,
                                 "preset": self.args.preset, "segment": self.segment_index})
            self.viewer.set_command_inputs((0, 0, 0))
            self.viewer.set_acquisition_status("扫描已更新，回合已重置，当前暂停。")
        return self.session

    def close(self, reason="exit"):
        if self._closed:
            return
        self._closed = True
        try:
            self._finish_segment(reason)
            summary = {
                "task": self.task_id, "checkpoint": str(self.checkpoint),
                "seed": self.args.seed, "requested_steps": self.args.steps,
                "control_steps": sum(e["control_steps"] for e in self.segments),
                "frame_encodings": sum(e["frame_encodings"] for e in self.segments),
                "preview_count": sum(e["preview_count"] for e in self.segments),
                "observation_perturbations": self.args.preset, "optimizer": False,
                "viewer": self.viewer is not None, "exit_reason": reason,
                "learning_effect_ready": False, "segments": self.segments,
                "weights_sha256": self.weights_sha256,
            }
            if self.session is not None:
                summary["command"] = self.session.runtime.current_command().detach().cpu().tolist()
                summary["scan_evaluation"] = self.session.scan_evaluation_report
            (self.output / "play_summary.json").write_text(json.dumps(summary, indent=2))
        finally:
            try:
                if self.viewer is not None:
                    self.viewer.detach_session()
                    self.viewer.close()
            finally:
                if self.session is not None:
                    self.session.close()
                self.session = self.env = None
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()


def run_play(session, viewer, *, steps, start_running=False, switch_acquisition=None):
    """Own physics/inference/reset; display changes use retained CPU snapshots."""
    from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play import resolve_point_pick

    paused = viewer is not None and not start_running
    row = 0
    total_controls = session.forward_count
    reason = "step_budget"
    if viewer is not None:
        session.refresh_idle_snapshot(reason="initial preview")
    snapshot = None
    dirty = viewer is not None
    deadline = time.monotonic()
    while True:
        single_step = False
        if viewer is not None:
            pending = viewer.pending.take()
            if pending["exit"]:
                reason = "viewer_exit"
                break
            if pending.get("acquisition") is not None and switch_acquisition is not None:
                # Drop this loop's reference before the owner releases old GPU
                # tensors. The owner preflights data and rebuilds one session.
                transition = None
                session = None
                session = switch_acquisition(pending["acquisition"])
                session.refresh_idle_snapshot(reason="acquisition selection preview; temporal state uncommitted")
                paused, dirty, row = True, True, 0
                snapshot = None
                deadline = time.monotonic()
                # Commands/steps queued for the old episode never reach the new one.
                continue
            dirty |= bool(pending["rerender"])
            if pending["env_row"] is not None:
                row = int(pending["env_row"])
                viewer.set_pick_result(None)
                dirty = True
            faulted = bool(session.runtime._faulted)
            stopped = bool(session.runtime._stopped)
            if pending["reset"] and not faulted:
                session.manual_reset()
                session.set_command((0, 0, 0))
                viewer.set_command_inputs((0, 0, 0))
                session.refresh_idle_snapshot(reason="manual reset preview")
                viewer.set_pick_result(None)
                paused, stopped, dirty = True, False, True
            if (pending["command"] is not None or pending["zero_command"]) and not (faulted or stopped):
                values = (0, 0, 0) if pending["zero_command"] else pending["command"]
                session.set_command(values)
                viewer.set_command_inputs(values)
                session.refresh_idle_snapshot(reason="command preview; temporal state uncommitted")
                dirty = True
            if pending["pause_toggle"] and not (faulted or stopped):
                paused = not paused
                dirty = True
                deadline = time.monotonic()
            single_step = bool(pending["step"]) and paused and not (faulted or stopped)
            if dirty or snapshot is None:
                snapshot = session.build_display_snapshot(row, paused=paused)
                viewer.render(snapshot)
                dirty = False
            if pending["pick"] is not None:
                viewer.set_pick_result(resolve_point_pick(snapshot, *pending["pick"]))
        if steps and total_controls >= steps:
            break
        if not paused or single_step:
            try:
                transition = session.step_once().transition
                total_controls += 1
            except Exception as error:
                if viewer is None or not session.runtime._faulted:
                    raise
                paused, dirty = True, True
                reason = "fault"
                print(f"Play fault: {error}; frozen display retained, Exit or Ctrl+C to close.", flush=True)
                continue
            dirty = viewer is not None
            if transition.engineering_stopped:
                paused, reason = True, "formal_fall"
                if viewer is None:
                    break
                print("Formal fall: paused. Reset episode to start with zero temporal state.", flush=True)
            if viewer is not None:
                snapshot = session.build_display_snapshot(row, paused=paused)
                viewer.render(snapshot)
                dirty = False
                deadline = max(deadline + 0.02, time.monotonic())
                time.sleep(max(0.0, deadline - time.monotonic()))
        elif viewer is not None:
            time.sleep(0.01)
    return reason, snapshot


def main(*, owner_type=PlayOwner, viewer_type=None):
    args = parse_args()
    import torch
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends(allow_tf32=False)
    rnn = getattr(torch.backends.cudnn, "rnn", None)
    if rnn is not None and hasattr(rnn, "fp32_precision"):
        rnn.fp32_precision = "ieee"
    owner = None
    reason = "initialization_failed"
    try:
        owner = owner_type(args)
        if not args.no_viewer:
            from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play import PlayViewer
            owner.viewer = (viewer_type or PlayViewer)(
                num_envs=owner.session.num_envs, mj_model=owner.env.sim.mj_model,
                pipeline=owner.session.runtime.pipeline,
                identity_header={"task": owner.task_id, "preset": args.preset, "segment": 0},
                acquisition_options=owner.acquisition_options, port=args.port)
            owner.viewer.set_command_inputs(args.command)
            print(f"Open http://127.0.0.1:{args.port} — {'running' if args.start_running else 'paused'}. Acquisition switches reset episode and GRU with the same weights.", flush=True)
        reason, _snapshot = run_play(
            owner.session, owner.viewer, steps=args.steps, start_running=args.start_running,
            switch_acquisition=owner.switch_acquisition)
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    except BaseException:
        reason = "error"
        raise
    finally:
        if owner is not None:
            owner.close(reason)
            print(f"Controls: {sum(e['control_steps'] for e in owner.segments)}; frame encodings: {sum(e['frame_encodings'] for e in owner.segments)}; exit: {reason}", flush=True)

