"""Tripod native mean-policy Play, with finite headless diagnostics available."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=None,
                        help="policy-step budget: interactive default 0 (unlimited), headless default 500")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--command", type=float, nargs=3, default=(.5, 0., 0.))
    parser.add_argument("--output-dir", "--output", dest="output_dir")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--start-running", action="store_true")
    args = parser.parse_args()
    args.steps = (500 if args.no_viewer else 0) if args.steps is None else args.steps
    if args.steps < 0 or (args.no_viewer and args.steps == 0):
        parser.error("--steps must be positive in headless mode, or nonnegative with a window")
    if args.gpu < 0 or not 0 <= args.seed < 2**32:
        parser.error("gpu must be nonnegative; seed must be in [0,2**32)")
    import math
    if any(not math.isfinite(v) or abs(v) > limit for v, limit in zip(args.command, (2.5, 1., 1.2))):
        parser.error("command outside legacy curriculum limits")
    if not Path(args.checkpoint).is_file():
        parser.error("--checkpoint must name an existing file")
    if args.output_dir and Path(args.output_dir).exists():
        parser.error("--output-dir must be a new directory")
    from hexapod_mjlab.tasks.confined_locomotion_2.visualization.native import NativePlayViewer, require_native_display
    from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play_cli import run_play
    from hexapod_mjlab.tasks.confined_locomotion_2.play_tripod import TripodPlaySession
    if not args.no_viewer:
        require_native_display()
    # Keep generated diagnostics available after exit.
    output = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="tripod-play-")) / "play"
    session = native = None
    reason = "initialization_failed"
    try:
        session = TripodPlaySession(checkpoint=args.checkpoint, output_dir=output,
                                   seed=args.seed, device=f"cuda:{args.gpu}")
        session.set_command(args.command)
        if not args.no_viewer:
            native = NativePlayViewer(session.runtime, command=args.command,
                                      command_limits=session.COMMAND_LIMITS)
            print(f"Native Tripod Play:\n{NativePlayViewer.KEY_HELP}", flush=True)
        reason, _ = run_play(session, None, steps=args.steps, start_running=args.start_running, native=native)
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    except BaseException:
        reason = "error"
        raise
    finally:
        try:
            if native is not None:
                native.close()
        finally:
            if session is not None:
                try:
                    (output / "play_summary.json").write_text(json.dumps({
                        "checkpoint": str(Path(args.checkpoint).resolve()), "seed": args.seed,
                        "policy_steps": session.forward_count,
                        "runtime_control_steps": session.runtime.control_steps_taken,
                        "warmup_control_steps": 1, "exit_reason": reason,
                        "native_viewer": native is not None, "weights_only": True,
                        "optimizer": False}, indent=2))
                finally:
                    session.close()
    print(f"Tripod Play diagnostics: {output}")


if __name__ == "__main__":
    main()
