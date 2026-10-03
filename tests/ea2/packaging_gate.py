"""Verify wheel assets and import EA2 from an isolated unpacked wheel."""

import _bootstrap
import argparse
import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wheel", type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    package = "hexapod_mjlab/tasks/envelope_adaptive_2/"
    forbidden = {"isaacgym", "legged_gym", "gymtorch"}
    with zipfile.ZipFile(args.wheel) as wheel:
        for name in wheel.namelist():
            if name.startswith(package) and name.endswith(".py"):
                tree = ast.parse(wheel.read(name), filename=name)
                for node in ast.walk(tree):
                    modules = (
                        [a.name for a in node.names]
                        if isinstance(node, ast.Import)
                        else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                    )
                    assert not any(m.split(".")[0] in forbidden for m in modules), name
        for asset in ["selected_airy_channels.pt", "attitude_traj_source.npz"]:
            packed = wheel.read(package + "assets/" + asset)
            local = (root / "src" / package / "assets" / asset).read_bytes()
            assert hashlib.sha256(packed).digest() == hashlib.sha256(local).digest(), asset
        with tempfile.TemporaryDirectory(prefix="ea2-wheel-") as directory:
            wheel.extractall(directory)
            code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch
import hexapod_mjlab.tasks.envelope_adaptive_2 as task
from hexapod_mjlab.tasks.envelope_adaptive_2.airy_mount import load_selected_channels
from hexapod_mjlab.tasks.envelope_adaptive_2.attitude_replay import load_source
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.model import EnvelopeNet
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.losses import frame_map_ids
assert Path(task.__file__).is_relative_to(Path(sys.argv[1]))
channels = load_selected_channels()
assert channels['ray_indices'].numel() == 187
source = load_source(Path(task.__file__).parent / 'assets/attitude_traj_source.npz')
assert len(source['pitch']) >= 200
assert EnvelopeNet()(torch.zeros(3, 2, 190)).shape == (3, 2, 5)
assert not any(name.split('.')[0] in {'isaacgym', 'legged_gym', 'gymtorch'} for name in sys.modules)
print('T6 PASS: isolated wheel imports, packaged assets, no old runtime imports')
"""
            subprocess.run([sys.executable, "-I", "-c", code, directory], cwd=directory, check=True)


if __name__ == "__main__":
    main()
