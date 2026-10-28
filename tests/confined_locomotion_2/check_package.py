"""Check all fixed-channel assets and catalog loading in an isolated wheel.

Build with ``python -m pip wheel --no-deps . -w /tmp/wheels`` first.
This gate uses installed dependencies but never imports the editable checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    package_assets = Path("hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels")
    local_assets = root / "src" / package_assets
    expected = sorted(p for p in local_assets.rglob("*") if p.suffix in {".json", ".npz"})
    if len(expected) != 45:
        raise AssertionError(f"baseline expects 45 fixed-channel assets, found {len(expected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with zipfile.ZipFile(args.wheel) as wheel:
        for path in expected:
            packed = wheel.read(str(package_assets / path.relative_to(local_assets)))
            if hashlib.sha256(packed).digest() != hashlib.sha256(path.read_bytes()).digest():
                raise AssertionError(f"wheel asset differs: {path.name}")
        with tempfile.TemporaryDirectory(prefix="confined-v2-wheel-") as temporary:
            wheel.extractall(temporary)
            result = Path(temporary) / "result.json"
            code = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
import hexapod_mjlab
from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.channels import build_channel_catalog, load_sparse_selection
package = Path(hexapod_mjlab.__file__).resolve().parent
assert package.is_relative_to(Path(sys.argv[1]).resolve())
catalog = build_channel_catalog()
assert catalog.directions_sensor.shape == (86400, 3)
assert catalog.near_mask.shape == (86400,)
assets = package / 'tasks/confined_locomotion_2/assets/fixed_channels'
selections = sorted(assets.rglob('*.npz'))
assert len(selections) == 21
for path in selections:
    ids, identity = load_sparse_selection(path, catalog)
    assert np.all(np.diff(ids) > 0)
    json_ids, _ = load_sparse_selection(path.with_suffix('.json'), catalog)
    assert np.array_equal(ids, json_ids)
hit10, _ = load_sparse_selection(assets / 'v1_hit10/near_x6_far_x10.npz', catalog)
assert len(hit10) == 2816
assert int(catalog.near_mask[hit10].sum()) == 1536
assert int((~catalog.near_mask[hit10]).sum()) == 1280
Path(sys.argv[2]).write_text(json.dumps({
    'isolated_wheel_import': True, 'catalog_ray_count': len(catalog.directions_sensor),
    'selection_count': len(selections), 'hit10_near_candidates': 1536,
    'hit10_far_candidates': 1280, 'catalog_identity': catalog.identity
}, indent=2))
"""
            subprocess.run([sys.executable, "-I", "-c", code, temporary, str(result)],
                           cwd=temporary, check=True)
            report = json.loads(result.read_text())
    report.update({"passed": True, "asset_count": len(expected), "asset_bytes_equal": True,
                   "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest()})
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS isolated wheel: {len(expected)} assets, 21 validated selections, full catalog + hit10")


if __name__ == "__main__":
    main()
