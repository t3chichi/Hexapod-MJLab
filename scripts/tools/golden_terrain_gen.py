#!/usr/bin/env python
"""M2 golden reference generator -- RUN IN THE ``li_leggym`` ENV ONLY.

Generates the golden heightfield for the terrain parity test (G2) by running
the *old* pipeline end-to-end:

    np.random.seed(42) -> legged_gym.utils.terrain.Terrain(
        El4090LidarHeadCfg.terrain, num_robots=num_rows*num_cols)

The old ``Terrain`` class consumes the global numpy RandomState; NumPy's
legacy ``RandomState`` stream is frozen (NEP 19 stream-compatibility policy),
so the same call sequence under ``np.random.seed(42)`` reproduces the field
bit-for-bit on any NumPy version (old env: numpy 1.24.3, new env: 2.5.3).
The parity-relevant terrain type (pillar_field) does not use interp2d, so no
float-path differences are expected; the new builder must match EXACTLY.

Run:

    conda run --no-capture-output -n li_leggym python scripts/tools/golden_terrain_gen.py

Outputs (into tests/data/golden_terrain/ of this repo):
    heightfield_seed42.npy   int16 (tot_rows, tot_cols) raw heightfield
    golden_config_seed42.json resolved config + provenance
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Old-repo checkout that provides legged_gym + the golden Terrain class.
# Override with the LIDAR_LEGGED_GYM env var on other machines.
OLD_REPO = Path(
    os.environ.get("LIDAR_LEGGED_GYM", "/home/t3chichi/Lidar_legged_gym")
)
SEED = 42

NEW_REPO = Path(__file__).resolve().parents[2]
OUT_DIR = NEW_REPO / "tests" / "data" / "golden_terrain"


def _git_info():
    def run(args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(OLD_REPO)] + args, text=True
            ).strip()
        except Exception as exc:  # pragma: no cover - provenance only
            return f"<unavailable: {exc}>"

    return {
        "head": run(["rev-parse", "HEAD"]),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty_files": run(["status", "--porcelain"]).splitlines(),
    }


def main():
    sys.path.insert(0, str(OLD_REPO / "legged_gym"))

    import numpy as np  # noqa: E402  (after sys.path setup)

    # Heavy import chain (torch, isaacgym bindings -- CPU-only import, no sim
    # is created).  Some imported modules draw from np.random at import time
    # (start/goal pair generation warnings), so the seed is applied AFTER all
    # imports, immediately before Terrain construction.
    from legged_gym.envs.el_4090.pd_gru_lidar_head.el_4090_lidar_head_config import (  # noqa: E402
        El4090LidarHeadCfg,
    )
    from legged_gym.utils.terrain import Terrain  # noqa: E402

    cfg = El4090LidarHeadCfg.terrain
    record = {
        "seed": SEED,
        "source_repo": str(OLD_REPO),
        "source_git": _git_info(),
        "old_class": "legged_gym.utils.terrain.Terrain",
        "isaacgym": "/home/t3chichi/isaacgym (Preview 1.0rc4)",
        "numpy_version": np.__version__,
        # resolved terrain config actually consumed by the old Terrain:
        "terrain": {
            "mesh_type": cfg.mesh_type,
            "curriculum": cfg.curriculum,
            "selected": cfg.selected,
            "terrain_length": cfg.terrain_length,
            "terrain_width": cfg.terrain_width,
            "border_size": cfg.border_size,
            "num_rows": cfg.num_rows,
            "num_cols": cfg.num_cols,
            "horizontal_scale": cfg.horizontal_scale,
            "vertical_scale": cfg.vertical_scale,
            "terrain_proportions": list(cfg.terrain_proportions),
            "difficulty_scale": cfg.difficulty_scale,
            "slope_treshold": cfg.slope_treshold,
            "stepping_stones_size": cfg.stepping_stones_size,
            "stepping_stones_distance": cfg.stepping_stones_distance,
            "stepping_stones_max_height": cfg.stepping_stones_max_height,
            "stepping_stones_platform_size": cfg.stepping_stones_platform_size,
            "pillar_count_min": cfg.pillar_count_min,
            "pillar_count_max": cfg.pillar_count_max,
            "pillar_size_x_min": cfg.pillar_size_x_min,
            "pillar_size_x_max": cfg.pillar_size_x_max,
            "pillar_size_y_min": cfg.pillar_size_y_min,
            "pillar_size_y_max": cfg.pillar_size_y_max,
            "pillar_height_min": cfg.pillar_height_min,
            "pillar_height_max": cfg.pillar_height_max,
            "pillar_min_separation": cfg.pillar_min_separation,
            "pillar_center_clear_radius": cfg.pillar_center_clear_radius,
            "pillar_spawn_radius": cfg.pillar_spawn_radius,
            "pillar_allow_height_variation": cfg.pillar_allow_height_variation,
        },
        "num_robots": cfg.num_rows * cfg.num_cols,
    }

    np.random.seed(SEED)
    terrain = Terrain(cfg, num_robots=record["num_robots"])

    hf = terrain.height_field_raw
    assert hf.dtype == np.int16

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"heightfield_seed{SEED}.npy", hf)
    (OUT_DIR / f"golden_config_seed{SEED}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"saved {OUT_DIR / f'heightfield_seed{SEED}.npy'} shape={hf.shape} "
          f"dtype={hf.dtype} min={hf.min()} max={hf.max()}")
    print(f"env_origins:\n{terrain.env_origins}")


if __name__ == "__main__":
    main()
