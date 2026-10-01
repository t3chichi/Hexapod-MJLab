#!/usr/bin/env python3
"""M1 gate G4 — mujoco-warp batched smoke check for el_4090 (brief GPU use).

Loads the converted MJCF with CPU MuJoCo, uploads it with mujoco_warp
(put_model), creates batched data for 8 worlds (put_data, nworld=8) starting
from the `stand` keyframe, steps 50 times (0.25 s at dt=0.005, feet settle
onto the ground plane so collision/solver kernels are exercised), and asserts
the batched state stays finite. CPU MjData runs the same 50 steps as a sanity
reference for the mean base height (not an equivalence test — warp's batched
solver differs numerically; this is a finiteness/stability smoke check only).
"""

from __future__ import annotations

import pathlib

import mujoco
import mujoco_warp
import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
XML = REPO / "src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090.xml"

NWORLD = 8
N_STEPS = 50


def main() -> None:
    # scene = robot MJCF + ground plane (same construction as verify_stance)
    spec = mujoco.MjSpec.from_file(str(XML))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, name="floor",
                            size=[0, 0, 0.1], friction=[1.0, 0.005, 0.0001])
    mjm = spec.compile()

    data = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, data, 0)  # stand keyframe, ctrl = 0

    mw_model = mujoco_warp.put_model(mjm)
    mw_data = mujoco_warp.put_data(mjm, data, nworld=NWORLD)

    for _ in range(N_STEPS):
        mujoco_warp.step(mw_model, mw_data)

    qpos = torch.as_tensor(mw_data.qpos)
    qvel = torch.as_tensor(mw_data.qvel)
    assert qpos.shape[0] == NWORLD, f"expected {NWORLD} worlds, got shape {tuple(qpos.shape)}"
    assert torch.isfinite(qpos).all(), "non-finite qpos after warp stepping"
    assert torch.isfinite(qvel).all(), "non-finite qvel after warp stepping"

    base_z = qpos[:, 2].cpu().numpy()
    print(f"[G4] warp: {NWORLD} worlds x {N_STEPS} steps finite; "
          f"base z per world: {np.round(base_z, 4)}")

    # CPU reference for context (same initial state, zero ctrl)
    for _ in range(N_STEPS):
        mujoco.mj_step(mjm, data)
    print(f"[G4] cpu reference base z: {data.qpos[2]:.4f} "
          f"(warp mean {base_z.mean():.4f}); "
          f"max |warp - cpu| = {np.max(np.abs(base_z - data.qpos[2])):.4f}")


if __name__ == "__main__":
    main()
