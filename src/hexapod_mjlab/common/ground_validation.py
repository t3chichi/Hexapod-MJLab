"""Shared compiled-scene validation for a single horizontal ground plane."""

from __future__ import annotations

import numpy as np

#: Confirmed flat-ground friction this task validates against (flat_env_cfg).
EXPECTED_PLANE_FRICTION = (1.0, 0.005, 0.0001)


class PlaneValidationError(RuntimeError):
    """The compiled scene does not match the single-horizontal-plane contract."""


def validate_single_horizontal_plane(
    mj_model,
    *,
    expected_friction: tuple[float, float, float] = EXPECTED_PLANE_FRICTION,
    horizontal_atol: float = 1e-9,
    friction_rtol: float = 1e-9,
) -> dict:
    """Validate the physical plane identity on the ACTUAL compiled model.

    Exactly one ``mjGEOM_PLANE`` geom; its world normal (static terrain body,
    evaluated once via a fresh host MjData forward) is vertical (aligned with
    world ±z to ``horizontal_atol``); its friction matches the confirmed
    (sliding, torsion, rolling) values. Returns the raw identity record (geom
    id/name, body, world position, normal, friction, geom_size); any deviation
    raises ``PlaneValidationError``.
    """
    import mujoco

    plane_ids = [
        int(g) for g in range(mj_model.ngeom)
        if int(mj_model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
    ]
    if len(plane_ids) != 1:
        raise PlaneValidationError(
            f"expected EXACTLY one mjGEOM_PLANE geom in the compiled scene, "
            f"found {len(plane_ids)}: {plane_ids}"
        )
    gid = plane_ids[0]
    data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, data)
    pos = np.asarray(data.geom_xpos[gid], dtype=np.float64).copy()
    mat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3).copy()
    normal = mat[:, 2].copy()  # plane normal in world coords (column convention)
    if not (
        abs(normal[0]) <= horizontal_atol
        and abs(normal[1]) <= horizontal_atol
        and abs(abs(normal[2]) - 1.0) <= horizontal_atol
    ):
        raise PlaneValidationError(
            f"plane geom {gid} is not horizontal: world normal {normal.tolist()}"
        )
    friction = np.asarray(mj_model.geom_friction[gid], dtype=np.float64).copy()
    expected = np.asarray(expected_friction, dtype=np.float64)
    if not np.allclose(friction, expected, rtol=friction_rtol, atol=0.0):
        raise PlaneValidationError(
            f"plane geom {gid} friction {friction.tolist()} != confirmed "
            f"{expected.tolist()}"
        )
    geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
    body_id = int(mj_model.geom_bodyid[gid])
    body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    return {
        "geom_id": gid,
        "geom_name": geom_name,
        "body_id": body_id,
        "body_name": body_name,
        "world_pos": pos.tolist(),
        "world_normal": normal.tolist(),
        "friction": friction.tolist(),
        "geom_size": np.asarray(mj_model.geom_size[gid], dtype=np.float64).tolist(),
    }
