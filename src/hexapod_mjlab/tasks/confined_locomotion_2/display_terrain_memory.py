"""Pre-action, CPU-only diagnostic retention for Terrain Memory play."""

import numpy as np
from . import display


def capture_inputs(session, prepared):
    captured = display.capture_inputs(session, prepared)
    labels = session.runtime.labels()
    height, envelope = labels["height"], labels["envelope"]
    height.validate(prepared.observation)
    quat = display.array(session.env.scene["robot"].data.root_link_quat_w)
    for row, entry in enumerate(captured):
        entry["terrain"] = {
            "root_pos_w": display.array(height.root_pos_w[row]),
            "root_quat_wxyz": quat[row].copy(),
            "query_xy_w": display.array(height.query_xy_w[row]),
            "target": display.array(height.heights[row]),
            "valid": display.array(height.valid[row]),
            "grid": dict(height.grid_identity),
            "terrain_identity": height.terrain_identity,
            "envelope_target": display.array(envelope.values[row]),
            "envelope_valid": bool(envelope.valid[row]),
            "identity": dict(entry["identity"]),
        }
    return captured


def retain_display(session, prepared, output, captured, *, preview, reason):
    positions = display.array(session.env.scene["robot"].data.root_link_pos_w)
    ground, valid = session.runtime.terrain_query.surface.reference_heights(positions[:, :2])
    display.retain_display(session, prepared, output, captured, preview=preview, reason=reason, ground_z_w=ground)
    prediction = display.array(output.height_prediction)
    if prediction.shape != (session.num_envs, 11, 17) or not np.isfinite(prediction).all():
        raise ValueError("height prediction must be finite [B,11,17]")
    for row, snapshot in enumerate(session._display_rows):
        snapshot["pose"]["ground_valid"] = bool(valid[row])
        if not valid[row]:
            snapshot["pose"].update(ground_z_m=None, root_height_m=None)
        diagnostic = captured[row]["terrain"]
        diagnostic["prediction"] = prediction[row].copy()
        snapshot["terrain"] = diagnostic
