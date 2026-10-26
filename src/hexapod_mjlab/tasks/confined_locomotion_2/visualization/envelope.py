"""Robot-frame envelope wireframe; display geometry only."""

import numpy as np

from hexapod_mjlab.tasks.confined_locomotion.visualization.geometry import (
    H_REF_M, HEX_VERTEX_ORDER, hex_vertices,
)

BOTTOM_ABOVE_REFERENCE_GROUND_M = 0.02


def envelope_prism_base(envelope):
    """18 edges: matching bottom/top hexagons plus six local-z edges.

    At the reference Root height H_REF_M, the bottom is 2 cm above the
    reference ground and the top is H_REF_M + delta_z above it. Both planes
    are fixed in the Root frame; no live terrain/Root-height compensation.
    delta_z remains a Root-height target offset, not a collision ceiling.
    """
    values = np.asarray(envelope, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("envelope must contain six finite metre values")
    vertices = hex_vertices(values)
    xy = np.asarray([vertices[name] for name in HEX_VERTEX_ORDER])
    bottom = np.column_stack((xy, np.full(6, BOTTOM_ABOVE_REFERENCE_GROUND_M - H_REF_M)))
    top = np.column_stack((xy, np.full(6, values[5])))
    return np.concatenate((
        np.stack((bottom, np.roll(bottom, -1, axis=0)), axis=1),
        np.stack((top, np.roll(top, -1, axis=0)), axis=1),
        np.stack((bottom, top), axis=1),
    ))
