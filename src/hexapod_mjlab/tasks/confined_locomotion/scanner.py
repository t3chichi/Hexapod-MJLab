"""M2 real scanner: task-local lidar config + plane-mesh raycast pipeline.

``ConfinedLidarCfg`` / ``ConfinedScanPipeline`` implement the REAL sensor path of
the confined_locomotion flat batch (docs/tasks/confined_locomotion/README.md,
lidar_setup/配置与边界.md, 平地学习与基准测量.md §2). This module owns:

* VALIDATION of the ACTUAL compiled scene: exactly ONE horizontal ``mjGEOM_PLANE``
  geom with friction (1.0, 0.005, 0.0001), read from ``env.sim.mj_model`` (host
  template; the terrain body is static so its world pose is constant);
* a NEW warp raycast MESH built from that plane (4 verts / 2 triangles at the
  plane's world z). The physical plane is INFINITE in MuJoCo — the mesh extent is
  an ENGINEERING parameter of the raycast proxy only (default 100 m half-extent),
  never a scene boundary. NO robot geometry and NO extra structures are included
  (explicitly no self-occlusion/self-echo in this batch). The mesh is a scene
  geometry proxy for raycasting — not synthetic sensor points, not ground
  auxiliary labels;
* a per-scan COVERAGE ASSERTION (before every raycast): the mesh rectangle must
  contain each scanned sensor's entire ``max_range`` reachable ray-sphere
  projection (bounding box of the radius-15 m disc around the sensor xy), so a
  finite mesh edge can NEVER become a false miss;
* the existing ``LidarSensor``/Warp raycasting (utils/lidar) with the corrected
  model-derived mount (``resolve_lidar_mount`` — the aligned ``lidar_sensor``
  site; NO extra 90° rotation, NO fallback; missing site is a hard failure);
* B8 near/far partition (stage-A chosen baseline):
  Airy 900 azimuth × 96 elevation at 0.4°, 15 m range; ALL 86,400 rays
  enter grouping (default stride=1, no azimuth pre-thinning); cone θ=28° /
  δ=56° on the corrected mount axis, plus BASE-x <= 3.5 m endpoint cutoff.
  Near 256 FPS / far 64 angular members per scan; both groups are disjoint
  and exhaustive over the current valid-ray pool (including 15m misses).
  The forward cutoff is BODY-FRAME x, NOT Euclidean range to the lidar;
* no-hit handling: the kernel's miss identity is EXACT — a miss writes
  ``dist == far_plane`` verbatim (sensor_kernels/lidar_kernels_warp.py) — and
  that exact identity is used as-is. Real hits below max_range (including
  15−ε) stay REAL HITS: no near-range promotion, no ε band. Misses become the
  max-range endpoint ALONG THE ORIGINAL RAY with ``valid=True``; endpoints are
  (re)constructed from the actual raw ray directions in the SENSOR frame
  BEFORE the sensor→BASE transform, on a COPY — the ORIGINAL backend pixels
  and dists are retained verbatim (``raw_points_sensor`` / ``raw_dists``).
  NaN/Inf or dist outside [0, max_range] are FAULTS (recorded + raised),
  never silently sanitized, never masked-as-valid, never clamped.
  NO new clipping, NO noise, NO dropout, NO domain
  randomization (sensor noise flags are all zero/false);
* sampling from the full-frame groups with exact index provenance:
  - near: farthest-point-sampling (FPS) over the near group; the selected-mark
    trick makes a repeated index impossible even on ties; if the group has fewer
    than 256 members the actual members are selected once and the missing slots
    are zero-filled with ``valid=False`` / source index −1 (never duplicated);
  - far: uniform angular-key sample over the far group (stable sort by the
    sensor-frame key φ·2π+azimuth, strictly-increasing positions ``floor(j·M/K)``
    when M ≥ K ⇒ distinct by construction); if the group has fewer than 64
    members each actual member is selected ONCE and the rest is zero-filled with
    ``valid=False`` / index −1. Distinct raw ray ids at the same coordinates are
    legitimate independent measurements and are never merged or invented away;
* full audit retention: per scan the pipeline keeps raw ray directions, raw
  dists, raw endpoints (sensor + BASE), no-hit masks, pool indices, group masks,
  chosen source indices, valid masks, sensor/root pose and scan identity/time
  (identity/time assigned by the runtime caller) so an independent analytic
  plane-intersection check can verify the ENTIRE raw ray output later (M3 /
  Codex acceptance). Coordinates are meters; BASE is the robot root body
  acquisition frame; quaternion conventions are explicit everywhere
  (mjlab root pose is wxyz, warp/mount math is xyzw).

Cadence is NOT owned here: ``LidarSensor.update`` performs the actual raycast
but does not enforce any schedule — the runtime (``runtime.py``) owns the
per-env 10 Hz clocks, scan/episode identities and window writes. This module
only provides ``scan_rows`` for the rows the runtime declares due (a physical
raycast is never launched for non-due envs; one single-env sensor per env row).

P6 (observation perturbations) adds an OPTIONAL consumption seam, OFF by
default and byte-identical to the clean path when off: ``scan_rows(rows,
perturbations=..., persistent_scan_numbers=...)`` first runs the unchanged
raw fault checks and keeps every raw array verbatim, then builds the grouping
input from a CONSUMED copy — gap-sector rays excluded before grouping, hit
rays carrying d_obs = clip(d_raw + ε, 0, 15) along the ORIGINAL unit ray,
cone angles/keys derived from raw unit directions (including zero-range
saturation); B8's endpoint-x cutoff uses consumed xyz and can change the
near/far group under range noise. Far is the valid pool complement of near,
and FPS uses consumed metric xyz. The gap/range streams live in
``perturbations.py``; this module only calls in per new scan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

from hexapod_mjlab.rl.confined_locomotion.contracts import (
    FAR_POINTS_PER_SCAN,
    FAR_WINDOW_SLOTS,
    NEAR_POINTS,
)
from hexapod_mjlab.tasks.el_4090_lidar.mdp.lidar import resolve_lidar_mount

#: Confirmed flat-ground friction this task validates against (flat_env_cfg).
EXPECTED_PLANE_FRICTION = (1.0, 0.005, 0.0001)

__all__ = [
    "ConfinedLidarCfg",
    "ConfinedScanPipeline",
    "PlaneValidationError",
    "ScanCoverageError",
    "ScanFaultError",
    "ScanFrame",
    "direction_angle_keys",
    "farthest_point_sampling_indices",
    "partition_cone_forward_x",
    "sensor_frame_angle_keys",
    "uniform_angular_sample_indices",
]


class PlaneValidationError(RuntimeError):
    """The compiled scene does not match the single-horizontal-plane contract."""


class ScanCoverageError(RuntimeError):
    """The finite raycast mesh does not cover a sensor's reachable ray sphere."""


class ScanFaultError(RuntimeError):
    """Non-finite/contract-violating raw raycast output (fault, never sanitized)."""


# ── xyzw quaternion helpers (same formulas as the el_4090_lidar mdp reference) ─


def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def _quat_apply_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv, qw = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + qw * t + torch.linalg.cross(qv, t, dim=-1)


# ── pure sampling / key functions (torch-only; unit-testable without an env) ─


def farthest_point_sampling_indices(
    points: torch.Tensor, group_mask: torch.Tensor, k: int
) -> tuple[torch.Tensor, int]:
    """FPS over the masked group; returns (original indices, count).

    Deterministic, index-based FPS with the confirmed no-repeat guarantee: the
    seed is the group member with max norm; after each pick the selected index's
    running min-distance is set to −1 so an already-selected index can NEVER be
    argmax again (ties among unselected members resolve to the first index).
    Returns exactly ``min(k, group_size)`` DISTINCT original indices; the caller
    zero-pads missing slots (valid=False, source index −1) — shortage is never
    filled by repeating a source index, and empty groups return 0 indices.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive int, got {k!r}")
    if points.dim() != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {tuple(points.shape)}")
    if group_mask.shape != points.shape[:1] or group_mask.dtype is not torch.bool:
        raise ValueError("group_mask must be (N,) bool matching points")
    member_idx = group_mask.nonzero(as_tuple=False).flatten()
    m = int(member_idx.numel())
    if m == 0:
        return member_idx.new_zeros(0), 0
    pts = points.index_select(0, member_idx)
    k_eff = min(k, m)
    # Seed: maximum norm in the supplied coordinate frame (BASE here).
    # Pairwise FPS distances use the same raw metric xyz; no normalization.
    seed = int(torch.argmax(pts.norm(dim=-1)).item())
    chosen = [seed]
    min_dist = ((pts - pts[seed]).square().sum(dim=-1)).clone()
    min_dist[seed] = -1.0
    for _ in range(1, k_eff):
        nxt = int(torch.argmax(min_dist).item())
        chosen.append(nxt)
        min_dist = torch.minimum(min_dist, (pts - pts[nxt]).square().sum(dim=-1))
        min_dist[nxt] = -1.0
    sel_member = torch.tensor(chosen, dtype=torch.long, device=points.device)
    return member_idx.index_select(0, sel_member), k_eff


def uniform_angular_sample_indices(
    angle_keys: torch.Tensor, group_mask: torch.Tensor, k: int
) -> tuple[torch.Tensor, int]:
    """Uniform angular-key sample over the masked group; returns (indices, count).

    The group is stably sorted by the caller-provided angular key; when the
    group has M ≥ K members the strictly-increasing positions ``floor(j·M/K)``
    select K DISTINCT members spread uniformly over the sorted order (consecutive
    positions differ by ≥1, so no repeats by construction — no rounding-tie
    duplicates). When M < K every actual member is selected EXACTLY ONCE (in
    ascending key order); the caller zero-pads the missing slots. Empty groups
    return 0 indices.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive int, got {k!r}")
    if angle_keys.shape != group_mask.shape or angle_keys.dtype is not torch.float32:
        raise ValueError("angle_keys must be (N,) float32 matching group_mask")
    member_idx = group_mask.nonzero(as_tuple=False).flatten()
    m = int(member_idx.numel())
    if m == 0:
        return member_idx.new_zeros(0), 0
    keys = angle_keys.index_select(0, member_idx)
    order = torch.argsort(keys, stable=True)
    k_eff = min(k, m)
    if m >= k:
        pos = (torch.arange(k_eff, device=keys.device, dtype=torch.long) * m) // k
        sel_member = order.index_select(0, pos)
    else:
        sel_member = order
    return member_idx.index_select(0, sel_member), k_eff


def partition_cone_forward_x(
    cos_alignment: torch.Tensor,
    points_base: torch.Tensor,
    *,
    theta_deg: float,
    forward_x_max_m: float | None,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exclusive/exhaustive near/far assignment for B8 on the valid source pool.

    The cone cosine comes from original unit ray directions on a P6 consumed
    scan, so noisy zero-distance endpoints never lose angular identity. The
    x cutoff ALWAYS tests the endpoint consumed by the network, in the
    scan-time BASE frame (including 15m original-direction miss endpoints).
    With no gap, valid=None means all source rays remain valid.
    """
    if points_base.ndim != 2 or points_base.shape[-1] != 3:
        raise ValueError("points_base must be (P,3)")
    if cos_alignment.shape != points_base.shape[:1]:
        raise ValueError("cos_alignment must be (P,)")
    if valid is not None and (
        valid.shape != cos_alignment.shape or valid.dtype != torch.bool
    ):
        raise ValueError("valid must be a (P,) bool mask or None")
    inside = cos_alignment >= math.sin(math.radians(theta_deg))
    if forward_x_max_m is not None:
        inside = inside & (points_base[:, 0] <= forward_x_max_m)
    if valid is None:
        return inside, ~inside
    return valid & inside, valid & ~inside


def sensor_frame_angle_keys(
    points_base: torch.Tensor,
    mount_pos: torch.Tensor,
    mount_quat_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Angular sort key φ·(2π)+azimuth of BASE-frame points, in the SENSOR frame.

    The mount transform (BASE→sensor) is a rigid CONSTANT, so the sensor-frame
    spherical coordinates of a stored acquisition-time BASE point are identical
    at scan time and at any later read time — the key is a pure function of the
    stored point. Elevation is computed as ``atan2(z, hypot(x, y))`` —
    numerically identical to ``asin(z/r)`` wherever asin is defined, but
    stable at the ±π/2 pole (no z/r division, no pole NaN, hence no epsilon
    guard and no arbitrary point clipping). Invalid (zero) slots produce
    finite dummy keys; the F1 sort contract ignores keys at invalid slots
    (invalid points are stable-sorted to the tail by their mask, not their
    key).
    """
    rel = points_base - mount_pos.to(points_base.dtype)
    conj = mount_quat_xyzw * torch.tensor(
        [-1.0, -1.0, -1.0, 1.0], device=points_base.device, dtype=points_base.dtype
    )
    pts_sensor = _quat_apply_xyzw(conj.expand(*rel.shape[:-1], 4), rel)
    x, y, z = pts_sensor.unbind(-1)
    azimuth = torch.atan2(y, x)
    phi = torch.atan2(z, torch.hypot(x, y))
    return phi * (2.0 * math.pi) + azimuth


def direction_angle_keys(unit_dirs_sensor: torch.Tensor) -> torch.Tensor:
    """Angular sort key φ·(2π)+azimuth of RAW UNIT RAYS (sensor frame).

    P6 consumption contract: the grouping keys come from the ORIGINAL unit
    ray directions, not from stored xyz — a range-noise-saturated d_obs = 0
    point is a zero VECTOR and must never generate a fake angle; with it the
    original ray keeps its angle key and cone identity (but B8's endpoint-x
    group can change with noise). The formula is the
    exact ``sensor_frame_angle_keys`` one evaluated on the direction itself
    (a positive-distance endpoint ``d·dir`` yields the same direction, so for
    unperturbed positive distances the two definitions agree).
    """
    if unit_dirs_sensor.dim() != 2 or tuple(unit_dirs_sensor.shape)[1] != 3:
        raise ValueError(
            f"unit_dirs_sensor must be (R, 3), got {tuple(unit_dirs_sensor.shape)}"
        )
    x, y, z = unit_dirs_sensor.unbind(-1)
    azimuth = torch.atan2(y, x)
    phi = torch.atan2(z, torch.hypot(x, y))
    return phi * (2.0 * math.pi) + azimuth


def assert_mesh_covers_sensor(
    mesh_bounds: tuple[float, float, float, float],
    sensor_xy: tuple[float, float],
    max_range: float,
    eps: float = 1e-3,
) -> None:
    """Assert the finite mesh rectangle contains the sensor's reachable sphere
    projection: every possible hit endpoint lies within ``max_range`` of the
    sensor, whose projection onto the ground plane is the radius-``max_range``
    disc around the sensor xy; the mesh must contain that disc's bounding box
    (checked with margin ``eps``) BEFORE any raycast is launched."""
    xmin, xmax, ymin, ymax = mesh_bounds
    sx, sy = sensor_xy
    if (
        xmin > sx - max_range - eps
        or xmax < sx + max_range + eps
        or ymin > sy - max_range - eps
        or ymax < sy + max_range + eps
    ):
        raise ScanCoverageError(
            f"raycast mesh bounds {mesh_bounds} do not cover the reachable ray "
            f"sphere projection of sensor at xy=({sx}, {sy}) with range "
            f"{max_range} m; refusing to scan (finite mesh edges must never "
            "become false misses)"
        )


# ── task-local lidar configuration ───────────────────────────────────────────


@dataclass(frozen=True)
class ConfinedLidarCfg:
    """Task-local scanner configuration with user-selected B8 grouping.

    The B8 cone + BASE-x limit is the production scan baseline, not a claim
    that final terrain/grid learning performance has been demonstrated.
    Raw sensor noise/dropout/domain-randomization flags stay OFF: separate
    P6 observation perturbations own consumption-only noise.

    ``plane_mesh_half_extent_m`` is an ENGINEERING parameter of the raycast
    proxy mesh only: the PHYSICAL compiled plane is infinite; the mesh must
    (and is asserted to) cover every scanned sensor's reachable ray sphere
    projection, and its identity/extent is recorded with each pipeline.
    """

    # Airy pattern (utils/lidar SpinningLidarGenerator; temporary values).
    airy_n_elevation: int = 96
    airy_n_azimuth: int = 900          # int(360/0.4); FP edge upstream is hardcoded
    airy_horizontal_resolution_deg: float = 0.4
    scan_frequency_hz: float = 10.0    # scanner rate; CADENCE is owned by the runtime
    max_range_m: float = 15.0
    min_range_m: float = 0.2           # kept for parity; the kernel applies no near clip

    # Candidate mapping (raw idx = az * n_elevation + el, C order).
    # Default 1 means ALL raw rays for BOTH groups; never load-adaptive.
    # An explicit stride is retained for a future hardware/load decision.
    pool_azimuth_stride: int = 1

    # Stage-A B8 baseline: real cone half-angle 90-theta=62 deg; tilted
    # corrected mount axis, followed by endpoint BASE-x cutoff. None restores
    # the legacy untruncated grouping for a separately identified comparison.
    split_theta_deg: float = 28.0
    split_axis_tilt_deg: float = 56.0
    split_max_forward_x_m: float | None = 3.5

    # Consumption sizes (must equal the F1 contract constants).
    near_points: int = NEAR_POINTS               # 256
    far_points_per_scan: int = FAR_POINTS_PER_SCAN   # 64
    far_window_slots: int = FAR_WINDOW_SLOTS     # 10

    # Raycast proxy mesh (engineering parameter; physical plane is infinite).
    plane_mesh_half_extent_m: float = 100.0
    expected_plane_friction: tuple[float, float, float] = EXPECTED_PLANE_FRICTION

    # ALL sensor noise flags OFF in this batch (documented zeros/false).
    enable_sensor_noise: bool = False
    pixel_dropout_prob: float = 0.0
    pixel_std_dev_multiplier: float = 0.0

    def __post_init__(self) -> None:
        c = self
        if abs(c.airy_n_azimuth * c.airy_horizontal_resolution_deg - 360.0) > 1e-9:
            raise ValueError(
                "airy azimuth count inconsistent with horizontal resolution "
                f"({c.airy_n_azimuth} × {c.airy_horizontal_resolution_deg}° != 360°)"
            )
        if (isinstance(c.pool_azimuth_stride, bool)
                or not isinstance(c.pool_azimuth_stride, int)
                or c.pool_azimuth_stride < 1):
            raise ValueError("pool_azimuth_stride must be a positive integer")
        if c.airy_n_azimuth % c.pool_azimuth_stride != 0:
            raise ValueError("airy_n_azimuth not divisible by pool_azimuth_stride")
        if (c.near_points, c.far_points_per_scan, c.far_window_slots) != (
            NEAR_POINTS, FAR_POINTS_PER_SCAN, FAR_WINDOW_SLOTS,
        ):
            raise ValueError("consumption sizes must equal the F1 contract constants")
        if not (0.0 < c.max_range_m and 0.0 < c.plane_mesh_half_extent_m):
            raise ValueError("max_range_m and plane_mesh_half_extent_m must be positive")
        if c.enable_sensor_noise or c.pixel_dropout_prob or c.pixel_std_dev_multiplier:
            raise ValueError("this diagnostic batch fixes ALL sensor noise flags OFF")
        if not (0.0 < c.scan_frequency_hz):
            raise ValueError("scan_frequency_hz must be positive")
        for field_name in ("split_theta_deg", "split_axis_tilt_deg"):
            angle = getattr(c, field_name)
            if isinstance(angle, bool) or not isinstance(angle, (int, float)) \
                    or not math.isfinite(float(angle)):
                raise ValueError(f"{field_name} must be a finite real angle")
        if not 0.0 < c.split_theta_deg < 90.0:
            raise ValueError("split_theta_deg must be in (0, 90)")
        cutoff = c.split_max_forward_x_m
        if cutoff is not None and (
            isinstance(cutoff, bool)
            or not isinstance(cutoff, (int, float))
            or not math.isfinite(float(cutoff))
            or float(cutoff) <= 0.0
        ):
            raise ValueError("split_max_forward_x_m must be finite and positive or None")


# ── scan frame (per-env products of ONE completed scan) ──────────────────────


@dataclass
class ScanFrame:
    """All products of one completed scan of ONE env row (raw audit unit).

    ``scan_id`` / ``time_s`` / ``episode_id`` are assigned by the runtime caller
    when it writes the ScanWindow; the pipeline records them verbatim into
    ``scan_records`` so geometry and identity stay paired for auditing and M3
    visualization. All coordinates are meters. Raw arrays are the FULL Airy
    frame (n_raw_rays); pool/group arrays cover all raw rays by default (stride=1); selected arrays are
    consumption-sized (256 / 64). Source indices are ORIGINAL ray ids (−1 = no
    source, shortage slot).

    P6 consumed-copy audit fields (default None/False = the CLEAN path, which
    is byte-identical to P0–P5): when lidar perturbations are active the
    pool/group/selected arrays above are the CONSUMED copy (gap rays excluded
    before grouping; hit rays carry d_obs = clip(d_raw + ε, 0, 15) along the
    ORIGINAL unit ray), and the raw arrays stay verbatim. ``consumed_valid``
    (R,) marks pool-valid rays (False = gap-excluded); ``consumed_dists`` is
    the full-frame d_obs; ``range_epsilon`` the full-frame drawn Normal
    (applied only where ``range_applied_mask``); ``near/far_angle_keys`` are
    the direction-derived sort keys aligned with the selected slots (dummy 0
    at shortage slots) for the window metadata. ``pool_dists`` stays the RAW
    gathered distances in BOTH paths (raw provenance; d_obs is retained in
    full via ``consumed_dists``).
    """

    env_row: int
    base_pos_w: torch.Tensor            # (3,) root body (BASE) world position
    base_quat_wxyz: torch.Tensor        # (4,) root world quaternion (mjlab wxyz)
    sensor_pos_w: torch.Tensor          # (3,)
    sensor_quat_xyzw: torch.Tensor      # (4,) warp convention
    raw_ray_dirs: torch.Tensor          # (R,3) unit dirs, sensor frame (actual pattern)
    raw_dists: torch.Tensor             # (R,) m; kernel miss convention == max_range (EXACT)
    raw_points_sensor: torch.Tensor     # (R,3) m; ORIGINAL backend pixels, VERBATIM
    points_sensor: torch.Tensor         # (R,3) m; consumed copy: no-hit endpoints
                                        #   reconstructed as max_range*dir (== raw
                                        #   except possible last-ulp differences)
    raw_points_base: torch.Tensor       # (R,3) m, acquisition-time BASE frame
    no_hit_mask: torch.Tensor           # (R,) bool; no-hit rays are VALID endpoints
    pool_raw_indices: torch.Tensor      # (P,) int64 pool→raw ray ids
    pool_points_base: torch.Tensor      # (P,3)
    pool_dists: torch.Tensor            # (P,)
    pool_angle_keys: torch.Tensor       # (P,) sensor-frame angular key
    near_mask: torch.Tensor             # (P,) bool cone AND BASE-x cutoff (B8)
    far_mask: torch.Tensor              # (P,) bool
    near_indices: torch.Tensor          # (256,) int64 raw ray ids, −1 = shortage slot
    near_valid: torch.Tensor            # (256,) bool
    near_points_base: torch.Tensor      # (256,3) zero-filled at invalid slots
    far_indices: torch.Tensor           # (64,) int64 raw ray ids, −1 = shortage slot
    far_valid: torch.Tensor             # (64,) bool
    far_points_base: torch.Tensor       # (64,3) zero-filled at invalid slots
    scan_id: int = -1
    episode_id: int = -1
    time_s: float = float("nan")
    # ── P6 consumed-copy audit (None/False on the clean path) ──────────────
    gap_fired: bool = False             # this scan carries the periodic gap
    consumed_valid: torch.Tensor | None = None    # (R,) bool; False = gap-excluded
    consumed_dists: torch.Tensor | None = None    # (R,) m; d_obs full frame
    range_epsilon: torch.Tensor | None = None     # (R,) m; drawn ε (applied subset only)
    range_applied_mask: torch.Tensor | None = None  # (R,) bool; hit & ~gap-excluded
    near_angle_keys: torch.Tensor | None = None   # (256,) direction keys, slot-aligned
    far_angle_keys: torch.Tensor | None = None    # (64,) direction keys, slot-aligned


# ── plane validation (pure; operates on a compiled host mj_model) ────────────


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


# ── the pipeline ─────────────────────────────────────────────────────────────


@dataclass
class _SensorBundle:
    """Strong owner of one single-env sensor and its pose tensors.

    The LidarSensor's warp arrays zero-copy alias these torch tensors; the
    bundle keeps them alive and every scan writes the fresh pose into the SAME
    storage (verified pattern from the el_4090 pipeline).
    """

    row: int
    sensor: object
    pos_tensor: torch.Tensor   # (1,3)
    quat_tensor: torch.Tensor  # (1,4) xyzw
    ray_dirs: torch.Tensor     # (R,3) unit dirs in sensor frame


class ConfinedScanPipeline:
    """Real scanner for the confined flat task (one shared plane mesh, per-env
    single-env ``LidarSensor`` raycast units).

    Construct AFTER a real reset geometry refresh (the runtime does so at its
    first explicit reset); the plane identity/mesh are read from the compiled
    template scene once and are state-independent (static terrain body). The
    pipeline owns NO clock: the runtime calls ``scan_rows`` exactly for the env
    rows whose per-env 10 Hz schedule is due (or the freshly reset rows for
    their first scan), so a physical raycast is never launched for non-due envs.
    """

    def __init__(self, env, cfg: ConfinedLidarCfg | None = None):
        self.env = env
        self.cfg = cfg or ConfinedLidarCfg()
        c = self.cfg
        self.device = env.device
        device_str = str(self.device)
        self.num_envs = int(env.num_envs)

        # ── 1. validate the ACTUAL physical plane + build the raycast mesh ──
        self.plane_record = validate_single_horizontal_plane(
            env.sim.mj_model, expected_friction=c.expected_plane_friction
        )
        px, py, pz = self.plane_record["world_pos"]
        h = float(c.plane_mesh_half_extent_m)
        self.mesh_bounds = (px - h, px + h, py - h, py + h)
        verts = np.asarray(
            [
                [px - h, py - h, pz],
                [px + h, py - h, pz],
                [px + h, py + h, pz],
                [px - h, py + h, pz],
            ],
            dtype=np.float32,
        )
        tris = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        wp.init()
        with wp.ScopedDevice(device_str):
            # Warp-OWNED copies (wp.from_torch aliases die with the torch
            # tensor); the pipeline keeps strong refs to mesh + arrays + ids.
            # Warp 1.17 wp.Mesh REQUIRES a flattened 1D int indices array —
            # a 2D (2,3) triangle table is rejected at construction (first
            # GPU attempt, logs/.../M2_short_attempt1). The 2x3 triangle
            # TABLE itself is retained verbatim in mesh_record; only the
            # warp array holds the row-major flattened copy.
            self._wp_points = wp.array(np.ascontiguousarray(verts), dtype=wp.vec3)
            self._wp_indices = wp.array(
                np.ascontiguousarray(tris.reshape(-1)), dtype=wp.int32
            )
            self._wp_mesh = wp.Mesh(points=self._wp_points, indices=self._wp_indices)
            self.mesh_ids = wp.array([self._wp_mesh.id], dtype=wp.uint64)
        self.mesh_record = {
            "source": "compiled infinite horizontal mjGEOM_PLANE (scene geometry proxy)",
            "half_extent_m": h,
            "center_xy": [px, py],
            "z": pz,
            "vertices": verts.tolist(),
            "triangles": tris.tolist(),
            "contains_robot_geometry": False,
            "note": (
                "physical plane is infinite; mesh extent is an engineering "
                "raycast parameter asserted to cover every scanned sensor's "
                "entire max-range reachable ray-sphere projection"
            ),
        }

        # ── 2. corrected model-derived mount (fail if missing; no fallback) ─
        mount = resolve_lidar_mount()
        if mount is None:
            raise RuntimeError(
                "resolve_lidar_mount() returned None: the corrected lidar_sensor "
                "site is missing from the radar MJCF — no fallback mount is "
                "permitted in this task"
            )
        mount_pos, mount_quat_xyzw = mount
        self.mount_pos = np.asarray(mount_pos, dtype=np.float32)
        self.mount_quat_xyzw = np.asarray(mount_quat_xyzw, dtype=np.float32)
        self._mount_t = torch.tensor(mount_pos, dtype=torch.float32, device=self.device).view(1, 3)
        self._mount_q = torch.tensor(mount_quat_xyzw, dtype=torch.float32, device=self.device).view(1, 4)

        # ── 3. grouping cone on the corrected mount axis (incl. mount tilt) ─
        delta = math.radians(c.split_axis_tilt_deg)
        theta = math.radians(c.split_theta_deg)
        dome_axis = _quat_apply_xyzw(
            self._mount_q, torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        )[0]
        cd, sd = math.cos(delta), math.sin(delta)
        axis = torch.stack(
            [
                cd * dome_axis[0] + sd * dome_axis[2],
                dome_axis[1],
                -sd * dome_axis[0] + cd * dome_axis[2],
            ]
        )
        self.cone_axis_body = axis / torch.linalg.norm(axis)
        self.cone_sin_thr = math.sin(theta)

        # ── 4. candidate indices (ALL raw rays by default, no DR) ───────────
        az = torch.arange(0, c.airy_n_azimuth, c.pool_azimuth_stride, device=self.device)
        el = torch.arange(c.airy_n_elevation, device=self.device)
        self.pool_indices = (
            (az[:, None] * c.airy_n_elevation + el[None, :]).reshape(-1).to(torch.long)
        )
        expected_candidates = (
            c.airy_n_azimuth // c.pool_azimuth_stride
        ) * c.airy_n_elevation
        if int(self.pool_indices.numel()) != expected_candidates:
            raise RuntimeError("candidate index count does not match configuration")

        # ── 5. per-env single-env sensors (raycast only for due rows) ───────
        from hexapod_mjlab.utils.lidar import LidarConfig, LidarSensor, LidarType

        lidar_sensor_cfg = LidarConfig(
            sensor_type=LidarType.AIRY,
            dt=float(env.step_dt),
            update_frequency=float(c.scan_frequency_hz),
            max_range=float(c.max_range_m),
            min_range=float(c.min_range_m),
            num_sensors=1,
            horizontal_line_num=int(c.airy_n_azimuth),
            vertical_line_num=int(c.airy_n_elevation),
            airy_horizontal_resolution_deg=float(c.airy_horizontal_resolution_deg),
            return_pointcloud=True,
            pointcloud_in_world_frame=False,   # sensor frame; BASE transform below
            randomize_placement=False,
            enable_sensor_noise=bool(c.enable_sensor_noise),       # False
            pixel_dropout_prob=float(c.pixel_dropout_prob),        # 0.0
            pixel_std_dev_multiplier=float(c.pixel_std_dev_multiplier),  # 0.0
        )
        self._sensors: list[_SensorBundle] = []
        for row in range(self.num_envs):
            pos_t = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
            quat_t = torch.zeros(1, 4, device=self.device, dtype=torch.float32)
            lidar_env = {
                "device": device_str,
                "num_envs": 1,
                "num_sensors": 1,
                "sensor_pos_tensor": pos_t,
                "sensor_quat_tensor": quat_t,
                "mesh_ids": self.mesh_ids,
            }
            sensor = LidarSensor(
                lidar_env, None, lidar_sensor_cfg, num_sensors=1, device=device_str
            )
            self._sensors.append(
                _SensorBundle(
                    row=row,
                    sensor=sensor,
                    pos_tensor=pos_t,
                    quat_tensor=quat_t,
                    ray_dirs=sensor.get_ray_directions().clone().contiguous(),
                )
            )
        self.num_raw_rays = int(self._sensors[0].ray_dirs.shape[0])
        expected_raw = c.airy_n_azimuth * c.airy_n_elevation
        if self.num_raw_rays != expected_raw:
            raise RuntimeError(
                f"actual AIRY rays {self.num_raw_rays} != configured {expected_raw}"
            )
        # AIRY source order is az*96+el with endpoint=False azimuths.
        # Classify the half-open gap by this source identity: atan2 of FP32
        # directions straddles 74/106 degrees across elevation channels.
        self._raw_azimuth_deg = (
            torch.arange(c.airy_n_azimuth, device=self.device, dtype=torch.float64)
            * (360.0 / c.airy_n_azimuth)
        ).repeat_interleave(c.airy_n_elevation)
        for bundle in self._sensors:
            actual = torch.rad2deg(torch.atan2(bundle.ray_dirs[:, 1].double(), bundle.ray_dirs[:, 0].double())) % 360
            delta = (actual - self._raw_azimuth_deg + 180) % 360 - 180
            # At the 90-degree pole atan2 is undefined; preserve its beam
            # source azimuth just like every other channel.
            nonpole = torch.hypot(bundle.ray_dirs[:, 0], bundle.ray_dirs[:, 1]) > 1e-6
            if bool((delta[nonpole].abs() > 1e-4).any()):
                raise RuntimeError("AIRY ray source order/azimuth contract changed")

        #: Full raw audit records (identity/time fields filled by the runtime).
        self.scan_records: list[dict] = []

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def mount_t(self) -> torch.Tensor:
        """(1,3) BASE→sensor translation (meters), torch view of the constant."""
        return self._mount_t

    @property
    def mount_q(self) -> torch.Tensor:
        """(1,4) BASE→sensor rotation (xyzw), torch view of the constant."""
        return self._mount_q

    def angle_keys(self, points_base: torch.Tensor) -> torch.Tensor:
        """Sensor-frame angular keys for (…,3) BASE points (batched, pure)."""
        return sensor_frame_angle_keys(points_base, self._mount_t, self._mount_q)

    def assert_coverage(self, sensor_pos_w: torch.Tensor) -> None:
        """Coverage assertion BEFORE any raycast (see assert_mesh_covers_sensor)."""
        assert_mesh_covers_sensor(
            self.mesh_bounds,
            (float(sensor_pos_w[0]), float(sensor_pos_w[1])),
            float(self.cfg.max_range_m),
        )

    # ── the scan ─────────────────────────────────────────────────────────────

    def scan_rows(
        self,
        rows,
        *,
        perturbations=None,
        persistent_scan_numbers: dict | None = None,
    ) -> list[ScanFrame]:
        """Raycast the given env rows NOW and build the full scan products.

        One single-env sensor per row ⇒ no physical raycast is launched for any
        non-listed env. Raises ``ScanFaultError`` (after recording the fault)
        on non-finite raw output or hit endpoint/dist inconsistency; the
        caller's fault policy (freeze lifecycle, record, raise) applies.

        P6 consumption seam (``perturbations`` = an
        ``ObservationPerturbations`` whose lidar streams are active,
        ``persistent_scan_numbers`` = {row: next persistent 1-based scan
        number} owned by the runtime): the RAW fault checks and raw arrays run
        FIRST and stay verbatim; the gap schedule is advanced per row IN SCAN
        ORDER (``begin_scan``); the consumed copy then EXCLUDES the gap-sector
        rays before grouping and carries d_obs = clip(d_raw + ε, 0, 15) on hit
        rays OUTSIDE the gap, always along the ORIGINAL unit ray. Grouping
        cones and angular keys are derived from the raw unit directions
        rotated into BASE (a d_obs = 0 point keeps its original
        angle/key, never a noisy-xyz radius division). B8's BASE-x cutoff
        DOES use consumed xyz and can change the group under range noise;
        ``far = pool_valid & ~near``; FPS runs on consumed METRIC xyz (noisy when range
        is on). Direction keys of the selected members are attached to the
        frame for the window metadata. With no perturbations (or only the
        proprio stream active) the path is EXACTLY the clean P0–P5 one — no
        re-defined sampling, no new numerics.
        """
        robot = self.env.scene["robot"]
        c = self.cfg
        max_range = float(c.max_range_m)
        is_cuda = str(self.device).startswith("cuda")
        lidar_active = perturbations is not None and bool(perturbations.lidar_active)
        if lidar_active and persistent_scan_numbers is None:
            raise ValueError(
                "perturbed scans require the runtime-owned persistent scan "
                "numbers (1-based, never reset) for the gap schedule"
            )
        if is_cuda:
            wp.synchronize_device()  # pending physics kernels -> pose reads
        base_pos = robot.data.root_link_pos_w          # (B,3)
        base_quat_wxyz = robot.data.root_link_quat_w   # (B,4) wxyz (mjlab)
        frames: list[ScanFrame] = []
        for row in rows:
            row = int(row)
            if not (0 <= row < self.num_envs):
                raise IndexError(f"env row {row} out of range [0, {self.num_envs})")
            bundle = self._sensors[row]
            base_p = base_pos[row].detach()
            base_q_xyzw = base_quat_wxyz[row].detach()[[1, 2, 3, 0]]  # wxyz→xyzw
            sensor_q = _quat_mul_xyzw(base_q_xyzw.view(1, 4), self._mount_q)[0]
            sensor_p = base_p + _quat_apply_xyzw(
                base_q_xyzw.view(1, 4), self._mount_t
            )[0]
            # P6 gap schedule advance happens in per-env persistent scan
            # order BEFORE any consumption processing (the event offset is
            # drawn/fixed before the earliest candidate frame per contract).
            gap_fired = False
            if lidar_active:
                gap_fired = bool(
                    perturbations.begin_scan(
                        row, int(persistent_scan_numbers[row])
                    )
                )
            # COVERAGE FIRST: never let a finite mesh edge become a false miss.
            self.assert_coverage(sensor_p)
            bundle.pos_tensor.copy_(sensor_p.view(1, 3))
            bundle.quat_tensor.copy_(sensor_q.view(1, 4))
            if is_cuda:
                torch.cuda.synchronize()  # pose writes -> raycast kernel
            points_sensor, dist = bundle.sensor.update()
            if is_cuda:
                wp.synchronize_device()  # raycast outputs -> torch
            points_backend = points_sensor.detach().view(-1, 3).clone()
            dist = dist.detach().view(-1).clone()

            # Fault checks: NaN/Inf are faults EVEN IF they would be maskable;
            # never silently sanitized, never disguised as valid.
            if not bool(torch.isfinite(dist).all()) or not bool(
                torch.isfinite(points_backend).all()
            ):
                self.scan_records.append(
                    {
                        "env_row": row,
                        "fault": "nonfinite_raw_output",
                        "sensor_pos_w": sensor_p.detach().cpu().numpy().tolist(),
                        "sensor_quat_xyzw": sensor_q.detach().cpu().numpy().tolist(),
                    }
                )
                raise ScanFaultError(
                    f"env {row}: raw raycast output contains NaN/Inf (fault; "
                    "recorded, never sanitized)"
                )

            # No-hit identity: the kernel writes far_plane VERBATIM for misses
            # (sensor_kernels/lidar_kernels_warp.py), so the EXACT documented
            # identity is dist == max_range. Real hits below max_range
            # (including 15−ε) stay REAL HITS — no near-range promotion, no ε
            # band. Any dist outside [0, max_range] is a fault, never clamped.
            unit_dirs = bundle.ray_dirs
            no_hit = dist == max_range
            if bool((dist < 0.0).any()) or bool((dist > max_range).any()):
                n_low = int((dist < 0.0).sum().item())
                n_high = int((dist > max_range).sum().item())
                self.scan_records.append(
                    {
                        "env_row": row,
                        "fault": "dist_out_of_range",
                        "sensor_pos_w": sensor_p.detach().cpu().numpy().tolist(),
                        "dist_min": float(dist.min().item()),
                        "dist_max": float(dist.max().item()),
                        "count_below_zero": n_low,
                        "count_above_max_range": n_high,
                    }
                )
                raise ScanFaultError(
                    f"env {row}: raw dist outside [0, {max_range}] m "
                    f"({n_low} below 0, {n_high} above {max_range}); fault "
                    "(recorded; never clamped)"
                )
            hit = ~no_hit
            if bool(hit.any()):
                norms = torch.linalg.vector_norm(points_backend[hit], dim=-1)
                if not bool(
                    torch.allclose(norms, dist[hit], atol=1e-3, rtol=1e-4)
                ):
                    self.scan_records.append(
                        {
                            "env_row": row,
                            "fault": "hit_endpoint_dist_mismatch",
                            "sensor_pos_w": sensor_p.detach().cpu().numpy().tolist(),
                        }
                    )
                    raise ScanFaultError(
                        f"env {row}: hit endpoint norm != recorded dist (fault)"
                    )

            # ── P6 consumed-copy extras (None on the clean path) ────────────
            consumed_valid_full = None
            consumed_dists_full = None
            range_epsilon = None
            range_applied = None
            if lidar_active:
                from .perturbations import apply_range_noise

                gap_cfg = perturbations.cfg.gap
                if gap_cfg is not None and gap_fired:
                    lo, hi = gap_cfg.sector
                    gap_mask = (self._raw_azimuth_deg >= lo) & (self._raw_azimuth_deg < hi)
                else:
                    gap_mask = torch.zeros_like(no_hit)
                consumed_valid_full = ~gap_mask
                epsilon = (
                    perturbations.draw_range_epsilon(row, self.num_raw_rays)
                    if perturbations.range_enabled
                    else None
                )
                clip = perturbations.cfg.range
                consumed_dists_full, range_applied = apply_range_noise(
                    dist,
                    no_hit,
                    consumed_valid_full,
                    epsilon,
                    clip_min_m=(
                        float(c.max_range_m) if clip is None else float(clip.clip_min_m)
                    ),
                    clip_max_m=(
                        float(c.max_range_m) if clip is None else float(clip.clip_max_m)
                    ),
                )
                range_epsilon = epsilon

            if lidar_active:
                # CONSUMED copy: no-hit keeps the EXACT max-range endpoint
                # along the original ray; hits carry d_obs along the ORIGINAL
                # unit ray (d_obs == raw at no-hit rows, so one uniform
                # product covers both; a clipped d_obs = 0 keeps direction,
                # angle/key identity; B8's x cutoff may change its group).
                points_sensor = consumed_dists_full.unsqueeze(-1) * unit_dirs
            else:
                # CONFIRMED miss policy applied to a COPY: the max-range
                # endpoint ALONG THE ORIGINAL RAY in the SENSOR frame, BEFORE
                # the sensor→BASE transform (defensive exactness: the kernel
                # already writes far_plane*dir for misses). The ORIGINAL
                # backend pixels stay verbatim in the frame for audit.
                points_sensor = torch.where(
                    no_hit.unsqueeze(-1), max_range * unit_dirs, points_backend
                )

            # Sensor frame -> acquisition-time BASE frame (root body BASE).
            points_base = _quat_apply_xyzw(
                self._mount_q.expand(self.num_raw_rays, 4), points_sensor
            ) + self._mount_t.expand(self.num_raw_rays, 3)

            # Candidate gather: stride=1 retains EVERY raw ray for both groups.
            # C-order identity az*96+el is kept for exact source provenance.
            pool_raw = self.pool_indices
            pool_points = points_base.index_select(0, pool_raw)
            pool_dists = dist.index_select(0, pool_raw)

            if lidar_active:
                # Direction-based grouping (P6): the cone membership and the
                # angular keys come from the raw unit rays — never from the
                # consumed xyz (zero-distance cannot invent an angle/key).
                # BASE-x cutoff uses consumed xyz; far is the valid complement.
                dirs_base = _quat_apply_xyzw(
                    self._mount_q.expand(self.num_raw_rays, 4), unit_dirs
                )
                cosang_pool = (dirs_base @ self.cone_axis_body).index_select(0, pool_raw)
                pool_valid = consumed_valid_full.index_select(0, pool_raw)
                near_mask, far_mask = partition_cone_forward_x(
                    cosang_pool, pool_points,
                    theta_deg=c.split_theta_deg,
                    forward_x_max_m=c.split_max_forward_x_m,
                    valid=pool_valid,
                )
                pool_keys = direction_angle_keys(unit_dirs).index_select(0, pool_raw)
            else:
                # Clean B8: direction cone + observed endpoint BASE-x cutoff.
                # Nonzero physical ray endpoints (including exact 15m misses)
                # yield the same cone identity as their original source rays.
                rel = pool_points - self._mount_t.expand(int(pool_raw.numel()), 3)
                r_pool = torch.linalg.vector_norm(rel, dim=-1)
                cosang = (rel @ self.cone_axis_body) / (r_pool + 1e-9)
                near_mask, far_mask = partition_cone_forward_x(
                    cosang, pool_points,
                    theta_deg=c.split_theta_deg,
                    forward_x_max_m=c.split_max_forward_x_m,
                )

                pool_keys = self.angle_keys(pool_points)

            # Near: FPS over the near group (no repeats, shortage zero-pads).
            near_idx, near_count = farthest_point_sampling_indices(
                pool_points, near_mask, c.near_points
            )
            near_valid = torch.zeros(c.near_points, dtype=torch.bool, device=self.device)
            near_points = torch.zeros(
                c.near_points, 3, dtype=torch.float32, device=self.device
            )
            near_indices = torch.full(
                (c.near_points,), -1, dtype=torch.long, device=self.device
            )
            if near_count > 0:
                near_valid[:near_count] = True
                near_points[:near_count] = pool_points.index_select(0, near_idx)
                near_indices[:near_count] = pool_raw.index_select(0, near_idx)

            # Far: uniform angular-key sample over the far group.
            far_idx, far_count = uniform_angular_sample_indices(
                pool_keys, far_mask, c.far_points_per_scan
            )
            far_valid = torch.zeros(
                c.far_points_per_scan, dtype=torch.bool, device=self.device
            )
            far_points = torch.zeros(
                c.far_points_per_scan, 3, dtype=torch.float32, device=self.device
            )
            far_indices = torch.full(
                (c.far_points_per_scan,), -1, dtype=torch.long, device=self.device
            )
            if far_count > 0:
                far_valid[:far_count] = True
                far_points[:far_count] = pool_points.index_select(0, far_idx)
                far_indices[:far_count] = pool_raw.index_select(0, far_idx)

            near_keys_sel = None
            far_keys_sel = None
            if lidar_active:
                # Direction-derived sort keys of the SELECTED members, aligned
                # with the slots (dummy 0 at shortage slots, whose points are
                # masked invalid) — the window carries them so later control
                # steps sort by the ORIGINAL ray direction, never by the
                # consumed xyz (zero-distance identity).
                near_keys_sel = torch.zeros(
                    c.near_points, dtype=torch.float32, device=self.device
                )
                far_keys_sel = torch.zeros(
                    c.far_points_per_scan, dtype=torch.float32, device=self.device
                )
                if near_count > 0:
                    near_keys_sel[:near_count] = pool_keys.index_select(0, near_idx)
                if far_count > 0:
                    far_keys_sel[:far_count] = pool_keys.index_select(0, far_idx)

            frame = ScanFrame(
                env_row=row,
                base_pos_w=base_p.clone(),
                base_quat_wxyz=base_quat_wxyz[row].detach().clone(),
                sensor_pos_w=sensor_p.detach().clone(),
                sensor_quat_xyzw=sensor_q.detach().clone(),
                raw_ray_dirs=unit_dirs.clone(),
                raw_dists=dist,
                raw_points_sensor=points_backend,
                points_sensor=points_sensor,
                raw_points_base=points_base,
                no_hit_mask=no_hit,
                pool_raw_indices=pool_raw.clone(),
                pool_points_base=pool_points,
                pool_dists=pool_dists,
                pool_angle_keys=pool_keys,
                near_mask=near_mask,
                far_mask=far_mask,
                near_indices=near_indices,
                near_valid=near_valid,
                near_points_base=near_points,
                far_indices=far_indices,
                far_valid=far_valid,
                far_points_base=far_points,
                gap_fired=gap_fired,
                consumed_valid=consumed_valid_full,
                consumed_dists=consumed_dists_full,
                range_epsilon=range_epsilon,
                range_applied_mask=range_applied,
                near_angle_keys=near_keys_sel,
                far_angle_keys=far_keys_sel,
            )
            frames.append(frame)
            # Full raw audit record (identity/time stamped by the runtime via
            # stamp_scan_record once it assigns scan_id/episode/time).
            self.scan_records.append({"frame": frame, "env_row": row})
        return frames

    def stamp_scan_record(self, frame: ScanFrame, *, scan_id: int, episode_id: int,
                          time_s: float) -> None:
        """Attach the runtime-assigned identity to the frame's audit record."""
        frame.scan_id = int(scan_id)
        frame.episode_id = int(episode_id)
        frame.time_s = float(time_s)
