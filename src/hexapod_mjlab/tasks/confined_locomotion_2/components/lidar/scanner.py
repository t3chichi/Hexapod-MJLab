"""Fixed T1-channel full/sparse physical scanner for task _2.

The complete real Airy catalog is classified once at the frozen T1 settled
pose and height. Runtime near/far membership is immutable by raw channel ID;
gap affects validity only, range noise affects consumed XYZ only. Both modes
use the same physical Warp query and deterministic XYZ FPS256/FPS128. Sparse
raw arrays are compact and carry raw_source_ids; selected indices always name
original Airy channels. Runtime owns scan cadence, identities and resets.
Flat uses its existing plane proxy. Explicit Terrain Memory collection uses
native compiled Heightfield rays before the same consumption/FPS processing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import warp as wp

from hexapod_mjlab.rl.confined_locomotion_2.contracts import (
    FAR_POINTS_PER_SCAN,
    FAR_WINDOW_SLOTS,
    NEAR_POINTS,
)
from hexapod_mjlab.common.ground_validation import (
    EXPECTED_PLANE_FRICTION,
    PlaneValidationError,
    validate_single_horizontal_plane,
)
from hexapod_mjlab.common.lidar import resolve_lidar_mount

from .sampling import (
    farthest_point_sampling_indices, farthest_point_sampling_masked,
    far_candidate_mask, sampling_identity,
)

__all__ = [
    "ConfinedLidarCfg",
    "training_lidar_defaults",
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
    "farthest_point_sampling_masked",
]


def training_lidar_defaults():
    """Published sparse hit-only recipe for new training, independent of cwd.

    The low-level scanner keeps its historical full/all-endpoint defaults for
    explicit fixtures and old checkpoint playback.
    """
    selection = Path(__file__).resolve().parents[2] / "assets/fixed_channels/v1_hit10/near_x6_far_x10.npz"
    return {"raycast_mode": "sparse", "far_sampling": "hit_only_10m",
            "sparse_selection_path": str(selection)}


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


def uniform_angular_sample_indices(
    angle_keys: torch.Tensor, group_mask: torch.Tensor, k: int
) -> tuple[torch.Tensor, int]:
    """Historical angular sampler for comparisons; unused by this task pipeline.

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
    original ray keeps its angle key and cone identity (the fixed grouping ignores endpoint-x
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
    """Task-local scanner configuration with fixed reference-channel grouping.

    The B8 cone + BASE-x rule is evaluated only at the frozen T1 reference.
    No claim of final terrain learning performance is made.
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
    raycast_mode: str = "full"
    sparse_selection_path: str | None = None
    # Explicit candidate policy; physical raycast range and fixed groups stay unchanged.
    far_sampling: str = "all_endpoints"

    # Stage-A B8 baseline: real cone half-angle 90-theta=62 deg; tilted
    # corrected mount axis, followed by endpoint BASE-x cutoff. None restores
    # the legacy untruncated grouping for a separately identified comparison.
    split_theta_deg: float = 28.0
    split_axis_tilt_deg: float = 56.0
    split_max_forward_x_m: float | None = 3.5

    # Consumption sizes (must equal the F1 contract constants).
    near_points: int = NEAR_POINTS               # 256
    far_points_per_scan: int = FAR_POINTS_PER_SCAN   # 128
    far_window_slots: int = FAR_WINDOW_SLOTS     # 1 (latest scan only)

    # Raycast proxy mesh (engineering parameter; physical plane is infinite).
    plane_mesh_half_extent_m: float = 100.0
    expected_plane_friction: tuple[float, float, float] = EXPECTED_PLANE_FRICTION

    # ALL sensor noise flags OFF in this batch (documented zeros/false).
    enable_sensor_noise: bool = False
    pixel_dropout_prob: float = 0.0
    pixel_std_dev_multiplier: float = 0.0

    def __post_init__(self) -> None:
        c = self
        sampling_identity(c.far_sampling)
        if c.far_sampling == "hit_only_10m" and c.max_range_m != 15.0:
            raise ValueError("hit_only_10m retains the physical 15 m raycast range")
        if c.raycast_mode not in ("full", "sparse"):
            raise ValueError("raycast_mode must be full or sparse")
        if (c.raycast_mode == "sparse") != (c.sparse_selection_path is not None):
            raise ValueError("sparse mode requires a selection path; full mode forbids one")
        if c.sparse_selection_path is not None and not str(c.sparse_selection_path).strip():
            raise ValueError("sparse selection path must not be empty")
        if (c.airy_n_azimuth, c.airy_n_elevation, c.airy_horizontal_resolution_deg) != (900, 96, 0.4):
            raise ValueError("fixed channel catalog requires the original 900x96 AIRY pattern")
        if c.pool_azimuth_stride != 1:
            raise ValueError("fixed channels use explicit sparse selections; stride must be 1")
        if c.split_max_forward_x_m is None:
            raise ValueError("fixed channel baseline requires a finite forward cutoff")
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
    """One physical scan: raw arrays follow compact raw_source_ids order.

    pool_raw_indices and selected near/far indices are original Airy IDs.
    Gap invalidates samples; raw no-hit remains a max-range endpoint. Only
    legacy far sampling consumes it; hit_only_10m filters far eligibility.
    Raw arrays stay verbatim, consumed arrays carry existing range noise.
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
    raw_points_base: torch.Tensor       # (R,3) unperturbed BASE endpoints;
                                        # no-hit uses the clean canonical endpoint
    no_hit_mask: torch.Tensor           # (R,) bool; no-hit rays are VALID endpoints
    pool_raw_indices: torch.Tensor      # (P,) int64 pool→raw ray ids
    pool_points_base: torch.Tensor      # (P,3)
    pool_dists: torch.Tensor            # (P,)
    pool_angle_keys: torch.Tensor       # (P,) sensor-frame angular key
    near_mask: torch.Tensor             # (P,) fixed near membership AND validity
    far_mask: torch.Tensor              # (P,) bool
    near_indices: torch.Tensor          # (256,) int64 raw ray ids, −1 = shortage slot
    near_valid: torch.Tensor            # (256,) bool
    near_points_base: torch.Tensor      # (256,3) zero-filled at invalid slots
    far_indices: torch.Tensor           # (128,) int64 raw ray ids, −1 = shortage slot
    far_valid: torch.Tensor             # (128,) bool
    far_points_base: torch.Tensor       # (128,3) zero-filled at invalid slots
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
    far_angle_keys: torch.Tensor | None = None    # (128,) direction keys, slot-aligned
    raw_source_ids: torch.Tensor | None = None   # (R,) local raw slot -> original AIRY ID


# ── the pipeline ─────────────────────────────────────────────────────────────


class ConfinedScanPipeline:
    """Real scanner using one shared plane mesh and batched due-row raycasts.

    Construct AFTER a real reset geometry refresh (the runtime does so at its
    first explicit reset); the plane identity/mesh are read from the compiled
    template scene once and are state-independent (static terrain body). The
    pipeline owns NO clock: the runtime calls ``scan_rows`` exactly for the env
    rows whose per-env 10 Hz schedule is due (or the freshly reset rows for
    their first scan), so a physical raycast is never launched for non-due envs.
    """

    def __init__(self, env, cfg: ConfinedLidarCfg | None = None, *, terrain_query=None):
        self.env = env
        self.cfg = cfg or ConfinedLidarCfg()
        c = self.cfg
        self.device = env.device
        device_str = str(self.device)
        self.num_envs = int(env.num_envs)

        self.terrain_query = terrain_query
        if terrain_query is not None and terrain_query.env is not env:
            raise ValueError("terrain query must own the same environment")
        if terrain_query is None:
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

        # One common catalog defines both modes and the immutable grouping.
        from .channels import build_channel_catalog, load_sparse_selection
        from .sparse_raycast import ChannelRaycaster
        self.catalog = build_channel_catalog(
            theta_deg=c.split_theta_deg, axis_tilt_deg=c.split_axis_tilt_deg,
            forward_x_max_m=c.split_max_forward_x_m, max_range_m=c.max_range_m)
        if c.raycast_mode == "sparse":
            source_ids, selection_identity = load_sparse_selection(c.sparse_selection_path, self.catalog)
        else:
            source_ids, selection_identity = self.catalog.source_ids, None
        self.scan_identity = {
            "grouping": "fixed_t1_channels/v1", "mode": c.raycast_mode,
            "catalog": self.catalog.identity, "selection": selection_identity,
        }
        if terrain_query is not None:
            self.scan_identity["terrain"] = terrain_query.surface.identity
            self.scan_identity["backend"] = "compiled_hfield_bvh/v1"
        # Legacy identities stay readable for explicit all-endpoint baselines.
        # New candidate semantics are never silently treated as that baseline.
        if c.far_sampling != "all_endpoints":
            self.scan_identity["sampling"] = sampling_identity(c.far_sampling)
        mask_grid = self.catalog.near_mask.reshape(c.airy_n_azimuth, c.airy_n_elevation)
        interior = mask_grid & np.roll(mask_grid, 1, 0) & np.roll(mask_grid, -1, 0)
        interior[:, 1:] &= mask_grid[:, :-1]
        interior[:, :-1] &= mask_grid[:, 1:]
        interior[:, (0, -1)] = False
        boundary = np.flatnonzero((mask_grid & ~interior).reshape(-1))
        boundary = boundary[::max(1, int(np.ceil(len(boundary) / 256)))]
        rotation = np.asarray(self.catalog.identity["mount_base_to_sensor"]["rotation_base_sensor"])
        directions_base = self.catalog.directions_sensor[boundary] @ rotation.T
        origins = np.broadcast_to(self.mount_pos, directions_base.shape)
        self.fixed_boundary_segments = np.stack((origins, origins + directions_base * min(3.0, c.max_range_m)), axis=1)
        self.full_ray_dirs = torch.tensor(self.catalog.directions_sensor, dtype=torch.float32, device=self.device)
        self.source_ray_ids = torch.tensor(source_ids, dtype=torch.long, device=self.device)
        self.num_full_rays = len(self.catalog.directions_sensor)
        self.num_raw_rays = len(source_ids)
        self.pool_indices = torch.arange(self.num_raw_rays, dtype=torch.long, device=self.device)
        full_near = torch.tensor(self.catalog.near_mask, dtype=torch.bool, device=self.device)
        self.fixed_near_mask = full_near.index_select(0, self.source_ray_ids)
        self._raw_azimuth_deg = (self.source_ray_ids // c.airy_n_elevation).double() * (360.0 / c.airy_n_azimuth)
        directions = self.full_ray_dirs.index_select(0, self.source_ray_ids).contiguous()
        if terrain_query is None:
            self._sensor = ChannelRaycaster(
                directions=directions,
                position=torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32),
                quaternion=torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32),
                mesh_ids=self.mesh_ids, max_range=c.max_range_m)

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
        """Raycast only due rows, retaining compact raw arrays and global IDs.

        Validity and consumed ranges use the existing per-env perturbation
        streams. Full-sized noise draws are gathered by source ID, so the
        full/subset comparison uses exactly the same per-channel noise.
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
        rows = [int(row) for row in rows]
        if any(row < 0 or row >= self.num_envs for row in rows):
            raise IndexError("scan row outside environment batch")
        if not rows:
            return []
        if is_cuda:
            wp.synchronize_device()  # pending physics kernels -> pose reads
        base_pos = robot.data.root_link_pos_w
        base_quat_wxyz = robot.data.root_link_quat_w
        selected = torch.tensor(rows, dtype=torch.long, device=self.device)
        base_q = base_quat_wxyz.index_select(0, selected)[:, [1, 2, 3, 0]]
        sensor_quats = _quat_mul_xyzw(base_q, self._mount_q.expand(len(rows), 4))
        sensor_positions = base_pos.index_select(0, selected) + _quat_apply_xyzw(
            base_q, self._mount_t.expand(len(rows), 3))
        ray_dirs = self.full_ray_dirs.index_select(0, self.source_ray_ids)
        prefix = len(rows)
        if self.terrain_query is not None:
            world_dirs = _quat_apply_xyzw(
                sensor_quats[:, None].expand(-1, self.num_raw_rays, -1),
                ray_dirs[None].expand(len(rows), -1, -1))
            raw, geom = self.terrain_query.raycast(
                selected, sensor_positions[:, None].expand_as(world_dirs), world_dirs)
            hit = (geom >= 0) & (raw >= 0) & (raw < max_range)
            batch_distances = torch.where(hit, raw, torch.full_like(raw, max_range))
            batch_points = batch_distances[:, :, None] * ray_dirs[None]
        else:
            # Preserve the Python-double coverage calculation and refuse to
            # raycast the first uncovered row or any later row. P6 side effects
            # and the original coverage error still occur in row order below.
            xy = sensor_positions[:, :2].double()
            xmin, xmax, ymin, ymax = self.mesh_bounds
            uncovered = ((xmin > xy[:, 0] - max_range - 1e-3) |
                         (xmax < xy[:, 0] + max_range + 1e-3) |
                         (ymin > xy[:, 1] - max_range - 1e-3) |
                         (ymax < xy[:, 1] + max_range + 1e-3)).cpu().tolist()
            prefix = next((i for i, bad in enumerate(uncovered) if bad), len(rows))
            if prefix:
                self._sensor.position[:prefix].copy_(sensor_positions[:prefix])
                self._sensor.quaternion[:prefix].copy_(sensor_quats[:prefix])
                points, distances = self._sensor.update(prefix)
                batch_points = points[:, 0, :, 0]
                batch_distances = distances[:, 0, :, 0]
        batch_checks = []
        if prefix:
            # One host transfer for all raw geometry flags. Interpret failures
            # below, before that row's range draw, preserving stream ordering.
            finite = (torch.isfinite(batch_distances).all(dim=1) &
                      torch.isfinite(batch_points).all(dim=(1, 2)))
            in_range = ((batch_distances >= 0) & (batch_distances <= max_range)).all(dim=1)
            norms = torch.linalg.vector_norm(batch_points, dim=-1)
            endpoint_ok = ((batch_distances == max_range) | torch.isclose(
                norms, batch_distances, atol=1e-3, rtol=1e-4)).all(dim=1)
            batch_checks = torch.stack((finite, in_range, endpoint_ok), dim=1).cpu().tolist()
        frames: list[ScanFrame] = []
        for scan_row, row in enumerate(rows):
            base_p = base_pos[row].detach()
            sensor_q, sensor_p = sensor_quats[scan_row], sensor_positions[scan_row]
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
            if scan_row >= prefix:
                self.assert_coverage(sensor_p)  # original error before any raycast for this row
            unit_dirs = ray_dirs
            # Retained frames own compact rows, not the reusable batch workspace.
            dist = batch_distances[scan_row].detach().clone()
            points_backend = batch_points[scan_row].detach().clone()

            # Fault checks: NaN/Inf are faults EVEN IF they would be maskable;
            # never silently sanitized, never disguised as valid.
            if not batch_checks[scan_row][0]:
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
            no_hit = dist == max_range
            if not batch_checks[scan_row][1]:
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
            if not batch_checks[scan_row][2]:
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
                from hexapod_mjlab.common.perturbations import apply_range_noise

                gap_cfg = perturbations.cfg.gap
                if gap_cfg is not None and gap_fired:
                    lo, hi = gap_cfg.sector
                    gap_mask = (self._raw_azimuth_deg >= lo) & (self._raw_azimuth_deg < hi)
                else:
                    gap_mask = torch.zeros_like(no_hit)
                consumed_valid_full = ~gap_mask
                epsilon = (
                    perturbations.draw_range_epsilon(row, self.num_full_rays).index_select(0, self.source_ray_ids)
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
                # angle/key identity; fixed group membership is unchanged).
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
            # Audit raw geometry independently of the consumed P6 copy. The
            # previous raw_points_base alias contained noisy points when P6
            # was active. Keep the existing canonical no-hit arithmetic so
            # raw BASE endpoints agree bitwise between clean and noisy scans.
            if lidar_active:
                unperturbed_sensor = torch.where(
                    no_hit.unsqueeze(-1), max_range * unit_dirs, points_backend
                )
                raw_points_base = _quat_apply_xyzw(
                    self._mount_q.expand(self.num_raw_rays, 4), unperturbed_sensor
                ) + self._mount_t.expand(self.num_raw_rays, 3)
            else:
                raw_points_base = points_base

            # Candidate gather: stride=1 retains EVERY raw ray for both groups.
            # C-order identity az*96+el is kept for exact source provenance.
            pool_raw = self.pool_indices
            pool_points = points_base.index_select(0, pool_raw)
            pool_dists = dist.index_select(0, pool_raw)

            # Membership is fixed by the reference catalog, never by an
            # actual/noisy endpoint. Gap removes observations from both pools.
            pool_valid = (torch.ones_like(no_hit) if consumed_valid_full is None
                          else consumed_valid_full).index_select(0, pool_raw)
            fixed_near = self.fixed_near_mask.index_select(0, pool_raw)
            near_mask = pool_valid & fixed_near
            far_mask = pool_valid & ~fixed_near
            far_mask = far_candidate_mask(
                far_mask, pool_dists, no_hit.index_select(0, pool_raw), c.far_sampling
            )
            pool_keys = direction_angle_keys(unit_dirs).index_select(0, pool_raw)

            # Sampling is deferred until every due row has produced its own
            # validated scan. Then both groups are sampled in batched GPU calls.
            near_valid = torch.zeros(c.near_points, dtype=torch.bool, device=self.device)
            far_valid = torch.zeros(c.far_points_per_scan, dtype=torch.bool, device=self.device)
            near_points = pool_points.new_zeros((c.near_points, 3))
            far_points = pool_points.new_zeros((c.far_points_per_scan, 3))
            near_indices = torch.full((c.near_points,), -1, dtype=torch.long, device=self.device)
            far_indices = torch.full((c.far_points_per_scan,), -1, dtype=torch.long, device=self.device)
            near_keys_sel = pool_keys.new_zeros(c.near_points)
            far_keys_sel = pool_keys.new_zeros(c.far_points_per_scan)

            frame = ScanFrame(
                env_row=row,
                base_pos_w=base_p.clone(),
                base_quat_wxyz=base_quat_wxyz[row].detach().clone(),
                sensor_pos_w=sensor_p.detach().clone(),
                sensor_quat_xyzw=sensor_q.detach().clone(),
                raw_ray_dirs=unit_dirs,
                raw_dists=dist,
                raw_points_sensor=points_backend,
                points_sensor=points_sensor,
                raw_points_base=raw_points_base,
                no_hit_mask=no_hit,
                pool_raw_indices=self.source_ray_ids.index_select(0, pool_raw),
                raw_source_ids=self.source_ray_ids,
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
        self._sample_frames(frames)
        return frames

    def _sample_frames(self, frames: list[ScanFrame]) -> None:
        """Sample only this call's due rows, preserving frame/source identities.

        Terrain raycasts run together; per-row perturbation draws keep their order.
        FPS for all resulting frames is independent per row and runs together;
        no non-due row is sampled, scanned or assigned a new identity here.
        """
        if not frames:
            return
        points = torch.stack([frame.pool_points_base for frame in frames])
        for prefix, count in (("near", self.cfg.near_points),
                              ("far", self.cfg.far_points_per_scan)):
            masks = torch.stack([getattr(frame, prefix + "_mask") for frame in frames])
            selected = farthest_point_sampling_masked(points, masks, count)
            valid = selected >= 0
            safe = selected.clamp_min(0)
            xyz = points.gather(1, safe[:, :, None].expand(-1, -1, 3))
            xyz = torch.where(valid[:, :, None], xyz, 0.0)
            for row, frame in enumerate(frames):
                slots = safe[row]
                ids = frame.pool_raw_indices.index_select(0, slots)
                ids = torch.where(valid[row], ids, -1)
                setattr(frame, prefix + "_indices", ids)
                setattr(frame, prefix + "_valid", valid[row])
                setattr(frame, prefix + "_points_base", xyz[row])
                if getattr(frame, prefix + "_angle_keys") is not None:
                    keys = frame.pool_angle_keys.index_select(0, slots)
                    setattr(frame, prefix + "_angle_keys", torch.where(valid[row], keys, 0.0))

    def stamp_scan_record(self, frame: ScanFrame, *, scan_id: int, episode_id: int,
                          time_s: float) -> None:
        """Attach the runtime-assigned identity to the frame's audit record."""
        frame.scan_id = int(scan_id)
        frame.episode_id = int(episode_id)
        frame.time_s = float(time_s)
