"""Real ground supervision: explicit grid config + flat-plane provider.

User-confirmed coordinate contract (2026-10-07; docs/research/训练架构对齐.md
§4, docs/tasks/confined_locomotion/平地学习与基准测量.md §3):

* the grid consists of FIXED body-xy positions ``[x_B, y_B]``; it follows the
  FULL body translation, yaw, pitch and roll;
* the query at each grid point runs along the SCAN-TIME BODY z AXIS and
  returns the SIGNED height ``h_B`` relative to the root reference point, so
  the world intersection point satisfies
  ``p_ground_W = p_root_W + R_WB @ [x_B, y_B, h_B]`` exactly;
* ``GroundLabel.positions`` declares the scan-time BODY coordinates
  ``[x_B, y_B, h_B]`` (NOT a world height, NOT the old world-vertical
  difference ``z_ground_W - z_root_W``); the world intersection point is kept
  separately as OFFLINE metadata (``GroundQueryResult.world_points`` / the
  runtime record), never in the label;
* the ground is usually BELOW the root, so ``h_B`` is usually negative; the
  signed line query may legitimately return positive or negative values and
  is NEVER clipped into a positive distance;
* this definition does NOT change the envelope Δz / h_init / h₀ vertical
  references (those stay world-vertical and ground-referenced).

This module owns three pieces and nothing else:

1. ``GroundGridCfg`` — an IMMUTABLE, EXPLICIT grid configuration:

   * direct construction from full ordered x/y sample coordinates, plus the
     engineering constructor ``from_ranges(x_range, y_range, nx, ny, ...)``;
     BOTH routes go through the SAME validation and produce the SAME class;
     no final range is defaulted here, so coordinates remain explicit;
   * ``num_points`` (M) is derived from the SAME ordered coordinates that
     define the grid, with a FIXED point order (``y`` outer ascending,
     ``x`` inner ascending, ``index = iy*nx + ix``) so grid identity is
     stable across construction routes;
   * every config DECLARES its semantics: ``body_xyz_semantics`` (fixed
     string), the near-parallel invalidity tolerance ``parallel_tol``
     (default 1e-6, RECORDED in the identity), and the explicit
     ``validity_mode`` (``"geometry_only_flat_training"``);
   * ``identity()`` / ``identity_hash()`` serialize the configuration
     content (route-independent, content-based);
   * strict rejection: bool/float/non-positive ``nx``/``ny`` (from_ranges),
     non-finite or non-strictly-increasing coordinates, empty axes, bad
     tolerance, undeclared/unknown validity mode.

2. ``FlatGroundProvider`` — the REAL flat-ground geometry source:

   * built from the ACTUAL compiled ``mj_model``: it reuses
     ``scanner.validate_single_horizontal_plane`` (exactly one horizontal
     ``mjGEOM_PLANE`` with the confirmed friction) and ADDITIONALLY verifies
     that the plane's body — and every ancestor up to the world body — is
     fully STATIC (no joints, no mocap); the REAL geometry identity (geom
     id/name, body id/name, world position, world normal, friction,
     geom_size) is retained verbatim in ``plane_record``;
   * NO hardcoded z=0, no old heightmap, no fitted lidar points, no
     synthetic labels: the analytic intersection uses the ACTUAL plane world
     position/normal read from the compiled model and the FULL scan-time
     root pose (position + quaternion);
   * ``query()`` validates the root pose strictly: position finite;
     quaternion finite AND canonical (‖q‖=1 within ``quat_norm_atol``) — a
     non-canonical quaternion is REJECTED, never silently renormalized;
   * near-parallel queries (|n·ẑ_B| < ``parallel_tol``, one scalar per root
     pose — the whole grid shares the body-z query direction) are ISOLATED
     as invalid with heights/positions/world points EXACTLY 0; no NaN ever
     leaks out of a query.

3. ``geometry_valid`` semantics (the ``validity_mode`` declaration):

   ``geometry_only_flat_training`` is the EXPLICIT declaration consumed by
   the training runtime:
   identical geometric-query semantics (the same validated real plane, the
   same signed body-z query, and the same near-parallel isolation). It
   carries the flat-training mask recipe (geometry-valid positions are
   ALL supervised; FPS/near-support is diagnostics only), but is NOT an
   observation-support or occlusion claim. External masks may be combined
   by callers explicitly (outside this module).

This module performs geometry queries only: it does not implement any loss,
weight, optimizer, training mask recipe, or scene beyond the validated flat
plane, and it never touches the online observation/network inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass

import torch

__all__ = [
    "BODY_XYZ_SEMANTICS",
    "DEFAULT_PARALLEL_TOL",
    "GRID_ORDER",
    "VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING",
    "ACCEPTED_VALIDITY_MODES",
    "FlatGroundProvider",
    "GroundGridCfg",
    "GroundGridError",
    "GroundQueryResult",
]

#: Fixed declaration of the position/height semantics carried by every grid
#: config (and thus by every label/identity produced from it).
BODY_XYZ_SEMANTICS = (
    "fixed body-xy grid positions; signed height h_B along the SCAN-TIME "
    "body z axis relative to the root reference point; world intersection "
    "p_ground_W = p_root_W + R_WB @ [x_B, y_B, h_B]; positions store "
    "scan-time body coordinates [x_B, y_B, h_B] (not world heights, not "
    "z_ground_W - z_root_W); usually negative (ground below root); never "
    "clipped to a positive distance"
)

#: The declared validity semantics for the production flat-training grid.
VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING = "geometry_only_flat_training"
ACCEPTED_VALIDITY_MODES = (VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,)

#: Default near-parallel invalidity tolerance (dimensionless, applied to
#: |n · z_body|), recorded in every config identity.
DEFAULT_PARALLEL_TOL = 1e-6

#: Fixed grid point order (part of the config identity): ``y`` outer loop
#: ascending, ``x`` inner loop ascending, flat index = iy*nx + ix.
GRID_ORDER = "y_outer_x_inner_index_iy*nx+ix"


class GroundGridError(ValueError):
    """Invalid ground grid configuration (counts, coordinates, modes)."""


def _require_coord_axis(name: str, values: object) -> tuple[float, ...]:
    """Validate one coordinate axis: non-empty, finite, strictly increasing.

    Accepts sequences of real numbers (int/float/numpy floats; bools are
    rejected). Returns the immutable float tuple with -0.0 normalized to
    +0.0 so content identity is stable against signed zeros.
    """
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise GroundGridError(
            f"{name} must be a non-empty sequence of real coordinates, got "
            f"{type(values).__name__}"
        )
    coords: list[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise GroundGridError(
                f"{name} contains a non-real coordinate: {v!r}"
            )
        fv = float(v)
        if not math.isfinite(fv):
            raise GroundGridError(f"{name} contains a non-finite coordinate: {v!r}")
        coords.append(fv + 0.0)  # normalize -0.0 -> +0.0
    if not coords:
        raise GroundGridError(f"{name} must contain at least one coordinate")
    for i in range(1, len(coords)):
        if not coords[i - 1] < coords[i]:
            raise GroundGridError(
                f"{name} must be STRICTLY increasing (element {i}: "
                f"{coords[i - 1]!r} >= {coords[i]!r})"
            )
    limit = torch.finfo(torch.float32).max
    if any(abs(c) > limit for c in coords):
        raise GroundGridError(f"{name} is not finite when represented as float32")
    represented = torch.tensor(coords, dtype=torch.float32)
    if represented.numel() > 1 and not bool((represented[1:] > represented[:-1]).all()):
        raise GroundGridError(f"{name} coordinates collapse under float32 representation")
    return tuple(coords)


def _require_count(name: str, value: object, *, minimum: int) -> int:
    """Strict integer count: bools and non-integer types are rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise GroundGridError(
            f"{name} must be an integer >= {minimum}, got {value!r} "
            f"(type {type(value).__name__})"
        )
    if value < minimum:
        raise GroundGridError(f"{name} must be >= {minimum}, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class GroundGridCfg:
    """Immutable explicit ground-supervision grid configuration.

    See the module docstring for the full contract. Fields:

    * ``x_coords`` / ``y_coords`` — the FULL ordered sample coordinates
      (each strictly increasing and finite; direct construction is the
      canonical route, ``from_ranges`` generates the same class);
    * ``validity_mode`` — REQUIRED (no default): one of the declared
      ``"geometry_only_flat_training"`` (included in the identity/hash);
    * ``parallel_tol`` — explicit near-parallel invalidity tolerance
      (0 < tol < 1; default 1e-6, recorded);
    * ``body_xyz_semantics`` — fixed semantics declaration (validated to the
      constant; carried into every identity).
    """

    x_coords: tuple[float, ...]
    y_coords: tuple[float, ...]
    validity_mode: str
    parallel_tol: float = DEFAULT_PARALLEL_TOL
    body_xyz_semantics: str = BODY_XYZ_SEMANTICS

    def __post_init__(self) -> None:
        if not isinstance(self.validity_mode, str) or (
            self.validity_mode not in ACCEPTED_VALIDITY_MODES
        ):
            raise GroundGridError(
                "validity_mode must be explicitly one of "
                f"{ACCEPTED_VALIDITY_MODES!r} (geometry-only flat validity; the "
                "formal training occlusion mask does not exist yet and is NOT "
                "defaulted here), got "
                f"{self.validity_mode!r}"
            )
        tol = self.parallel_tol
        if isinstance(tol, bool) or not isinstance(tol, numbers.Real):
            raise GroundGridError(f"parallel_tol must be a real number, got {tol!r}")
        tol = float(tol)
        if not math.isfinite(tol) or not (0.0 < tol < 1.0):
            raise GroundGridError(
                f"parallel_tol must satisfy 0 < tol < 1 (finite), got {tol!r}"
            )
        if not isinstance(self.body_xyz_semantics, str) or (
            self.body_xyz_semantics != BODY_XYZ_SEMANTICS
        ):
            raise GroundGridError(
                "body_xyz_semantics must be the fixed declared semantics "
                "(see ground.BODY_XYZ_SEMANTICS); custom semantics are not "
                "supported"
            )
        object.__setattr__(self, "x_coords", _require_coord_axis("x_coords", self.x_coords))
        object.__setattr__(self, "y_coords", _require_coord_axis("y_coords", self.y_coords))
        object.__setattr__(self, "parallel_tol", tol)

    # ── derived grid ────────────────────────────────────────────────────────

    @property
    def nx(self) -> int:
        return len(self.x_coords)

    @property
    def ny(self) -> int:
        return len(self.y_coords)

    @property
    def num_points(self) -> int:
        """M — derived from the SAME ordered coordinates that define the grid."""
        return self.nx * self.ny

    def grid_xy(
        self, *, dtype: torch.dtype = torch.float64, device=None
    ) -> torch.Tensor:
        """(M, 2) body-xy grid positions in the FIXED order (y outer, x inner)."""
        x = torch.tensor(self.x_coords, dtype=dtype, device=device)
        y = torch.tensor(self.y_coords, dtype=dtype, device=device)
        xx = x.repeat(self.ny)
        yy = y.repeat_interleave(self.nx)
        return torch.stack([xx, yy], dim=1)

    def grid_points_body(
        self, *, dtype: torch.dtype = torch.float64, device=None
    ) -> torch.Tensor:
        """(M, 3) grid points ``[x_B, y_B, 0]`` (query origins, z filled at query)."""
        xy = self.grid_xy(dtype=dtype, device=device)
        return torch.cat([xy, torch.zeros_like(xy[:, :1])], dim=1)

    # ── serialized identity ─────────────────────────────────────────────────

    def identity(self) -> dict:
        """JSON-serializable content identity (route-independent, content-based).

        Two configs with identical coordinates/semantics/tolerance/mode have
        identical identities and hashes, no matter whether they were built
        directly or via ``from_ranges``.
        """
        return {
            "type": "GroundGridCfg/v1",
            "grid_order": GRID_ORDER,
            "body_xyz_semantics": self.body_xyz_semantics,
            "validity_mode": self.validity_mode,
            "parallel_tol": float(self.parallel_tol),
            "x_coords": [float(v) for v in self.x_coords],
            "y_coords": [float(v) for v in self.y_coords],
            "num_points": int(self.num_points),
        }

    def identity_hash(self) -> str:
        """Stable sha256 of the canonical identity serialization."""
        payload = json.dumps(
            self.identity(), sort_keys=True, allow_nan=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── engineering constructor (same class, same validation) ───────────────

    @classmethod
    def from_ranges(
        cls,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        nx: int,
        ny: int,
        *,
        validity_mode: str,
        parallel_tol: float = DEFAULT_PARALLEL_TOL,
    ):
        """Build a config from ranges and point counts (same class/interface).

        ``nx``/``ny`` must be integers >= 2 and each range a pair of finite
        numbers with start < stop; coordinates are generated at
        ``start + i*(stop-start)/(n-1)`` with the last point exactly ``stop``
        (linspace convention). The generated coordinates go through the SAME
        strict validation as direct construction (a degenerate range whose
        points collide under float rounding is rejected there).
        """
        nx = _require_count("nx", nx, minimum=2)
        ny = _require_count("ny", ny, minimum=2)

        def _axis(name: str, rng: object, n: int) -> tuple[float, ...]:
            if isinstance(rng, (str, bytes)) or not hasattr(rng, "__iter__"):
                raise GroundGridError(f"{name} must be a (start, stop) pair")
            pair = list(rng)
            if len(pair) != 2:
                raise GroundGridError(f"{name} must be a (start, stop) pair, got {pair!r}")
            out = []
            for v in pair:
                if isinstance(v, bool) or not isinstance(v, numbers.Real):
                    raise GroundGridError(f"{name} contains a non-real bound: {v!r}")
                fv = float(v)
                if not math.isfinite(fv):
                    raise GroundGridError(f"{name} contains a non-finite bound: {v!r}")
                out.append(fv)
            start, stop = out
            if not start < stop:
                raise GroundGridError(
                    f"{name} requires start < stop, got ({start!r}, {stop!r})"
                )
            step = (stop - start) / (n - 1)
            coords = [start + i * step for i in range(n)]
            coords[-1] = stop  # exact endpoint (linspace convention)
            return tuple(coords)

        return cls(
            x_coords=_axis("x_range", x_range, nx),
            y_coords=_axis("y_range", y_range, ny),
            validity_mode=validity_mode,
            parallel_tol=parallel_tol,
        )


@dataclass(frozen=True)
class GroundQueryResult:
    """Products of ONE ground query at ONE scan-time root pose.

    All tensors are fresh CPU storages (float32 / bool), never aliases of any
    cached or caller tensor. ``positions`` are the scan-time body coordinates
    ``[x_B, y_B, h_B]`` for the GroundLabel; ``world_points`` are the OFFLINE
    world-intersection metadata; invalid points are EXACTLY 0 everywhere.
    """

    heights: torch.Tensor      # (M,) float32, signed meters
    positions: torch.Tensor    # (M, 3) float32 = [x_B, y_B, h_B]
    world_points: torch.Tensor # (M, 3) float32 = p_root + R_WB @ positions
    valid: torch.Tensor        # (M,) bool, geometry-only flat-training semantics


def _rotation_matrix_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """(3, 3) float64 rotation matrix from a finite canonical wxyz quaternion."""
    w, x, y, z = (float(v) for v in q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.tensor(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=torch.float64,
    )


def _verify_plane_body_static(mj_model, body_id: int) -> list[str]:
    """The plane body AND every ancestor up to the world body must be fully
    static: no joints, no mocap. Returns the checked body-name chain (plane
    body first, world last); raises on the first violation."""
    import mujoco

    from .scanner import PlaneValidationError

    checked: list[str] = []
    bid = int(body_id)
    hops = 0
    while True:
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"<id {bid}>"
        jntnum = int(mj_model.body_jntnum[bid])
        if jntnum != 0:
            raise PlaneValidationError(
                f"plane body ancestor {name!r} has {jntnum} joint(s); the real "
                "flat ground provider requires a fully static plane body chain"
            )
        mocapid = int(mj_model.body_mocapid[bid])
        if mocapid != -1:
            raise PlaneValidationError(
                f"plane body ancestor {name!r} is a mocap body (mocap id "
                f"{mocapid}); the real flat ground provider requires a fully "
                "static plane body chain"
            )
        checked.append(name)
        if bid == 0:
            break
        bid = int(mj_model.body_parentid[bid])
        hops += 1
        if hops > int(mj_model.nbody):
            raise PlaneValidationError(
                "body_parentid chain exceeded nbody (cyclic body tree?)"
            )
    return checked


class FlatGroundProvider:
    """Real flat-plane ground provider for an explicit training grid.

    Construct from the ACTUAL compiled host ``mj_model`` (the runtime passes
    ``env.sim.mj_model``). Validation: ``scanner.validate_single_horizontal_plane``
    (reused verbatim — exactly one horizontal ``mjGEOM_PLANE`` with the
    confirmed friction) plus this module's own static-plane-body-chain check.
    The real geometry identity is retained in ``plane_record``; every query
    intersects the analytic line ``p(t) = p_root_W + R_WB @ [x_B, y_B, t]``
    with the ACTUAL plane world position/normal in float64. No z=0
    assumption, no heightmap, no lidar fitting, no synthetic labels.
    """

    quat_norm_atol_default = 1e-6

    def __init__(
        self,
        mj_model,
        grid_cfg: GroundGridCfg,
        *,
        quat_norm_atol: float = quat_norm_atol_default,
    ):
        if not isinstance(grid_cfg, GroundGridCfg):
            raise TypeError(
                f"grid_cfg must be a GroundGridCfg, got {type(grid_cfg).__name__}"
            )
        atol = float(quat_norm_atol)
        if not math.isfinite(atol) or atol <= 0.0:
            raise ValueError(f"quat_norm_atol must be positive and finite, got {atol!r}")
        # Lazy import keeps this module free of warp/mjlab at import time.
        from .scanner import validate_single_horizontal_plane

        self.grid_cfg = grid_cfg
        self.quat_norm_atol = atol
        self.plane_record = validate_single_horizontal_plane(mj_model)
        self.static_chain = _verify_plane_body_static(
            mj_model, int(self.plane_record["body_id"])
        )
        self.ground_identity_record = {
            **self.plane_record,
            "static_body_chain": list(self.static_chain),
            "quat_norm_atol": atol,
            "grid_identity": grid_cfg.identity(),
            "grid_identity_hash": grid_cfg.identity_hash(),
        }
        self._plane_pos = torch.tensor(
            self.plane_record["world_pos"], dtype=torch.float64
        )
        normal = torch.tensor(
            self.plane_record["world_normal"], dtype=torch.float64
        )
        self._plane_normal = normal / normal.norm()

    # ── the query ───────────────────────────────────────────────────────────

    def query(
        self, root_pos_w: torch.Tensor, root_quat_wxyz: torch.Tensor
    ) -> GroundQueryResult:
        """Signed body-z ground heights for the FULL grid at ONE root pose.

        Inputs: world root position (3,) and root quaternion (4,) in mjlab
        wxyz convention (any tensor-like; moved to CPU float64 internally).
        Invalid poses are REJECTED (raise ``ValueError``): non-finite
        position; non-finite quaternion; quaternion whose norm deviates from
        1 beyond ``quat_norm_atol`` (no silent renormalization).

        The whole grid shares the body-z query direction
        ``d = R_WB @ [0,0,1]``; with ``s = n · d`` the plane equation
        ``n · (o + h·d − p0) = 0`` gives ``h = n·(p0 − o) / s`` per grid
        origin ``o = p_root + R_WB @ [x_B, y_B, 0]``. ``h`` is SIGNED and
        never clipped. ``|s| < parallel_tol`` (near-parallel query) isolates
        the whole grid as invalid with values EXACTLY 0 — no NaN leaks.
        """
        pos = torch.as_tensor(root_pos_w).detach().reshape(-1).to(
            device="cpu", dtype=torch.float64
        )
        quat = torch.as_tensor(root_quat_wxyz).detach().reshape(-1).to(
            device="cpu", dtype=torch.float64
        )
        if pos.numel() != 3:
            raise ValueError(f"root position must have 3 elements, got {pos.numel()}")
        if quat.numel() != 4:
            raise ValueError(f"root quaternion must have 4 elements, got {quat.numel()}")
        if not bool(torch.isfinite(pos).all()):
            raise ValueError(f"root position contains NaN/Inf: {pos.tolist()}")
        if not bool(torch.isfinite(quat).all()):
            raise ValueError(f"root quaternion contains NaN/Inf: {quat.tolist()}")
        norm = float(quat.norm())
        if abs(norm - 1.0) > self.quat_norm_atol:
            raise ValueError(
                f"root quaternion norm {norm!r} deviates from 1 beyond "
                f"quat_norm_atol={self.quat_norm_atol!r}; refusing (no silent "
                "renormalization of a non-canonical pose)"
            )

        rotation = _rotation_matrix_from_quat_wxyz(quat)
        z_body_w = rotation[:, 2]
        xy = self.grid_cfg.grid_xy(dtype=torch.float64)
        m = int(xy.shape[0])
        origins = pos.unsqueeze(0) + xy @ rotation[:, :2].T   # (M, 3)
        denom = float(self._plane_normal @ z_body_w)          # signed scalar
        parallel = abs(denom) < float(self.grid_cfg.parallel_tol)
        if parallel:
            heights = torch.zeros(m, dtype=torch.float64)
            valid = torch.zeros(m, dtype=torch.bool)
        else:
            offsets = (self._plane_pos.unsqueeze(0) - origins) @ self._plane_normal
            h = offsets / denom
            valid = torch.isfinite(h)
            heights = torch.where(valid, h, torch.zeros_like(h))
        world = origins + heights.unsqueeze(1) * z_body_w.unsqueeze(0)
        positions = torch.cat([xy, heights.unsqueeze(1)], dim=1)
        if bool((~valid).any()):
            world = torch.where(valid.unsqueeze(1), world, torch.zeros_like(world))
            positions = torch.where(
                valid.unsqueeze(1), positions, torch.zeros_like(positions)
            )
        heights32 = heights.to(torch.float32)
        positions32 = positions.to(torch.float32)
        world32 = world.to(torch.float32)
        # A finite float64 solve may overflow in the actual float32 contract.
        valid = (valid & torch.isfinite(heights32)
                 & torch.isfinite(positions32).all(dim=1)
                 & torch.isfinite(world32).all(dim=1))
        return GroundQueryResult(
            heights=torch.where(valid, heights32, torch.zeros_like(heights32)),
            positions=torch.where(valid[:, None], positions32, torch.zeros_like(positions32)),
            world_points=torch.where(valid[:, None], world32, torch.zeros_like(world32)),
            valid=valid,
        )
