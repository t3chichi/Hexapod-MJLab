"""Pure left/right mirror mappings for confined-locomotion training.

This module contains tensor/data mappings only. It does not augment batches,
change probabilities, calculate PPO losses, touch a simulator, or alter model
or T1 reference values. The chosen map is the same-sign joint permutation
RF↔LF, RM↔LM, RB↔LB, with HAA/HFE/KFE paired by name.

Point blocks use source unit rays in the actual sensor optical frame. A ray is
rotated by the compiled sensor-to-body mount, reflected by Q=diag(1,-1,1) in
BODY, then rotated back by the same optical mount before calculating the
prescribed key ``phi * 2*pi + azimuth``. This works for zero-range and
max-range rays without guessing a direction from endpoint coordinates. BODY
points themselves are reflected in their acquisition-time BODY coordinates;
there is no motion compensation or new scan.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Sequence

import torch

from .contracts import (
    COMMAND_DIM,
    ENVELOPE_DIM,
    FLOAT_DTYPE,
    INT_DTYPE,
    JOINT_DIM,
    PROPRIO_DIM,
    OnlineObservation,
)

__all__ = [
    "MIRROR_MAPPING_SCHEMA_VERSION",
    "GridPermutation",
    "JointPermutation",
    "MirrorMappingError",
    "MirroredActionDistribution",
    "MirroredPointBlock",
    "mirror_action_distribution",
    "mirror_axial_vector",
    "mirror_command",
    "mirror_envelope_values",
    "mirror_ground_positions",
    "mirror_ground_values",
    "mirror_mean_mse",
    "mirror_observation",
    "mirror_point_block",
    "mirror_polar_vector",
    "mirror_proprio63",
]


MIRROR_MAPPING_SCHEMA_VERSION = "confined_locomotion.mirror_mapping/v1"

_LEG_PAIRS = (("RF", "LF"), ("RM", "LM"), ("RB", "LB"))
_JOINT_KINDS = ("HAA", "HFE", "KFE")
_ANGLE_SCALE = 2.0 * math.pi
_UNIT_TOL = 1e-5


class MirrorMappingError(ValueError):
    """Invalid mirror mapping input or incompatible metadata."""


def _float_tensor(name: str, value: object, *, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype is not FLOAT_DTYPE:
        dtype = getattr(value, "dtype", None)
        raise MirrorMappingError(f"{name} must be a float32 tensor, got {dtype}")
    if shape is not None and tuple(value.shape) != shape:
        raise MirrorMappingError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise MirrorMappingError(f"{name} contains NaN/Inf")
    return value


def _same_shape(name: str, value: object, *, shape: tuple[int, ...], dtype: torch.dtype,
                device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype is not dtype:
        raise MirrorMappingError(f"{name} must be a {dtype} tensor")
    if tuple(value.shape) != shape or value.device != device:
        raise MirrorMappingError(
            f"{name} must have shape {shape} on {device}, got "
            f"{tuple(value.shape)} on {value.device}"
        )
    return value


def _require_matrix(name: str, value: object, width: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype is not FLOAT_DTYPE \
            or value.dim() != 2 or int(value.shape[0]) < 1 or int(value.shape[1]) != width:
        raise MirrorMappingError(
            f"{name} must be (B,{width}) float32, got "
            f"{tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__}"
        )
    if not bool(torch.isfinite(value).all()):
        raise MirrorMappingError(f"{name} contains NaN/Inf")
    return value


def mirror_polar_vector(values: torch.Tensor) -> torch.Tensor:
    """Reflect BODY polar vectors with Q=diag(1,-1,1)."""
    if not isinstance(values, torch.Tensor) or values.dtype is not FLOAT_DTYPE \
            or values.dim() < 1 or int(values.shape[-1]) != 3:
        raise MirrorMappingError("polar vectors must be float32 with final dimension 3")
    if not bool(torch.isfinite(values).all()):
        raise MirrorMappingError("polar vectors contain NaN/Inf")
    return torch.stack((values[..., 0], -values[..., 1], values[..., 2]), dim=-1)


def mirror_axial_vector(values: torch.Tensor) -> torch.Tensor:
    """Reflect BODY axial vectors with det(Q)Q=diag(-1,1,-1)."""
    if not isinstance(values, torch.Tensor) or values.dtype is not FLOAT_DTYPE \
            or values.dim() < 1 or int(values.shape[-1]) != 3:
        raise MirrorMappingError("axial vectors must be float32 with final dimension 3")
    if not bool(torch.isfinite(values).all()):
        raise MirrorMappingError("axial vectors contain NaN/Inf")
    return torch.stack((-values[..., 0], values[..., 1], -values[..., 2]), dim=-1)


def mirror_command(command: torch.Tensor) -> torch.Tensor:
    """Map ``[vx, vy, omega_z]`` to ``[vx, -vy, -omega_z]``."""
    command = _require_matrix("command", command, COMMAND_DIM)
    return torch.stack((command[:, 0], -command[:, 1], -command[:, 2]), dim=1)


@dataclass(frozen=True)
class JointPermutation:
    """Name-derived same-sign 18-joint involution.

    ``indices[j]`` is the source column used for output joint column ``j``.
    Names may arrive in any model order; names are never inferred from numeric
    slices or from a legacy task layout.
    """

    joint_names: tuple[str, ...]
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        indices = tuple(self.indices)
        if len(names) != JOINT_DIM or any(not isinstance(name, str) for name in names) \
                or len(set(names)) != JOINT_DIM:
            raise MirrorMappingError(f"joint_names must contain {JOINT_DIM} unique names")
        expected = {f"{leg}_{kind}" for pair in _LEG_PAIRS for leg in pair for kind in _JOINT_KINDS}
        if set(names) != expected:
            missing = sorted(expected - set(names))
            extra = sorted(set(names) - expected)
            raise MirrorMappingError(f"joint names do not match the 18 paired joints; missing={missing}, extra={extra}")
        if len(indices) != JOINT_DIM or any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        ) or sorted(indices) != list(range(JOINT_DIM)):
            raise MirrorMappingError("indices must be a permutation of 0..17")
        if any(indices[indices[i]] != i for i in range(JOINT_DIM)):
            raise MirrorMappingError("joint permutation must be an involution")
        name_to_index = {name: index for index, name in enumerate(names)}
        expected_indices = []
        for name in names:
            leg, kind = name.split("_", maxsplit=1)
            pair = next(pair for pair in _LEG_PAIRS if leg in pair)
            partner_leg = pair[1] if leg == pair[0] else pair[0]
            expected_indices.append(name_to_index[f"{partner_leg}_{kind}"])
        if indices != tuple(expected_indices):
            raise MirrorMappingError("indices do not implement the named left/right joint pairs")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "indices", indices)

    @classmethod
    def from_joint_names(cls, joint_names: Sequence[str]) -> "JointPermutation":
        names = tuple(joint_names)
        if len(names) != JOINT_DIM or any(not isinstance(name, str) for name in names):
            raise MirrorMappingError(f"joint_names must be a sequence of {JOINT_DIM} strings")
        positions = {name: index for index, name in enumerate(names)}
        if len(positions) != JOINT_DIM:
            raise MirrorMappingError("joint_names contains duplicates")
        indices: list[int] = []
        for name in names:
            leg, kind = name.split("_", maxsplit=1) if "_" in name else ("", "")
            pair = next((pair for pair in _LEG_PAIRS if leg in pair), None)
            if pair is None or kind not in _JOINT_KINDS:
                raise MirrorMappingError(f"unrecognized paired joint name {name!r}")
            partner_leg = pair[1] if leg == pair[0] else pair[0]
            partner = f"{partner_leg}_{kind}"
            if partner not in positions:
                raise MirrorMappingError(f"missing mirror joint {partner!r} for {name!r}")
            indices.append(positions[partner])
        return cls(joint_names=names, indices=tuple(indices))

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``A values`` on the same device without mutating the input."""
        if not isinstance(values, torch.Tensor) or values.dim() < 1 \
                or int(values.shape[-1]) != JOINT_DIM:
            raise MirrorMappingError(f"joint tensor must have final dimension {JOINT_DIM}")
        index = torch.tensor(self.indices, dtype=torch.long, device=values.device)
        return values.index_select(-1, index)


def mirror_proprio63(proprio: torch.Tensor, permutation: JointPermutation) -> torch.Tensor:
    """Map the 63-D unscaled proprio vector; the 3-D command stays separate."""
    proprio = _require_matrix("proprio", proprio, PROPRIO_DIM)
    if not isinstance(permutation, JointPermutation):
        raise MirrorMappingError("permutation must be a JointPermutation")
    return torch.cat((
        mirror_polar_vector(proprio[:, 0:3]),
        mirror_axial_vector(proprio[:, 3:6]),
        mirror_polar_vector(proprio[:, 6:9]),
        permutation.apply(proprio[:, 9:27]),
        permutation.apply(proprio[:, 27:45]),
        permutation.apply(proprio[:, 45:63]),
    ), dim=1)


def mirror_observation(
    observation: OnlineObservation, permutation: JointPermutation
) -> OnlineObservation:
    """Return an independent mirrored OnlineObservation with the same identity."""
    if not isinstance(observation, OnlineObservation):
        raise MirrorMappingError("observation must be OnlineObservation")
    if not isinstance(permutation, JointPermutation):
        raise MirrorMappingError("permutation must be a JointPermutation")
    observation.validate()
    return OnlineObservation(
        env_ids=observation.env_ids.detach().clone(),
        episode_ids=observation.episode_ids.detach().clone(),
        step_ids=observation.step_ids.detach().clone(),
        time_s=observation.time_s.detach().clone(),
        lin_vel_b=mirror_polar_vector(observation.lin_vel_b),
        ang_vel_b=mirror_axial_vector(observation.ang_vel_b),
        gravity_b=mirror_polar_vector(observation.gravity_b),
        joint_pos_rel_ref=permutation.apply(observation.joint_pos_rel_ref),
        joint_vel=permutation.apply(observation.joint_vel),
        prev_action=permutation.apply(observation.prev_action),
        command=mirror_command(observation.command),
    )


@dataclass(frozen=True)
class MirroredActionDistribution:
    """Purely permuted old distribution fields and raw action."""

    mean: torch.Tensor
    std: torch.Tensor
    raw_action: torch.Tensor


def mirror_action_distribution(
    mean: torch.Tensor,
    std: torch.Tensor,
    raw_action: torch.Tensor,
    permutation: JointPermutation,
) -> MirroredActionDistribution:
    """Apply the same A to mean, independent std, and un-clipped raw action.

    No clipping, q_ref offset, probability recomputation, or parameter tying is
    performed. The diagonal-Normal log probability is invariant to this pure
    permutation when all three old fields are pushed forward together.
    """
    mean = _require_matrix("mean", mean, JOINT_DIM)
    std = _require_matrix("std", std, JOINT_DIM)
    raw_action = _require_matrix("raw_action", raw_action, JOINT_DIM)
    if mean.shape != std.shape or mean.shape != raw_action.shape \
            or mean.device != std.device or mean.device != raw_action.device:
        raise MirrorMappingError("mean, std, and raw_action must share shape and device")
    if bool((std <= 0).any()):
        raise MirrorMappingError("std must be positive")
    if not isinstance(permutation, JointPermutation):
        raise MirrorMappingError("permutation must be a JointPermutation")
    return MirroredActionDistribution(
        mean=permutation.apply(mean),
        std=permutation.apply(std),
        raw_action=permutation.apply(raw_action),
    )


def _require_point_block(
    points: object, valid: object, source_dirs_sensor: object
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(points, torch.Tensor) or points.dtype is not FLOAT_DTYPE \
            or points.dim() != 3 or int(points.shape[0]) < 1 or int(points.shape[1]) < 1 \
            or int(points.shape[2]) != 3:
        raise MirrorMappingError("points must be (B,N,3) float32")
    batch, count, _ = points.shape
    device = points.device
    valid_t = _same_shape("valid", valid, shape=(batch, count), dtype=torch.bool, device=device)
    dirs = _same_shape(
        "source_dirs_sensor", source_dirs_sensor,
        shape=(batch, count, 3), dtype=FLOAT_DTYPE, device=device,
    )
    if bool(valid_t.any()) and not bool(torch.isfinite(points[valid_t]).all()):
        raise MirrorMappingError("valid points must be finite")
    if bool((~valid_t).any()) and not torch.equal(points[~valid_t], torch.zeros_like(points[~valid_t])):
        raise MirrorMappingError("invalid point slots must be exactly zero")
    if bool(valid_t.any()):
        if not bool(torch.isfinite(dirs[valid_t]).all()):
            raise MirrorMappingError("valid source directions must be finite")
        norms = torch.linalg.vector_norm(dirs[valid_t], dim=-1)
        if not bool(torch.isclose(norms, torch.ones_like(norms), atol=_UNIT_TOL, rtol=0).all()):
            raise MirrorMappingError("valid source directions must be unit sensor-frame vectors")
    if bool((~valid_t).any()) and not torch.equal(dirs[~valid_t], torch.zeros_like(dirs[~valid_t])):
        raise MirrorMappingError("invalid source-direction slots must be exactly zero")
    return points, valid_t, dirs


def _metadata_tensor(
    name: str, value: torch.Tensor | None, *, shape: tuple[int, int],
    dtype: torch.dtype, device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    result = _same_shape(name, value, shape=shape, dtype=dtype, device=device)
    if dtype is FLOAT_DTYPE and not bool(torch.isfinite(result).all()):
        raise MirrorMappingError(f"{name} contains NaN/Inf")
    return result


def _quat_apply_xyzw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit quaternions in xyzw order; no normalization."""
    q_xyz = quat[..., :3]
    q_w = quat[..., 3:4]
    uv = torch.cross(q_xyz, vec, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return vec + 2.0 * (q_w * uv + uuv)


def _direction_keys(directions: torch.Tensor) -> torch.Tensor:
    x, y, z = directions.unbind(dim=-1)
    azimuth = torch.atan2(y, x)
    elevation = torch.atan2(z, torch.hypot(x, y))
    return elevation * _ANGLE_SCALE + azimuth


@dataclass(frozen=True)
class MirroredPointBlock:
    """Reflected and stably sorted point block with aligned source metadata.

    ``permutation[b,j]`` is the input column gathered into output column j.
    ``source_order`` preserves the initial F1-prepared stable order across maps;
    it only restores that original input order before the stable angle sort and
    is never used as an alternate geometric sort key.
    """

    points: torch.Tensor
    valid: torch.Tensor
    source_dirs_sensor: torch.Tensor
    angle_keys: torch.Tensor
    scan_ids: torch.Tensor | None
    times: torch.Tensor | None
    source_indices: torch.Tensor | None
    source_order: torch.Tensor
    permutation: torch.Tensor


def mirror_point_block(
    points: torch.Tensor,
    valid: torch.Tensor,
    source_dirs_sensor: torch.Tensor,
    mount_quat_xyzw: torch.Tensor,
    *,
    scan_ids: torch.Tensor | None = None,
    times: torch.Tensor | None = None,
    source_indices: torch.Tensor | None = None,
    source_order: torch.Tensor | None = None,
) -> MirroredPointBlock:
    """BODY-reflect, optical-angle sort, and carry all per-point provenance.

    ``source_dirs_sensor`` are the original unit rays aligned with ``points``;
    each valid point must have one even at zero range. ``mount_quat_xyzw`` is
    the actual sensor-to-BODY orientation from the compiled model. Point xyz
    uses Q directly in acquisition-time BODY coordinates. Optical directions
    are transformed by R_mount^-1 Q R_mount, not by negating a scalar key.

    Points are stably sorted by the prescribed ``phi*2*pi + azimuth`` key;
    invalid zero-filled slots go to the tail. ``source_order`` should begin as
    arange(N) after original F1 preparation and be carried through each mirror
    operation. It reconstructs the original input order before a subsequent
    stable sort, so even exact-key ties/poles restore under a double map.
    """
    points, valid, dirs = _require_point_block(points, valid, source_dirs_sensor)
    batch, count, _ = points.shape
    device = points.device
    quat = _same_shape(
        "mount_quat_xyzw", mount_quat_xyzw,
        shape=(4,), dtype=FLOAT_DTYPE, device=device,
    )
    if not bool(torch.isfinite(quat).all()):
        raise MirrorMappingError("mount_quat_xyzw contains NaN/Inf")
    q_norm = torch.linalg.vector_norm(quat)
    if not bool(torch.isclose(q_norm, torch.ones_like(q_norm), atol=_UNIT_TOL, rtol=0)):
        raise MirrorMappingError("mount_quat_xyzw must be unit length; it is not normalized here")

    metadata_shape = (batch, count)
    scan_ids = _metadata_tensor(
        "scan_ids", scan_ids, shape=metadata_shape, dtype=INT_DTYPE, device=device
    )
    times = _metadata_tensor("times", times, shape=metadata_shape, dtype=FLOAT_DTYPE, device=device)
    source_indices = _metadata_tensor(
        "source_indices", source_indices, shape=metadata_shape, dtype=INT_DTYPE, device=device
    )
    if scan_ids is not None:
        if bool((scan_ids[valid] < 0).any()):
            raise MirrorMappingError("valid points must carry non-negative source scan IDs")
        if bool((scan_ids[~valid] != -1).any()):
            raise MirrorMappingError("invalid point slots must carry source scan ID -1")
    if times is not None and bool((times[~valid] != 0).any()):
        raise MirrorMappingError("invalid point slots must carry source time 0")
    if source_indices is not None:
        if bool((source_indices[valid] < 0).any()):
            raise MirrorMappingError("valid points must carry non-negative source indices")
        if bool((source_indices[~valid] != -1).any()):
            raise MirrorMappingError("invalid point slots must carry source index -1")
    if source_order is None:
        source_order = torch.arange(count, dtype=INT_DTYPE, device=device).expand(batch, count)
    else:
        source_order = _same_shape(
            "source_order", source_order, shape=metadata_shape, dtype=INT_DTYPE, device=device
        )
        for row in source_order:
            if torch.unique(row).numel() != count:
                raise MirrorMappingError("source_order must be unique within each point row")

    # R^-1 Q R is a Householder reflection in sensor coordinates:
    # H = I - 2 n nᵀ, n = Rᵀ e_y. Construct n in float64, normalize that
    # numeric vector, then cast H to float32 for direction application. The
    # direct FP32 quaternion round trip loses optical azimuth precision for
    # near-pole rays by subtracting O(1) z terms to recover O(1e-8) x/y terms.
    quat64 = quat.to(torch.float64)
    conjugate64 = torch.cat((-quat64[:3], quat64[3:4]))
    ey64 = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float64, device=device)
    normal_sensor64 = _quat_apply_xyzw(conjugate64, ey64)
    normal_sensor64 = normal_sensor64 / torch.linalg.vector_norm(normal_sensor64)
    identity64 = torch.eye(3, dtype=torch.float64, device=device)
    reflection_sensor = (identity64 - 2.0 * torch.outer(normal_sensor64, normal_sensor64)).to(FLOAT_DTYPE)
    reflected_dirs = torch.matmul(dirs, reflection_sensor)
    reflected_dirs = torch.where(valid.unsqueeze(-1), reflected_dirs, torch.zeros_like(reflected_dirs))
    keys = _direction_keys(reflected_dirs)
    keys = torch.where(valid, keys, torch.zeros_like(keys))

    reflected_points = mirror_polar_vector(points)
    reflected_points = torch.where(valid.unsqueeze(-1), reflected_points, torch.zeros_like(reflected_points))

    # Recover the original stable input order (F1's stable angle-sort input),
    # then apply only F1's stable angle-key sort with invalids at the tail.
    base_order = torch.argsort(source_order, dim=1, stable=True)
    base_valid = torch.gather(valid, 1, base_order)
    base_keys = torch.gather(keys, 1, base_order)
    effective_keys = torch.where(base_valid, base_keys, torch.full_like(base_keys, float("inf")))
    local_order = torch.argsort(effective_keys, dim=1, stable=True)
    order = torch.gather(base_order, 1, local_order)

    def gather_points(value: torch.Tensor) -> torch.Tensor:
        return torch.gather(value, 1, order.unsqueeze(-1).expand_as(value))

    def gather_rows(value: torch.Tensor | None) -> torch.Tensor | None:
        return torch.gather(value, 1, order) if value is not None else None

    return MirroredPointBlock(
        points=gather_points(reflected_points),
        valid=gather_rows(valid),  # type: ignore[arg-type]
        source_dirs_sensor=gather_points(reflected_dirs),
        angle_keys=gather_rows(keys),  # type: ignore[arg-type]
        scan_ids=gather_rows(scan_ids),
        times=gather_rows(times),
        source_indices=gather_rows(source_indices),
        source_order=gather_rows(source_order),  # type: ignore[arg-type]
        permutation=order,
    )


def mirror_envelope_values(values: torch.Tensor) -> torch.Tensor:
    """Copy the six symmetric envelope labels unchanged under y reflection."""
    values = _require_matrix("envelope values", values, ENVELOPE_DIM)
    return values.clone()


@dataclass(frozen=True)
class GridPermutation:
    """Metadata-derived point permutation for a y-reflected rectangular grid.

    Grid order is ``y`` outer, ``x`` inner, matching GroundGridCfg. Each output
    destination reads the input point at the same x and reflected y coordinate.
    """

    x_coords: tuple[float, ...]
    y_coords: tuple[float, ...]
    source_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        xs = _float32_axis("x_coords", self.x_coords)
        ys = _float32_axis("y_coords", self.y_coords)
        indices = tuple(self.source_indices)
        nx, ny = len(xs), len(ys)
        if nx < 1 or ny < 1 or len(self.source_indices) != nx * ny:
            raise MirrorMappingError("grid permutation dimensions are inconsistent")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices) \
                or sorted(indices) != list(range(nx * ny)):
            raise MirrorMappingError("grid source_indices must be a bijection")
        if any(indices[indices[i]] != i for i in range(nx * ny)):
            raise MirrorMappingError("grid permutation must be an involution")
        expected_indices = _derive_grid_indices(xs, ys)
        if indices != expected_indices:
            raise MirrorMappingError("grid source_indices do not match reflected coordinate metadata")
        object.__setattr__(self, "x_coords", xs)
        object.__setattr__(self, "y_coords", ys)
        object.__setattr__(self, "source_indices", indices)

    @classmethod
    def from_axes(
        cls, x_coords: Sequence[float], y_coords: Sequence[float]
    ) -> "GridPermutation":
        xs = _float32_axis("x_coords", x_coords)
        ys = _float32_axis("y_coords", y_coords)
        return cls(xs, ys, _derive_grid_indices(xs, ys))

    @property
    def num_points(self) -> int:
        return len(self.source_indices)

    def apply(self, values: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        """Gather grid values along an explicit point dimension (copy, no alias)."""
        if not isinstance(values, torch.Tensor) or values.dim() < 1:
            raise MirrorMappingError("grid values must be a tensor with a point dimension")
        dim = dim if dim >= 0 else values.dim() + dim
        if dim < 0 or dim >= values.dim() or int(values.shape[dim]) != self.num_points:
            raise MirrorMappingError(
                f"grid point dimension must have length {self.num_points}, got {tuple(values.shape)}"
            )
        index = torch.tensor(self.source_indices, dtype=torch.long, device=values.device)
        return values.index_select(dim, index)


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _derive_grid_indices(xs: tuple[float, ...], ys: tuple[float, ...]) -> tuple[int, ...]:
    x_index = {value: index for index, value in enumerate(xs)}
    y_index = {value: index for index, value in enumerate(ys)}
    if len(x_index) != len(xs) or len(y_index) != len(ys):
        raise MirrorMappingError("grid axes must remain unique in float32")
    indices: list[int] = []
    nx = len(xs)
    for y in ys:
        reflected_y = _positive_zero(-y)
        if reflected_y not in y_index:
            raise MirrorMappingError(
                f"grid y axis is not paired under reflection: {y!r} has no {reflected_y!r}"
            )
        iy_source = y_index[reflected_y]
        for x in xs:
            indices.append(iy_source * nx + x_index[x])
    return tuple(indices)


def _float32_axis(name: str, coords: Sequence[float]) -> tuple[float, ...]:
    if isinstance(coords, (str, bytes)):
        raise MirrorMappingError(f"{name} must be a numeric coordinate sequence")
    try:
        raw = tuple(coords)
    except TypeError as exc:
        raise MirrorMappingError(f"{name} must be a numeric coordinate sequence") from exc
    if not raw:
        raise MirrorMappingError(f"{name} must be non-empty")
    values: list[float] = []
    for coord in raw:
        if isinstance(coord, bool) or not isinstance(coord, numbers.Real):
            raise MirrorMappingError(f"{name} contains a non-real coordinate {coord!r}")
        value = float(coord)
        if not math.isfinite(value):
            raise MirrorMappingError(f"{name} contains a non-finite coordinate")
        values.append(value)
    represented = torch.tensor(values, dtype=FLOAT_DTYPE)
    if not bool(torch.isfinite(represented).all()):
        raise MirrorMappingError(f"{name} is not finite in float32")
    if represented.numel() > 1 and not bool((represented[1:] > represented[:-1]).all()):
        raise MirrorMappingError(f"{name} must be strictly increasing in float32")
    return tuple(_positive_zero(float(value)) for value in represented.tolist())


def mirror_ground_values(
    values: torch.Tensor, valid: torch.Tensor, grid: GridPermutation
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map heights/masks by the metadata-derived grid pair; heights are unchanged."""
    if not isinstance(grid, GridPermutation):
        raise MirrorMappingError("grid must be a GridPermutation")
    if not isinstance(values, torch.Tensor) or values.dtype is not FLOAT_DTYPE or values.dim() < 1 \
            or int(values.shape[-1]) != grid.num_points:
        raise MirrorMappingError(f"values must be float32 and end in the grid size {grid.num_points}")
    if not isinstance(valid, torch.Tensor) or valid.dtype is not torch.bool \
            or tuple(valid.shape) != tuple(values.shape) or valid.device != values.device:
        raise MirrorMappingError("valid must be a bool tensor matching values")
    if bool(valid.any()) and not bool(torch.isfinite(values[valid]).all()):
        raise MirrorMappingError("valid ground values contain NaN/Inf")
    if bool((~valid).any()) and not torch.equal(values[~valid], torch.zeros_like(values[~valid])):
        raise MirrorMappingError("invalid ground values must be exactly zero")
    return grid.apply(values, dim=-1), grid.apply(valid, dim=-1)


def mirror_ground_positions(
    positions: torch.Tensor, valid: torch.Tensor, grid: GridPermutation
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid-pair [x_B,y_B,h_B] positions and reflect their BODY y coordinate."""
    if not isinstance(grid, GridPermutation):
        raise MirrorMappingError("grid must be a GridPermutation")
    if not isinstance(positions, torch.Tensor) or positions.dtype is not FLOAT_DTYPE \
            or positions.dim() < 2 or tuple(positions.shape[-2:]) != (grid.num_points, 3):
        raise MirrorMappingError(
            f"positions must end in ({grid.num_points},3) float32, got "
            f"{tuple(positions.shape) if isinstance(positions, torch.Tensor) else type(positions).__name__}"
        )
    if not isinstance(valid, torch.Tensor) or valid.dtype is not torch.bool \
            or tuple(valid.shape) != tuple(positions.shape[:-1]) or valid.device != positions.device:
        raise MirrorMappingError("valid must match positions without their final xyz dimension")
    if bool(valid.any()) and not bool(torch.isfinite(positions[valid]).all()):
        raise MirrorMappingError("valid ground positions contain NaN/Inf")
    if bool((~valid).any()) and not torch.equal(
        positions[~valid], torch.zeros_like(positions[~valid])
    ):
        raise MirrorMappingError("invalid ground positions must be exactly zero")
    nx, ny = len(grid.x_coords), len(grid.y_coords)
    expected_x = torch.tensor(grid.x_coords, dtype=FLOAT_DTYPE, device=positions.device).repeat(ny)
    expected_y = torch.tensor(grid.y_coords, dtype=FLOAT_DTYPE, device=positions.device).repeat_interleave(nx)
    expected_xy = torch.stack((expected_x, expected_y), dim=-1)
    if bool(valid.any()) and bool(
        ((positions[..., :2][valid] - expected_xy.expand_as(positions[..., :2])[valid]).abs() > 1e-5).any()
    ):
        raise MirrorMappingError("valid ground positions do not match grid coordinate metadata")
    paired = grid.apply(positions, dim=-2)
    paired_valid = grid.apply(valid, dim=-1)
    reflected = mirror_polar_vector(paired)
    reflected = torch.where(paired_valid.unsqueeze(-1), reflected, torch.zeros_like(reflected))
    return reflected, paired_valid


def mirror_mean_mse(
    mirrored_mean: torch.Tensor,
    original_mean: torch.Tensor,
    permutation: JointPermutation,
) -> torch.Tensor:
    """Stop-gradient-target action mean MSE, averaged over batch×18.

    The returned scalar has no coefficient; the configured 0.6 multiplier is
    applied by the loss aggregation layer.
    """
    mirrored_mean = _require_matrix("mirrored_mean", mirrored_mean, JOINT_DIM)
    original_mean = _require_matrix("original_mean", original_mean, JOINT_DIM)
    if mirrored_mean.shape != original_mean.shape or mirrored_mean.device != original_mean.device:
        raise MirrorMappingError("mirrored_mean and original_mean must share shape and device")
    if not isinstance(permutation, JointPermutation):
        raise MirrorMappingError("permutation must be a JointPermutation")
    target = permutation.apply(original_mean).detach()
    return (mirrored_mean - target).square().mean()
