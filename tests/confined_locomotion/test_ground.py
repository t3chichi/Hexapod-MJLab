"""CPU contracts for the explicit training grid and flat-plane queries."""
from __future__ import annotations

import math
import traceback

import torch

from support.fixtures import make_plane_model, m12_training_grid


def _rotation_wxyz(quaternion):
    w, x, y, z = [float(value) for value in quaternion]
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=torch.float64)


def _roll_pitch_quaternion(roll_degrees: float, pitch_degrees: float):
    roll = math.radians(roll_degrees) / 2
    pitch = math.radians(pitch_degrees) / 2
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return (cr * cp, sr * cp, cr * sp, -sr * sp)


def test_training_grid_has_fixed_order_and_content_identity():
    from hexapod_mjlab.tasks.confined_locomotion.ground import GroundGridCfg

    grid = m12_training_grid()
    assert (grid.nx, grid.ny, grid.num_points) == (4, 3, 12)
    assert grid.validity_mode == "geometry_only_flat_training"
    assert torch.allclose(
        torch.tensor(grid.x_coords, dtype=torch.float64),
        torch.linspace(0.1, 0.8, 4, dtype=torch.float64),
        atol=1e-15,
        rtol=0.0,
    )
    assert grid.y_coords == (-0.3, 0.0, 0.3)
    expected = [[x, y] for y in grid.y_coords for x in grid.x_coords]
    assert grid.grid_xy(dtype=torch.float64).tolist() == expected
    ident = grid.identity()
    assert ident["num_points"] == 12
    assert ident["validity_mode"] == "geometry_only_flat_training"
    assert ident["x_coords"] == list(grid.x_coords)
    assert ident["y_coords"] == list(grid.y_coords)
    direct = GroundGridCfg(
        x_coords=grid.x_coords,
        y_coords=grid.y_coords,
        validity_mode=grid.validity_mode,
        parallel_tol=grid.parallel_tol,
        body_xyz_semantics=grid.body_xyz_semantics,
    )
    assert direct.identity_hash() == grid.identity_hash()

    invalid = (
        {"x_coords": (0.2, 0.1)},
        {"x_coords": (0.1, 0.1)},
        {"x_coords": (0.1, float("nan"))},
        {"x_coords": (True, 0.8)},
        {"y_coords": ()},
    )
    for changes in invalid:
        fields = {
            "x_coords": grid.x_coords,
            "y_coords": grid.y_coords,
            "validity_mode": grid.validity_mode,
        }
        fields.update(changes)
        try:
            GroundGridCfg(**fields)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid ground grid accepted: {changes!r}")
    try:
        GroundGridCfg.from_ranges(
            (0.1, 0.8), (-0.3, 0.3), 4.0, 3,
            validity_mode="geometry_only_flat_training",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("float sample count was accepted")


def test_s1_187_explicit_grid_and_signed_body_z_ground_labels():
    from hexapod_mjlab.tasks.confined_locomotion.ground import FlatGroundProvider
    from hexapod_mjlab.tasks.confined_locomotion.ground_presets import (
        S1_187_X_RANGE_M, S1_187_Y_RANGE_M,
        S1_187_X_COORDS, S1_187_Y_COORDS, S1_187_NX, S1_187_NY,
        ground_grid_from_rectangle,
        resolve_explicit_ground_grid, resolve_task_ground_grid, s1_187_ground_grid,
    )
    from hexapod_mjlab.rl.confined_locomotion.symmetry import GridPermutation

    grid = s1_187_ground_grid()
    assert S1_187_X_RANGE_M == (0.8, 3.2)
    assert S1_187_Y_RANGE_M == (-0.5, 0.5)
    assert (S1_187_NX, S1_187_NY) == (17, 11)
    assert (grid.nx, grid.ny, grid.num_points) == (17, 11, 187)
    assert grid.identity_hash() == (
        "e837e7edbb0b60478bab21dc51967bc09d6144e1e729657a405b5802acddc8ec"
    )  # The existing accepted checkpoint must keep its precise coordinate identity.
    assert grid == ground_grid_from_rectangle(
        x_range_m=S1_187_X_RANGE_M, y_range_m=S1_187_Y_RANGE_M
    )
    # Four bounds, same fixed network head M=187. Check exact endpoints,
    # ordering and mirror behavior when BOTH forward/lateral spans change.
    resized = ground_grid_from_rectangle(
        x_range_m=(0.8, 3.3), y_range_m=(-0.6, 0.6),
    )
    assert (resized.nx, resized.ny, resized.num_points) == (17, 11, 187)
    assert resized.x_coords[0] == 0.8 and resized.x_coords[-1] == 3.3
    assert resized.y_coords[0] == -0.6 and resized.y_coords[-1] == 0.6
    assert resized.y_coords[5] == 0.0
    assert resized.identity_hash() != grid.identity_hash()
    assert resized.grid_xy()[0].tolist() == [0.8, -0.6]
    assert resized.grid_xy()[-1].tolist() == [3.3, 0.6]
    resized_mirror = GridPermutation.from_axes(resized.x_coords, resized.y_coords)
    resized_mirror_idx = torch.tensor(resized_mirror.source_indices)
    assert torch.equal(
        resized_mirror_idx[resized_mirror_idx], torch.arange(187, dtype=torch.int64)
    )
    for bounds in (
        dict(x_range_m=(0.8, 3.3), y_range_m=(-0.4, 0.6)),
        dict(x_range_m=(0.8, 0.8), y_range_m=(-0.6, 0.6)),
        dict(x_range_m=(float("nan"), 3.3), y_range_m=(-0.6, 0.6)),
    ):
        try:
            ground_grid_from_rectangle(**bounds)
        except ValueError:
            pass
        else:
            raise AssertionError(f"malformed or unpaired rectangle accepted: {bounds}")
    assert grid.x_coords == S1_187_X_COORDS
    assert grid.y_coords == S1_187_Y_COORDS
    assert grid.x_coords[0] == 0.8 and grid.x_coords[-1] == 3.2
    assert grid.y_coords[0] == -0.5 and grid.y_coords[-1] == 0.5
    assert all(abs(grid.x_coords[i+1]-grid.x_coords[i]-0.15)<1e-12 for i in range(16))
    assert all(abs(grid.y_coords[i+1]-grid.y_coords[i]-0.10)<1e-12 for i in range(10))
    assert grid.y_coords[5] == 0.0
    assert grid.grid_xy()[0].tolist() == [0.8, -0.5]
    assert grid.grid_xy()[-1].tolist() == [3.2, 0.5]
    perm=GridPermutation.from_axes(grid.x_coords, grid.y_coords)
    order=torch.tensor(perm.source_indices)
    assert torch.equal(order[order], torch.arange(187, dtype=torch.int64))
    assert resolve_explicit_ground_grid(
        grid_preset="s1_187", grid_x=None, grid_y=None
    ).identity_hash() == grid.identity_hash()
    assert resolve_explicit_ground_grid(
        grid_preset=None, grid_x=grid.x_coords, grid_y=grid.y_coords
    ).identity_hash() == grid.identity_hash()
    # Registered task and both CLI entrypoints now select this current
    # baseline when no override is supplied. Historical explicit-grid
    # helper keeps the strict "no implicit grid" contract for old callers.
    assert resolve_task_ground_grid(
        grid_preset=None, grid_x=None, grid_y=None,
    ).identity_hash() == grid.identity_hash()
    assert resolve_task_ground_grid(
        grid_preset="s1_187", grid_x=None, grid_y=None,
    ).identity_hash() == grid.identity_hash()
    custom = resolve_task_ground_grid(
        grid_preset=None, grid_x=(0.1, 0.5, 0.9),
        grid_y=(-0.2, 0.0, 0.2),
    )
    assert custom.num_points == 9 and custom.identity_hash() != grid.identity_hash()
    for params in (
        dict(grid_preset=None, grid_x=(0.8, 3.2), grid_y=None),
        dict(grid_preset="s1_187", grid_x=(0.8, 3.2), grid_y=None),
    ):
        try:
            resolve_task_ground_grid(**params)
        except ValueError:
            pass
        else:
            raise AssertionError(f"task grid accepted partial/mixed override: {params}")
    for parameters in (
        dict(grid_preset=None,grid_x=None,grid_y=None),
        dict(grid_preset=None,grid_x=[0.8,3.2],grid_y=None),
        dict(grid_preset="s1_187",grid_x=[0.8,3.2],grid_y=None),
        dict(grid_preset="unknown",grid_x=None,grid_y=None),
    ):
        try:
            resolve_explicit_ground_grid(**parameters)
        except ValueError:
            pass
        else:
            raise AssertionError(f"implicit or mixed grid preset accepted: {parameters}")

    provider=FlatGroundProvider(make_plane_model(pos=(0.,0.,0.)),grid)
    root=torch.tensor([0.,0.,0.5])
    identity=torch.tensor([1.,0.,0.,0.])
    flat=provider.query(root,identity)
    assert bool(flat.valid.all()) and flat.heights.shape==(187,)
    assert torch.allclose(flat.heights,torch.full((187,),-0.5),atol=1e-6)
    raised=provider.query(root+torch.tensor([0.,0.,0.1]),identity)
    assert torch.allclose(raised.heights,flat.heights-0.1,atol=1e-6)

    # Nonzero roll+pitch changes signed BODY-z height across grid even for a
    # completely flat WORLD plane. Ground labels are not world-Z elevations.
    tilted_q=torch.tensor(_roll_pitch_quaternion(8.,5.),dtype=torch.float32)
    tilted=provider.query(root,tilted_q)
    R=_rotation_wxyz(tilted_q.tolist())
    world=root.double()[None,:]+tilted.positions.double()@R.T
    assert tilted.positions.shape==(187,3) and bool(tilted.valid.all())
    assert torch.allclose(world[:,2],torch.zeros(187,dtype=torch.float64),atol=3e-6)
    assert not torch.allclose(flat.heights,tilted.heights,atol=1e-3)
    assert torch.allclose(tilted.positions[:,2],tilted.heights)


def test_flat_plane_query_uses_full_body_pose_and_signed_heights():
    from hexapod_mjlab.tasks.confined_locomotion.ground import FlatGroundProvider

    grid = m12_training_grid()
    plane_position = (0.3, -0.2, 0.5)
    provider = FlatGroundProvider(make_plane_model(pos=plane_position), grid)
    assert provider.plane_record["friction"] == [1.0, 0.005, 0.0001]
    assert provider.static_chain and provider.static_chain[-1] == "world"

    root = torch.tensor([0.4, 0.2, 1.0], dtype=torch.float64)
    quaternion = _roll_pitch_quaternion(25.0, 20.0)
    rotation = _rotation_wxyz(quaternion)
    xy = grid.grid_xy(dtype=torch.float64)
    denominator = rotation[2, 2]
    expected_heights = (
        plane_position[2] - root[2] - xy[:, 0] * rotation[2, 0] - xy[:, 1] * rotation[2, 1]
    ) / denominator
    local_points = torch.cat((xy, expected_heights[:, None]), dim=1)
    expected_world = root[None] + local_points @ rotation.T

    result = provider.query(root.float(), torch.tensor(quaternion, dtype=torch.float32))
    assert result.valid.shape == (12,) and bool(result.valid.all())
    assert result.heights.shape == (12,) and result.positions.shape == (12, 3)
    assert result.world_points.shape == (12, 3)
    assert torch.equal(result.positions[:, :2], grid.grid_xy(dtype=torch.float32))
    assert torch.equal(result.positions[:, 2], result.heights)
    assert torch.allclose(result.heights.double(), expected_heights, atol=2e-6, rtol=1e-5)
    assert torch.allclose(result.world_points.double(), expected_world, atol=3e-6, rtol=1e-5)
    assert torch.allclose(result.world_points[:, 2], torch.full((12,), 0.5), atol=3e-6)
    assert not result.heights.requires_grad and not result.positions.requires_grad

    above = provider.query(
        torch.tensor([0.4, 0.2, 0.3]), torch.tensor([1.0, 0.0, 0.0, 0.0])
    )
    assert bool(above.valid.all())
    assert torch.allclose(above.heights, torch.full((12,), 0.2), atol=1e-6)
    assert float(above.heights.min()) > 0.0


def test_near_parallel_ground_queries_are_isolated_as_exact_zeros():
    from hexapod_mjlab.tasks.confined_locomotion.ground import FlatGroundProvider

    grid = m12_training_grid()
    provider = FlatGroundProvider(make_plane_model(pos=(0.3, -0.2, 0.5)), grid)
    roll = math.radians(90.0) / 2
    quaternion = torch.tensor([math.cos(roll), math.sin(roll), 0.0, 0.0])
    result = provider.query(torch.tensor([0.4, 0.2, 1.0]), quaternion)
    assert not bool(result.valid.any())
    assert torch.equal(result.heights, torch.zeros(12))
    assert torch.equal(result.positions, torch.zeros(12, 3))
    assert torch.equal(result.world_points, torch.zeros(12, 3))
    assert bool(torch.isfinite(result.heights).all())

    for position, quat in (
        ([0.4, 0.2, 1.0], [1.0, 0.0, 0.0, float("nan")]),
        ([0.4, 0.2, 1.0], [1.0, 0.0, 0.0]),
        ([float("nan"), 0.2, 1.0], [1.0, 0.0, 0.0, 0.0]),
        ([0.4, 0.2, 1.0], [1.0, 0.0, 0.0, 1.0]),
    ):
        try:
            provider.query(torch.tensor(position), torch.tensor(quat))
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid root pose accepted: {position!r} {quat!r}")


def test_compiled_scene_validation_rejects_non_training_geometry():
    from hexapod_mjlab.tasks.confined_locomotion.ground import FlatGroundProvider
    from hexapod_mjlab.tasks.confined_locomotion.scanner import PlaneValidationError

    grid = m12_training_grid()
    for label, model in (
        ("multiple planes", make_plane_model(second_plane=True)),
        ("tilted plane", make_plane_model(tilt_deg=5.0)),
    ):
        try:
            FlatGroundProvider(model, grid)
        except PlaneValidationError:
            pass
        else:
            raise AssertionError(f"invalid scene accepted: {label}")


def run_cpu_checks() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} CPU checks passed")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(run_cpu_checks())

