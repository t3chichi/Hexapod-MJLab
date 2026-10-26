"""CPU checks for the unregistered Terrain Memory geometry boundary."""

import importlib
from contextlib import contextmanager

import mujoco
import numpy as np
import torch

from mjlab.tasks.registry import list_tasks

from hexapod_mjlab.tasks.confined_locomotion_2.config.terrain_geometry import (
    HeightfieldTerrainCfg,
    TERRAIN_GROUP,
    make_geometry_env_cfg,
)
from hexapod_mjlab.tasks.confined_locomotion_2.terrain_geometry import (
    CompiledHeightfield,
    HeightGridCfg,
    grid_world_xy,
)


@contextmanager
def _raises(exception_type, *, match=None):
    try:
        yield
    except exception_type as error:
        if match is not None:
            assert match in str(error), (match, str(error))
    else:
        raise AssertionError(f"expected {exception_type.__name__}")


def _elevations(rows=4, cols=5):
    # Asymmetric values expose both the hfield row/column orientation and its
    # two different triangle-interpolation branches.
    return tuple(
        tuple(0.07 * row + 0.031 * col + 0.009 * row * col for col in range(cols))
        for row in range(rows)
    )


def _terrain_cfg(*, elevations=None, size=(4.8, 3.0), spawn_origin=(0.0, 0.0, 0.0)):
    return HeightfieldTerrainCfg(
        size=size,
        elevations_m=_elevations() if elevations is None else elevations,
        spawn_origin=spawn_origin,
    )


def _compile_generated(cfg):
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg.function(0.0, spec, np.random.default_rng(17))
    return spec.compile()


def _compile_test_hfield(*, group=TERRAIN_GROUP, contype=1, conaffinity=1,
                         dynamic=False, add_extra_group_geom=False):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="terrain")
    if dynamic:
        body.add_freejoint(name="terrain_free")
    heights = np.asarray(_elevations(3, 4), dtype=np.float64)
    floor = float(heights.min())
    span = float(heights.max() - floor)
    hfield = spec.add_hfield(
        name="test_hfield", size=[2.0, 1.5, span, 0.1],
        nrow=heights.shape[0], ncol=heights.shape[1],
        userdata=((heights - floor) / span).ravel().tolist(),
    )
    body.add_geom(
        name="test_surface", type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=hfield.name, pos=[2.0, 1.5, floor], group=group,
        contype=contype, conaffinity=conaffinity,
    )
    if add_extra_group_geom:
        spec.worldbody.add_geom(
            name="unexpected_terrain_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.1, 0.0, 0.0], pos=[0.0, 0.0, 2.0], group=TERRAIN_GROUP,
        )
    return spec.compile()


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray((
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ))


def _quat_axis(axis, angle):
    half = angle / 2
    return np.asarray((np.cos(half), *(np.asarray(axis) * np.sin(half))))


def test_compiled_non_square_heightfield_triangle_interpolation_matches_mj_ray():
    model = _compile_generated(_terrain_cfg())
    surface = CompiledHeightfield(model)
    assert surface.samples.shape == (4, 5)
    np.testing.assert_allclose(surface.size[:2], [2.4, 1.5], atol=0, rtol=0)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Query off-grid points in both diagonal halves of multiple cells.
    uv = np.asarray(((0.2, 0.7), (1.4, 0.2), (2.6, 1.1), (3.7, 2.6)))
    xy = np.column_stack((
        surface.position[0] - surface.size[0] + uv[:, 0] * 2 * surface.size[0] / 4,
        surface.position[1] - surface.size[1] + uv[:, 1] * 2 * surface.size[1] / 3,
    ))
    reference, valid = surface.reference_heights(xy)
    assert valid.all()
    ray_heights = []
    for point in xy:
        origin = np.asarray((point[0], point[1], surface.top_z + 2.0), dtype=np.float64)
        geom_id = np.asarray([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            model, data, origin, np.asarray((0.0, 0.0, -1.0)),
            None, True, -1, geom_id,
        )
        assert geom_id[0] == surface.geom_id
        ray_heights.append(origin[2] - distance)
    np.testing.assert_allclose(reference, ray_heights, atol=2e-6, rtol=0)


def test_root_grid_is_yaw_only_and_uses_y_rows_x_columns_with_reflection():
    grid = HeightGridCfg(x_range_m=(-1.2, 1.2), y_range_m=(-0.8, 0.8))
    root = np.asarray((0.4, -0.3, 0.9))
    yaw, roll, pitch = 0.73, -0.51, 0.34
    q_yaw = _quat_axis((0, 0, 1), yaw)
    q_tilted = _quat_mul(
        _quat_mul(q_yaw, _quat_axis((0, 1, 0), pitch)),
        _quat_axis((1, 0, 0), roll),
    )
    poses = torch.tensor(np.stack((q_yaw, q_tilted)), dtype=torch.float32)
    positions = torch.tensor(np.stack((root, root)), dtype=torch.float32)
    xy = grid_world_xy(positions, poses, grid)
    torch.testing.assert_close(xy[0], xy[1], atol=2e-6, rtol=0)

    points = grid.xy().reshape(11, 17, 2)
    assert points.shape == (11, 17, 2)
    torch.testing.assert_close(points[0, :, 0], torch.linspace(-1.2, 1.2, 17))
    torch.testing.assert_close(points[:, 0, 1], torch.linspace(-0.8, 0.8, 11))
    mirrored = points.flip(0)
    torch.testing.assert_close(mirrored[..., 0], points[..., 0])
    torch.testing.assert_close(mirrored[..., 1], -points[..., 1])
    values = torch.arange(187).reshape(11, 17)
    assert torch.equal(values.flip(0)[0], values[-1])


def test_geometry_cfg_has_no_flat_plane_training_terms_or_task_registration():
    registered_before = list_tasks()
    cfg = make_geometry_env_cfg(terrain=_terrain_cfg(), num_envs=2, seed=21)
    assert cfg.rewards == {}
    assert cfg.terminations == {}
    assert cfg.commands == {}
    assert cfg.curriculum == {}
    generator = cfg.scene.terrain.terrain_generator
    assert set(generator.sub_terrains) == {"terrain_memory_geometry"}
    model = _compile_generated(generator.sub_terrains["terrain_memory_geometry"])
    assert model.ngeom == 1
    assert model.geom_type.tolist() == [mujoco.mjtGeom.mjGEOM_HFIELD]

    # Geometry construction does not add or replace registered learning tasks.
    importlib.import_module("hexapod_mjlab.tasks.confined_locomotion_2")
    assert list_tasks() == registered_before
    assert not hasattr(cfg, "rl_cfg")


def test_heightfield_configuration_rejects_invalid_geometry():
    for kwargs in (
        {"elevations": ((0.0,),)},
        {"elevations": ((0.0, float("nan")), (0.1, 0.2))},
        {"size": (0.0, 2.0)},
        {"size": (float("inf"), 2.0)},
        {"spawn_origin": (2.4, 0.0, 0.0)},
        {"spawn_origin": (0.0, 0.0, float("nan"))},
    ):
        with _raises(ValueError):
            _terrain_cfg(**kwargs)


def test_environment_factory_rejects_invalid_count_and_seed():
    terrain = _terrain_cfg()
    for num_envs, seed in ((True, 1), (0, 1), (1, True), (1, -1), (1, 2**32)):
        with _raises(ValueError):
            make_geometry_env_cfg(terrain=terrain, num_envs=num_envs, seed=seed)


def test_compiled_surface_rejects_wrong_group_dynamic_or_noncolliding_geometry():
    for options in (
        {"group": 2},
        {"add_extra_group_geom": True},
        {"dynamic": True},
        {"contype": 0, "conaffinity": 0},
    ):
        with _raises(ValueError):
            CompiledHeightfield(_compile_test_hfield(**options))


def test_compiled_surface_requires_exactly_one_heightfield_in_terrain_group():
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(
        name="terrain_group_sphere", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[1.0, 0.0, 0.0], group=TERRAIN_GROUP,
    )
    with _raises(ValueError, match="exactly one Heightfield"):
        CompiledHeightfield(spec.compile())


def test_heightfield_compiled_identity_is_deterministic_and_content_sensitive():
    first = CompiledHeightfield(_compile_generated(_terrain_cfg()))
    same = CompiledHeightfield(_compile_generated(_terrain_cfg()))
    changed_values = [list(row) for row in _elevations()]
    changed_values[1][2] += 0.02
    changed = CompiledHeightfield(
        _compile_generated(_terrain_cfg(elevations=tuple(tuple(row) for row in changed_values)))
    )
    assert first.identity == same.identity
    assert first.identity["sha256"] == same.identity["sha256"]
    assert first.identity["samples_sha256"] != changed.identity["samples_sha256"]
    assert first.identity["sha256"] != changed.identity["sha256"]
