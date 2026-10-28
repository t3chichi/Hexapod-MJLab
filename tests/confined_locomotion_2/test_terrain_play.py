"""CPU checks for checkpoint-bound terrain play and display geometry."""

import numpy as np
import torch
import mujoco

from hexapod_mjlab.tasks.confined_locomotion_2.play_terrain_memory import checkpoint_grid
from hexapod_mjlab.tasks.confined_locomotion_2.play import PlayConfigError
from hexapod_mjlab.tasks.confined_locomotion_2.rl.checkpoint import (
    SCHEMA,
    TASK_ID,
    TERRAIN_TASK_ID,
    architecture_identity,
)
from hexapod_mjlab.tasks.confined_locomotion_2.components.terrain_query import (
    CompiledHeightfield,
    HeightGridCfg,
)
from hexapod_mjlab.tasks.confined_locomotion_2.visualization.grid import configured_grid
from hexapod_mjlab.tasks.confined_locomotion_2.visualization.terrain_debug import (
    height_points_world,
    heightfield_mesh,
)


def _save_checkpoint(path, *, grid, task=TERRAIN_TASK_ID, grid_identity=None):
    terrain = task == TERRAIN_TASK_ID
    saved_grid = grid.identity() if grid_identity is None else grid_identity
    torch.save({
        "schema": SCHEMA,
        "task": task,
        "kind": "full",
        "resumable": True,
        "identity": {
            "architecture": architecture_identity("hit_only_10m", terrain_memory=terrain),
            "recipe": {"height_grid": saved_grid},
        },
    }, path)


def test_checkpoint_grid_uses_saved_ranges_and_rejects_flat_or_wrong_semantics(tmp_path):
    saved = HeightGridCfg((-1.7, 1.7), (-0.9, 0.9))
    path = tmp_path / "terrain.pt"
    _save_checkpoint(path, grid=saved)

    loaded = checkpoint_grid(path)
    assert loaded.identity() == saved.identity()
    assert loaded.identity() != configured_grid().identity()

    _save_checkpoint(path, grid=saved, task=TASK_ID)
    try:
        checkpoint_grid(path)
    except ValueError:
        pass
    else:
        raise AssertionError("Flat checkpoint was accepted for Terrain Memory play")

    wrong = saved.identity()
    wrong["order"] = "ix*11+iy"
    _save_checkpoint(path, grid=saved, grid_identity=wrong)
    try:
        checkpoint_grid(path)
    except PlayConfigError as error:
        assert "semantics" in str(error)
    else:
        raise AssertionError("checkpoint with incorrect grid semantics was accepted")


def test_height_points_use_retained_root_z_and_valid_mask_only():
    xy = np.zeros((11, 17, 2), dtype=np.float64)
    xy[..., 0], xy[..., 1] = np.meshgrid(np.arange(17), np.arange(11))
    valid = np.ones((11, 17), dtype=bool)
    valid[0, 0] = False
    prediction = np.full((11, 17), 0.25)
    target = np.full((11, 17), -0.1)
    prediction[0, 0] = np.inf
    target[0, 0] = np.nan
    data = {
        "query_xy_w": xy,
        "prediction": prediction,
        "target": target,
        "valid": valid,
        "root_pos_w": np.asarray((4.0, -3.0, 1.6)),
        # A later display pose is deliberately unrelated to retained acquisition.
        "current_pose": {"root_pos_w": np.asarray((-9.0, 2.0, 8.0))},
    }

    predicted_points = height_points_world(data, "prediction")
    target_points = height_points_world(data, "target")
    assert predicted_points.shape == target_points.shape == (186, 3)
    np.testing.assert_allclose(predicted_points[:, 2], 1.85)
    np.testing.assert_allclose(target_points[:, 2], 1.5)
    np.testing.assert_array_equal(predicted_points[:, :2], xy[valid])
    np.testing.assert_array_equal(target_points[:, :2], xy[valid])


def _compile_test_heightfield():
    heights = np.asarray(
        ((0.00, 0.03, 0.10, 0.16),
         (0.08, 0.15, 0.19, 0.31),
         (0.21, 0.27, 0.39, 0.48)),
        dtype=np.float64,
    )
    low, high = float(heights.min()), float(heights.max())
    spec = mujoco.MjSpec()
    hfield = spec.add_hfield(
        name="mesh_test_hfield", size=[2.0, 1.5, high - low, 0.1],
        nrow=heights.shape[0], ncol=heights.shape[1],
        userdata=((heights - low) / (high - low)).ravel().tolist(),
    )
    spec.worldbody.add_geom(
        name="mesh_test_surface", type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=hfield.name, pos=[2.0, 1.5, low], group=3,
        contype=1, conaffinity=1,
    )
    return spec.compile()


def test_heightfield_mesh_triangle_centroids_match_label_interpolation():
    model = _compile_test_heightfield()
    surface = CompiledHeightfield(model)
    vertices, faces = heightfield_mesh(model)
    centroids = vertices[faces].mean(axis=1)
    reference, valid = surface.reference_heights(centroids[:, :2])
    assert valid.all()
    np.testing.assert_allclose(reference, centroids[:, 2], atol=2e-7, rtol=0)


def test_missing_post_step_ground_is_display_invalidity_not_an_inference_fault():
    from types import SimpleNamespace
    from hexapod_mjlab.tasks.confined_locomotion_2 import display_terrain_memory as module
    session = SimpleNamespace(
        num_envs=1,
        env=SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[20., 0., 0.5]])))}),
        runtime=SimpleNamespace(terrain_query=SimpleNamespace(surface=SimpleNamespace(
            reference_heights=lambda xy: (np.zeros(1), np.zeros(1, dtype=bool))))),
    )
    original = module.display.retain_display
    def retain(*args, **kwargs):
        session._display_rows = [{"pose": {"ground_z_m": 0., "root_height_m": .5}}]
    module.display.retain_display = retain
    try:
        module.retain_display(session, None, SimpleNamespace(height_prediction=torch.zeros(1, 11, 17)),
                              [{"terrain": {}}], preview=False, reason="test")
    finally:
        module.display.retain_display = original
    assert session._display_rows[0]["pose"] == {
        "ground_z_m": None, "root_height_m": None, "ground_valid": False}
