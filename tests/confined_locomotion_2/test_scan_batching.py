"""Batched terrain scan checks preserve row order and perturbation boundaries."""
from types import SimpleNamespace
import torch
from hexapod_mjlab.tasks.confined_locomotion_2.scanner import ConfinedLidarCfg, ConfinedScanPipeline, ScanFaultError


def pipeline(distances, directions=None):
    count, rays = distances.shape
    directions = (torch.tensor([[1., 0., 0.]]).repeat(rays, 1) if directions is None else directions)
    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.zeros(count, 3),
        root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1)))
    pipe = object.__new__(ConfinedScanPipeline)
    pipe.env = SimpleNamespace(scene={"robot": robot})
    pipe.cfg = ConfinedLidarCfg(far_sampling="hit_only_10m")
    pipe.device, pipe.num_envs = "cpu", count
    pipe.terrain_query = SimpleNamespace(raycast=lambda rows, *_: (
        distances.index_select(0, rows), torch.zeros(len(rows), rays, dtype=torch.int32)))
    pipe.full_ray_dirs, pipe.source_ray_ids = directions, torch.arange(rays)
    pipe.num_raw_rays = pipe.num_full_rays = rays
    pipe._mount_q, pipe._mount_t = torch.tensor([[0., 0., 0., 1.]]), torch.zeros(1, 3)
    pipe.pool_indices = torch.arange(rays)
    pipe.fixed_near_mask = torch.arange(rays) < rays // 2
    pipe.scan_records = []
    return pipe


class CountingNoise:
    lidar_active = range_enabled = True
    cfg = SimpleNamespace(gap=None, range=SimpleNamespace(clip_min_m=.01, clip_max_m=15.))
    def __init__(self):
        self.events = []
    def begin_scan(self, row, number):
        self.events.append(("begin", row, number))
        return False
    def last_event(self, row):
        return None
    def draw_range_epsilon(self, row, count):
        self.events.append(("range", row, count))
        return torch.zeros(count)


def test_batch_checks_keep_fault_row_order_and_noise_consumption_boundary():
    distances = torch.full((3, 8), 5.)
    distances[2] = 15.  # The first due row is all misses, so endpoint checks skip it.
    directions = torch.tensor([[1., 0., 0.]]).repeat(8, 1)
    directions[0, 0] = 2.  # A real hit in the next row fails endpoint consistency.
    pipe, noise = pipeline(distances, directions), CountingNoise()
    try:
        pipe.scan_rows([2, 0, 1], perturbations=noise, persistent_scan_numbers={2: 7, 0: 8, 1: 9})
    except ScanFaultError as error:
        assert "env 0: hit endpoint norm" in str(error)
    else:
        raise AssertionError("inconsistent hit endpoint accepted")
    assert noise.events == [("begin", 2, 7), ("range", 2, 8), ("begin", 0, 8)]
    assert [record["env_row"] for record in pipe.scan_records] == [2, 0]
    assert pipe.scan_records[-1]["fault"] == "hit_endpoint_dist_mismatch"


def test_batch_checks_still_reject_nonfinite_endpoints_even_for_misses():
    for value in (float("nan"), float("inf")):
        directions = torch.tensor([[1., 0., 0.]]).repeat(8, 1)
        directions[0, 0] = value
        pipe = pipeline(torch.full((2, 8), 15.), directions)
        try:
            pipe.scan_rows([1, 0])
        except ScanFaultError as error:
            assert "env 1: raw raycast output contains NaN/Inf" in str(error)
        else:
            raise AssertionError("nonfinite miss endpoint accepted")
        assert pipe.scan_records[-1]["fault"] == "nonfinite_raw_output"


def test_batch_checks_keep_native_miss_conversion_and_owned_rows():
    # Preserve existing geom/range-to-miss conversion before endpoint validation.
    distances = torch.tensor([[float("nan"), float("inf"), -1., 16., 15., 0., 10., 10.001]])
    pipe, noise = pipeline(distances), CountingNoise()
    frame = pipe.scan_rows([0], perturbations=noise, persistent_scan_numbers={0: 1})[0]
    torch.testing.assert_close(frame.raw_dists, torch.tensor([15., 15., 15., 15., 15., 0., 10., 10.001]))
    assert frame.no_hit_mask.tolist() == [True, True, True, True, True, False, False, False]
    assert noise.events == [("begin", 0, 1), ("range", 0, 8)]
    for tensor in (frame.raw_dists, frame.raw_points_sensor):
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    retained = frame.raw_points_sensor.clone()
    pipe.scan_rows([0])
    assert torch.equal(frame.raw_points_sensor, retained)


def flat_pipeline():
    pipe = pipeline(torch.zeros(3,8))
    pipe.terrain_query = None
    pipe.mesh_bounds = (-30.,30.,-30.,30.)
    pipe.env.scene["robot"].data.root_link_pos_w[:,0] = torch.tensor([40.,1.,2.])
    calls = []
    sensor = SimpleNamespace(position=torch.zeros(3,3),quaternion=torch.zeros(3,4))
    def update(count):
        calls.append(count)
        distances = (sensor.position[:count,0]+5)[:,None].expand(-1,8).clone()
        points = distances[:,:,None]*pipe.full_ray_dirs[None]
        return points[:,None,:,None],distances[:,None,:,None]
    sensor.update = update
    pipe._sensor = sensor
    return pipe,calls


def test_flat_batches_only_due_rows_and_retained_frames_own_workspace_copies():
    pipe,calls = flat_pipeline()
    # An uncovered row that is not due must never be queried or checked.
    frames = pipe.scan_rows([2,1])
    assert calls == [2] and [f.env_row for f in frames] == [2,1]
    retained = frames[0].raw_points_sensor.clone()
    pipe.scan_rows([1])
    assert calls == [2,1] and torch.equal(frames[0].raw_points_sensor,retained)
    assert pipe.scan_rows([]) == [] and calls == [2,1]
    for f in frames:
        assert f.raw_points_sensor.untyped_storage().nbytes() == f.raw_points_sensor.numel()*4


def test_flat_coverage_failure_never_queries_bad_or_later_rows():
    from hexapod_mjlab.tasks.confined_locomotion_2.scanner import ScanCoverageError
    pipe,calls = flat_pipeline()
    noise = CountingNoise()
    try:
        pipe.scan_rows([2,0,1],perturbations=noise,persistent_scan_numbers={2:7,0:8,1:9})
    except ScanCoverageError:
        pass
    else:
        raise AssertionError("uncovered pose accepted")
    assert calls == [1]  # Only the valid prefix was physically raycast.
    assert noise.events == [("begin",2,7),("range",2,8),("begin",0,8)]
    assert [r["env_row"] for r in pipe.scan_records] == [2]
    pipe,calls = flat_pipeline()
    try:
        pipe.scan_rows([0,2])
    except ScanCoverageError:
        pass
    else:
        raise AssertionError("uncovered first row accepted")
    assert calls == [] and pipe.scan_records == []


def test_runtime_batches_scan_metadata_and_leaves_non_due_rows_untouched():
    from hexapod_mjlab.tasks.confined_locomotion_2.runtime import TrainingRuntime
    from hexapod_mjlab.rl.confined_locomotion_2.history import ScanWindow
    pipe = pipeline(torch.full((3, 8), 5.))
    runtime = object.__new__(TrainingRuntime)
    runtime.device, runtime.pipeline = "cpu", pipe
    runtime._perturbations = CountingNoise()
    runtime.episode_ids = torch.tensor([10, 11, 12])
    runtime.scan_counters = torch.zeros(3, dtype=torch.long)
    runtime.sensor_scan_counts = torch.tensor([6, 7, 8])
    runtime._control_steps, runtime.scan_log = 5, []
    runtime._play_source_scans = runtime.scan_audit_bound = None
    runtime.window = ScanWindow(3, episode_ids=runtime.episode_ids)
    rows = runtime._scan_rows_now([2, 0])
    assert [w["env_row"] for w in rows] == [2, 0]
    assert [w["persistent_scan_count"] for w in rows] == [9, 7]
    assert runtime.scan_counters.tolist() == [1, 0, 1]
    assert runtime.sensor_scan_counts.tolist() == [7, 7, 9]
    assert runtime.window.read().near_scan_ids.tolist() == [1, -1, 1]
    assert [w["episode_id"] for w in rows] == [12, 10]
    for record, frame_record in zip(rows, pipe.scan_records):
        frame = frame_record["frame"]
        assert record["near_valid_count"] == int(frame.near_valid.sum())
        assert record["far_valid_count"] == int(frame.far_valid.sum())
    assert runtime._scan_rows_now([]) == []
    runtime._control_steps += 5
    runtime._scan_rows_now([0])
    assert runtime.scan_counters.tolist() == [2, 0, 1]
    assert runtime.sensor_scan_counts.tolist() == [8, 7, 9]
    assert runtime.window.read().near_scan_ids.tolist() == [2, -1, 1]
