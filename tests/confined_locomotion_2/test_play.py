from types import SimpleNamespace

import torch
from torch import nn

from hexapod_mjlab.tasks.confined_locomotion_2.play import (
    PlayConfigError,
    PlaySession,
    make_play_env_cfg,
)
from hexapod_mjlab.tasks.confined_locomotion_2.rl.checkpoint import (
    SCHEMA,
    TASK_ID,
    architecture_identity,
)


class FakeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.25))
        self.encode_count = 0

    @staticmethod
    def initial_hidden(batch, device):
        return torch.zeros(batch, 187, device=device)

    def encode_points(self, near, near_valid, far, far_valid, **kwargs):
        self.encode_count += 1
        return torch.ones(near.shape[0], 187, device=near.device)

    def forward_from_features(self, observation, frame_feature, hidden):
        return SimpleNamespace(
            action_mean=self.bias.expand(observation.env_ids.shape[0], 18),
            hidden=hidden + 1,
        )


class FakeRuntime:
    mode = "play"
    device = torch.device("cpu")
    num_envs = 1
    _stopped = False
    scan_identity = {"catalog": "test-catalog", "mode": "full"}

    def __init__(self):
        self.episode_ids = torch.zeros(1, dtype=torch.long)
        self.control_steps_taken = 0
        self.scan_ids = torch.ones(1, dtype=torch.long)
        self.wrapper = None
        self.env = None
        self._commands = SimpleNamespace(
            RANGE_VX=(-1.0, 1.0), RANGE_VY=(-1.0, 1.0), RANGE_OMEGA=(-1.0, 1.0)
        )
        self.actions = []
        self.closed = False

    def prepare(self):
        obs = SimpleNamespace(
            env_ids=torch.zeros(1, dtype=torch.long),
            episode_ids=self.episode_ids.clone(),
        )
        return SimpleNamespace(
            observation=obs,
            near_scan_ids=self.scan_ids.clone(),
            near_points=torch.zeros(1, 256, 3),
            near_valid=torch.ones(1, 256, dtype=torch.bool),
            far_points=torch.zeros(1, 128, 3),
            far_valid=torch.ones(1, 128, dtype=torch.bool),
            near_angle_keys=None,
            far_angle_keys=None,
        )

    def step(self, action):
        self.actions.append(action.clone())
        self.control_steps_taken += 1
        if self.control_steps_taken % 5 == 0:
            self.scan_ids += 1
        return SimpleNamespace(step=self.control_steps_taken)

    def reset(self, rows):
        assert torch.equal(rows, torch.tensor([0]))
        self.episode_ids += 1
        self.control_steps_taken = 0
        self.scan_ids += 1
        self._stopped = False
        return {"episode_ids": self.episode_ids.clone()}

    def set_manual_command(self, rows, command):
        return {"rows": rows.clone(), "command": command.clone()}

    def close(self):
        self.closed = True


def _checkpoint(path, network):
    torch.save(
        {
            "schema": SCHEMA,
            "task": TASK_ID,
            "kind": "full",
            "resumable": True,
            "identity": {"architecture": architecture_identity(),
                         "scan_identity": FakeRuntime.scan_identity},
            "network": network.state_dict(),
        },
        path,
    )


def test_play_mean_steps_reuse_scan_features_and_reset_only_episode_state(tmp_path):
    runtime = FakeRuntime()
    network = FakeNetwork()
    checkpoint = tmp_path / "current.pt"
    _checkpoint(checkpoint, network)
    session = PlaySession(
        runtime=runtime,
        network=network,
        checkpoint=checkpoint,
        output_dir=tmp_path / "play",
        seed=9,
    )
    assert not hasattr(session, "optimizer")
    assert not hasattr(session.network, "ground_head")
    for _ in range(50):
        session.step_once()
    assert session.forward_count == 50
    assert network.encode_count == 10
    assert session.frame_encoding_count == 10
    assert all(torch.equal(action, torch.full((1, 18), 0.25)) for action in runtime.actions)
    retained_hidden = session.hidden.clone()
    assert session.peek() is not None
    assert torch.equal(session.hidden, retained_hidden)

    session.manual_reset()
    assert torch.count_nonzero(session.hidden) == 0
    assert session.peek() is None
    session.step_once()
    assert session.forward_count == 51
    assert network.encode_count == 11
    session.close()
    assert runtime.closed


def test_play_cfg_removes_timeout_but_keeps_formal_falls():
    cfg = make_play_env_cfg(num_envs=1, seed=3)
    assert set(cfg.terminations) == {
        "fall_low_height",
        "fall_tilt_sustained",
        "fall_flipped",
    }


def test_play_rejects_legacy_checkpoint(tmp_path):
    runtime = FakeRuntime()
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"schema": "confined_locomotion.full/v1"}, checkpoint)
    try:
        PlaySession(
            runtime=runtime,
            network=FakeNetwork(),
            checkpoint=checkpoint,
            output_dir=tmp_path / "play",
            seed=1,
        )
    except PlayConfigError:
        pass
    else:
        raise AssertionError("legacy checkpoint must be rejected")


def test_play_rejects_checkpoint_from_different_scan_identity(tmp_path):
    runtime = FakeRuntime()
    checkpoint = tmp_path / "other-scan.pt"
    network = FakeNetwork()
    _checkpoint(checkpoint, network)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["identity"]["scan_identity"] = {"catalog": "other-catalog", "mode": "full"}
    torch.save(payload, checkpoint)
    try:
        PlaySession(
            runtime=runtime,
            network=network,
            checkpoint=checkpoint,
            output_dir=tmp_path / "play",
            seed=1,
        )
    except PlayConfigError as error:
        assert "scan identity" in str(error)
    else:
        raise AssertionError("weights from another scan identity must be rejected")
    assert runtime.closed


def test_play_rejects_checkpoint_from_different_far_sampling_mode(tmp_path):
    runtime = FakeRuntime()
    runtime.lidar_cfg = SimpleNamespace(far_sampling="all_endpoints")
    environment_closed = []
    runtime.env = SimpleNamespace(close=lambda: environment_closed.append(True))
    network = FakeNetwork()
    checkpoint = tmp_path / "hit-only.pt"
    _checkpoint(checkpoint, network)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["identity"]["architecture"] = architecture_identity("hit_only_10m")
    torch.save(payload, checkpoint)
    try:
        PlaySession(
            runtime=runtime,
            network=network,
            checkpoint=checkpoint,
            output_dir=tmp_path / "play",
            seed=1,
            own_environment=True,
            allow_scan_override=True,
        )
    except PlayConfigError as error:
        assert "far sampling mode" in str(error)
    else:
        raise AssertionError("play must reject weights from another far sampling mode")
    assert runtime.closed
    assert environment_closed == [True]


def test_play_scan_override_allows_same_catalog_mode_change_and_reports_it(tmp_path):
    runtime = FakeRuntime()
    catalog = {"schema": "confined_locomotion_2.fixed_channels/v1", "ray_count": 86400}
    grouping = "fixed_t1_channels/v1"
    runtime.scan_identity = {
        "catalog": catalog, "grouping": grouping, "mode": "sparse", "selection": {"ids": [0]}
    }
    network = FakeNetwork()
    checkpoint = tmp_path / "full.pt"
    _checkpoint(checkpoint, network)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["identity"]["scan_identity"] = {
        "catalog": catalog, "grouping": grouping, "mode": "full", "selection": None
    }
    torch.save(payload, checkpoint)
    session = PlaySession(
        runtime=runtime,
        network=network,
        checkpoint=checkpoint,
        output_dir=tmp_path / "play",
        seed=1,
        allow_scan_override=True,
    )
    assert session.scan_evaluation_report == {
        "checkpoint_scan_identity": {
            "catalog": catalog, "grouping": grouping, "mode": "full", "selection": None
        },
        "runtime_scan_identity": {
            "catalog": catalog, "grouping": grouping, "mode": "sparse", "selection": {"ids": [0]}
        },
        "cross_configuration": True,
        "override_enabled": True,
    }
    session.close()


def test_play_scan_override_allows_selection_change_with_same_fixed_grouping(tmp_path):
    catalog = {"schema": "confined_locomotion_2.fixed_channels/v1", "ray_count": 86400,
               "mount": {"yaw": 0.0}}
    grouping = "fixed_t1_channels/v1"
    checkpoint_identity = {
        "catalog": catalog,
        "grouping": grouping,
        "mode": "sparse",
        "selection": {"ids": [0]},
    }
    runtime = FakeRuntime()
    runtime.scan_identity = {
        "catalog": catalog,
        "grouping": grouping,
        "mode": "sparse",
        "selection": {"ids": [1]},
    }
    network = FakeNetwork()
    checkpoint = tmp_path / "selected.pt"
    _checkpoint(checkpoint, network)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["identity"]["scan_identity"] = checkpoint_identity
    torch.save(payload, checkpoint)
    session = PlaySession(
        runtime=runtime,
        network=network,
        checkpoint=checkpoint,
        output_dir=tmp_path / "play",
        seed=1,
        allow_scan_override=True,
    )
    assert session.scan_evaluation_report["cross_configuration"] is True
    assert session.scan_evaluation_report["checkpoint_scan_identity"] == checkpoint_identity
    assert session.scan_evaluation_report["runtime_scan_identity"] == runtime.scan_identity
    session.close()


def test_play_scan_override_rejects_different_catalog_or_grouping(tmp_path):
    catalog = {"schema": "confined_locomotion_2.fixed_channels/v1", "ray_count": 86400}
    grouping = "fixed_t1_channels/v1"
    for runtime_identity in (
        {"catalog": {**catalog, "ray_count": 1}, "grouping": grouping, "mode": "sparse"},
        {"catalog": catalog, "grouping": "other-group", "mode": "sparse"},
        {"catalog": catalog, "mode": "sparse"},
    ):
        runtime = FakeRuntime()
        runtime.scan_identity = runtime_identity
        checkpoint = tmp_path / "full.pt"
        _checkpoint(checkpoint, FakeNetwork())
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload["identity"]["scan_identity"] = {
            "catalog": catalog, "grouping": grouping, "mode": "full", "selection": None
        }
        torch.save(payload, checkpoint)
        try:
            PlaySession(
                runtime=runtime,
                network=FakeNetwork(),
                checkpoint=checkpoint,
                output_dir=tmp_path / "play",
                seed=1,
                allow_scan_override=True,
            )
        except PlayConfigError as error:
            assert "scan identity" in str(error)
        else:
            raise AssertionError("override must reject another catalog or grouping")
        assert runtime.closed


def test_play_scan_override_must_be_bool(tmp_path):
    runtime = FakeRuntime()
    try:
        PlaySession(
            runtime=runtime,
            network=FakeNetwork(),
            checkpoint=tmp_path / "does-not-exist.pt",
            output_dir=tmp_path / "play",
            seed=1,
            allow_scan_override=1,
        )
    except PlayConfigError as error:
        assert "allow_scan_override must be bool" in str(error)
    else:
        raise AssertionError("non-bool scan override flag must be rejected")
    assert not runtime.closed


def test_scan_override_rejects_far_sampling_change():
    from hexapod_mjlab.tasks.confined_locomotion_2.sampling import sampling_identity

    catalog = {"schema": "confined_locomotion_2.fixed_channels/v1", "ray_count": 86400}
    checkpoint_identity = {
        "catalog": catalog, "grouping": "fixed_t1_channels/v1",
        "mode": "full", "selection": None,
        "sampling": sampling_identity("all_endpoints"),
    }
    runtime_identity = {
        "catalog": catalog, "grouping": "fixed_t1_channels/v1",
        "mode": "sparse", "selection": {"ids": [1]},
        "sampling": sampling_identity("hit_only_10m"),
    }
    assert not PlaySession._is_allowed_scan_override(checkpoint_identity, runtime_identity)


def test_play_rejects_invalid_lidar_mapping_before_checkpoint_or_runtime_use(tmp_path):
    runtime = FakeRuntime()
    try:
        PlaySession(
            runtime=runtime,
            checkpoint=tmp_path / "does-not-exist.pt",
            output_dir=tmp_path / "play",
            seed=1,
            lidar_cfg={"raycast_mode": "adaptive"},
        )
    except PlayConfigError as error:
        assert "invalid lidar_cfg" in str(error)
    else:
        raise AssertionError("invalid mapping must be rejected before checkpoint loading")
    assert not runtime.closed
