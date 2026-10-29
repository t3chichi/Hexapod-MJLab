from types import SimpleNamespace

import torch

from hexapod_mjlab.rl.confined_locomotion.contracts import OnlineObservation, EnvelopeLabel
from hexapod_mjlab.rl.confined_locomotion_2.distribution import DirectStdNormalProvider
from hexapod_mjlab.rl.confined_locomotion.symmetry import JointPermutation
from hexapod_mjlab.rl.confined_locomotion_2.integration import PreparedControlInputs
from hexapod_mjlab.tasks.confined_locomotion_2.collector import TrainingCollector
from hexapod_mjlab.tasks.confined_locomotion_2.flat_env_cfg import JOINT_NAMES


class CountingNetwork:
    @staticmethod
    def initial_hidden(batch, device):
        return torch.zeros(batch, 187, device=device)

    @staticmethod
    def encode_points(near, *args, **kwargs):
        return torch.zeros(near.shape[0], 187)

    @staticmethod
    def forward_from_features(obs, feature, hidden):
        return SimpleNamespace(hidden=hidden + 1, action_mean=torch.zeros(1, 18),
                               value=hidden[:, :1] + 1)


class CountingRuntime:
    device = torch.device("cpu")
    num_envs = 1

    def __init__(self):
        self.t = 0
        self.episode_ids = torch.zeros(1, dtype=torch.long)

    def prepare(self):
        z3, z18 = torch.zeros(1, 3), torch.zeros(1, 18)
        obs = OnlineObservation(
            env_ids=torch.zeros(1, dtype=torch.long), episode_ids=self.episode_ids.clone(),
            step_ids=torch.tensor([self.t]), time_s=torch.tensor([self.t * .02]),
            lin_vel_b=z3.clone(), ang_vel_b=z3.clone(), gravity_b=z3.clone(),
            joint_pos_rel_ref=z18.clone(), joint_vel=z18.clone(), prev_action=z18.clone(),
            command=z3.clone())
        return PreparedControlInputs(
            observation=obs, near_points=torch.zeros(1, 256, 3),
            near_valid=torch.zeros(1, 256, dtype=torch.bool),
            far_points=torch.zeros(1, 128, 3), far_valid=torch.zeros(1, 128, dtype=torch.bool),
            near_filled=torch.ones(1, dtype=torch.bool),
            near_scan_ids=torch.tensor([self.t // 5 + 1]),
            near_times=torch.tensor([self.t // 5 * .1]),
            far_point_scan_ids=torch.full((1, 128), -1, dtype=torch.long),
            far_point_times=torch.zeros(1, 128),
            near_angle_keys=torch.zeros(1, 256), far_angle_keys=torch.zeros(1, 128))

    def labels(self):
        obs = self.prepare().observation
        return {"envelope": EnvelopeLabel(obs.env_ids, obs.episode_ids, obs.step_ids,
                                           torch.zeros(1, 6), torch.ones(1, dtype=torch.bool))}

    def step(self, action):
        self.t += 1
        return SimpleNamespace(final_inputs=self.prepare(), reward=torch.zeros(1),
                               terminated=torch.zeros(1, dtype=torch.bool),
                               truncated=torch.zeros(1, dtype=torch.bool),
                               engineering_stopped=False)


def test_bootstrap_is_a_fork_and_both_live_histories_advance_only_once():
    runtime = CountingRuntime()
    collector = TrainingCollector(
        runtime, CountingNetwork(), DirectStdNormalProvider(generator_seed=3),
        symmetry_enabled=True, joint_permutation=JointPermutation.from_joint_names(JOINT_NAMES),
        mount_quat_xyzw=torch.tensor([0., 0., 0., 1.]))
    buffer = collector.collect()
    assert torch.equal(collector.hidden, torch.full((1, 187), 24.))
    assert torch.equal(collector.mirror_hidden, torch.full((1, 187), 24.))
    assert all(float(step.hidden_before[0, 0]) == t for t,step in enumerate(buffer.steps))
    assert all(float(step.next_value[0]) == t + 2 for t,step in enumerate(buffer.steps))


def test_batched_feature_cache_owns_outputs_and_keeps_mirror_and_reset_independent():
    count = 3
    parameter = torch.nn.Parameter(torch.tensor(1.))
    calls = []
    def encode(near, *args, **kwargs):
        calls.append(len(near))
        return (near[:, 0, :1] * parameter).expand(-1, 187)
    runtime = SimpleNamespace(device="cpu", num_envs=count, episode_ids=torch.zeros(count, dtype=torch.long))
    network = SimpleNamespace(initial_hidden=lambda b, d: torch.zeros(b, 187), encode_points=encode)
    collector = TrainingCollector(runtime, network, None)
    prepared = SimpleNamespace(
        observation=SimpleNamespace(episode_ids=torch.zeros(count, dtype=torch.long)),
        near_scan_ids=torch.zeros(count, dtype=torch.long),
        near_points=torch.arange(1, count + 1).float()[:, None, None].expand(-1, 256, 3).clone(),
        near_valid=torch.ones(count, 256, dtype=torch.bool),
        far_points=torch.zeros(count, 128, 3), far_valid=torch.zeros(count, 128, dtype=torch.bool),
        near_angle_keys=None, far_angle_keys=None)
    first = collector._frame(prepared)
    assert not first.requires_grad
    # Caller writes and later cache replacements cannot mutate each other.
    first[0] = -5
    assert collector._frame(prepared)[0, 0] == 1
    prepared.near_points[1] = 7
    prepared.observation.episode_ids[1] += 1
    second = collector._frame(prepared)
    assert first[1, 0] == 2 and second[1, 0] == 7
    prepared.near_points[0] = 9
    mirrored = collector._frame(prepared, mirror=True)
    assert mirrored[0, 0] == 9 and collector._frame(prepared)[0, 0] == 1
    state = collector.state_dict()
    collector.load_state_dict(state)
    refreshed = collector._frame(prepared)
    assert refreshed[0, 0] == 9 and second[0, 0] == 1
    assert calls == [3, 1, 3, 3]
