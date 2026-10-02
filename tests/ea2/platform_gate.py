"""Real GPU gate for the approved jointless mjlab architecture."""
import torch
from hexapod_mjlab.tasks.envelope_adaptive_2.env import EA2Env, SLAdapter, make_platform_cfg, PlatformState


def main():
    env = EA2Env(make_platform_cfg(2, episode_length_s=.06), state=PlatformState(2,"cuda:0"))
    try:
        assert env.sim.mj_model.nq == 0 and env.sim.mj_model.nv == 0
        assert env.single_action_space.shape == (5,)
        adapter = SLAdapter(env)
        assert adapter.reset().shape == (2, 190)
        assert env.ea2.logical_step == 1
        before = env.ea2.age.clone()
        env.get_observations(); env.get_observations()
        assert torch.equal(env.ea2.age, before)
        for _ in range(1):
            adapter.step(torch.zeros((2, 5), device=env.device))
        assert env.ea2.logical_step == 2
        env.reset(env_ids=torch.tensor([0], device=env.device))
        assert env.ea2.age.tolist() == [0, 2]
        _, _, done, _ = adapter.step(torch.zeros((2, 5), device=env.device))
        assert done.tolist() == [False, True]
        assert env.ea2.age.tolist() == [1, 0]
        assert abs(env.step_dt - .02) < 1e-10
        print("T1 PASS: jointless GPU scene, 190/5, 50Hz, cached reads, explicit/partial/auto reset")
    finally:
        env.close()


if __name__ == "__main__":
    main()
