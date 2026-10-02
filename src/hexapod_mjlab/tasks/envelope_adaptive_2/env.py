"""mjlab lifecycle and the explicit-reset adapter for independent SL callers."""
import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from .mdp.actions import EnvelopeActionCfg


def carrier_spec():
    return mujoco.MjSpec.from_string('''<mujoco><worldbody>
      <body name="carrier" mocap="true"><geom type="sphere" size="0.03"
      contype="0" conaffinity="0" rgba="0.2 0.5 1 1"/></body>
      </worldbody></mujoco>''')


def observation(env):
    return env.ea2.observation


def reset_state(env, env_ids):
    env.ea2.reset(env_ids)
    if hasattr(env.ea2, "base_pos"):
        write_carrier_pose(env, env_ids)


def write_carrier_pose(env, env_ids=None):
    """Use public buffers to adapt mjlab 1.6's batched mocap writer shape bug."""
    ids = slice(None) if env_ids is None else env_ids
    index = env.scene["carrier"].data.indexing.mocap_id
    pose = env.ea2.carrier_pose()[ids]
    env.sim.data.mocap_pos[ids, index:index+1] = pose[:, :3].unsqueeze(1)
    env.sim.data.mocap_quat[ids, index:index+1] = pose[:, 3:].unsqueeze(1)


def timeout(env):
    return env.episode_length_buf >= env.max_episode_length


def reward(env):
    return env.ea2.reward


def path_failed(env):
    return env.ea2._replan_reset_buf


def make_platform_cfg(num_envs=2, seed=1, episode_length_s=45.0):
    return ManagerBasedRlEnvCfg(
        seed=seed,
        scene=SceneCfg(num_envs=num_envs, env_spacing=0.0,
                       entities={"carrier": EntityCfg(spec_fn=carrier_spec)}),
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=.02)),
        decimation=1, episode_length_s=episode_length_s, scale_rewards_by_dt=False,
        observations={"policy": ObservationGroupCfg(
            terms={"envelope": ObservationTermCfg(func=observation)},
            concatenate_terms=True, enable_corruption=False)},
        actions={"envelope": EnvelopeActionCfg(entity_name="carrier")},
        events={"reset": EventTermCfg(func=reset_state, mode="reset")},
        rewards={"envelope": RewardTermCfg(func=reward, weight=1.)},
        terminations={"time_out": TerminationTermCfg(func=timeout, time_out=True)},
    )


class EA2Env(ManagerBasedRlEnv):
    def __init__(self, cfg, device="cuda:0", *, state, **kwargs):
        self.ea2 = state
        super().__init__(cfg, device=device, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if hasattr(self.ea2, "executed_frame"):
            info["executed_frame"] = self.ea2.executed_frame
            info.update(self.ea2.extras)
            info["time_outs"] = truncated
        return obs, reward, terminated, truncated, info

    def update_visualizers(self, visualizer):
        super().update_visualizers(visualizer)
        if hasattr(self.ea2, "sensor"):
            from .visualization import draw_state
            draw_state(self.ea2, visualizer)

    def close(self):
        try:
            super().close()
        finally:
            if hasattr(self.ea2, "close"):
                self.ea2.close()


def make_env(parameters=None, device="cuda:0", capture_debug=False, **kwargs):
    from .config.parameters import EA2Cfg
    from .mdp.state import EnvelopeState
    parameters = parameters or EA2Cfg()
    if parameters.sim.dt != .02 or parameters.env.num_observations != 190:
        raise ValueError("EA2 contract requires dt=.02 and 190 observations")
    state = EnvelopeState(parameters, device, capture_debug)
    cfg = make_platform_cfg(parameters.env.num_envs, parameters.seed,
                            parameters.env.episode_length_s)
    from .visualization import map_spec
    cfg.scene.entities["map"] = EntityCfg(spec_fn=lambda: map_spec(state.map_data))
    from mjlab.viewer.viewer_config import ViewerConfig
    cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    cfg.viewer.entity_name = "carrier"
    cfg.viewer.distance = 6.
    cfg.viewer.height, cfg.viewer.width = 720, 960
    cfg.terminations["path_failed"] = TerminationTermCfg(func=path_failed)
    try:
        return EA2Env(cfg, device, state=state, **kwargs)
    except Exception:
        state.close()
        raise


class SLAdapter:
    """Preserve old explicit reset's extra zero-action step, outside mjlab API."""
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.env.reset()
        obs, _, _, _, _ = self.env.step(
            torch.zeros((self.env.num_envs, 5), device=self.env.device))
        return obs["policy"]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs["policy"], reward, terminated | truncated, info


class PlatformState:
    """T1 lifecycle probe only; not a runnable EA2 data-generation task."""
    def __init__(self, num_envs, device):
        self.observation = torch.zeros((num_envs, 190), device=device)
        self.reward = torch.zeros(num_envs, device=device)
        self.logical_step = 0
        self.age = torch.zeros(num_envs, dtype=torch.long, device=device)

    def advance(self, action):
        self.logical_step += 1
        self.age += 1
        self.observation[:, -3] = self.age * .02

    def reset(self, env_ids):
        self.age[env_ids] = 0
        self.observation[env_ids] = 0
