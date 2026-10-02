"""Rollouts with episode-aware recurrence and explicit observation/target timing."""
from collections import deque
from dataclasses import asdict, dataclass
import numpy as np
import torch

from .config import ACTION_SIGN, SLConfig, SLModelConfig, env_action_scale
from .model import EnvelopeNet
from .losses import s_to_params
from ..env import SLAdapter


def load_checkpoint(path, device="cpu"):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    model_cfg = SLModelConfig(**raw["cfg_model"])
    net = EnvelopeNet(model_cfg).to(device)
    net.load_state_dict(raw["state_dict"])
    net.eval()
    return net, {k:v for k,v in raw.items() if k != "state_dict"}


def s_to_action(s, device=None):
    return s.new_tensor(ACTION_SIGN) * (s-.5) / env_action_scale()


def hex_area(s):
    p = s_to_params(s.clamp(0,1))
    return (p[...,0]+p[...,1])*p[...,3] + (p[...,2]+p[...,1])*p[...,4].abs()


def legacy_area(s):
    p = s_to_params(s.clamp(0,1))
    return p[...,0]*p[...,3] + p[...,2]*p[...,4].abs()


class Predictor:
    def __init__(self, net, mode="stateful", seq_len=400):
        if mode not in {"stateful","window"}:
            raise ValueError(mode)
        self.net, self.mode, self.seq_len = net, mode, seq_len
        self.hidden = None
        self.history = deque(maxlen=seq_len)
        self.lengths = None

    @torch.no_grad()
    def __call__(self, obs):
        if self.mode == "stateful":
            prediction,self.hidden = self.net.step(obs.unsqueeze(0),self.hidden)
            return prediction.clamp(0,1)
        if self.lengths is None:
            self.lengths = torch.zeros(obs.shape[0],device=obs.device,dtype=torch.long)
        self.history.append(obs.detach().clone())
        self.lengths.add_(1).clamp_(max=self.seq_len)
        if len(self.history)<self.seq_len:
            return torch.full((obs.shape[0],5),.5,device=obs.device)
        prediction = self.net(torch.stack(tuple(self.history)))[-1].clamp(0,1)
        return torch.where((self.lengths>=self.seq_len)[:,None],prediction,.5)

    def reset(self, done=None):
        if done is None:
            self.hidden = None
            self.history.clear()
            self.lengths = None
            return
        if self.hidden is not None:
            self.hidden[:,done] = 0
        if self.lengths is not None:
            self.lengths[done] = 0
            for frame in self.history:
                frame[done] = 0


@dataclass
class RolloutResult:
    step_reward: float
    pred_target_mse: float | None
    same_frame_mse: float | None
    executed_frame_mse: float | None
    collision_rate: float
    temporal_std: float | None
    area: float | None
    oracle_area: float | None
    legacy_area: float | None
    legacy_oracle_area_last_frame: float | None
    n_steps: int

    def as_dict(self):
        return asdict(self)


@torch.no_grad()
def closed_loop_rollout(env, net, cfg:SLConfig, mode="stateful", num_steps=700,
                        warmup_steps=30, device="cuda", zero_action=False, constant_s=None):
    if num_steps<=warmup_steps:
        raise ValueError("Evaluation requires steps beyond warmup")
    adapter=SLAdapter(env);obs=adapter.reset();state=env.ea2
    predictor=None if zero_action or constant_s is not None else Predictor(net,mode,cfg.train.seq_len)
    last_action=torch.zeros((env.num_envs,5),device=env.device)
    last_prediction=None
    rewards=[];collisions=[];legacy_mse=[];same_mse=[];executed_mse=[];preds=[];oracles=[]
    for t in range(num_steps):
        before_target=state.oracle_s.clone()
        if zero_action:
            last_prediction=None
        elif constant_s is not None:
            last_prediction=constant_s.to(env.device).expand(env.num_envs,5)
            last_action=s_to_action(last_prediction)
        elif t % cfg.data.lidar_decimation == 0:
            last_prediction=predictor(obs)
            last_action=s_to_action(last_prediction)
        obs,reward,done,info=adapter.step(last_action)
        frame=info["executed_frame"]
        if predictor is not None:
            predictor.reset(done)
        if t<warmup_steps:
            continue
        rewards.append(float(reward.mean()))
        collisions.append(float((frame["collision_hard"]>0).float().mean()))
        if last_prediction is not None:
            # Original field compares obs_t prediction with returned post-step target.
            legacy_mse.append(float(((last_prediction-state.oracle_s)**2).mean()))
            same_mse.append(float(((last_prediction-before_target)**2).mean()))
            executed_mse.append(float(((last_prediction-frame["target"])**2).mean()))
            preds.append(last_prediction.cpu().clone())
            oracles.append(frame["target"].cpu().clone())
    p=torch.stack(preds) if preds else None
    o=torch.stack(oracles) if oracles else None
    mean=lambda values:float(np.mean(values)) if values else None
    return RolloutResult(
        step_reward=mean(rewards),pred_target_mse=mean(legacy_mse),same_frame_mse=mean(same_mse),
        executed_frame_mse=mean(executed_mse),collision_rate=mean(collisions),
        temporal_std=float(p.std(0).mean()) if p is not None else None,
        area=float(hex_area(p).mean()) if p is not None else None,
        oracle_area=float(hex_area(o).mean()) if o is not None else None,
        legacy_area=float(legacy_area(p).mean()) if p is not None else None,
        legacy_oracle_area_last_frame=float(legacy_area(state.oracle_s).mean()) if p is not None else None,
        n_steps=len(rewards),
    )


def evaluate_closed_loop(cfg, net, seed=1, num_envs=64, num_steps=700, device="cuda:0",
                         env=None, modes=("stateful","window"), oracle_mean=None):
    from .dataset import build_env
    owned=env is None
    if owned:
        env=build_env(seed,num_envs,sim_device=device)
    try:
        if oracle_mean is None:
            oracle_mean=estimate_oracle_mean(env,cfg,device=device)
        results={}
        for name,kw in [("zero_action",{"zero_action":True}),
                        ("constant_oracle_mean",{"constant_s":oracle_mean})]:
            results[name]=closed_loop_rollout(env,net,cfg,num_steps=num_steps,device=device,**kw).as_dict()
        for mode in modes:
            results[f"sl_{mode}"]=closed_loop_rollout(env,net,cfg,mode=mode,num_steps=num_steps,device=device).as_dict()
        return results
    finally:
        if owned:env.close()


@torch.no_grad()
def estimate_oracle_mean(env,cfg,num_steps=300,device="cuda:0"):
    adapter=SLAdapter(env);adapter.reset();targets=[]
    for t in range(num_steps):
        adapter.step(torch.zeros((env.num_envs,5),device=env.device))
        if t>=cfg.data.warmup_steps:targets.append(env.ea2.oracle_s.mean(0).clone())
    if not targets:
        raise ValueError("Oracle mean requires steps beyond warmup")
    return torch.stack(targets).mean(0).to(device)
