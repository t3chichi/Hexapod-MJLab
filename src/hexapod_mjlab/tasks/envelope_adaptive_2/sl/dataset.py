"""Real mjlab collection, quantised windows, per-environment splits."""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from .config import SLConfig

@dataclass
class MapData:
    """One collected map: per-frame tensors plus the static distance field."""
    seed: int
    obs: torch.Tensor
    target: torch.Tensor
    done: torch.Tensor
    heading: torch.Tensor
    pos: torch.Tensor
    distance_field: torch.Tensor
    meta: Dict

    def __post_init__(self) -> None:
        assert self.obs.ndim == 3 and self.obs.shape[-1] == 190
        assert self.target.ndim == 3 and self.target.shape[-1] == 5
        assert self.obs.shape[:2] == self.target.shape[:2] == self.done.shape
        assert self.target.dtype == torch.float32

def save_map(data: MapData, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    torch.save({'seed': data.seed, 'obs': data.obs, 'target': data.target, 'done': data.done, 'heading': data.heading, 'pos': data.pos, 'distance_field': data.distance_field, 'meta': data.meta}, path)

def load_map(path: str) -> MapData:
    raw = torch.load(path, map_location='cpu')
    return MapData(seed=int(raw['seed']), obs=raw['obs'].float(), target=raw['target'].float(), done=raw['done'].bool(), heading=raw['heading'].float(), pos=raw['pos'].float(), distance_field=raw['distance_field'].float(), meta=raw.get('meta', {}))

def build_windows(obs: torch.Tensor, target: torch.Tensor, done: torch.Tensor, heading: torch.Tensor, pos: torch.Tensor, seq_len: int, warmup_frames: int, stride: int=2) -> Dict[str, torch.Tensor]:
    """Slice per-env time series into fixed-length windows.

    Windows that contain a ``done`` are rejected so that no sequence spans an
    episode boundary.  Returns CPU tensors shaped ``(S, seq_len, ...)``.
    """
    n_frames, n_envs = (obs.shape[0], obs.shape[1])
    starts: List[Tuple[int, int]] = []
    for n in range(n_envs):
        for st in range(warmup_frames, n_frames - seq_len + 1, stride):
            sl = slice(st, st + seq_len)
            if bool(done[sl, n].any()):
                continue
            starts.append((st, n))
    S = len(starts)
    out = {'obs': torch.zeros(S, seq_len, obs.shape[-1]), 'target': torch.zeros(S, seq_len, target.shape[-1]), 'heading': torch.zeros(S, seq_len), 'pos': torch.zeros(S, seq_len, 2), 'env_id': torch.zeros(S, dtype=torch.long)}
    for i, (st, n) in enumerate(starts):
        sl = slice(st, st + seq_len)
        out['obs'][i] = obs[sl, n]
        out['target'][i] = target[sl, n]
        out['heading'][i] = heading[sl, n]
        out['pos'][i] = pos[sl, n]
        out['env_id'][i] = n
    return out

def env_split(env_ids: torch.Tensor, val_fraction: float=0.2, seed: int=1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split window indices into train/val **by environment**.

    Splitting by timestep would leak: consecutive windows from the same env
    overlap heavily and share the same obstacle context.
    """
    uniq = torch.unique(env_ids)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    n_val = max(1, int(round(len(uniq) * val_fraction)))
    val_envs = set(uniq[perm[:n_val]].tolist())
    is_val = torch.tensor([int(e) in val_envs for e in env_ids.tolist()])
    idx = torch.arange(len(env_ids))
    return (idx[~is_val], idx[is_val])

class SLDataset:
    """Concatenated windows from one or more collected maps.

    Memory note
    -----------
    Windowing expands the source data by roughly ``seq_len / window_stride``
    (about 15x at the default seq_len=40, stride=2): four maps of ~90 MiB become
    ~1.4 GiB of float32 tensors.  ``cfg.train.quantise_obs`` stores the
    observations as ``uint8`` instead, cutting that ~4x at the cost of a
    ~1.3 cm quantisation step on the range channels (which span 3.2 m).
    """

    def __init__(self, cfg: SLConfig, maps: Sequence[MapData]):
        self.cfg = cfg
        self.quantised = bool(cfg.train.quantise_obs)
        warmup_frames = max(1, cfg.data.warmup_steps // cfg.data.lidar_decimation)
        obs_all, tgt_all, head_all, pos_all, env_all, src_all = ([], [], [], [], [], [])
        offset = 0
        self.maps = list(maps)
        for m in maps:
            w = build_windows(m.obs, m.target, m.done, m.heading, m.pos, seq_len=cfg.train.seq_len, warmup_frames=warmup_frames, stride=cfg.train.window_stride)
            obs_all.append(self._encode(w['obs']))
            tgt_all.append(w['target'])
            head_all.append(w['heading'])
            pos_all.append(w['pos'])
            env_all.append(w['env_id'] + offset)
            src_all.append(torch.full((w['obs'].shape[0],), len(src_all), dtype=torch.long))
            offset += m.obs.shape[1]
        if not obs_all or sum(len(x) for x in obs_all) == 0:
            raise ValueError("No valid windows: check steps, seq_len, warmup and reset boundaries")
        self.obs = torch.cat(obs_all)
        self.target = torch.cat(tgt_all)
        self.heading = torch.cat(head_all)
        self.pos = torch.cat(pos_all)
        self.env_id = torch.cat(env_all)
        self.map_id = torch.cat(src_all)
        self.train_idx, self.val_idx = env_split(self.env_id, cfg.train.val_fraction, cfg.train.split_seed)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Quantise observations to uint8 when enabled (range is within [-1, 1]).

        The range channels occupy [0, 1] and the ego channels [-1, 1], so a
        single affine map ``(x + 1) / 2 -> [0, 255]`` covers both.
        """
        if not self.quantised:
            return obs
        return torch.clamp(((obs + 1.0) * 127.5).round(), 0, 255).to(torch.uint8)

    def _decode(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.to(torch.float32) / 127.5 - 1.0 if self.quantised else obs

    def batch(self, idx: torch.Tensor) -> torch.Tensor:
        """Materialise a batch of windows as float32, ready for the model."""
        return self._decode(self.obs[idx])

    def __len__(self) -> int:
        return self.obs.shape[0]

    def summary(self) -> Dict:
        tgt = self.target.reshape(-1, 5)
        corr = np.corrcoef(tgt.numpy().T)
        return {'n_windows': int(self.obs.shape[0]), 'n_train': int(self.train_idx.numel()), 'n_val': int(self.val_idx.numel()), 'n_envs': int(self.env_id.unique().numel()), 'obs_min': float(self.obs.min()), 'obs_max': float(self.obs.max()), 'target_mean': [round(float(x), 4) for x in tgt.mean(0)], 'target_std': [round(float(x), 4) for x in tgt.std(0)], 'target_var_mean': float(tgt.var(0).mean()), 'corr_fw_fl': float(corr[0, 3]), 'corr_bw_bl': float(corr[2, 4]), 'frac_saturated_1': float((tgt > 0.995).float().mean()), 'frac_saturated_0': float((tgt < 0.005).float().mean())}


def build_env(cfg_seed, num_envs, pillar_count=None, sim_device="cuda:0", headless=True, attitude=False):
    from ..config.parameters import EA2Cfg
    from ..env import make_env
    cfg = EA2Cfg(seed=int(cfg_seed))
    cfg.env.num_envs = int(num_envs)
    if pillar_count is not None:
        cfg.obstacles.pillar_count_min = cfg.obstacles.pillar_count_max = int(pillar_count)
    cfg.attitude.enable = attitude
    return make_env(cfg, device=sim_device, capture_debug=not headless)


def collect_map(seed, num_envs=96, num_steps=None, lidar_decimation=None,
                pillar_count=None, device="cuda:0", attitude=False):
    from dataclasses import asdict
    from ..env import SLAdapter
    cfg = SLConfig()
    num_steps = cfg.data.num_steps if num_steps is None else int(num_steps)
    decimation = cfg.data.lidar_decimation if lidar_decimation is None else int(lidar_decimation)
    if num_steps < 1 or decimation < 1:
        raise ValueError("steps and sampling decimation must be positive")
    env = build_env(seed, num_envs, pillar_count, sim_device=device, attitude=attitude)
    try:
        adapter = SLAdapter(env)
        with torch.no_grad():
            adapter.reset()
            zero = torch.zeros((num_envs,5),device=env.device)
            frames = []
            for t in range(num_steps):
                obs, _, done, _ = adapter.step(zero)
                if t % decimation == 0:
                    state = env.ea2
                    frames.append((obs.cpu().clone(), state.oracle_s.cpu().clone(),
                                   done.cpu().clone(), state.heading.cpu().clone(),
                                   state.base_pos[:,:2].cpu().clone()))
            obs,target,done,heading,pos = (torch.stack([f[i] for f in frames]) for i in range(5))
        meta = {"data_version":2,"seed":seed,"num_envs":num_envs,"num_steps":num_steps,
                "lidar_decimation":decimation,"pillar_count":pillar_count,
                "warmup_steps":cfg.data.warmup_steps,"attitude":attitude,
                "environment_config":asdict(env.ea2.cfg),
                "map_acceptance":dict(env.ea2.map_data.acceptance),
                "reward_scales":dict(env.ea2.reward_scales),"n_frames":len(frames),
                "frame_semantics":"post-step/post-reset observation, pose and target"}
        return MapData(seed,obs,target,done,heading,pos,env.ea2.distance_field.cpu().clone(),meta)
    finally:
        env.close()
