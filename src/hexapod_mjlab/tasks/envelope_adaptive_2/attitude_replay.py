"""Continuous attitude replay generator for EA2 collection (design v3.1).

Replays a measured real-robot attitude trajectory (pitch/roll/height at 50 Hz,
straight walking) as a *continuous* per-env disturbance so collection covers
the body-tilt-induced ground-return compression that caused open-field false
contraction (see cascade tests/ea2/cascade/diag_openfield_contraction.py).

Signal model per env:

    pitch(t)  = mean_p + trim + k(t)·scale·osc_p(∫rate dt) + ar_p(t)
    roll(t)   = mean_r +           k(t)·scale·osc_r(∫rate dt) + ar_r(t)
    height(t) = mean_h +           k(t)·scale·osc_h(∫rate dt)

* ``osc_*`` is the mean-removed recorded trajectory; the three channels share
  trim/scale/k/phase so the measured pitch-height coupling is preserved.
* ``k(t) = lowpass(std(v)/std_ref, tau)`` carries the measured speed
  dependence (pitch std ≈ 0.858·v + 0.318 deg, R²=0.93; gait frequency is
  speed-invariant at 4 Hz — speed acts through amplitude only).
* Loops over the source use a cosine crossfade; measured seam step is below
  the in-segment p99 (no visible jumps). Episode-boundary trim jumps coincide
  with env reset + GRU hidden reset (invisible to the policy).
* Sign convention (pinned numerically): pitch > 0 is nose-down
  (Ry(+θ) applied to body-frame rays); at pitch=+4.6°, h=0.521 the far-row
  ground return is ≈1.80 m, matching the real robot's recorded 1.85 m.

Pure torch/numpy, no Isaac dependencies — unit-testable
(tests/ea2/test_attitude_replay.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_source(path: str | Path) -> dict:
    """Load and validate the pinned attitude source npz."""
    data = np.load(Path(path))
    for key in ("pitch", "roll", "height"):
        if key not in data:
            raise ValueError(f"attitude source {path} missing channel {key!r}")
    pitch = np.asarray(data["pitch"], dtype=np.float64)
    roll = np.asarray(data["roll"], dtype=np.float64)
    height = np.asarray(data["height"], dtype=np.float64)
    if not (len(pitch) == len(roll) == len(height)) or len(pitch) < 200:
        raise ValueError(f"attitude source {path} too short/degenerate")
    fs = float(data.get("fs", np.float32(50.0)))
    return {"pitch": pitch, "roll": roll, "height": height, "fs": fs}


def build_episode_buffer(
    osc: np.ndarray,
    rate: float,
    phase: int,
    fade: int,
    length: int,
    dt: float,
) -> np.ndarray:
    """Resample one channel at ``rate`` and tile to ``length`` steps with
    cosine crossfades at loop junctions (random start offset via ``phase``)."""
    n_src = len(osc)
    m = max(2, int(n_src / max(rate, 1e-6)))
    t_src = np.arange(n_src) * dt
    t_new = np.arange(m) * (dt / rate)
    base = np.interp(t_new, t_src, osc)
    start = int(phase) % m
    stream = [base[start:]]
    while sum(len(x) for x in stream) < length + fade:
        prev = stream[-1]
        k = min(fade, len(prev))  # 短首片（起始点近段尾）时整个尾部参与渐变
        w = 0.5 - 0.5 * np.cos(np.pi * (np.arange(k) + 0.5) / k)
        prev[-k:] = prev[-k:] * (1 - w) + base[:k] * w
        stream.append(base[k:])
    buf = np.concatenate(stream)[: length + fade]
    return buf[:length]


class AttitudeReplay:
    """Batched per-env continuous attitude generator."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        dt: float,
        max_episode_length: int,
        source_path: str | Path,
        cfg,
        seed: int,
    ) -> None:
        """``cfg`` fields (deg units where noted): trim_range, scale_range,
        rate_range, ar_std, ratio_tau, fade_steps, std_law_k, std_law_b."""
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dt = float(dt)
        self.max_episode_length = int(max_episode_length)
        self.cfg = cfg

        source = load_source(source_path)
        self.fs = float(source["fs"])
        self.means = torch.tensor(
            [source["pitch"].mean(), source["roll"].mean(), source["height"].mean()],
            dtype=torch.float32, device=self.device,
        )
        osc = np.stack([
            source["pitch"] - source["pitch"].mean(),
            source["roll"] - source["roll"].mean(),
            source["height"] - source["height"].mean(),
        ])
        self.osc = torch.tensor(osc, dtype=torch.float32, device=self.device)  # (3, N)
        # reference std = pitch std of the vx=1.0 recording (rad)
        self.ref_std = float(self.osc[0].std())

        length = self.max_episode_length + 2
        self.buf = torch.zeros(self.num_envs, 3, length, dtype=torch.float32,
                               device=self.device)
        self.idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.k = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.trim = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.scale = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.ar_p = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.ar_r = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(seed))

    def _law_std(self, v: torch.Tensor) -> torch.Tensor:
        """Measured pitch-oscillation std at speed ``v`` (deg law → rad)."""
        deg = float(self.cfg.std_law_k) * v + float(self.cfg.std_law_b)
        return torch.deg2rad(deg.clamp_min(float(self.cfg.std_law_b)))

    def reset(self, env_ids: torch.Tensor, v0: torch.Tensor) -> None:
        """Per-episode draws + episode buffer build for the given envs."""
        if env_ids.numel() == 0:
            return
        ids = env_ids.to(self.device, dtype=torch.long)
        trim = (torch.rand(ids.numel(), generator=self._gen, device=self.device)
                * (self.cfg.trim_range[1] - self.cfg.trim_range[0])
                + self.cfg.trim_range[0])
        scale = (torch.rand(ids.numel(), generator=self._gen, device=self.device)
                 * (self.cfg.scale_range[1] - self.cfg.scale_range[0])
                 + self.cfg.scale_range[0])
        rate = (torch.rand(ids.numel(), generator=self._gen, device=self.device)
                * (self.cfg.rate_range[1] - self.cfg.rate_range[0])
                + self.cfg.rate_range[0])
        self.trim[ids] = torch.deg2rad(trim)
        self.scale[ids] = scale
        self.k[ids] = self._law_std(v0.to(self.device)) / self.ref_std
        self.ar_p[ids] = 0.0
        self.ar_r[ids] = 0.0

        fade = int(self.cfg.fade_steps)
        osc_np = self.osc.cpu().numpy()
        for i, e in enumerate(ids.tolist()):
            phase = int(torch.randint(0, self.osc.shape[1], (1,),
                                      generator=self._gen, device=self.device))
            for ch in range(3):
                buf = build_episode_buffer(
                    osc_np[ch], float(rate[i]), phase, fade,
                    self.buf.shape[2], 1.0 / self.fs,
                )
                self.buf[e, ch] = torch.tensor(buf, dtype=torch.float32,
                                               device=self.device)
        self.idx[ids] = 0

    @torch.no_grad()
    def step(self, v: torch.Tensor):
        """Advance one control step; return ``(pitch, roll, height)`` (rad, rad, m)."""
        v = v.to(self.device)
        k_target = self._law_std(v) / self.ref_std
        alpha = 1.0 - float(np.exp(-self.dt / max(float(self.cfg.ratio_tau), 1e-6)))
        self.k += alpha * (k_target - self.k)

        amp = self.k * self.scale
        idx = self.idx.clamp_max(self.buf.shape[2] - 1)
        ar = torch.arange(self.num_envs, device=self.device)
        buf_p = self.buf[ar, 0, idx]
        buf_r = self.buf[ar, 1, idx]
        buf_h = self.buf[ar, 2, idx]

        a = float(np.exp(-2.0 * np.pi * 1.0 * self.dt))  # AR residual fc=1 Hz
        ar_scale = float(np.deg2rad(float(self.cfg.ar_std)))
        root = float(np.sqrt(max(1 - a * a, 0.0)))
        self.ar_p = a * self.ar_p + root * ar_scale * torch.randn(
            self.num_envs, generator=self._gen, device=self.device)
        self.ar_r = a * self.ar_r + root * ar_scale * torch.randn(
            self.num_envs, generator=self._gen, device=self.device)

        pitch = self.means[0] + self.trim + amp * buf_p + self.ar_p
        roll = self.means[1] + amp * buf_r + self.ar_r
        height = self.means[2] + amp * buf_h
        self.idx = (self.idx + 1).clamp_max(self.buf.shape[2] - 1)
        return pitch, roll, height
