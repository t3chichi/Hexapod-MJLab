"""Independent SL defaults and explicit artifact locations."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path
import os as _os
@dataclass
class SLDataConfig:
    """Data-collection options."""

    #: Map seeds to collect, each paired with an obstacle-count override in
    #: ``pillar_counts``.  ``env.py`` reads ``cfg.seed`` as ``map_seed``, so each
    #: seed yields a different obstacle layout.  ``1`` is the production value
    #: (``task_registry`` copies ``train_cfg.seed`` into ``env_cfg.seed``).
    #:
    #: Original map recipe; historical team measurements are not mjlab results.
    seeds: List[int] = field(default_factory=lambda: [1, 7, 13, 21])

    #: Obstacle-count overrides, positionally paired with ``seeds``.  ``None``
    #: keeps the config default (24).  Use :meth:`seed_specs` to iterate the two
    #: lists together safely.
    pillar_counts: List[Optional[int]] = field(
        default_factory=lambda: [None, None, None, 28]
    )

    def seed_specs(self) -> List["tuple[int, Optional[int]]"]:
        """Zip ``seeds`` with ``pillar_counts``, padding with ``None``."""
        counts = list(self.pillar_counts) + [None] * (
            len(self.seeds) - len(self.pillar_counts)
        )
        return list(zip(self.seeds, counts[: len(self.seeds)]))

    #: Parallel environments used during collection.
    num_envs: int = 96

    #: Control steps per collection episode.  2200 < the 2250-step (45 s)
    #: episode cap, so every rollout stays a single episode with no done.
    num_steps: int = 2200

    #: LiDAR refresh decimation.  The sensor runs at 10 Hz while control runs at
    #: 50 Hz, so with ``1`` every control step is stored and the SL cadence
    #: matches PPO/play exactly (both act on every 50 Hz observation).  The
    #: historical ``5`` (10 Hz frames) trained the GRU on a 5x sparser cadence
    #: than deployment and cost ~6x oracle-MSE when the same weights acted at
    #: 50 Hz (measured: 0.011 -> 0.067 on the baseline weights).
    lidar_decimation: int = 1

    #: Steps discarded at the start of every episode.  On reset the rate limiter
    #: seeds ``prev_s = 1`` (fully open) and shrinks at ``shrink_rate = 2.0 m/s``;
    #: with a max span of 0.4 m that settles in ~10 steps.  30 keeps a 3x margin.
    warmup_steps: int = 30

@dataclass
class SLModelConfig:
    """Model architecture.  Must stay isomorphic to rsl_rl's actor."""

    obs_dim: int = 190  # 187 range channels + 3 ego motion
    action_dim: int = 5  # normalised envelope params [0, 1]
    rnn_hidden_dim: int = 187
    rnn_num_layers: int = 1
    rnn_type: str = "gru"
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    activation: str = "elu"

    #: Training-only auxiliary memory heads (a single Linear(hidden, action)
    #: probe per k, attached to the GRU output).  Empty list = disabled; the
    #: heads exist only during SL training and are never exported.
    #: ``aux_mode="recall"`` asks h_t to reconstruct s_{t-k} (pure memory
    #: demand, cannot be satisfied by spatial extrapolation of the current
    #: view); ``"forward"`` asks h_t to predict s_{t+k} (mixed extrapolation
    #: + retention, gradient lands at the write moment).  k is in frames at
    #: the control rate (75 = 1.5 s at 50 Hz ~= the pass-by invisible tail).
    aux_ks: List[int] = field(default_factory=list)
    aux_mode: str = "recall"

@dataclass
class SLTrainConfig:
    """Optimisation options."""

    #: Frames per training sequence.  200 frames @ 50 Hz == 4 s of memory,
    #: the horizon where the offline observability scan saturated (R2 = 1.0)
    #: when it was expressed as 40 frames @ 10 Hz.
    seq_len: int = 400  # 8s memory

    #: Stride (in frames) between consecutive window start positions.
    #: 10 frames @ 50 Hz == 0.2 s, the overlap semantics of the historical
    #: stride 2 @ 10 Hz.
    window_stride: int = 15

    #: Store windows as ``uint8``-quantised observations to cut memory ~4x.
    #:
    #: Windowing expands the source data by roughly ``seq_len / stride`` (~27x
    #: at seq_len=400, stride=15): the corpus would be ~14 GiB of float32 and
    #: the build-time concat peak ~28 GiB.  Quantising the range image to 8
    #: bits keeps the peak ~7 GiB.  NOTE: the range channels span [0, 1] --
    #: only the upper half of the [-1, 1] uint8 range -- so their effective
    #: step is ~2.5 cm (not the 1.3 cm a full-scale read would give).  That is
    #: well inside the 2 % multiplicative sensor noise the net already sees,
    #: and deployment consumes float observations (strictly finer).
    quantise_obs: bool = True

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    grad_clip: float = 1.0

    #: Fraction of environments held out for validation.  Splitting is done by
    #: environment (never by timestep) so no trajectory straddles the split.
    val_fraction: float = 0.2

    split_seed: int = 1

    #: ``std`` written into the exported ActorCriticRecurrent.

    #: Weight of the auxiliary memory loss (see ``SLModelConfig.aux_ks``).
    #: 0 disables the loss even when heads exist.  Monitor the main val MSE:
    #: if it degrades, lower this.
    aux_beta: float = 0.5

    #: Weight of the differentiable safety (collision) loss on the policy's own
    #: prediction: mean bilinear-sampled violation over the 24 boundary hex
    #: samples, with floor-pinned frames masked out.  0 disables it.  The loss
    #: is zero-gradient on safe frames, so it only acts where the prediction
    #: actually violates geometry.  Requires >=1 for effect; watch the main val
    #: MSE and the policy clearance (collapse guard).
    safe_lambda: float = 0.0

@dataclass
class SLConfig:
    data: SLDataConfig = field(default_factory=SLDataConfig)
    model: SLModelConfig = field(default_factory=SLModelConfig)
    train: SLTrainConfig = field(default_factory=SLTrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

ACTION_SIGN = [1.,1.,1.,1.,-1.]
PARAM_NAMES = ["front_width","middle_width","back_width","forward_limit","backward_limit"]

def env_action_scale():
    from ..config.parameters import EnvelopeCfg
    cfg = EnvelopeCfg()
    return cfg.soft_dof_pos_limit / (2 * cfg.action_max)

# CLI may override these paths. No installed package directory is used for logs.
SL_LOGS = str(Path.cwd() / "logs" / "ea2")
SL_DATA_DIR = str(Path(SL_LOGS) / "data")
SL_RUNS_DIR = str(Path(SL_LOGS) / "runs")

def data_path(seed):
    return str(Path(SL_DATA_DIR) / f"map_seed{int(seed)}.pt")

def run_dir(run_name):
    return str(Path(SL_RUNS_DIR) / run_name)

def available_seeds(data_dir=None):
    return sorted(int(p.stem.removeprefix("map_seed")) for p in Path(data_dir or SL_DATA_DIR).glob("map_seed*.pt"))
