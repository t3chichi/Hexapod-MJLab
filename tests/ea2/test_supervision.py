"""CPU recurrence, supervision masks, metrics, datasets, losses and checkpoints."""

import _bootstrap
from pathlib import Path
import tempfile
import torch
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.config import SLConfig, SLModelConfig
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.model import EnvelopeNet
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import (
    MapData,
    SLDataset,
    build_windows,
    save_map,
    load_map,
)
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.train import train, save_checkpoint, compute_metrics
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.evaluate import (
    load_checkpoint,
    Predictor,
    hex_area,
    s_to_action,
    closed_loop_rollout,
    checkpoint_teacher,
)
from hexapod_mjlab.tasks.envelope_adaptive_2.math_utils import (
    map_actions_to_params,
    normalized_envelope_params,
)
import numpy as np
from hexapod_mjlab.tasks.envelope_adaptive_2.teachers.batch import teacher_identity
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.losses import (
    masked_mse,
    aux_memory_loss,
    batch_safety_loss,
    frame_map_ids,
)


def check_learning(legacy_source=False):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    ref = None
    if legacy_source:
        from source_reference import load_reference

        ref = load_reference()
    net = EnvelopeNet()
    x = torch.rand(35, 3, 190)
    if ref is not None:
        gold = ref["sl_model"].EnvelopeNet()
        gold.load_state_dict(net.state_dict())
        assert torch.allclose(net(x), gold(x), atol=1e-06, rtol=0)
    h = None
    out = []
    for obs in x:
        p, h = net.step(obs[None], h)
        out.append(p)
    assert torch.allclose(net(x), torch.stack(out), atol=1e-06, rtol=0)
    predictor = Predictor(net)
    predictor(x[0])
    predictor(x[1])
    before = predictor.hidden.clone()
    predictor.reset(torch.tensor([False, True, False]))
    assert not predictor.hidden[:, 1].any() and torch.equal(
        predictor.hidden[:, [0, 2]], before[:, [0, 2]]
    )
    p = predictor(x[2])
    independent, _ = net.step(x[2, 1:2][None])
    assert torch.allclose(p[1], independent[0].clamp(0, 1), atol=1e-06, rtol=0)
    window = Predictor(net, "window", 4)
    for obs in x[:4]:
        window(obs)
    window.reset(torch.tensor([False, True, False]))
    for i, obs in enumerate(x[4:8]):
        p = window(obs)
        if i < 3:
            assert torch.equal(p[1], torch.full((5,), 0.5))
    assert torch.allclose(p[1], net(x[4:8, 1:2])[-1, 0].clamp(0, 1), atol=1e-06, rtol=0)
    assert torch.allclose(p[0], net(x[4:8, 0:1])[-1, 0].clamp(0, 1), atol=1e-06, rtol=0)
    from hexapod_mjlab.tasks.envelope_adaptive_2.geometry import compute_hex_vertices
    from hexapod_mjlab.tasks.envelope_adaptive_2.sl.train import s_to_params

    s = torch.rand(7, 5)
    params = s_to_params(s)
    v = compute_hex_vertices(*params.unbind(-1))
    shoelace = (v[..., 0] * v.roll(-1, 1)[..., 1] - v[..., 1] * v.roll(-1, 1)[..., 0]).sum(
        1
    ).abs() * 0.5
    assert torch.allclose(hex_area(s), shoelace, atol=1e-06, rtol=0)
    lo = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.9])
    hi = torch.tensor([0.6, 0.7, 0.6, 0.9, -0.6])
    realized = map_actions_to_params(s_to_action(s), lo, hi, 0.95, 4.0)
    assert torch.allclose(normalized_envelope_params(realized, lo, hi), s, atol=1e-06, rtol=0)
    from types import SimpleNamespace

    class TimingEnv:
        device = "cpu"
        num_envs = 1

        def __init__(self):
            self.ea2 = SimpleNamespace(oracle_s=torch.zeros(1, 5))
            self.t = 0

        def reset(self):
            self.t = 0
            self.ea2.oracle_s.zero_()
            return ({"policy": torch.zeros(1, 190)}, {})

        def step(self, a):
            self.t += 1
            terminal = torch.full((1, 5), self.t / 10)
            done = torch.tensor([self.t == 3])
            self.ea2.oracle_s.fill_(0.9 if done.item() else self.t / 10)
            frame = {"target": terminal, "collision_hard": torch.zeros(1)}
            return (
                {"policy": torch.zeros(1, 190)},
                torch.ones(1),
                done,
                torch.zeros(1, dtype=torch.bool),
                {"executed_frame": frame},
            )

    metric = closed_loop_rollout(
        TimingEnv(),
        None,
        SLConfig(),
        num_steps=3,
        warmup_steps=0,
        constant_s=torch.full((1, 5), 0.5),
        device="cpu",
    )
    for name, targets in [
        ("same_frame_mse", [0.1, 0.2, 0.9]),
        ("executed_frame_mse", [0.2, 0.3, 0.4]),
        ("pred_target_mse", [0.2, 0.9, 0.4]),
    ]:
        expected = ((torch.tensor(targets) - 0.5) ** 2).mean().item()
        assert abs(getattr(metric, name) - expected) < 1e-06, name
    expected = hex_area(torch.tensor([[0.2] * 5, [0.3] * 5, [0.4] * 5])).mean().item()
    assert abs(metric.oracle_area - expected) < 1e-06
    import numpy as np

    target = torch.rand(30, 3, 5) * torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    metric = compute_metrics(target, target)
    eigenvalues = np.linalg.eigvalsh(np.cov(target.reshape(-1, 5).numpy().T))[::-1]
    assert np.allclose(metric["pc_var"], eigenvalues / eigenvalues.sum(), atol=1e-06)
    assert np.allclose(metric["legacy_pc_var"], eigenvalues**2 / (eigenvalues**2).sum(), atol=1e-6)
    if ref is not None:
        assert np.allclose(
            metric["legacy_pc_var"],
            ref["sl_train"].compute_metrics(target, target)["pc_var"],
            atol=1e-06,
        )
    auxnet = EnvelopeNet(SLModelConfig(aux_ks=[25, 50, 100, 200, 300]))
    y = torch.rand(400, 2, 5)
    main, h = auxnet.forward_with_aux(torch.rand(400, 2, 190))
    for k in auxnet.aux_ks:
        auxnet.zero_grad()
        loss = aux_memory_loss(auxnet.aux_heads[str(k)](h), y, k, "recall")
        loss.backward(retain_graph=True)
        for param in [auxnet.gru.weight_ih_l0, auxnet.aux_heads[str(k)].weight]:
            assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    yy = -37 + (torch.arange(740) + 0.5) * 0.1
    corridor = (0.55 - yy.abs()).clamp_min(0)[:, None].expand(740, 740).contiguous()
    dfs = torch.stack([torch.full_like(corridor, 5.0), corridor])
    ids = frame_map_ids(torch.tensor([0, 1]), 3)
    pred = torch.full((3, 2, 5), 0.55, requires_grad=True)
    loss, rate = batch_safety_loss(pred.reshape(-1, 5), torch.zeros(6), torch.zeros(6, 2), ids, dfs)
    loss.backward()
    assert (
        loss > 0 and rate > 0 and (pred.grad[:, 1].abs().sum() > 0) and (not pred.grad[:, 0].any())
    )
    assert torch.isfinite(pred.grad).all()
    for field in [torch.full_like(corridor, 5.0), torch.zeros_like(corridor)]:
        q = torch.full((2, 5), 0.55, requires_grad=True)
        loss, _ = batch_safety_loss(
            q, torch.zeros(2), torch.zeros(2, 2), torch.zeros(2, dtype=torch.long), field[None]
        )
        loss.backward()
        assert loss == 0 and (not q.grad.any())
    obs = torch.rand(50, 6, 190)
    target = torch.rand(50, 6, 5)
    done = torch.zeros(50, 6, dtype=torch.bool)
    done[20, 0] = True
    heading = torch.zeros(50, 6)
    pos = torch.zeros(50, 6, 2)
    a = build_windows(obs, target, done, heading, pos, 12, 2, 3)
    if ref is not None:
        b = ref["sl_dataset"].build_windows(obs, target, done, heading, pos, 12, 2, 3)
        for key in b:
            assert torch.equal(a[key], b[key]), key
    assert a["valid"].all()
    cfg = SLConfig()
    cfg.train.seq_len = 12
    cfg.train.window_stride = 3
    cfg.train.epochs = 1
    cfg.data.warmup_steps = 2
    data = MapData(1, obs, target, done, heading, pos, corridor, {})
    dataset = SLDataset(cfg, [data])
    if ref is not None:
        original = ref["sl_dataset"].SLDataset(cfg, [data])
        for name in ["obs", "target", "train_idx", "val_idx", "env_id", "map_id"]:
            assert torch.equal(getattr(dataset, name), getattr(original, name)), name
    assert not set(dataset.env_id[dataset.train_idx].tolist()) & set(
        dataset.env_id[dataset.val_idx].tolist()
    )
    result = train(cfg, dataset, device="cpu", verbose=False)
    assert torch.isfinite(torch.tensor(result["val"]["mse"]))
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "model.pt")
        save_checkpoint(result, cfg, path)
        loaded, _ = load_checkpoint(path)
        assert torch.allclose(result["net"](x), loaded(x), atol=1e-06, rtol=0)


def check_validity():
    torch.manual_seed(2)
    valid = torch.rand(400, 2) > 0.2
    target = torch.rand(400, 2, 5)
    bad = target.clone()
    bad[~valid] = float("nan")
    prediction = torch.rand_like(target, requires_grad=True)
    for mode in ["main", "recall", "forward"]:
        for k in [0] if mode == "main" else [25, 50, 100, 200, 300]:
            loss = lambda y: (
                masked_mse(prediction, y, valid)
                if mode == "main"
                else aux_memory_loss(prediction, y, k, mode, valid)
            )
            a = loss(target)
            ga = torch.autograd.grad(a, prediction)[0]
            b = loss(bad)
            gb = torch.autograd.grad(b, prediction)[0]
            assert torch.equal(a, b) and torch.equal(ga, gb) and (ga.abs().sum() > 0)
            if mode == "main":
                assert not ga[~valid].any()
    q = torch.full((2, 5), float("nan"), requires_grad=True)
    loss, _, count = batch_safety_loss(
        q,
        torch.zeros(2),
        torch.zeros(2, 2),
        torch.zeros(2, dtype=torch.long),
        torch.ones(1, 740, 740),
        valid=torch.zeros(2, dtype=torch.bool),
        return_count=True,
    )
    assert loss == 0 and count == 0 and (not torch.autograd.grad(loss, q)[0].any())
    empty = masked_mse(prediction, bad, torch.zeros_like(valid))
    assert empty == 0 and (not torch.autograd.grad(empty, prediction)[0].any())
    a = compute_metrics(prediction.detach().transpose(0, 1), target.transpose(0, 1), valid.T)
    b = compute_metrics(prediction.detach().transpose(0, 1), bad.transpose(0, 1), valid.T)
    for key in ["mse", "r2", "temporal_std_oracle", "pc_var"]:
        assert a[key] == b[key]
    obs = torch.rand(40, 2, 190)
    y = torch.rand(40, 2, 5)
    done = torch.zeros(40, 2, dtype=torch.bool)
    done[20, 0] = True
    mask = torch.ones_like(done)
    mask[10, 1] = False
    w = build_windows(obs, y, done, torch.zeros(40, 2), torch.zeros(40, 2, 2), 12, 1, 1, mask)
    assert (~w["valid"]).any()
    for i, e in enumerate(w["env_id"]):
        start = next((t for t in range(40) if torch.equal(obs[t, e], w["obs"][i, 0])))
        assert not done[start : start + 12, e].any()
    data = MapData(
        1, obs, y, done, torch.zeros(40, 2), torch.zeros(40, 2, 2), torch.ones(740, 740), {}, mask
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "map.pt")
        save_map(data, path)
        saved = load_map(path)
        assert torch.equal(saved.supervision_valid, mask)
        raw = torch.load(path, weights_only=True)
        for key in [
            "supervision_valid",
            "geometric_valid",
            "emergency",
            "rate_checked",
            "rate_valid",
        ]:
            raw.pop(key)
        torch.save(raw, path)
        legacy = load_map(path)
        assert legacy.supervision_valid.all() and (not legacy.geometric_valid.any())
        raw["meta"] = {"teacher": teacher_identity()}
        torch.save(raw, path)
        try:
            load_map(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing exact certificates accepted")
    assert checkpoint_teacher({})["kind"] == "legacy"
    assert checkpoint_teacher({"teacher": teacher_identity()})["kind"] == "exact"
    cfg = SLConfig()
    cfg.train.epochs = 1
    cfg.train.seq_len = 12
    cfg.train.window_stride = 5
    cfg.data.warmup_steps = 1
    data.meta = {"teacher": teacher_identity()}
    data.geometric_valid = torch.ones_like(done)
    data.target[~mask] = float("nan")
    ds = SLDataset(cfg, [data])
    result = train(cfg, ds, device="cpu", verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = str(Path(tmp) / "model.pt")
        save_checkpoint(result, cfg, checkpoint)
        _, meta = load_checkpoint(checkpoint)
        assert checkpoint_teacher(meta) == teacher_identity()
    from types import SimpleNamespace

    class AreaEnv:
        device = "cpu"
        num_envs = 1

        def __init__(self):
            self.ea2 = SimpleNamespace(
                oracle_s=torch.zeros(1, 5),
                oracle_supervision_valid=torch.ones(1, dtype=torch.bool),
                _teacher=object(),
            )

        def reset(self):
            self.t = 0
            return ({"policy": torch.zeros(1, 190)}, {})

        def step(self, action):
            self.t += 1
            target = torch.full((1, 5), self.t / 10)
            self.ea2.oracle_s = target
            valid = torch.tensor([self.t % 2 == 0])
            self.ea2.oracle_supervision_valid = valid
            frame = {
                "target": target,
                "collision_hard": torch.zeros(1),
                "geometric_valid": valid,
                "supervision_valid": valid,
                "emergency": torch.zeros(1, dtype=torch.bool),
            }
            return (
                {"policy": torch.zeros(1, 190)},
                torch.zeros(1),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                {"executed_frame": frame},
            )

    rollout = closed_loop_rollout(
        AreaEnv(),
        None,
        SLConfig(),
        num_steps=3,
        warmup_steps=0,
        constant_s=torch.full((1, 5), 0.5),
        device="cpu",
    )
    assert rollout.paired_area_frames == 2 and rollout.total_frames == 3
    assert abs(rollout.area - float(hex_area(torch.full((1, 5), 0.5)))) < 1e-06
    assert (
        abs(rollout.oracle_area - float(hex_area(torch.tensor([[0.2] * 5, [0.4] * 5])).mean()))
        < 1e-06
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-source", action="store_true", help="Also compare frozen original modules"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    check_learning(args.legacy_source)
    print("PASS recurrence, metrics, gradients and checkpoint", flush=True)
    check_validity()
    print("PASS masks, provenance and dataset validity", flush=True)


if __name__ == "__main__":
    main()
