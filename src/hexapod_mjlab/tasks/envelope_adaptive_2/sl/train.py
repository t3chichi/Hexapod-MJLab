"""Supervised optimisation and reusable recall/safety auxiliary losses."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .dataset import SLDataset
from .model import EnvelopeNet
from .config import PARAM_NAMES, SLConfig

from .losses import masked_mse, aux_pair_mask, aux_memory_loss, aux_head_weights, s_to_params, batch_safety_loss, frame_map_ids


@dataclass
class History:
    train_mse: List[float] = field(default_factory=list)
    val_mse: List[float] = field(default_factory=list)
    val_r2: List[float] = field(default_factory=list)

    def best_epoch(self) -> int:
        return int(np.argmax(self.val_r2)) if self.val_r2 else -1


def _predict(
    net: nn.Module, dataset, idx: torch.Tensor, device: str, batch: int = 128
):
    """Run the net over ``dataset`` windows ``idx``.

    Takes the *dataset* rather than a raw tensor so that quantised storage
    (``SLDataset.quantised``) is decoded transparently.
    """
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            chunk = dataset.batch(idx[i : i + batch]).transpose(0, 1).to(device)
            preds.append(net(chunk).transpose(0, 1).cpu())
    return torch.cat(preds, dim=0).clamp(0.0, 1.0)










def compute_metrics(pred: torch.Tensor, target: torch.Tensor, valid=None) -> Dict:
    """Offline metrics for normalised-envelope predictions."""
    mask = torch.ones(target.shape[:-1], dtype=torch.bool) if valid is None else valid.cpu()
    p, t = pred[mask], target[mask]
    if len(t) < 2:
        raise ValueError("Metrics require at least two valid frames")
    err = (p - t) ** 2
    mse_per_dim = err.mean(0)
    var_per_dim = t.var(0)
    r2_per_dim = 1.0 - mse_per_dim / var_per_dim.clamp_min(1e-8)
    mse = float(err.mean())
    var = float(var_per_dim.mean())
    r2 = 1.0 - mse / max(var, 1e-8)
    def temporal_std(values):
        # Missing labels do not compress time or link windows/episodes.
        count = mask.sum(1)
        clean = torch.where(mask[..., None], values, 0.)
        mu = clean.sum(1) / count.clamp_min(1)[:, None]
        delta = torch.where(mask[..., None], values - mu[:, None], 0.)
        variance = delta.square().sum(1) / (count - 1).clamp_min(1)[:, None]
        return float(variance[count > 1].sqrt().mean()) if bool((count > 1).any()) else None

    # --- PCA decomposition of the target, then R2 along each component ---
    flat_t = t.numpy()
    mu = flat_t.mean(0)
    centred = flat_t - mu
    _, s, vt = np.linalg.svd(np.cov(centred.T), hermitian=True)
    # Covariance singular values are already variances, not data singular values.
    ev = s / max(float(s.sum()), 1e-20)
    legacy_ev = (s**2) / max(float((s**2).sum()), 1e-20)
    flat_p = p.numpy()

    pc_var, pc_r2 = [], []
    for j in range(5):
        v = vt[j]
        yt = centred @ v
        pt = (flat_p - mu) @ v
        denom = float(yt.var())
        pc_var.append(float(ev[j]))
        pc_r2.append(float(1.0 - ((pt - yt) ** 2).mean() / denom) if denom > 1e-10 else float("nan"))

    # subspace R2 for the top-k components
    sub_r2 = {}
    for k in (1, 2, 3):
        P = vt[:k]
        yt = centred @ P.T
        pt = (flat_p - mu) @ P.T
        denom = float(yt.var(0).sum()) * len(yt)
        sub_r2[f"top{k}"] = float(1.0 - ((pt - yt) ** 2).sum() / denom) if denom > 1e-10 else float("nan")

    return {
        "mse": mse,
        "r2": r2,
        "target_var": var,
        "mse_per_dim": {n: float(x) for n, x in zip(PARAM_NAMES, mse_per_dim)},
        "r2_per_dim": {n: float(x) for n, x in zip(PARAM_NAMES, r2_per_dim)},
        "target_std_per_dim": {n: float(x) for n, x in zip(PARAM_NAMES, t.std(0))},
        "pc_var": pc_var,
        "legacy_pc_var": [float(x) for x in legacy_ev],
        "pc_r2": pc_r2,
        "subspace_r2": sub_r2,
        # dynamic-ness: std over time, compared against oracle's own
        "valid_frames": int(mask.sum()),
        "total_frames": int(mask.numel()),
        "temporal_std": temporal_std(pred),
        "temporal_std_oracle": temporal_std(target),
    }


def train(
    cfg: SLConfig,
    dataset: SLDataset,
    device: str = "cuda",
    log_every: int = 5,
    verbose: bool = True,
    patience: int = 0,
    val_every: int = 1,
) -> Dict:
    """Train ``EnvelopeNet``; returns metrics plus the best-epoch weights.

    Args:
        patience: stop after this many *validated* epochs without improvement
            (0 disables early stopping).  Validation improvements are only
            observable on epochs where validation actually runs.
        val_every: run validation every N epochs.  Validation is a full forward
            pass over the validation split and is easily the most expensive part
            of an epoch, so raising this is a cheap speed-up for large datasets.
            The final epoch is always validated.
    """
    dev = device if torch.cuda.is_available() else "cpu"
    net = EnvelopeNet(cfg.model).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.train.learning_rate)
    hist = History()

    tr, va = dataset.train_idx, dataset.val_idx
    if not len(tr) or not len(va):
        raise ValueError("Training requires nonempty environment-separated train and validation windows")
    tgt = dataset.target
    bs = cfg.train.batch_size

    # per-map distance fields on device for the differentiable safety loss
    safe_lambda = float(getattr(cfg.train, "safe_lambda", 0.0))
    dfs = None
    if safe_lambda > 0:
        dfs = torch.stack(
            [torch.as_tensor(m.distance_field, dtype=torch.float32) for m in dataset.maps]
        ).to(dev)

    best_r2, best_state, best_epoch = -np.inf, None, -1
    epochs_without_gain = 0
    stopped_early = False
    t0 = time.time()

    for epoch in range(cfg.train.epochs):
        net.train()
        perm = torch.randperm(len(tr))
        running = 0.0
        main_frames = 0
        aux_counts = {}
        aux_running = {}      # per-k raw aux loss sums (pre-weight)
        running_safe, safe_frames = 0.0, 0
        aux_beta = float(getattr(cfg.train, "aux_beta", 0.5))
        aux_w = (
            aux_head_weights(net.aux_ks, cfg.train.seq_len, 1.0)
            if net.aux_heads else {}
        )  # relative shares; beta applied per term below
        for i in range(0, len(tr), bs):
            b = tr[perm[i : i + bs]]
            x = dataset.batch(b).transpose(0, 1).to(dev)
            y = tgt[b].transpose(0, 1).to(dev)
            valid = dataset.valid[b].transpose(0, 1).to(dev)
            count = int(valid.sum())
            if not count:
                continue
            use_aux = net.aux_heads and aux_beta > 0
            if use_aux or safe_lambda > 0:
                main_out, h_seq = net.forward_with_aux(x)
            else:
                main_out = net(x)
            main_mse = masked_mse(main_out, y, valid)
            loss = main_mse
            running += float(main_mse.detach()) * count
            main_frames += count
            if use_aux:
                for k_str, head in net.aux_heads.items():
                    aux_l = aux_memory_loss(head(h_seq), y, int(k_str), cfg.model.aux_mode, valid)
                    loss = loss + aux_beta * aux_w[int(k_str)] * aux_l
                    aux_count = int(aux_pair_mask(valid, int(k_str)).sum())
                    aux_running[int(k_str)] = aux_running.get(int(k_str), 0.0) + float(aux_l.detach()) * aux_count
                    aux_counts[int(k_str)] = aux_counts.get(int(k_str), 0) + aux_count
            if safe_lambda > 0:
                # flatten the (L, B, ...) window batch into frames: the loss
                # contract is a flat (N, 5) frame batch with per-frame fields
                L_frames = main_out.shape[0]
                safe_l, trig, n_safe = batch_safety_loss(
                    main_out.reshape(-1, 5),
                    dataset.heading[b].transpose(0, 1).reshape(-1).to(dev),
                    dataset.pos[b].transpose(0, 1).reshape(-1, 2).to(dev),
                    frame_map_ids(dataset.map_id[b], L_frames).to(dev),
                    dfs,
                    valid=valid.reshape(-1) if dataset.teacher["kind"] == "exact" else None,
                    return_count=True,
                )
                loss = loss + safe_lambda * safe_l
                running_safe += float(safe_l.detach()) * n_safe
                safe_frames += n_safe
            opt.zero_grad()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite supervised loss")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
            opt.step()

        if not main_frames:
            raise ValueError("Training split has no valid supervision")
        train_mse = running / main_frames
        aux_mse_epoch = (
            {k: v / max(1, aux_counts[k]) for k, v in aux_running.items()} if use_aux else None
        )
        aux_str = (
            "[" + " ".join(f"k{k}:{v:.4f}" for k, v in sorted(aux_mse_epoch.items())) + "]"
            if aux_mse_epoch else ""
        )
        safe_epoch = running_safe / max(1, safe_frames) if safe_lambda > 0 else None
        is_last = epoch == cfg.train.epochs - 1
        do_val = is_last or ((epoch + 1) % val_every == 0)
        val_metrics = None

        if do_val:
            val_pred = _predict(net, dataset, va, dev)
            val_metrics = compute_metrics(val_pred, tgt[va], dataset.valid[va])
            hist.train_mse.append(train_mse)
            hist.val_mse.append(val_metrics["mse"])
            hist.val_r2.append(val_metrics["r2"])

            if val_metrics["r2"] > best_r2:
                best_r2 = val_metrics["r2"]
                best_epoch = epoch + 1
                epochs_without_gain = 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                epochs_without_gain += 1

            if patience and epochs_without_gain >= patience and not is_last:
                stopped_early = True

        if verbose and (epoch % log_every == 0 or is_last or stopped_early):
            vstr = (
                f"val_r2={val_metrics['r2']:.4f}  val_mse={val_metrics['mse']:.5f}"
                if val_metrics
                else "val skipped"
            )
            astr = (
                f"  aux_mse({cfg.model.aux_mode}){aux_str}"
                if aux_mse_epoch is not None
                else ""
            )
            sstr = (
                f"  safe_violation={safe_epoch:.5f}"
                if safe_epoch is not None
                else ""
            )
            print(
                f"  [train] ep{epoch + 1:>3}  train_mse={train_mse:.5f}  {vstr}"
                f"{astr}{sstr}  ({time.time() - t0:.0f}s)"
            )
        if stopped_early:
            if verbose:
                print(f"  [train] early stop at epoch {epoch + 1} (patience={patience})")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    final_val = compute_metrics(_predict(net, dataset, va, dev), tgt[va], dataset.valid[va])
    final_train = compute_metrics(_predict(net, dataset, tr, dev), tgt[tr], dataset.valid[tr])

    # per-k aux loss on the val split: how well h decodes the (recalled or
    # future) envelope -- the direct memory-utilisation readout
    aux_val_mse = None
    if net.aux_heads:
        aux_val_mse = {}
        with torch.no_grad():
            for k_str, head in net.aux_heads.items():
                error_sum, aux_count = 0., 0
                for i in range(0, len(va), 128):
                    x = dataset.batch(va[i : i + 128]).transpose(0, 1).to(dev)
                    y = tgt[va[i : i + 128]].transpose(0, 1).to(dev)
                    valid = dataset.valid[va[i : i + 128]].transpose(0, 1).to(dev)
                    _, h_seq = net.forward_with_aux(x)
                    count = int(aux_pair_mask(valid, int(k_str)).sum())
                    error_sum += float(aux_memory_loss(head(h_seq), y, int(k_str), cfg.model.aux_mode, valid)) * count
                    aux_count += count
                aux_val_mse[f"k{int(k_str)}"] = error_sum / aux_count if aux_count else None

    # val-set safety metrics: trigger rate (frames whose prediction violates)
    # and mean violation -- the collapse guard for the safety loss
    safe_val = None
    if safe_lambda > 0:
        trig, viol, n = 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va), 128):
                idx = va[i : i + 128]
                x = dataset.batch(idx).transpose(0, 1).to(dev)
                main_out = net(x)
                L_frames = main_out.shape[0]
                l, t, b_n = batch_safety_loss(
                    main_out.reshape(-1, 5),
                    dataset.heading[idx].transpose(0, 1).reshape(-1).to(dev),
                    dataset.pos[idx].transpose(0, 1).reshape(-1, 2).to(dev),
                    frame_map_ids(dataset.map_id[idx], L_frames).to(dev),
                    dfs,
                    valid=dataset.valid[idx].transpose(0, 1).reshape(-1).to(dev) if dataset.teacher["kind"] == "exact" else None,
                    return_count=True,
                )
                viol += float(l) * b_n
                trig += float(t) * b_n
                n += b_n
        safe_val = {"trigger_rate": trig / n if n else None, "mean_violation": viol / n if n else None, "valid_frames": n}

    return {
        "teacher": dataset.teacher,
        "net": net,
        "history": hist,
        "best_epoch": best_epoch,
        "best_val_r2": float(best_r2),
        "val": final_val,
        "train": final_train,
        "aux_val_mse": aux_val_mse,
        "safe_val": safe_val,
        "elapsed_s": time.time() - t0,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "stopped_early": stopped_early,
        "epochs_run": epoch + 1 if cfg.train.epochs else 0,
    }


def save_checkpoint(result: Dict, cfg: SLConfig, path: str, provenance=None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": result["net"].state_dict(),
            "config": provenance['config'] if provenance is not None else cfg.to_dict(),
            "provenance": provenance,
            "data_version": 3,
            "teacher": provenance.get("teacher", result["teacher"]) if provenance else result["teacher"],
            "cfg_model": cfg.model.__dict__,
            "best_epoch": result["best_epoch"],
            "best_val_r2": result["best_val_r2"],
            "val": result["val"],
        },
        path,
    )
    with open(os.path.splitext(path)[0] + "_metrics.json", "w") as f:
        json.dump(
            {
                "best_epoch": result["best_epoch"],
                "best_val_r2": result["best_val_r2"],
                "val": result["val"],
                # train metrics are needed to compute the generalisation gap
                "train": {
                    "r2": result["train"]["r2"],
                    "mse": result["train"]["mse"],
                    "r2_per_dim": result["train"]["r2_per_dim"],
                },
                "n_train": result["n_train"],
                "n_val": result["n_val"],
                "elapsed_s": result["elapsed_s"],
                "history": result["history"].__dict__,
                "aux_val_mse": result["aux_val_mse"],
                "safe_val": result["safe_val"],
            },
            f,
            indent=2,
        )
