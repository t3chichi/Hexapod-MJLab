"""Reload a checkpoint in a fresh process and evaluate real mjlab rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from ..config import SLConfig,SLModelConfig,SLDataConfig,SLTrainConfig,SL_DATA_DIR
from ..dataset import load_map
from ..evaluate import load_checkpoint,evaluate_closed_loop


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint',type=Path)
    ap.add_argument('--seeds',default='1')
    ap.add_argument('--num-envs',type=int,default=64)
    ap.add_argument('--num-steps',type=int,default=700)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--data-dir',type=Path,default=Path(SL_DATA_DIR))
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args();torch.set_num_threads(1)
    net,meta=load_checkpoint(str(args.checkpoint),args.device)
    c=meta.get('config',SLConfig().to_dict())
    cfg=SLConfig(SLDataConfig(**c['data']),SLModelConfig(**c['model']),SLTrainConfig(**c['train']))
    results={}
    probe_path=args.checkpoint.parent/'reload_probe.pt'
    if probe_path.exists():
        probe=torch.load(probe_path,weights_only=True)
        with torch.no_grad():actual=net(probe['observation'].to(args.device)).cpu()
        error=float((actual-probe['prediction']).abs().max())
        if error>1e-6:raise AssertionError(f'Checkpoint reload error {error}')
        results['reload_max_error']=error
    for seed in [int(s) for s in args.seeds.split(',')]:
        path=args.data_dir/f'map_seed{seed}.pt'
        # Frozen-data mean avoids adding a second collection budget for this baseline.
        mean=load_map(str(path)).target[cfg.data.warmup_steps:].mean((0,1)) if path.exists() else None
        results[f'map_seed{seed}']={
            'constant_baseline':{'source':'frozen_collected_data' if mean is not None else 'live_oracle_rollout',
                                 'mean':mean.tolist() if mean is not None else None,
                                 'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                                 'warmup_steps':cfg.data.warmup_steps},
            'rollouts':evaluate_closed_loop(cfg,net,seed,args.num_envs,args.num_steps,
                                            device=args.device,oracle_mean=mean)}
        print(f'EVAL seed={seed} completed',flush=True)
    out=args.out or args.checkpoint.parent/'rollout_metrics.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(f'EVAL SAVED {out}',flush=True)


if __name__=='__main__':main()
