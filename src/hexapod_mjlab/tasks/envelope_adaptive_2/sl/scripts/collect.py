"""Collect real EA2 zero-action trajectories on fixed maps."""
import argparse
from pathlib import Path
from ..config import SLConfig, SL_DATA_DIR
from ..dataset import collect_map, save_map


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds",default="1,7,13")
    ap.add_argument("--pillar-counts",default=None,help="comma-separated counts; '-' keeps original default")
    ap.add_argument("--num-envs",type=int,default=96)
    ap.add_argument("--num-steps",type=int,default=SLConfig().data.num_steps)
    ap.add_argument("--out-dir",type=Path,default=Path(SL_DATA_DIR))
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--attitude",action="store_true")
    args=ap.parse_args()
    seeds=[int(s) for s in args.seeds.split(',')]
    counts=args.pillar_counts.split(',') if args.pillar_counts else ['-']*len(seeds)
    if len(counts)==1:counts*=len(seeds)
    if len(counts)!=len(seeds):ap.error("pillar counts must match seeds")
    for seed,count in zip(seeds,counts):
        data=collect_map(seed,args.num_envs,args.num_steps,pillar_count=None if count=='-' else int(count),
                         device=args.device,attitude=args.attitude)
        path=args.out_dir/f'map_seed{seed}.pt';save_map(data,str(path))
        print(f"COLLECT seed={seed} shape={tuple(data.obs.shape)} -> {path}",flush=True)


if __name__=='__main__':main()
