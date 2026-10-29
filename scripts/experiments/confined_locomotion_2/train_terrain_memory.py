"""Explicit local Terrain Memory training; defaults do not authorize a budget."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--num-envs',type=int,required=True)
    parser.add_argument('--iterations',type=int,required=True)
    parser.add_argument('--height-loss-weight',type=float,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--log-root',default='logs/confined_locomotion_2')
    parser.add_argument('--engineering-validation',action='store_true')
    parser.add_argument('--execution-mode',choices=('low_memory','speed'),default='low_memory',
                        help='low_memory recomputes point-GRU activations; speed retains them')
    parser.add_argument('--point-replay-chunk',type=int,default=1024,
                        help='positive point-GRU replay scan batch cap (default 1024; try 256/128 for less VRAM)')
    parser.add_argument('--resume-checkpoint')
    parser.add_argument('--save-interval',type=int,default=50)
    parser.add_argument('--raycast-mode',choices=('full','sparse'),default='sparse')
    parser.add_argument('--far-sampling',choices=('all_endpoints','hit_only_10m'),default='hit_only_10m')
    parser.add_argument('--sparse-selection','--sparse-selection-path',dest='sparse_selection_path')
    args=vars(parser.parse_args());args['gpu_ids']=(args.pop('gpu'),);args['max_iterations']=args.pop('iterations')
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.terrain_entry import launch_terrain_training
    result=launch_terrain_training(**args)
    print(f'Checkpoint: {result.checkpoint_path}')
    print('Engineering completion does not establish terrain memory or locomotion performance.')


if __name__=='__main__':main()
