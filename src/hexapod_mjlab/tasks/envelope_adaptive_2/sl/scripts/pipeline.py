"""Reproducible SL-only pipeline; --small runs the approved migration acceptance."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from ..config import SLConfig
from ..dataset import load_map,SLDataset

PREFIX='hexapod_mjlab.tasks.envelope_adaptive_2.sl.scripts.'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--small',action='store_true')
    ap.add_argument('--root',type=Path,default=Path.cwd()/'logs/ea2')
    ap.add_argument('--run-name',default='migration-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--play',action='store_true',help='finish with a ten-step native viewer smoke')
    args=ap.parse_args();torch.set_num_threads(1)
    cfg=SLConfig()
    if args.small:
        cfg.data.seeds=[1,7];cfg.data.pillar_counts=[None,None]
        cfg.data.num_envs=8;cfg.data.num_steps=800;cfg.train.epochs=5
    data_dir=(args.root/'data'/args.run_name).resolve()
    evidence=(args.root/'acceptance'/args.run_name).resolve();evidence.mkdir(parents=True,exist_ok=True)
    status={'small':args.small,'config':cfg.to_dict(),'data_dir':str(data_dir),'stages':[]}
    status_path=evidence/'pipeline.json'
    def save():status_path.write_text(json.dumps(status,indent=2)+'\n')
    def run(stage,module,options):
        entry={'stage':stage,'state':'running','command':[sys.executable,'-m',PREFIX+module,*map(str,options)]}
        status['stages'].append(entry);save();print('START '+stage,flush=True)
        with (evidence/(stage+'.log')).open('w') as stream:
            try:subprocess.run(entry['command'],stdout=stream,stderr=subprocess.STDOUT,check=True)
            except Exception:
                entry['state']='failed';save();raise
        entry['state']='passed';save();print('PASS '+stage,flush=True)
    try:
        for seed,count in cfg.data.seed_specs():
            options=['--seeds',seed,'--num-envs',cfg.data.num_envs,'--num-steps',cfg.data.num_steps,
                     '--out-dir',data_dir,'--device',args.device]
            if count is not None:options+=['--pillar-counts',count]
            run(f'collect_seed{seed}','collect',options)
        paths=[data_dir/f'map_seed{s}.pt' for s in cfg.data.seeds]
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        maps=[load_map(str(p)) for p in paths]
        targets=torch.cat([m.target[cfg.data.warmup_steps:][~m.done[cfg.data.warmup_steps:]] for m in maps])
        std=targets.std(0);contracted=(targets<=.8).any(-1).float().mean()
        if not torch.isfinite(targets).all() or float(std.max())<1e-3 or float(contracted)<.01:
            raise AssertionError(f'Data quality gate failed: std={std.tolist()}, contraction={float(contracted)}')
        dataset=SLDataset(cfg,maps)
        if not len(dataset.train_idx) or not len(dataset.val_idx):raise AssertionError('Empty train/val split')
        status['data_sha256']=hashes;status['data_quality']={'target_std':std.tolist(),'contracted_frame_fraction':float(contracted),'windows':len(dataset),'train_windows':len(dataset.train_idx),'val_windows':len(dataset.val_idx)};save()
        seed_arg=','.join(map(str,cfg.data.seeds));run_dirs={}
        for recipe in ['baseline','aux_safe']:
            out=(args.root/'runs'/f'{args.run_name}-{recipe}').resolve();run_dirs[recipe]=out
            if {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}!=hashes:
                raise AssertionError('Frozen data changed')
            options=['--data-dir',data_dir,'--seeds',seed_arg,'--out-dir',out,
                     '--epochs',cfg.train.epochs,'--seq-len',cfg.train.seq_len,
                     '--window-stride',cfg.train.window_stride,'--batch-size',cfg.train.batch_size,
                     '--lr',cfg.train.learning_rate,'--seed',1,'--device',args.device]
            if recipe=='aux_safe':options+=['--aux-ks','25,50,100,200,300','--aux-mode','recall','--aux-beta',.5,'--safe-lambda',1.]
            run(f'train_{recipe}','train',options)
            metrics=json.loads((out/'model_metrics.json').read_text())
            mse=metrics['history']['train_mse'];improvement=(mse[0]-mse[-1])/mse[0]
            if len(mse)!=cfg.train.epochs or not all(torch.isfinite(torch.tensor(mse))) or improvement<.05:
                raise AssertionError(f'{recipe} learning gate failed: history={mse}')
            if not torch.isfinite(torch.tensor(metrics['val']['mse'])):
                raise AssertionError('Nonfinite validation MSE')
            status.setdefault('learning',{})[recipe]={'mse_history':mse,'relative_decrease':improvement,
                                                    'validation_mse':metrics['val']['mse'],'validation_r2':metrics['val']['r2']};save()
            run(f'eval_{recipe}','eval',[out/'model.pt','--seeds',seed_arg,'--num-envs',cfg.data.num_envs,
                                        '--num-steps',700,'--data-dir',data_dir,'--device',args.device])
        if args.play:
            run('play_baseline','play',[run_dirs['baseline']/'model.pt','--num-envs',1,'--steps',10,
                                        '--snapshot',evidence/'trained_play.png','--device',args.device])
        run('report','report',['--root',args.root,'--run-name',args.run_name])
        status['result']='passed';save()
        print(f'PIPELINE PASS {status_path}',flush=True)
    except Exception as error:
        status['result']='failed';status['error']=repr(error);save()
        print(f'PIPELINE FAILED; preserved evidence: {status_path}',flush=True)
        raise


if __name__=='__main__':main()
