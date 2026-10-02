"""Frozen release data, two training recipes, actual rollouts and inference export."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import load_map,SLDataset
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.config import SLConfig
from hexapod_mjlab.tasks.envelope_adaptive_2.teachers.batch import teacher_identity
from hexapod_mjlab.tasks.envelope_adaptive_2.map_generator import generate_map
from hexapod_mjlab.tasks.envelope_adaptive_2 import contracts
from whole_polygon_reference import WholePolygonChecker


def main():
    torch.set_num_threads(1)
    root=Path('logs/ea2'); evidence=root/'acceptance/ea2-exact-release-20261003'
    evidence.mkdir(parents=True,exist_ok=True)
    status_path=evidence/'pipeline.json'
    if status_path.exists():raise FileExistsError(status_path)
    data_dir=root/'data/ea2-exact-release-20261003'
    paths=[data_dir/f'map_seed{s}.pt' for s in [1,7]]
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    maps=[load_map(str(p)) for p in paths]
    status={'teacher':teacher_identity(),'data_sha256':hashes,'stages':[],'device':'cuda:0',
            'protocol':'two maps, 800 x 4 raw frames/map; 5 epochs, seq400, stride30, batch8; seed1; 450 x4 real rollout steps/mode/map'}
    def save():status_path.write_text(json.dumps(status,indent=2,allow_nan=False)+'\n')
    save()
    try:
        data_records=[]
        for data in maps:
            assert data.meta['teacher']==teacher_identity()
            assert len(data.teacher_status)==len(data.obs) and len(data.teacher_reason)==len(data.obs)
            config=data.meta['environment_config']
            geom=generate_map(contracts.MapGenCfg(**config['map']),contracts.PillarFieldCfg(**{k.removeprefix('pillar_'):v for k,v in config['obstacles'].items()}),data.seed)
            checker=WholePolygonChecker(geom.rects)
            from hexapod_mjlab.tasks.envelope_adaptive_2.sl.losses import s_to_params
            params=s_to_params(data.target).numpy()
            for t,e in data.geometric_valid.nonzero().tolist():
                assert checker.clearance(params[t,e],data.pos[t,e].numpy(),float(data.heading[t,e]))>=.1-1e-6
            assert not (data.supervision_valid & (~data.geometric_valid | data.emergency)).any()
            assert not (data.supervision_valid & data.rate_checked & ~data.rate_valid).any()
            data_records.append({'seed':data.seed,'total_frames':data.done.numel(),'geometric_frames':int(data.geometric_valid.sum()),'supervision_frames':int(data.supervision_valid.sum()),'emergency_frames':int(data.emergency.sum()),'done_frames':int(data.done.sum())})
        status['data_correctness']=data_records
        cfg=SLConfig();cfg.train.seq_len=400;cfg.train.window_stride=30
        ds=SLDataset(cfg,maps);status['dataset']=ds.summary();save()
        def run(stage,module,options):
            entry={'stage':stage,'state':'running','command':[sys.executable,'-m','hexapod_mjlab.tasks.envelope_adaptive_2.sl.scripts.'+module,*map(str,options)]}
            status['stages'].append(entry);save()
            with (evidence/(stage+'.log')).open('w') as stream:
                subprocess.run(entry['command'],stdout=stream,stderr=subprocess.STDOUT,check=True)
            entry['state']='passed';save();print('PASS',stage,flush=True)
        for recipe in ['baseline','aux_safe']:
            assert hashes=={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            output=root/'runs'/f'exact-release-20261003-{recipe}'
            options=['--data-dir',data_dir,'--out-dir',output,'--epochs',5,'--batch-size',8,'--seq-len',400,'--window-stride',30]
            if recipe=='aux_safe':options+=['--aux-ks','25,50,100,200,300','--safe-lambda',.1]
            run('train_'+recipe,'train',options)
            metric=json.loads((output/'model_metrics.json').read_text())
            assert len(metric['history']['train_mse'])==5 and all(torch.isfinite(torch.tensor(metric['history']['train_mse'])))
            assert metric['val']['valid_frames']>0
            status.setdefault('learning',{})[recipe]={'validation_mse':metric['val']['mse'],'validation_r2':metric['val']['r2'],'training_mse':metric['history']['train_mse']};save()
            run('eval_'+recipe,'eval',[output/'model.pt','--seeds','1,7','--num-envs',4,'--num-steps',450,'--data-dir',data_dir])
            run('export_'+recipe,'export',[output/'model.pt'])
        run('play_baseline','play',[root/'runs/exact-release-20261003-baseline/model.pt','--steps',10,'--num-envs',1,'--snapshot',evidence/'trained_play.png'])
        status['result']='passed';save();print('RELEASE PIPELINE PASS',status_path,flush=True)
    except Exception as error:
        status['result']='failed';status['error']=repr(error);save();raise


if __name__=='__main__':main()
