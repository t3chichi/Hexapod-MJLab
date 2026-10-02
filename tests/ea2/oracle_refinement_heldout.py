"""Fixed-candidate audit on independently collected, previously unused maps."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
import oracle_refinement_compare as compare
from hexapod_mjlab.tasks.envelope_adaptive_2 import contracts
from hexapod_mjlab.tasks.envelope_adaptive_2.map_generator import generate_map
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import load_map
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.static_qp import StaticOracle,MIN
from whole_polygon_reference import WholePolygonChecker


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,nargs='+',default=[89,101])
    ap.add_argument('--data-dir',type=Path,default=Path('logs/ea2/data/ea2-oracle-final-heldout-20261003'))
    ap.add_argument('--source-dir',type=Path,default=Path('logs/ea2/oracle_refinement/heldout_source_final_20261003'))
    ap.add_argument('--tag',default='heldout_axial015_v19')
    args=ap.parse_args()
    torch.set_num_threads(1)
    compare.DATA=args.data_dir
    compare.SOURCE=args.source_dir
    compare.SOURCE.mkdir(parents=True,exist_ok=True)
    provenance={'protocol':'candidate one-way axial reserve .15 selected using map1 and analytic direction/shape requirements; prior variants retained; final config fixed before this replay, not tuned using seeds89/101','selected_config':{'horizon_s':.12,'headroom':.15,'floor_cap':.05,'yaw_reserve_rad':0.,'weight':4.,'cooldown_s':.1},'files':{},'source_code':{}}
    for p in [Path(__file__),Path(compare.__file__),*[Path('src/hexapod_mjlab/tasks/envelope_adaptive_2/experiments')/f for f in ['static_qp.py','refined_oracle.py','lookahead_oracle.py']],*Path('src/hexapod_mjlab/tasks/envelope_adaptive_2/teachers').glob('*.py')]:
        h=hashlib.sha256(p.read_bytes()).hexdigest();provenance['source_code'][str(p)]=h
        (compare.OUT/'source_snapshots'/f'{h}.py').write_bytes(p.read_bytes())
    compare.RUN_PROVENANCE=provenance['source_code']
    for seed in args.seeds:
        path=compare.DATA/f'map_seed{seed}.pt';provenance['files'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        data=load_map(str(path));cfg=data.meta['environment_config'];valid=~data.done.numpy();valid[:30]=False
        time,env=np.where(valid);pos=data.pos.numpy()[time,env].astype(float);heading=data.heading.numpy()[time,env].astype(float)
        geometry=generate_map(contracts.MapGenCfg(**cfg['map']),contracts.PillarFieldCfg(**{k.removeprefix('pillar_'):v for k,v in cfg['obstacles'].items()}),seed)
        checker=WholePolygonChecker(geometry.rects)
        base={'time':time,'env':env,'pos':pos,'heading':heading,'minimum_clearance':np.array([checker.clearance(MIN,p,h) for p,h in zip(pos,heading)])}
        for margin in [.1,.2]:
            solver=StaticOracle(geometry.rects,margin);s=np.full((len(time),5),np.nan)
            for i,(p,h) in enumerate(zip(pos,heading)):
                result=solver.solve(p,h)
                if result.status=='feasible':s[i]=result.s
            base[f'new_{margin:.2f}_s']=s
        out=compare.SOURCE/f'map_seed{seed}.npz'
        if out.exists():
            saved=dict(np.load(out))
            assert set(saved)==set(base) and all(np.allclose(saved[k],base[k],equal_nan=True) for k in base),'baseline changed'
        else:np.savez_compressed(out,**base)
        (compare.SOURCE/f'provenance_{args.tag}.json').write_text(json.dumps(provenance,indent=2)+'\n')
        for margin in [.1,.2]:compare.run(seed,margin,'lookahead',4.,1,args.tag,.12,0.,False,0.,.1,0.,.15)
    print('HELDOUT FINISHED',json.dumps(provenance))


if __name__=='__main__':main()
