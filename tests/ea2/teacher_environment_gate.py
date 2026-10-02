"""Real mjlab teacher timing, reset isolation and exact geometry checks."""
import json
from pathlib import Path
import numpy as np
import torch
from hexapod_mjlab.tasks.envelope_adaptive_2.config.parameters import EA2Cfg
from hexapod_mjlab.tasks.envelope_adaptive_2.env import make_env
from hexapod_mjlab.tasks.envelope_adaptive_2.teachers.batch import BatchTeacher
from whole_polygon_reference import WholePolygonChecker


def main():
    torch.set_num_threads(1)
    cfg=EA2Cfg();cfg.env.num_envs=2;cfg.path.path_plan_workers=0
    env=make_env(cfg)
    times=[];valid=emergency=total=0
    try:
        env.reset();s=env.ea2;mirror=BatchTeacher(s.map_data.rects)
        checker=WholePolygonChecker(s.map_data.rects)
        def initial(ids):
            mirror.reset(ids)
            idx=torch.tensor(ids,device=env.device)
            r=mirror.evaluate(s.base_pos[idx,:2],s.heading[idx],s.ego_motion[idx],ids)
            assert torch.allclose(r['s'],s.oracle_s[idx],atol=1e-6,rtol=0)
            assert not s.oracle_rate_checked[idx].any()
        initial([0,1])
        zero=torch.zeros((2,5),device=env.device)
        for t in range(100):
            prev={k:v.copy() for k,v in s._teacher.oracle.prev.items()}
            counter={k:v.copy() for k,v in s._teacher.oracle.counter.items()}
            env.get_observations();env.get_observations()
            assert prev.keys()==s._teacher.oracle.prev.keys()
            for key in prev:assert np.array_equal(prev[key],s._teacher.oracle.prev[key])
            for key in counter:assert np.array_equal(counter[key],s._teacher.oracle.counter[key])
            _,_,terminated,truncated,info=env.step(zero)
            frame=info['executed_frame']
            ego=frame['observation'][:,-3:]*torch.tensor([1.5,1.,1.5],device=env.device)
            result=mirror.evaluate(frame['pos'],frame['heading'],ego)
            assert torch.allclose(result['s'],frame['target'],atol=1e-5,rtol=0)
            for name in ['geometric_valid','supervision_valid','emergency','rate_checked','rate_valid']:
                assert torch.equal(result[name],frame[name]),(t,name)
            for e in range(2):
                if frame['geometric_valid'][e]:
                    assert checker.clearance(result['params'][e].cpu().numpy(),frame['pos'][e].cpu().numpy(),float(frame['heading'][e]))>=.1-1e-6
            valid+=int(frame['supervision_valid'].sum());emergency+=int(frame['emergency'].sum());total+=2
            times.append(s.teacher_elapsed_s)
            done=(terminated|truncated).nonzero().flatten().tolist()
            if done:initial(done)
            if t==10:
                other=s._teacher.oracle.prev[1].copy()
                env.reset(env_ids=torch.tensor([0],device=env.device))
                assert np.array_equal(other,s._teacher.oracle.prev[1])
                initial([0])
            if t==40:env.cfg.episode_length_s=.1;env.reset();initial([0,1])
        report={'frames':total,'supervision_frames':valid,'emergency_frames':emergency,
                'mean_teacher_batch_s':float(np.mean(times)), 'p95_teacher_batch_s':float(np.quantile(times,.95)),
                'num_envs':2,'cached_reads':'pass','partial_reset':'pass','online_offline':'pass','whole_polygon':'pass'}
        path=Path('logs/ea2/oracle_refinement/20261003/production_environment_gate.json')
        path.write_text(json.dumps(report,indent=2)+'\n')
        print('TEACHER GPU PASS',json.dumps(report),flush=True)
    finally:env.close()


if __name__=='__main__':main()
