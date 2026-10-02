"""Validity, temporal endpoints, gradient isolation and legacy compatibility."""
from pathlib import Path
import tempfile
import numpy as np
import torch
from hexapod_mjlab.tasks.envelope_adaptive_2.contracts import RectPrimitive
from hexapod_mjlab.tasks.envelope_adaptive_2.teachers.batch import BatchTeacher, teacher_identity
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.losses import masked_mse, aux_memory_loss, batch_safety_loss
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import MapData, save_map, load_map, build_windows, SLDataset
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.train import compute_metrics, train, save_checkpoint
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.config import SLConfig
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.evaluate import checkpoint_teacher, load_checkpoint, closed_loop_rollout, hex_area
from whole_polygon_reference import WholePolygonChecker


def main():
    torch.manual_seed(2)
    valid = torch.rand(400, 2) > .2
    target = torch.rand(400, 2, 5)
    bad = target.clone(); bad[~valid] = float('nan')
    prediction = torch.rand_like(target, requires_grad=True)
    for mode in ['main', 'recall', 'forward']:
        for k in ([0] if mode == 'main' else [25, 50, 100, 200, 300]):
            loss = lambda y: masked_mse(prediction, y, valid) if mode == 'main' else aux_memory_loss(prediction, y, k, mode, valid)
            a = loss(target); ga = torch.autograd.grad(a, prediction)[0]
            b = loss(bad); gb = torch.autograd.grad(b, prediction)[0]
            assert torch.equal(a, b) and torch.equal(ga, gb) and ga.abs().sum() > 0
            if mode == 'main': assert not ga[~valid].any()
    q = torch.full((2,5),float('nan'),requires_grad=True)
    loss, _, count = batch_safety_loss(q,torch.zeros(2),torch.zeros(2,2),torch.zeros(2,dtype=torch.long),torch.ones(1,740,740),valid=torch.zeros(2,dtype=torch.bool),return_count=True)
    assert loss == 0 and count == 0 and not torch.autograd.grad(loss,q)[0].any()
    empty = masked_mse(prediction, bad, torch.zeros_like(valid))
    assert empty == 0 and not torch.autograd.grad(empty, prediction)[0].any()
    a = compute_metrics(prediction.detach().transpose(0,1), target.transpose(0,1), valid.T)
    b = compute_metrics(prediction.detach().transpose(0,1), bad.transpose(0,1), valid.T)
    for key in ['mse', 'r2', 'temporal_std_oracle', 'pc_var']: assert a[key] == b[key]
    # Invalid frame remains in the sequence; windows never join across reset.
    obs=torch.rand(40,2,190); y=torch.rand(40,2,5)
    done=torch.zeros(40,2,dtype=torch.bool); done[20,0]=True
    mask=torch.ones_like(done); mask[10,1]=False
    w=build_windows(obs,y,done,torch.zeros(40,2),torch.zeros(40,2,2),12,1,1,mask)
    assert (~w['valid']).any()
    for i,e in enumerate(w['env_id']):
        start = next(t for t in range(40) if torch.equal(obs[t,e],w['obs'][i,0]))
        assert not done[start:start+12,e].any()
    data=MapData(1,obs,y,done,torch.zeros(40,2),torch.zeros(40,2,2),torch.ones(740,740),{},mask)
    with tempfile.TemporaryDirectory() as tmp:
        path=str(Path(tmp)/'map.pt'); save_map(data,path); saved=load_map(path)
        assert torch.equal(saved.supervision_valid,mask)
        raw=torch.load(path,weights_only=True)
        for key in ['supervision_valid','geometric_valid','emergency','rate_checked','rate_valid']:raw.pop(key)
        torch.save(raw,path); legacy=load_map(path)
        assert legacy.supervision_valid.all() and not legacy.geometric_valid.any()
        raw['meta']={'teacher':teacher_identity()};torch.save(raw,path)
        try:load_map(path)
        except ValueError:pass
        else:raise AssertionError('Missing exact certificates accepted')
    assert checkpoint_teacher({})['kind']=='legacy'
    assert checkpoint_teacher({'teacher':teacher_identity()})['kind']=='exact'
    # Direct public API must not relabel a newly trained model as legacy.
    cfg=SLConfig();cfg.train.epochs=1;cfg.train.seq_len=12;cfg.train.window_stride=5;cfg.data.warmup_steps=1
    data.meta={'teacher':teacher_identity()}
    data.geometric_valid=torch.ones_like(done)
    data.target[~mask]=float('nan')
    ds=SLDataset(cfg,[data]);result=train(cfg,ds,device='cpu',verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint=str(Path(tmp)/'model.pt');save_checkpoint(result,cfg,checkpoint)
        _,meta=load_checkpoint(checkpoint)
        assert checkpoint_teacher(meta)==teacher_identity()
    # Area pairs must share the geometry mask, including emergency frames.
    from types import SimpleNamespace
    class AreaEnv:
        device='cpu';num_envs=1
        def __init__(self):
            self.ea2=SimpleNamespace(oracle_s=torch.zeros(1,5),oracle_supervision_valid=torch.ones(1,dtype=torch.bool),_teacher=object())
        def reset(self):self.t=0;return {'policy':torch.zeros(1,190)},{}
        def step(self,action):
            self.t+=1;target=torch.full((1,5),self.t/10)
            self.ea2.oracle_s=target
            valid=torch.tensor([self.t%2==0]);self.ea2.oracle_supervision_valid=valid
            frame={'target':target,'collision_hard':torch.zeros(1),'geometric_valid':valid,'supervision_valid':valid,'emergency':torch.zeros(1,dtype=torch.bool)}
            return {'policy':torch.zeros(1,190)},torch.zeros(1),torch.zeros(1,dtype=torch.bool),torch.zeros(1,dtype=torch.bool),{'executed_frame':frame}
    rollout=closed_loop_rollout(AreaEnv(),None,SLConfig(),num_steps=3,warmup_steps=0,constant_s=torch.full((1,5),.5),device='cpu')
    assert rollout.paired_area_frames==2 and rollout.total_frames==3
    assert abs(rollout.area-float(hex_area(torch.full((1,5),.5))))<1e-6
    assert abs(rollout.oracle_area-float(hex_area(torch.tensor([[.2]*5,[.4]*5])).mean()))<1e-6
    # Pure frontal wall must not needlessly narrow middle/back/front width.
    wall=[RectPrimitive((1.03,0),(.6,4))]
    teacher=BatchTeacher(wall)
    pos=torch.zeros(2,2);yaw=torch.zeros(2);ego=torch.zeros(2,3)
    out=teacher.evaluate(pos,yaw,ego)
    expected=np.array([.975,.975,.975,.1,.975])
    assert np.max(abs(out['s'][0].numpy()-expected))<1e-4
    assert out['geometric_valid'].all() and out['supervision_valid'].all()
    assert not out['rate_checked'].any()  # initialization is shape-only
    checker=WholePolygonChecker(wall)
    assert checker.clearance(out['params'][0].numpy(),[0,0],0)>=.1-1e-6
    other=teacher.oracle.prev[1].copy();teacher.reset([0])
    assert np.array_equal(other,teacher.oracle.prev[1]) and 0 not in teacher.oracle.prev
    out=teacher.evaluate(pos,yaw,ego)
    assert out['rate_checked'].tolist()==[False,True] and out['rate_valid'].tolist()==[False,True]
    narrow=[RectPrimitive((0,z),(8,.6)) for z in [-.715,.715]]
    narrow_teacher=BatchTeacher(narrow)
    narrow_out=narrow_teacher.evaluate(pos[:1],yaw[:1],ego[:1])
    expected=np.array([.05,.0375,.05,.05+(.975-.05)/.15*.05,.05+(.975-.05)/.15*.05])
    assert np.max(abs(narrow_out['s'][0].numpy()-expected))<1e-4
    assert WholePolygonChecker(narrow).clearance(narrow_out['params'][0].numpy(),[0,0],0)>=.1-1e-6
    blocked=BatchTeacher([RectPrimitive((0,z),(8,.6)) for z in [-.65,.65]])
    out=blocked.evaluate(pos[:1],yaw[:1],ego[:1])
    assert not out['geometric_valid'].any() and not out['supervision_valid'].any()
    assert torch.isfinite(out['s']).all() and not blocked.oracle.prev
    try: teacher.evaluate(torch.full_like(pos,float('nan')),yaw,ego)
    except ValueError: pass
    else: raise AssertionError('Nonfinite pose certified')
    print('TEACHER CPU PASS: masks/gradients, five aux horizons both directions, resets, legacy compatibility, frontal shape')


if __name__=='__main__':main()
