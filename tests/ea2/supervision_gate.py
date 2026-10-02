"""Numerical SL parity, reset isolation, gradients, split and checkpoint gate."""
from pathlib import Path
import tempfile
import torch
from source_reference import load_reference
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.config import SLConfig, SLModelConfig
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.model import EnvelopeNet
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import MapData, SLDataset, build_windows
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.train import train, save_checkpoint, aux_memory_loss, batch_safety_loss, frame_map_ids, compute_metrics
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.evaluate import load_checkpoint, Predictor, hex_area, s_to_action, closed_loop_rollout
from hexapod_mjlab.tasks.envelope_adaptive_2.math_utils import map_actions_to_params, normalized_envelope_params


def main():
    torch.set_num_threads(1);torch.manual_seed(1);ref=load_reference()
    net=EnvelopeNet();gold=ref['sl_model'].EnvelopeNet()
    gold.load_state_dict(net.state_dict())
    x=torch.rand(35,3,190)
    assert torch.allclose(net(x),gold(x),atol=1e-6,rtol=0)
    h=None;out=[]
    for obs in x:
        p,h=net.step(obs[None],h);out.append(p)
    assert torch.allclose(net(x),torch.stack(out),atol=1e-6,rtol=0)
    predictor=Predictor(net);predictor(x[0]);predictor(x[1]);before=predictor.hidden.clone()
    predictor.reset(torch.tensor([False,True,False]))
    assert not predictor.hidden[:,1].any() and torch.equal(predictor.hidden[:,[0,2]],before[:,[0,2]])
    p=predictor(x[2]);independent,_=net.step(x[2,1:2][None])
    assert torch.allclose(p[1],independent[0].clamp(0,1),atol=1e-6,rtol=0)
    window=Predictor(net,"window",4)
    for obs in x[:4]:window(obs)
    window.reset(torch.tensor([False,True,False]))
    for i,obs in enumerate(x[4:8]):
        p=window(obs)
        if i<3:assert torch.equal(p[1],torch.full((5,),.5))
    assert torch.allclose(p[1],net(x[4:8,1:2])[-1,0].clamp(0,1),atol=1e-6,rtol=0)
    assert torch.allclose(p[0],net(x[4:8,0:1])[-1,0].clamp(0,1),atol=1e-6,rtol=0)
    # Physical area checked against the independent polygon shoelace formula.
    from hexapod_mjlab.tasks.envelope_adaptive_2.geometry import compute_hex_vertices
    from hexapod_mjlab.tasks.envelope_adaptive_2.sl.train import s_to_params
    s=torch.rand(7,5);params=s_to_params(s);v=compute_hex_vertices(*params.unbind(-1))
    shoelace=((v[...,0]*v.roll(-1,1)[...,1]-v[...,1]*v.roll(-1,1)[...,0]).sum(1).abs()*.5)
    assert torch.allclose(hex_area(s),shoelace,atol=1e-6,rtol=0)
    lo=torch.tensor([.3,.3,.3,.6,-.9]);hi=torch.tensor([.6,.7,.6,.9,-.6])
    realized=map_actions_to_params(s_to_action(s),lo,hi,.95,4.)
    assert torch.allclose(normalized_envelope_params(realized,lo,hi),s,atol=1e-6,rtol=0)
    # Deliberately distinct before, terminal and post-reset labels catch timing bugs.
    from types import SimpleNamespace
    class TimingEnv:
        device='cpu';num_envs=1
        def __init__(self):self.ea2=SimpleNamespace(oracle_s=torch.zeros(1,5));self.t=0
        def reset(self):self.t=0;self.ea2.oracle_s.zero_();return {'policy':torch.zeros(1,190)},{}
        def step(self,a):
            self.t+=1;terminal=torch.full((1,5),self.t/10)
            done=torch.tensor([self.t==3])
            self.ea2.oracle_s.fill_(.9 if done.item() else self.t/10)
            frame={'target':terminal,'collision_hard':torch.zeros(1)}
            return {'policy':torch.zeros(1,190)},torch.ones(1),done,torch.zeros(1,dtype=torch.bool),{'executed_frame':frame}
    metric=closed_loop_rollout(TimingEnv(),None,SLConfig(),num_steps=3,warmup_steps=0,
                               constant_s=torch.full((1,5),.5),device='cpu')
    for name,targets in [('same_frame_mse',[.1,.2,.9]),('executed_frame_mse',[.2,.3,.4]),('pred_target_mse',[.2,.9,.4])]:
        expected=((torch.tensor(targets)-.5)**2).mean().item()
        assert abs(getattr(metric,name)-expected)<1e-6,name
    expected=hex_area(torch.tensor([[.2]*5,[.3]*5,[.4]*5])).mean().item()
    assert abs(metric.oracle_area-expected)<1e-6
    import numpy as np
    target=torch.rand(30,3,5)*torch.tensor([1.,2.,3.,4.,5.])
    metric=compute_metrics(target,target)
    eigenvalues=np.linalg.eigvalsh(np.cov(target.reshape(-1,5).numpy().T))[::-1]
    assert np.allclose(metric['pc_var'],eigenvalues/eigenvalues.sum(),atol=1e-6)
    assert np.allclose(metric['legacy_pc_var'],ref['sl_train'].compute_metrics(target,target)['pc_var'],atol=1e-6)
    # All approved recall horizons have finite nonzero gradients through GRU and heads.
    auxnet=EnvelopeNet(SLModelConfig(aux_ks=[25,50,100,200,300]))
    y=torch.rand(400,2,5);main,h=auxnet.forward_with_aux(torch.rand(400,2,190))
    for k in auxnet.aux_ks:
        auxnet.zero_grad();loss=aux_memory_loss(auxnet.aux_heads[str(k)](h),y,k,"recall")
        loss.backward(retain_graph=True)
        for param in [auxnet.gru.weight_ih_l0,auxnet.aux_heads[str(k)].weight]:
            assert torch.isfinite(param.grad).all() and param.grad.abs().sum()>0
    # Different fields prove time-major flattened map ids match the actual frames.
    yy=-37+(torch.arange(740)+.5)*.1
    corridor=(.55-yy.abs()).clamp_min(0)[:,None].expand(740,740).contiguous()
    dfs=torch.stack([torch.full_like(corridor,5.),corridor])
    ids=frame_map_ids(torch.tensor([0,1]),3)
    pred=torch.full((3,2,5),.55,requires_grad=True)
    loss,rate=batch_safety_loss(pred.reshape(-1,5),torch.zeros(6),torch.zeros(6,2),ids,dfs)
    loss.backward()
    assert loss>0 and rate>0 and pred.grad[:,1].abs().sum()>0 and not pred.grad[:,0].any()
    assert torch.isfinite(pred.grad).all()
    for field in [torch.full_like(corridor,5.),torch.zeros_like(corridor)]:
        q=torch.full((2,5),.55,requires_grad=True)
        loss,_=batch_safety_loss(q,torch.zeros(2),torch.zeros(2,2),torch.zeros(2,dtype=torch.long),field[None])
        loss.backward();assert loss==0 and not q.grad.any()
    # Synthetic data checks exact original window/quantisation/split semantics.
    obs=torch.rand(50,6,190);target=torch.rand(50,6,5);done=torch.zeros(50,6,dtype=torch.bool);done[20,0]=True
    heading=torch.zeros(50,6);pos=torch.zeros(50,6,2)
    a=build_windows(obs,target,done,heading,pos,12,2,3)
    b=ref['sl_dataset'].build_windows(obs,target,done,heading,pos,12,2,3)
    for key in b:assert torch.equal(a[key],b[key]),key
    assert a["valid"].all()
    cfg=SLConfig();cfg.train.seq_len=12;cfg.train.window_stride=3;cfg.train.epochs=1;cfg.data.warmup_steps=2
    data=MapData(1,obs,target,done,heading,pos,corridor,{})
    dataset=SLDataset(cfg,[data])
    original=ref['sl_dataset'].SLDataset(cfg,[data])
    for name in ['obs','target','train_idx','val_idx','env_id','map_id']:
        assert torch.equal(getattr(dataset,name),getattr(original,name)),name
    assert not set(dataset.env_id[dataset.train_idx].tolist()) & set(dataset.env_id[dataset.val_idx].tolist())
    result=train(cfg,dataset,device="cpu",verbose=False)
    assert torch.isfinite(torch.tensor(result['val']['mse']))
    with tempfile.TemporaryDirectory() as directory:
        path=str(Path(directory)/'model.pt');save_checkpoint(result,cfg,path)
        loaded,_=load_checkpoint(path)
        assert torch.allclose(result['net'](x),loaded(x),atol=1e-6,rtol=0)
    print('T3 CPU PASS: source model/windows/split parity, sequence-stateful, reset isolation, area/mapping, all aux horizons, multi-map safety/floor masks, optimiser and checkpoint')


if __name__=='__main__':main()
