"""Short real collection smoke; full 400-frame learning happens at T5."""
import torch
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import collect_map,save_map,load_map
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.evaluate import closed_loop_rollout
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.config import SLConfig
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.model import EnvelopeNet
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import build_env
from pathlib import Path
import tempfile


def main():
    torch.set_num_threads(1)
    data=collect_map(1,num_envs=2,num_steps=37)
    assert data.obs.shape==(37,2,190) and data.target.shape==(37,2,5)
    assert torch.isfinite(data.obs).all() and torch.isfinite(data.target).all()
    assert ((data.target>=0)&(data.target<=1)).all()
    with tempfile.TemporaryDirectory() as directory:
        p=str(Path(directory)/'data.pt');save_map(data,p);loaded=load_map(p)
        assert torch.equal(loaded.obs,data.obs) and torch.equal(loaded.target,data.target)
    env=build_env(1,2)
    try:
        net=EnvelopeNet().to(env.device)
        result=closed_loop_rollout(env,net,SLConfig(),num_steps=36)
        assert result.n_steps==6
        for name in ['step_reward','same_frame_mse','executed_frame_mse','pred_target_mse','area','oracle_area']:
            assert torch.isfinite(torch.tensor(getattr(result,name))),name
    finally:env.close()
    print('T3 GPU PASS: real same-frame data collection, save/load, rollout metrics')


if __name__=='__main__':main()
