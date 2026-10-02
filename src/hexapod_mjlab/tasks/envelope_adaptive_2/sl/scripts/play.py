"""Interactive native mjlab viewer for an independent SL checkpoint."""
import argparse
from pathlib import Path
import torch
from mjlab.viewer.native.viewer import NativeMujocoViewer
from ...config.parameters import EA2Cfg
from ...env import make_env,SLAdapter
from ..evaluate import load_checkpoint,Predictor,s_to_action


class PlaySession:
    """Viewer protocol adapter also clears prediction memory on reset/done."""
    def __init__(self,env,net,mode="stateful",seq_len=400):
        self.env,self.predictor=env,Predictor(net,mode,seq_len)

    @property
    def unwrapped(self):return self.env
    @property
    def cfg(self):return self.env.cfg
    @property
    def device(self):return self.env.device
    @property
    def num_envs(self):return self.env.num_envs

    def reset(self):
        self.predictor.reset()
        return {"policy":SLAdapter(self.env).reset()},{}

    def step(self,action):
        out=self.env.step(action)
        self.predictor.reset(out[2]|out[3])
        return out

    def get_observations(self):return self.env.get_observations()
    def policy(self,obs):return s_to_action(self.predictor(obs['policy']))
    def close(self):self.env.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint',type=Path)
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--num-envs',type=int,default=1)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--mode',choices=['stateful','window'],default='stateful')
    ap.add_argument('--full-lidar',action='store_true')
    ap.add_argument('--point-stride',type=int,default=1)
    ap.add_argument('--steps',type=int,default=None,help='optional control-step limit')
    ap.add_argument('--snapshot',type=Path,default=None)
    args=ap.parse_args();torch.set_num_threads(1)
    cfg=EA2Cfg(seed=args.seed);cfg.env.num_envs=args.num_envs
    cfg.lidar.use_reduced_raycast=not args.full_lidar
    cfg.lidar.debug_env_ids=list(range(args.num_envs));cfg.lidar.debug_point_stride=args.point_stride
    net,meta=load_checkpoint(str(args.checkpoint),args.device)
    env=make_env(cfg,args.device,capture_debug=True,
                 render_mode='rgb_array' if args.snapshot else None)
    session=PlaySession(env,net,args.mode,meta.get('config',{}).get('train',{}).get('seq_len',400))
    try:
        session.reset()
        NativeMujocoViewer(session,session.policy).run(num_steps=args.steps)
        if args.snapshot:
            from PIL import Image
            args.snapshot.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(env.render()).save(args.snapshot)
    finally:session.close()


if __name__=='__main__':main()
