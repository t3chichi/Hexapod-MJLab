"""Actual native viewer controls, rendered overlay and shared drawing example."""
from pathlib import Path
import argparse
import numpy as np
import torch
from mjlab.viewer.native.viewer import NativeMujocoViewer
from hexapod_mjlab.utils.visualization.example import draw_example
from hexapod_mjlab.tasks.envelope_adaptive_2.config.parameters import EA2Cfg
from hexapod_mjlab.tasks.envelope_adaptive_2.env import make_env
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.model import EnvelopeNet
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.scripts.play import PlaySession


class RecordingVisualizer:
    def __init__(self):self.lines=[];self.points=[]
    def add_sphere(self,center,radius,color):self.points.append((np.array(center),radius,color))
    def add_cylinder(self,start,end,radius,color):self.lines.append((np.array(start),np.array(end),radius,color))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full-lidar',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1)
    record=RecordingVisualizer();draw_example(record)
    assert len(record.points)==3 and len(record.lines)==4
    from hexapod_mjlab.utils.visualization import draw_points
    draw_points(record,np.zeros((20000,3)),max_points=10)
    assert len(record.points)==13
    cfg=EA2Cfg();cfg.env.num_envs=2;cfg.lidar.debug_env_ids=[0,1]
    cfg.lidar.use_reduced_raycast=not args.full_lidar
    env=make_env(cfg,capture_debug=True,render_mode='rgb_array')
    session=PlaySession(env,EnvelopeNet().to(env.device))
    viewer=NativeMujocoViewer(session,session.policy)
    try:
        session.reset();viewer.setup();assert viewer.is_running()
        viewer.pause();before=env.ea2.logical_step;viewer.tick()
        assert env.ea2.logical_step==before
        viewer.request_single_step();viewer.tick()
        assert env.ea2.logical_step==before+1
        viewer.request_reset();viewer.tick()
        assert env.ea2.logical_step==before+2 and session.predictor.hidden is None
        viewer.request_action('NEXT_ENV');viewer.tick();assert viewer.env_idx==1
        viewer.request_action('PREV_ENV');viewer.tick();assert viewer.env_idx==0
        viewer.resume();assert not viewer.get_status().paused
        viewer.pause();viewer.sync_env_to_viewer()
        assert viewer.viewer.user_scn.ngeom>=13
        assert viewer.viewer.user_scn.ngeom<=cfg.lidar.debug_geom_budget+env.sim.mj_model.ngeom+1
        image=env.render();assert image.shape==(720,960,3) and image.std()>5
        from PIL import Image
        path=Path('logs/ea2/acceptance/viewer_full_smoke.png' if args.full_lidar else 'logs/ea2/acceptance/viewer_smoke.png');path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(image).save(path)
        # Shared helpers plus EA2 overlay do not advance the state.
        assert env.ea2.logical_step==before+2
        print(f'T4 PASS full={args.full_lidar}: native viewer pause/single-step/reset/env selection/resume, bounded map + overlay render, independent utils example')
    finally:
        viewer.close();session.close()


if __name__=='__main__':main()
