"""Same-scene sparse scanner/cache comparison; no PPO or optimizer updates."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/experiments/confined_locomotion_2"))
from benchmark_collection import reference_method
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends
from hexapod_mjlab.tasks.confined_locomotion_2 import collector as collector_module
from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar import scanner as scanner_module
from hexapod_mjlab.tasks.confined_locomotion_2.config.terrain_memory_config import make_training_env_cfg
from hexapod_mjlab.tasks.confined_locomotion_2.components.terrain_query import TerrainGeometryQuery, HeightGridCfg
from hexapod_mjlab.common.perturbations import ALL_THREE_PRESET, ObservationPerturbations
from hexapod_mjlab.common.reset import reset_env_preserving_unselected_physics
from hexapod_mjlab.rl.confined_locomotion_2.network import TerrainMemoryNetwork


def same_state(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same_state(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same_state(x, y)
    else:
        assert a == b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--flat", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--reference-rev", default="2c08bab09c2d97f966bfc6905b21562e9ef81978")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "task": "flat" if args.flat else "terrain", "max_scan_error": 0., "max_feature_error": 0., "batches": [], "optimizer_updates": 0}
    env = None
    try:
        device = f"cuda:{args.gpu}"
        assert torch.cuda.is_available()
        configure_torch_backends(allow_tf32=False)
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
        torch.manual_seed(20261027)
        if args.flat:
            from hexapod_mjlab.tasks.confined_locomotion_2.flat_env_cfg import make_training_env_cfg as env_cfg
        else:
            env_cfg = make_training_env_cfg
        env = ManagerBasedRlEnv(env_cfg(num_envs=4, seed=20261027), device=device)
        poses = torch.tensor([[-1.3,.2,1.0],[0.,1.8,1.0],[1.3,-.2,.85],[.1,-1.,.9]], device=device)
        rotations = Rotation.from_euler("xyz", [[.19,-.13,.7],[-.2,.1,-.4],[.1,.2,1.1],[-.15,-.1,-1.2]]).as_quat()[:,[3,0,1,2]]
        env.scene["robot"].write_root_link_pose_to_sim(torch.cat((poses, torch.tensor(rotations,dtype=torch.float32,device=device)),dim=1))
        env.sim.forward()
        query = None if args.flat else TerrainGeometryQuery(env)
        cfg = scanner_module.ConfinedLidarCfg(raycast_mode="sparse", far_sampling="hit_only_10m",
            sparse_selection_path=str(ROOT / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_hit10/near_x6_far_x10.npz"))
        reference = object.__new__(scanner_module.ConfinedScanPipeline)
        reference_method(scanner_module,scanner_module.ConfinedScanPipeline,"__init__",args.reference_rev)(reference,env,cfg,terrain_query=query)
        pipelines = [reference,scanner_module.ConfinedScanPipeline(env,cfg,terrain_query=query)]
        pipelines[0].scan_rows = MethodType(reference_method(scanner_module, scanner_module.ConfinedScanPipeline, "scan_rows", args.reference_rev), pipelines[0])
        noises = [ObservationPerturbations(4,cfg=ALL_THREE_PRESET,seed=20261027,device=device) for _ in range(2)]
        network = TerrainMemoryNetwork(mlp_activation=torch.nn.ELU(),init_fn=None,envelope_mode="raw").to(device)
        runtime = SimpleNamespace(device=device,num_envs=4,episode_ids=torch.zeros(4,dtype=torch.long,device=device))
        caches = [collector_module.TrainingCollector(runtime,network,None) for _ in range(2)]
        caches[0]._frame = MethodType(reference_method(collector_module,collector_module.TrainingCollector,"_frame",args.reference_rev),caches[0])
        counts, frame_rows = {r:0 for r in range(4)}, {}
        retained = None
        for index, rows in enumerate(([3,1,0,2], [2], [], [0,3], [3,1,0])):
            if index == 4:
                reset_env_preserving_unselected_physics(env,torch.tensor([1],device=device))
                runtime.episode_ids[1] += 1
            numbers = {row:counts[row]+1 for row in rows}
            frames = [pipe.scan_rows(rows,perturbations=noise,persistent_scan_numbers=numbers) for pipe,noise in zip(pipelines,noises)]
            for a,b in zip(*frames):
                assert a.env_row == b.env_row
                for name in vars(a):
                    x,y = getattr(a,name),getattr(b,name)
                    if torch.is_tensor(x):
                        if x.is_floating_point():
                            delta = (x-y).abs().max().item() if x.numel() else 0.
                            report["max_scan_error"] = max(report["max_scan_error"],delta)
                            torch.testing.assert_close(x,y,atol=2e-5,rtol=1e-5)
                        else:
                            assert torch.equal(x,y), name
                    elif isinstance(x,float) and math.isnan(x):
                        assert math.isnan(y)
                    else:
                        assert x == y, name
                for tensor in (b.raw_dists,b.raw_points_sensor):
                    assert tensor.untyped_storage().nbytes() == tensor.numel()*tensor.element_size()
                frame_rows[b.env_row] = b
                counts[b.env_row] += 1
            same_state(noises[0].state_dict(),noises[1].state_dict())
            prepared = SimpleNamespace(
                observation=SimpleNamespace(episode_ids=runtime.episode_ids.clone()),
                near_scan_ids=torch.tensor([counts[r] for r in range(4)],device=device),
                near_points=torch.stack([frame_rows[r].near_points_base for r in range(4)]),
                near_valid=torch.stack([frame_rows[r].near_valid for r in range(4)]),
                far_points=torch.stack([frame_rows[r].far_points_base for r in range(4)]),
                far_valid=torch.stack([frame_rows[r].far_valid for r in range(4)]),
                near_angle_keys=torch.stack([frame_rows[r].near_angle_keys for r in range(4)]),
                far_angle_keys=torch.stack([frame_rows[r].far_angle_keys for r in range(4)]))
            with torch.no_grad():
                for mirror in (False,True):
                    # Distinct mirror data exercises cache independence.
                    view = SimpleNamespace(**vars(prepared))
                    if mirror:
                        view.near_points = prepared.near_points * torch.tensor([1.,-1.,1.],device=device)
                        view.far_points = prepared.far_points * torch.tensor([1.,-1.,1.],device=device)
                    values = [cache._frame(view,mirror=mirror) for cache in caches]
                    report["max_feature_error"] = max(report["max_feature_error"],(values[0]-values[1]).abs().max().item())
                    torch.testing.assert_close(*values,atol=2e-5,rtol=1e-5)
                    assert caches[0].frame_encoding_rows == caches[1].frame_encoding_rows
            if retained is None:
                retained = (frames[1][0],frames[1][0].raw_points_sensor.clone(),values[1],values[1].clone())
            assert torch.equal(retained[0].raw_points_sensor,retained[1])
            assert torch.equal(retained[2],retained[3])
            report["batches"].append({"rows":rows,"noise_state_equal":True,"feature_rows":caches[1].frame_encoding_rows})
        if not args.flat:
            labels = query.height_labels(torch.arange(4,device=device),runtime.episode_ids,env.episode_length_buf.clone(),HeightGridCfg((-.75,.75),(-.65,.65)))
            z,valid = query.surface.reference_heights(labels.query_xy_w.cpu().numpy())
            expected = z-labels.root_pos_w[:,None,None,2].cpu().numpy()
            assert np.array_equal(valid,labels.valid.cpu().numpy())
            report["label_max_error_m"] = float(np.abs(labels.heights.cpu().numpy()[valid]-expected[valid]).max())
            assert report["label_max_error_m"] < 2e-5
        report["passed"] = True
    finally:
        if env is not None:
            env.close()
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
