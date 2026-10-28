"""Measure real-terrain height loss/gradients before choosing its coefficient."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from check_terrain_geometry_gpu import fixture_terrain


def gradient_norm(loss, parameters):
    grads = torch.autograd.grad(loss, tuple(parameters), retain_graph=True, allow_unused=True)
    values = [g.detach().square().sum() for g in grads if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--weight',type=float,help='Explicit coefficient approved after the probe; permits one PPO update')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report={'passed':False,'probe_only':args.weight is None};runtime=env=None
    try:
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        from hexapod_mjlab.tasks.confined_locomotion_2.config.terrain_memory_config import make_collection_env_cfg
        from hexapod_mjlab.tasks.confined_locomotion_2.runtime import TrainingRuntime
        from hexapod_mjlab.tasks.confined_locomotion_2.collector import TrainingCollector
        from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.scanner import ConfinedLidarCfg
        from hexapod_mjlab.tasks.confined_locomotion_2.components.terrain_query import HeightGridCfg
        from hexapod_mjlab.common.perturbations import ALL_THREE_PRESET
        from hexapod_mjlab.tasks.confined_locomotion_2.flat_env_cfg import JOINT_NAMES
        from hexapod_mjlab.rl.confined_locomotion_2.network import TerrainMemoryNetwork
        from hexapod_mjlab.rl.confined_locomotion_2.training_rollout import replay_sequence
        from hexapod_mjlab.rl.confined_locomotion_2.auxiliary import HeightReconstructionAuxiliary
        from hexapod_mjlab.rl.confined_locomotion.distribution import IndependentDiagonalNormalProvider
        from hexapod_mjlab.rl.confined_locomotion.symmetry import JointPermutation, mirror_action_distribution, mirror_envelope_values, mirror_mean_mse
        from hexapod_mjlab.rl.confined_locomotion.losses import normalize_rollout_advantage, ppo_policy_loss, value_clipped_loss, envelope_sum_loss
        torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.fp32_precision='ieee';torch.backends.cudnn.rnn.fp32_precision='ieee'
        torch.manual_seed(20261025);device='cuda:0'
        env=ManagerBasedRlEnv(make_collection_env_cfg(terrain=fixture_terrain(),num_envs=2,seed=20261025),device=device)
        env.scene.env_origins[1]=torch.tensor([1.8,.2,.18],device=device)
        wrapper=RslRlVecEnvWrapper(env,clip_actions=None)
        selection=ROOT/'src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_hit10/near_x6_far_x10.npz'
        runtime=TrainingRuntime(wrapper,seed=20261025,output_dir=args.output/'runtime',mode='training',height_grid=HeightGridCfg((-1.2,1.2),(-.9,.9)),lidar_cfg=ConfinedLidarCfg(raycast_mode='sparse',sparse_selection_path=str(selection),far_sampling='hit_only_10m'),perturbation_cfg=ALL_THREE_PRESET)
        network=TerrainMemoryNetwork(mlp_activation=torch.nn.ELU(),init_fn=None,envelope_mode='raw').to(device)
        provider=IndependentDiagonalNormalProvider(device=device,generator_seed=20261025)
        permutation=JointPermutation.from_joint_names(JOINT_NAMES);mount=runtime.pipeline.mount_q[0].clone()
        collector=TrainingCollector(runtime,network,provider,symmetry_enabled=True,joint_permutation=permutation,mount_quat_xyzw=mount)
        buffer=collector.collect();steps=buffer.steps
        out=replay_sequence(network,steps)
        mirrored=replay_sequence(network,steps,mirrored=True,joint_permutation=permutation,mount_quat_xyzw=mount)
        mask=torch.ones(24,2,dtype=torch.bool,device=device)
        auxiliary=HeightReconstructionAuxiliary(1.)
        height=auxiliary(out,{'steps':steps,'loss_mask':mask,'mirrored_outputs':mirrored})
        advantage,returns=buffer.compute_gae();advantage=normalize_rollout_advantage(advantage.reshape(-1)).reshape(24,2).detach();returns=returns.detach()
        old_values=torch.stack([s.old_value for s in steps]);old_lp=torch.stack([s.old_log_prob for s in steps]);targets=torch.stack([s.envelope_label.values for s in steps]);valid=torch.stack([s.envelope_label.valid for s in steps])
        def motion_loss(outputs,mirror):
            logp=torch.stack([provider(o.action_mean).log_prob(mirror_action_distribution(s.old_mean,s.old_std,s.action,permutation).raw_action if mirror else s.action) for o,s in zip(outputs,steps)])
            entropy=torch.stack([provider(o.action_mean).entropy() for o in outputs]).mean()
            values=torch.stack([o.value.squeeze(-1) for o in outputs]);envelope=torch.stack([o.envelope for o in outputs])
            et=mirror_envelope_values(targets.reshape(-1,6)).reshape_as(targets) if mirror else targets
            return ppo_policy_loss(logp.reshape(-1),old_lp.reshape(-1),advantage.reshape(-1))+value_clipped_loss(values.reshape(-1),old_values.reshape(-1),returns.reshape(-1))-.01*entropy+envelope_sum_loss(envelope.reshape(-1,6),et.reshape(-1,6),valid.reshape(-1))
        base=.5*(motion_loss(out,False)+motion_loss(mirrored,True))+.6*mirror_mean_mse(torch.stack([o.action_mean for o in mirrored]).reshape(-1,18),torch.stack([o.action_mean for o in out]).reshape(-1,18),permutation)
        groups={'frame_encoder':tuple(network.frame_encoder.parameters()),'temporal_encoder':tuple(network.temporal_encoder.parameters()),'height_head':tuple(network.height_head.parameters())}
        norms={name:{'height_unweighted':gradient_norm(height.loss,params),'motion':gradient_norm(base,params)} for name,params in groups.items()}
        control=tuple(network.action_head.parameters())+tuple(network.value_head.parameters())+tuple(network.morphology_head.parameters())
        control_grads=torch.autograd.grad(height.loss,control,retain_graph=True,allow_unused=True)
        assert all(g is None for g in control_grads),'height loss reached control/envelope head parameters'
        assert all(norms[name]['height_unweighted']>0 for name in groups),'missing encoder/head gradient'
        assert all(p.grad is None for p in network.parameters())
        report.update(raw_height_mse_m2=float(height.loss.detach()),motion_loss=float(base.detach()),gradient_norms=norms,control_head_auxiliary_gradients_absent=True,head_shape=list(network.height_head.weight.shape),head_bias_shape=list(network.height_head.bias.shape),valid_cells=int(torch.stack([s.height_label.valid for s in steps]).sum()),target_min_m=float(torch.stack([s.height_label.heights for s in steps]).min()),target_max_m=float(torch.stack([s.height_label.heights for s in steps]).max()),control_steps=24,num_envs=2,perturbations='ALL_THREE',mirrored_branch_weight=.5,optimizer_updates=0)
        del out,mirrored,height,base
        if args.weight is not None:
            from hexapod_mjlab.rl.confined_locomotion_2.ppo import TemporalPPOTrainer
            trainer=TemporalPPOTrainer(network=network,distribution_provider=provider,device=device,symmetry_enabled=True,joint_permutation=permutation,mount_quat_xyzw=mount,seed=20261025,auxiliary=HeightReconstructionAuxiliary(args.weight))
            ids=[id(p) for group in trainer.optimizer.param_groups for p in group['params']]
            assert all(ids.count(id(p))==1 for p in network.height_head.parameters())
            before={name:[p.detach().clone() for p in params] for name,params in groups.items()}
            metrics=trainer.update(buffer)
            assert trainer.optimizer_step_count==20
            report.update(height_loss_weight=args.weight,optimizer_updates=20,metrics=metrics,changed_groups={name:any(not torch.equal(a,p) for a,p in zip(before[name],params)) for name,params in groups.items()})
            assert all(report['changed_groups'].values())
        report['passed']=True
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        if runtime is not None:runtime.close()
        if env is not None:env.close()
        (args.output/'checks.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
