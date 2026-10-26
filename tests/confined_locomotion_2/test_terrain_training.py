"""Focused Terrain Memory model/config/strict task-boundary checks."""
from dataclasses import asdict
import torch
from torch import nn

from hexapod_mjlab.rl.confined_locomotion_2.network import ConfinedLocomotionNetwork, TerrainMemoryNetwork
from hexapod_mjlab.rl.confined_locomotion_2.contracts import OnlineObservation
from hexapod_mjlab.tasks.confined_locomotion_2.rl.terrain_entry import build_terrain_train_config
from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import resolve_recipe


def test_height_head_is_one_affine_layer_and_does_not_feed_control():
    torch.manual_seed(7)
    flat=ConfinedLocomotionNetwork(mlp_activation=nn.ELU(),init_fn=None,envelope_mode='raw')
    torch.manual_seed(7)
    terrain=TerrainMemoryNetwork(mlp_activation=nn.ELU(),init_fn=None,envelope_mode='raw')
    assert isinstance(terrain.height_head,nn.Linear)
    assert terrain.height_head.weight.shape==(187,187) and terrain.height_head.bias.shape==(187,)
    assert sum(p.numel() for p in terrain.parameters())-sum(p.numel() for p in flat.parameters())==187*187+187
    assert all(torch.equal(value,terrain.state_dict()[name]) for name,value in flat.state_dict().items())
    z3=torch.zeros(1,3);z18=torch.zeros(1,18);ids=torch.zeros(1,dtype=torch.int64)
    obs=OnlineObservation(env_ids=ids,episode_ids=ids.clone(),step_ids=ids.clone(),time_s=torch.zeros(1),lin_vel_b=z3,ang_vel_b=z3.clone(),gravity_b=z3.clone(),joint_pos_rel_ref=z18,joint_vel=z18.clone(),prev_action=z18.clone(),command=z3.clone())
    feature=torch.randn(1,187);hidden=torch.zeros(1,187)
    first=terrain.forward_from_features(obs,feature,hidden)
    reference=flat.forward_from_features(obs,feature,hidden)
    assert first.height_prediction.shape==(1,11,17) and reference.height_prediction is None
    assert torch.equal(first.action_mean,reference.action_mean)
    with torch.no_grad():terrain.height_head.weight.add_(10.)
    second=terrain.forward_from_features(obs,feature,hidden)
    assert not torch.equal(second.height_prediction,first.height_prediction)
    for name in ('action_mean','value','envelope','hidden'):
        assert torch.equal(getattr(first,name),getattr(second,name))


def test_terrain_entry_requires_explicit_weight_and_retains_shared_ppo():
    for invalid in (None,0,-1,float('nan'),True):
        try:build_terrain_train_config(seed=1,num_envs=2,max_iterations=1,height_loss_weight=invalid,engineering_validation=True)
        except (ValueError,TypeError):pass
        else:raise AssertionError('invalid auxiliary coefficient accepted')
    cfg=build_terrain_train_config(seed=1,num_envs=2,max_iterations=1,height_loss_weight=1.,engineering_validation=True)
    recipe=resolve_recipe(asdict(cfg.agent))
    assert recipe['height_loss_weight']==1.
    assert recipe['height_grid'].identity()['order']=='iy*17+ix'
    assert recipe['ppo'].rollout_steps==24 and recipe['ppo'].epochs==5 and recipe['ppo'].minibatches==4
    assert cfg.env.scene.terrain.terrain_type=='generator'
    for count,budget in ((3,1),(2,2)):
        try:build_terrain_train_config(seed=1,num_envs=count,max_iterations=budget,height_loss_weight=1.,engineering_validation=True)
        except ValueError:pass
        else:raise AssertionError('unbounded engineering launch accepted')


def test_full_checkpoint_task_and_architecture_boundaries_are_strict(tmp_path):
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.checkpoint import SCHEMA,TASK_ID,TERRAIN_TASK_ID,architecture_identity,read_checkpoint
    path=tmp_path/'terrain.pt'
    torch.save({'schema':SCHEMA,'task':TERRAIN_TASK_ID,'kind':'full','resumable':True,'identity':{'architecture':architecture_identity('hit_only_10m',terrain_memory=True)}},path)
    assert read_checkpoint(path,expected_task_id=TERRAIN_TASK_ID,expected_far_sampling='hit_only_10m')['task']==TERRAIN_TASK_ID
    try:read_checkpoint(path)
    except ValueError:pass
    else:raise AssertionError('Flat accepted Terrain Memory checkpoint')
    torch.save({'schema':SCHEMA,'task':TERRAIN_TASK_ID,'kind':'full','resumable':True,'identity':{'architecture':architecture_identity('hit_only_10m')}},path)
    try:read_checkpoint(path,expected_task_id=TERRAIN_TASK_ID)
    except ValueError:pass
    else:raise AssertionError('Terrain Memory accepted Flat architecture')
