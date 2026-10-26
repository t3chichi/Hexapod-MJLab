"""Strict Terrain Memory checkpoint boundary and no-update recurrent restore gate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    runner=env=None;result={'passed':False}
    try:
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        from hexapod_mjlab.tasks.confined_locomotion_2.rl.checkpoint import read_checkpoint,TERRAIN_TASK_ID,cpu_tree
        from hexapod_mjlab.tasks.confined_locomotion_2.rl.terrain_entry import build_terrain_train_config
        from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import ConfinedLocomotionRunner
        from check_gpu import differences
        payload=read_checkpoint(args.checkpoint,expected_task_id=TERRAIN_TASK_ID)
        seed=payload['identity']['seed'];count=payload['identity']['num_envs']
        weight=payload['identity']['recipe']['height_loss_weight']
        selection=ROOT/'src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_hit10/near_x6_far_x10.npz'
        cfg=build_terrain_train_config(seed=seed,num_envs=count,max_iterations=1,height_loss_weight=weight,engineering_validation=True,raycast_mode='sparse',sparse_selection_path=str(selection),resume_checkpoint=args.checkpoint)
        env=ManagerBasedRlEnv(cfg.env,device='cuda:0');wrapper=RslRlVecEnvWrapper(env,clip_actions=None)
        runner=ConfinedLocomotionRunner(wrapper,asdict(cfg.agent),str(args.output/'restored'),'cuda:0')
        runner.load(str(args.checkpoint))
        for name,live in [('network',runner.network.state_dict()),('provider',runner.provider.state_dict()),('trainer',runner.trainer.state_dict()),('runtime',runner.runtime.state_dict()),('collector',runner.collector.state_dict()),('learning_metrics',runner.learning_metrics.state_dict())]:
            delta=differences(payload[name],cpu_tree(live))
            if delta:raise AssertionError(f'{name} restored boundary differs: {delta[:4]}')
        a=runner.runtime.labels()['height'];b=runner.runtime.labels()['height']
        assert a is b and torch.equal(a.step_ids,runner.runtime.step_in_episode)
        assert torch.equal(a.root_pos_w,runner.runtime.env.scene['robot'].data.root_link_pos_w)
        original=runner.recipe_identity['height_loss_weight']
        runner.recipe_identity['height_loss_weight']=original+.1
        try:runner.load(str(args.checkpoint))
        except ValueError:pass
        else:raise AssertionError('different auxiliary coefficient restored')
        runner.recipe_identity['height_loss_weight']=original
        from hexapod_mjlab.rl.confined_locomotion_2.training_rollout import replay_sequence
        before={name:value.detach().clone() for name,value in runner.network.state_dict().items()}
        buf=runner.collector.collect()
        with torch.no_grad():out=replay_sequence(runner.network,buf.steps)
        assert all(o.height_prediction.shape==(count,11,17) for o in out)
        assert all(torch.equal(value,runner.network.state_dict()[name]) for name,value in before.items())
        result.update(passed=True,checkpoint=str(args.checkpoint),strict_boundary_bit_exact=True,changed_weight_rejected=True,labels_regenerated_after_restore=True,subsequent_collection_steps=24,optimizer_updates_after_restore=0)
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        if runner is not None:runner.close()
        elif env is not None:env.close()
        (args.output/'checks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
