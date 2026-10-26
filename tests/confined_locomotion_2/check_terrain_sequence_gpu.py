"""Finite real-terrain collection/replay gate; no head, loss or optimizer."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from check_terrain_geometry_gpu import fixture_terrain, quat


def check(value, message):
    if not bool(value):
        raise AssertionError(message)


def fingerprint(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--flat", action="store_true", help="same collection/replay gate on unchanged Flat defaults")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"stage": "collection_replay_only", "flat": args.flat,
              "auxiliary_head": False, "loss_computed": False, "optimizer_created": False,
              "optimizer_updates": 0}
    env, runtime = None, None
    started = time.monotonic()
    try:
        check(torch.cuda.is_available(), "real CUDA required")
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        from mjlab.tasks.registry import list_tasks
        from hexapod_mjlab.tasks.confined_locomotion_2.config.terrain_memory import make_collection_env_cfg, TASK_ID
        from hexapod_mjlab.tasks.confined_locomotion_2.flat_env_cfg import make_training_env_cfg, JOINT_NAMES
        from hexapod_mjlab.tasks.confined_locomotion_2.runtime import TrainingRuntime
        from hexapod_mjlab.tasks.confined_locomotion_2.scanner import ConfinedLidarCfg
        from hexapod_mjlab.tasks.confined_locomotion_2.collector import TrainingCollector
        from hexapod_mjlab.tasks.confined_locomotion_2.terrain_geometry import HeightGridCfg
        from hexapod_mjlab.tasks.confined_locomotion_2.perturbations import ALL_THREE_PRESET
        from hexapod_mjlab.rl.confined_locomotion.distribution import IndependentDiagonalNormalProvider, derive_action_generator_seed
        from hexapod_mjlab.rl.confined_locomotion.symmetry import JointPermutation
        from hexapod_mjlab.rl.confined_locomotion_2.network import ConfinedLocomotionNetwork
        from hexapod_mjlab.rl.confined_locomotion_2.symmetry import mirror_height_targets
        from hexapod_mjlab.rl.confined_locomotion_2.training_rollout import replay_sequence

        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
        device, seed = f"cuda:{args.gpu}", 20261025
        torch.manual_seed(seed)
        registered_before = list_tasks()
        cfg = (make_training_env_cfg(num_envs=2, seed=seed) if args.flat else
               make_collection_env_cfg(terrain=fixture_terrain(), num_envs=2, seed=seed))
        env = ManagerBasedRlEnv(cfg, device=device)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        selection = ROOT / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_hit10/near_x6_far_x10.npz"
        runtime = TrainingRuntime(
            wrapper, seed=seed, output_dir=args.output/"runtime", mode="training_lifecycle_fixture",
            lidar_cfg=ConfinedLidarCfg(raycast_mode="sparse", sparse_selection_path=str(selection), far_sampling="hit_only_10m"),
            perturbation_cfg=ALL_THREE_PRESET,
            **({} if args.flat else {"height_grid": HeightGridCfg((-1.2,1.2),(-.9,.9))}))
        check(list_tasks() == registered_before, "geometry/collection changed the task registry")
        check(runtime.terrain_query is None if args.flat else runtime.terrain_query is not None, "wrong task sensing path")
        if not args.flat:
            # Put one row on the genuinely non-flat region before collection;
            # update the ordinary selected-row lifecycle, no mocked observations.
            runtime.env.scene.env_origins[1, 0] = 1.8
            runtime.env.scene.env_origins[1, 1] = .2
            runtime.env.scene.env_origins[1, 2] = .18
            runtime.reset(torch.tensor([1], device=device))
        network = ConfinedLocomotionNetwork(mlp_activation=torch.nn.ELU(), init_fn=None, envelope_mode="raw").to(device)
        provider = IndependentDiagonalNormalProvider(device=device, generator_seed=derive_action_generator_seed(seed))
        weights_before, distribution_before = fingerprint(network), fingerprint(provider)
        permutation = JointPermutation.from_joint_names(JOINT_NAMES)
        mount = runtime.pipeline.mount_q[0].clone()
        live_outputs = []
        def observe(transition, output, envelope):
            check(torch.equal(envelope.step_ids, transition.step_ids), "observer/envelope control identity mismatch")
            live_outputs.append((transition.episode_ids.clone(), transition.step_ids.clone(), output.temporal_feature.clone()))
        collector = TrainingCollector(runtime, network, provider, symmetry_enabled=True,
                                      joint_permutation=permutation, mount_quat_xyzw=mount, transition_observer=observe)
        if not args.flat:
            label_calls = []
            original_labels = runtime.terrain_query.height_labels
            def counted_labels(*values):
                label = original_labels(*values)
                check(torch.equal(label.root_pos_w, env.scene["robot"].data.root_link_pos_w), "label root is not current physical root")
                label_calls.append((label.episode_ids.clone(), label.step_ids.clone()))
                return label
            runtime.terrain_query.height_labels = counted_labels
            a, b = runtime.labels()["height"], runtime.labels()["height"]
            check(a is b and len(label_calls)==1, "same-identity labels repeat geometry query")
        retained = None
        max_replay, max_mirror, max_height = 0., 0., 0.
        scan_counts = []
        buffers = []
        for batch_index in range(2 if args.flat else 3):
            if batch_index == 1:
                # Explicit lifecycle fixture: actual manager timeout next step,
                # no fabricated done/reward; the other row remains untouched.
                runtime.step_in_episode[0] = 999
                env.episode_length_buf[0] = 999
                runtime._input_cache.invalidate()
                if not args.flat:
                    runtime._height_label = None
                report["timeout_fixture"] = {"row":0,"pre_action_episode_step":999}
            elif batch_index == 2:
                # A real flipped pose above the terrain triggers the existing
                # immediate-fall manager term. Construct it inside a real new
                # episode reset, before its first scan/observation is created.
                root_reset = env.event_manager.get_term_cfg("reset_root_witness")
                original_reset = root_reset.func
                def flipped_reset(target_env, env_ids, **kwargs):
                    original_reset(target_env, env_ids, **kwargs)
                    pose = torch.cat((torch.tensor([1.8,.2,1.5],device=device), quat(np.pi,0,.3,device)))[None]
                    target_env.scene["robot"].write_root_link_pose_to_sim(pose,env_ids=env_ids)
                root_reset.func = flipped_reset
                try:
                    runtime.reset(torch.tensor([1],device=device))
                finally:
                    root_reset.func = original_reset
                report["termination_fixture"] = {"row":1,"roll_rad":float(np.pi),"root_z_m":1.5}
            initial_hidden = collector.hidden.clone()
            initial_mirror = collector.mirror_hidden.clone()
            changed = runtime.episode_ids != collector.episode_ids
            initial_hidden[changed] = 0
            initial_mirror[changed] = 0
            initial_scan = runtime.sensor_scan_counts.clone()
            offset = len(live_outputs)
            buffer = collector.collect()
            buffers.append(buffer)
            check(torch.equal(buffer.steps[0].hidden_before, initial_hidden), "rollout boundary cleared history")
            check(torch.equal(buffer.steps[0].mirror_hidden_before, initial_mirror), "rollout boundary cleared mirror history")
            with torch.no_grad():
                outputs = replay_sequence(network, buffer.steps)
                mirrored = replay_sequence(network, buffer.steps, mirrored=True,
                                           joint_permutation=permutation, mount_quat_xyzw=mount)
            for t, (step, output) in enumerate(zip(buffer.steps, outputs)):
                episode, control, online = live_outputs[offset+t]
                check(torch.equal(episode, step.observation.episode_ids) and torch.equal(control, step.observation.step_ids), "output/rollout control identity mismatch")
                error = float((online-output.temporal_feature).abs().max())
                max_replay = max(max_replay,error)
                check(error < 2e-6, "collection/replay temporal output mismatch")
                if t+1 < 24:
                    expected_hidden = output.hidden.clone()
                    expected_hidden[step.terminated | step.truncated] = 0
                    check(torch.allclose(buffer.steps[t+1].hidden_before, expected_hidden, atol=2e-6,rtol=0), "done boundary or hidden progression is wrong")
                if args.flat:
                    check(step.height_label is None, "Flat created terrain labels")
                else:
                    label = step.height_label
                    check(label is not None, "Terrain Memory lost height labels")
                    label.validate(step.observation)
                    z, valid = runtime.terrain_query.surface.reference_heights(label.query_xy_w.cpu().numpy())
                    target = z-label.root_pos_w[:,None,None,2].cpu().numpy()
                    check(np.array_equal(valid,label.valid.cpu().numpy()), "stored geometric mask mismatch")
                    max_height = max(max_height,float(abs(label.heights.cpu().numpy()[valid]-target[valid]).max()))
                    mh,mv = mirror_height_targets(label.heights,label.valid)
                    check(torch.equal(mh.flip(-2),label.heights) and torch.equal(mv.flip(-2),label.valid), "mirrored grid target is wrong")
            check(torch.allclose(collector.hidden,outputs[-1].hidden * (~(buffer.steps[-1].terminated|buffer.steps[-1].truncated))[:,None],atol=2e-6,rtol=0), "bootstrap committed an extra hidden update")
            max_mirror = max(max_mirror,float((collector.mirror_hidden-mirrored[-1].hidden * (~(buffer.steps[-1].terminated|buffer.steps[-1].truncated))[:,None]).abs().max()))
            check(max_mirror < 2e-6,"mirror online/replay history mismatch")
            if batch_index == 1:
                check(bool(buffer.steps[0].truncated[0]) and not bool(buffer.steps[0].terminated[0]), "real timeout fixture did not truncate")
                check(int(buffer.steps[1].observation.step_ids[0])==0, "reset row did not enter fresh episode")
                check(int(buffer.steps[1].observation.step_ids[1])==25, "timeout reset changed unselected row")
            elif batch_index == 2:
                check(bool(buffer.steps[0].terminated[1]) and not bool(buffer.steps[0].truncated[1]), "real flip did not terminate")
                check(int(buffer.steps[1].observation.step_ids[1])==0, "terminated row did not enter fresh episode")
                check(int(buffer.steps[1].observation.step_ids[0])==24, "termination reset changed unselected row")
                check(float(buffer.steps[0].next_value[1])==0., "true terminal retained bootstrap value")
            scan_counts.append((runtime.sensor_scan_counts-initial_scan).tolist())
            if retained is not None:
                check(torch.equal(retained[0].heights,retained[1]),"retained rollout label mutated after another rollout/reset")
            if not args.flat:
                retained=(buffer.steps[0].height_label,buffer.steps[0].height_label.heights.clone())
            check(not bool(wp.to_torch(env.sim.wp_data.overflow).any()),"physics overflow")
        check(weights_before==fingerprint(network) and distribution_before==fingerprint(provider),"collection/replay changed network/distribution weights")
        check(all(p.grad is None for p in network.parameters()), "gate accumulated gradients")
        report.update(passed=True, control_steps=24*len(buffers), environment_transitions=48*len(buffers),
                      temporal_replay_max_error=max_replay, mirror_history_max_error=max_mirror,
                      height_reference_max_error_m=max_height, scan_counts_per_rollout=scan_counts,
                      labels_created=0 if args.flat else len(label_calls),
                      weights_unchanged=True, gradients_absent=True, registry_unmodified=True,
                      reset_and_bootstrap_checked=True, perturbations="ALL_THREE",
                      observations=[f.name for f in fields(buffers[0].steps[0].observation)],
                      network_sha256=weights_before)
    except BaseException as error:
        report.update(passed=False,error=repr(error),traceback=traceback.format_exc())
    finally:
        if runtime is not None:
            runtime.close()
        if env is not None:
            env.close()
        report["elapsed_s"]=time.monotonic()-started
        (args.output/"checks.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
        print(json.dumps(report,indent=2,ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
