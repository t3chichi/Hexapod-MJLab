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


CROSS_PATH_TOLERANCE = 1e-5
# cuDNN picks different GRU kernels/accumulation orders for a PackedSequence and
# a one-step call. The bound was measured on the RTX 4050 Laptop run
# (PyTorch 2.11, CUDA 13.0, IEEE fp32); it is not a cross-device guarantee.


def stepwise_replay(network, steps, *, mirrored=False, joint_permutation=None, mount_quat_xyzw=None):
    """Same-path reference: recompute frames and GRU one control step at a time.

    Reads only the rollout's stored raw inputs and its stored initial hidden
    state; the recurrent state is advanced here, never taken from the batched
    replay output under comparison.
    """
    from hexapod_mjlab.rl.confined_locomotion_2.symmetry import mirror_prepared

    inputs = []
    for step in steps:
        check(step.prepared is not None, "stepwise replay requires stored prepared inputs")
        inputs.append(mirror_prepared(step.prepared, joint_permutation, mount_quat_xyzw=mount_quat_xyzw)
                      if mirrored else step.prepared)
    initial = steps[0].mirror_hidden_before if mirrored else steps[0].hidden_before
    check(initial is not None, "stepwise replay requires the stored initial hidden state")
    hidden = initial.detach().clone()
    outputs = []
    with torch.no_grad():
        for t, (step, p) in enumerate(zip(steps, inputs)):
            if t:
                done = steps[t-1].terminated | steps[t-1].truncated
                hidden = torch.where(done[:, None], torch.zeros_like(hidden), hidden)
            frame = network.encode_points(
                p.near_points, p.near_valid, p.far_points, p.far_valid,
                near_angle_keys=p.near_angle_keys, far_angle_keys=p.far_angle_keys)
            output = network.forward_from_features(p.observation, frame, hidden)
            outputs.append(output)
            hidden = output.hidden
    return outputs


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
        from hexapod_mjlab.common.perturbations import ALL_THREE_PRESET
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
        transitions = []
        def observe(transition, output, envelope):
            check(torch.equal(envelope.step_ids, transition.step_ids), "observer/envelope control identity mismatch")
            live_outputs.append((transition.episode_ids.clone(), transition.step_ids.clone(),
                                 output.frame_feature.clone(), output.temporal_feature.clone(),
                                 output.hidden.clone()))
            transitions.append(transition)
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
        stepwise_frame_max = stepwise_temporal_max = stepwise_hidden_max = 0.
        stepwise_progress_max = stepwise_mirror_max = 0.
        cross_frame_max = cross_temporal_max = cross_hidden_max = 0.
        cross_progress_max = cross_bootstrap_max = cross_mirror_max = 0.
        max_height = 0.
        scan_counts = []
        buffers = []
        for batch_index in range(2 if args.flat else 4):
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
            elif batch_index == 3:
                # Root remains inside, but foot ground queries cross the edge.
                root_reset = env.event_manager.get_term_cfg("reset_root_witness")
                original_reset = root_reset.func
                def edge_reset(target_env, env_ids, **kwargs):
                    original_reset(target_env, env_ids, **kwargs)
                    pose = torch.cat((torch.tensor([4.95,0.,.8],device=device), quat(0,0,0,device)))[None]
                    target_env.scene["robot"].write_root_link_pose_to_sim(pose,env_ids=env_ids)
                root_reset.func = edge_reset
                try:
                    runtime.reset(torch.tensor([0],device=device))
                finally:
                    root_reset.func = original_reset
                unselected_step = int(runtime.step_in_episode[1])
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
                stepwise = stepwise_replay(network, buffer.steps)
                stepwise_mirror = stepwise_replay(
                    network, buffer.steps, mirrored=True,
                    joint_permutation=permutation, mount_quat_xyzw=mount)
            for t, (step, output, reference) in enumerate(zip(buffer.steps, outputs, stepwise)):
                episode, control, online_frame, online, online_hidden = live_outputs[offset+t]
                check(torch.equal(episode, step.observation.episode_ids) and torch.equal(control, step.observation.step_ids), "output/rollout control identity mismatch")
                stepwise_frame = float((online_frame-reference.frame_feature).abs().max())
                stepwise_temporal = float((online-reference.temporal_feature).abs().max())
                stepwise_hidden = float((online_hidden-reference.hidden).abs().max())
                stepwise_frame_max = max(stepwise_frame_max,stepwise_frame)
                stepwise_temporal_max = max(stepwise_temporal_max,stepwise_temporal)
                stepwise_hidden_max = max(stepwise_hidden_max,stepwise_hidden)
                check(stepwise_frame < 2e-6, "stepwise frame encoding differs from online collection")
                check(stepwise_temporal < 2e-6, "stepwise temporal output differs from online collection")
                check(stepwise_hidden < 2e-6, "stepwise hidden output differs from online collection")
                if t+1 < 24:
                    expected_hidden = reference.hidden.clone()
                    expected_hidden[step.terminated | step.truncated] = 0
                    progress = float((buffer.steps[t+1].hidden_before-expected_hidden).abs().max())
                    stepwise_progress_max = max(stepwise_progress_max,progress)
                    check(progress < 2e-6, "done boundary or hidden progression is wrong")
                    expected_mirror = stepwise_mirror[t].hidden.clone()
                    expected_mirror[step.terminated | step.truncated] = 0
                    mirror_progress = float((buffer.steps[t+1].mirror_hidden_before-expected_mirror).abs().max())
                    stepwise_mirror_max = max(stepwise_mirror_max,mirror_progress)
                    check(mirror_progress < 2e-6, "mirror hidden progression is wrong")
                cross_frame = float((reference.frame_feature-output.frame_feature).abs().max())
                cross_temporal = float((online-output.temporal_feature).abs().max())
                cross_hidden = float((reference.hidden-output.hidden).abs().max())
                cross_frame_max = max(cross_frame_max,cross_frame)
                cross_temporal_max = max(cross_temporal_max,cross_temporal)
                cross_hidden_max = max(cross_hidden_max,cross_hidden)
                check(cross_frame < CROSS_PATH_TOLERANCE, "packed-sequence frame encoding exceeds the documented cross-path numerical bound")
                check(cross_temporal < CROSS_PATH_TOLERANCE, "packed-sequence temporal output exceeds the documented cross-path numerical bound")
                check(cross_hidden < CROSS_PATH_TOLERANCE, "packed-sequence hidden output exceeds the documented cross-path numerical bound")
                if t+1 < 24:
                    expected_batch = output.hidden.clone()
                    expected_batch[step.terminated | step.truncated] = 0
                    batch_progress = float((buffer.steps[t+1].hidden_before-expected_batch).abs().max())
                    cross_progress_max = max(cross_progress_max,batch_progress)
                    check(batch_progress < CROSS_PATH_TOLERANCE, "batch replay hidden progression exceeds the documented cross-path bound")
                    batch_mirror = mirrored[t].hidden.clone()
                    batch_mirror[step.terminated | step.truncated] = 0
                    cross_mirror_max = max(cross_mirror_max,float((buffer.steps[t+1].mirror_hidden_before-batch_mirror).abs().max()))
                    check(cross_mirror_max < CROSS_PATH_TOLERANCE, "packed-sequence mirror history exceeds the documented cross-path bound")
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
            committed = (~(buffer.steps[-1].terminated|buffer.steps[-1].truncated))[:,None]
            check(torch.allclose(collector.hidden,stepwise[-1].hidden * committed,atol=2e-6,rtol=0), "bootstrap committed an extra hidden update")
            bootstrap = float((collector.hidden-outputs[-1].hidden * committed).abs().max())
            cross_bootstrap_max = max(cross_bootstrap_max,bootstrap)
            check(bootstrap < CROSS_PATH_TOLERANCE, "packed-sequence bootstrap hidden exceeds the documented cross-path bound")
            stepwise_mirror_max = max(stepwise_mirror_max,float((collector.mirror_hidden-stepwise_mirror[-1].hidden * committed).abs().max()))
            check(torch.allclose(collector.mirror_hidden,stepwise_mirror[-1].hidden * committed,atol=2e-6,rtol=0), "mirror online/stepwise history mismatch")
            mirror_final = float((collector.mirror_hidden-mirrored[-1].hidden * committed).abs().max())
            cross_mirror_max = max(cross_mirror_max,mirror_final)
            check(mirror_final < CROSS_PATH_TOLERANCE, "packed-sequence mirror history exceeds the documented cross-path bound")
            if batch_index == 1:
                check(bool(buffer.steps[0].truncated[0]) and not bool(buffer.steps[0].terminated[0]), "real timeout fixture did not truncate")
                check(int(buffer.steps[1].observation.step_ids[0])==0, "reset row did not enter fresh episode")
                check(int(buffer.steps[1].observation.step_ids[1])==25, "timeout reset changed unselected row")
            elif batch_index == 2:
                check(bool(buffer.steps[0].terminated[1]) and not bool(buffer.steps[0].truncated[1]), "real flip did not terminate")
                check(int(buffer.steps[1].observation.step_ids[1])==0, "terminated row did not enter fresh episode")
                check(int(buffer.steps[1].observation.step_ids[0])==24, "termination reset changed unselected row")
                check(float(buffer.steps[0].next_value[1])==0., "true terminal retained bootstrap value")
            elif batch_index == 3:
                first, following = buffer.steps[:2]
                transition = transitions[offset]
                check(bool(first.truncated[0]) and not bool(first.terminated[0]), "field boundary did not truncate")
                check(transition.reset_env_ids.tolist() == [0], "field boundary reset another row")
                check(bool(transition.diagnostics["termination_reasons"]["terrain_boundary"][0]), "boundary reason missing")
                check(not bool(transition.diagnostics["termination_reasons"]["timeout"][0]), "boundary misreported as timeout")
                check(int(following.observation.step_ids[0]) == 0 and int(following.observation.step_ids[1]) == unselected_step+1, "boundary reset identity mismatch")
                final = transition.final_inputs
                check(torch.equal(final.observation.episode_ids, first.observation.episode_ids), "bootstrap used reset episode")
                with torch.no_grad():
                    frame = network.encode_points(final.near_points,final.near_valid,final.far_points,final.far_valid,
                                                  near_angle_keys=final.near_angle_keys,far_angle_keys=final.far_angle_keys)
                    value = network.forward_from_features(final.observation,frame,live_outputs[offset][4]).value.squeeze(-1)
                check(torch.allclose(first.next_value[0],value[0],atol=2e-6,rtol=0), "boundary lost pre-reset bootstrap")
                check(not runtime._faulted and bool(torch.isfinite(first.reward).all()), "boundary faulted or produced invalid reward")
                report["terrain_boundary"] = {"truncated_only_row":0,"bootstrap_pre_reset":True,"other_row_preserved":True}
            scan_counts.append((runtime.sensor_scan_counts-initial_scan).tolist())
            if retained is not None:
                check(torch.equal(retained[0].heights,retained[1]),"retained rollout label mutated after another rollout/reset")
            if not args.flat:
                retained=(buffer.steps[0].height_label,buffer.steps[0].height_label.heights.clone())
            check(not bool(wp.to_torch(env.sim.wp_data.overflow).any()),"physics overflow")
        check(weights_before==fingerprint(network) and distribution_before==fingerprint(provider),"collection/replay changed network/distribution weights")
        check(all(p.grad is None for p in network.parameters()), "gate accumulated gradients")
        report.update(passed=True, control_steps=24*len(buffers), environment_transitions=48*len(buffers),
                      stepwise_reference={
                          "path": "per-control-step encode_points + forward_from_features",
                          "tolerance": 2e-6,
                          "frame_max_error": stepwise_frame_max,
                          "temporal_max_error": stepwise_temporal_max,
                          "hidden_max_error": stepwise_hidden_max,
                          "hidden_progression_max_error": stepwise_progress_max,
                          "mirror_history_max_error": stepwise_mirror_max,
                      },
                      packed_sequence_cross_path={
                          "path": "batched encode_point_sequence + forward_sequence_from_features",
                          "tolerance": CROSS_PATH_TOLERANCE,
                          "scope": "measured on RTX 4050 Laptop / PyTorch 2.11 / CUDA 13.0 / cuDNN IEEE fp32; not a cross-device guarantee",
                          "frame_max_error": cross_frame_max,
                          "temporal_max_error": cross_temporal_max,
                          "hidden_max_error": cross_hidden_max,
                          "hidden_progression_max_error": cross_progress_max,
                          "bootstrap_max_error": cross_bootstrap_max,
                          "mirror_history_max_error": cross_mirror_max,
                      },
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
