"""Production provider replay against every selected frozen .10 result."""

import _bootstrap
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from hexapod_mjlab.tasks.envelope_adaptive_2 import contracts
from hexapod_mjlab.tasks.envelope_adaptive_2.map_generator import generate_map
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import load_map
from hexapod_mjlab.tasks.envelope_adaptive_2.teachers.batch import BatchTeacher, teacher_identity
from whole_polygon_reference import WholePolygonChecker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="release")
    args = parser.parse_args()
    torch.set_num_threads(1)
    outdir = Path("logs/ea2/oracle_refinement/20261003")
    destination = outdir / f"production_frozen_parity_{args.tag}.json"
    if destination.exists():
        raise FileExistsError(destination)
    records = []
    for seed, folder, tag in [
        (1, "ea2-migration-small-20261002", "axial_reserve015_v18"),
        (7, "ea2-migration-small-20261002", "frozen_axial015_v19"),
        (89, "ea2-oracle-final-heldout-20261003", "heldout_axial015_v19"),
        (101, "ea2-oracle-final-heldout-20261003", "heldout_axial015_v19"),
    ]:
        data = load_map(str(Path("logs/ea2/data") / folder / f"map_seed{seed}.pt"))
        cfg = data.meta["environment_config"]
        geom = generate_map(
            contracts.MapGenCfg(**cfg["map"]),
            contracts.PillarFieldCfg(
                **{k.removeprefix("pillar_"): v for k, v in cfg["obstacles"].items()}
            ),
            seed,
        )
        teacher = BatchTeacher(geom.rects)
        checker = WholePolygonChecker(geom.rects)
        gold = np.load(outdir / f"lookahead_w4_seed{seed}_m0.10_stride1_{tag}.npz")
        index = {(int(t), int(e)): i for i, (t, e) in enumerate(zip(gold["time"], gold["env"]))}
        errors = []
        emergencies = 0
        checked = 0
        supervised = 0
        for t in range(len(data.obs)):
            reset_ids = data.done[t].nonzero().flatten().tolist()
            teacher.reset(reset_ids)
            ego = data.obs[t, :, -3:].double() * torch.tensor([1.5, 1.0, 1.5], dtype=torch.float64)
            result = teacher.evaluate(data.pos[t].double(), data.heading[t].double(), ego)
            for e in range(data.obs.shape[1]):
                i = index.get((t, e))
                if i is None:
                    continue
                feasible = gold["status"][i] == "feasible"
                assert feasible == bool(result["geometric_valid"][e]), (seed, t, e)
                assert bool(gold["emergency"][i]) == bool(result["emergency"][e]), (
                    seed,
                    t,
                    e,
                    "emergency",
                )
                if feasible:
                    error = float(np.max(abs(gold["s"][i] - result["s"][e].numpy())))
                    errors.append(error)
                    assert error <= 1e-6, (seed, t, e, error)
                    assert (
                        checker.clearance(
                            result["params"][e].numpy(),
                            data.pos[t, e].numpy(),
                            float(data.heading[t, e]),
                        )
                        >= 0.1 - 1e-6
                    )
                    assert bool(result["supervision_valid"][e]) == (
                        not bool(result["emergency"][e])
                    )
                emergencies += int(result["emergency"][e])
                checked += 1
                supervised += int(result["supervision_valid"][e])
            if t % 100 == 0:
                print("PARITY", seed, t, flush=True)
        records.append(
            {
                "seed": seed,
                "frames": checked,
                "max_error": max(errors),
                "emergencies": emergencies,
                "supervision_frames": supervised,
            }
        )
    report = {"teacher": teacher_identity(), "results": records}
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print("PRODUCTION FROZEN PARITY PASS", json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
