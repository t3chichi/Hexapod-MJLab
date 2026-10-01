#!/usr/bin/env python
"""M5a friction-combination semantics probe: Isaac Gym/PhysX avg vs MuJoCo max-mix.

Why this matters for M5 baseline comparison: the old stack randomizes the
robot's shape friction U(0.3, 1.25) per env against a fixed terrain friction
1.0, and the two engines COMBINE the two geoms' friction differently at
contact time, so the effective friction distributions differ:

* Isaac Gym / PhysX: pairwise average of the two shapes' friction
  (documented in the old repo config itself:
  ``el_4090_spider_config.py``: "on ground planes the friction combination
  mode is averaging, i.e total friction = (foot_friction + 1.)/2.",
  with terrain ``static_friction = dynamic_friction = 1.0``).
* MuJoCo: element-wise MAX of the two geoms' friction vectors when geom
  priorities are equal (default 0); if priorities differ, the higher-priority
  geom's friction wins entirely (MuJoCo docs, Solver > Contact parameters).
  Verified empirically below with a two-geom pull test on CPU mujoco.

Probe (cheap, exact): a settled box on a plane in plain mujoco is pulled
horizontally with force F; the friction-cone boundary (box starts sliding)
is found by bisection, so mu_eff = F_threshold / (m g).  Computed for the
pairs of the task's DR ranges plus a priority override check.

Outputs ``tests/data/m5/friction_semantics.{json,md}`` (table for M5
calibration: how to reproduce the old effective friction range if desired).

Usage (mjlab conda env)::

    python scripts/friction_probe.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "tests" / "data" / "m5"

# Task-relevant friction pairs: (robot_geom_mu, terrain_geom_mu).
# Terrain default in both stacks is 1.0; old DR range [0.3, 1.25].
PAIRS = [
  (0.3, 1.0),
  (0.4, 1.0),
  (0.65, 1.0),
  (1.0, 1.0),
  (1.125, 1.0),
  (1.25, 1.0),
  (0.4, 0.8),
  (1.25, 0.8),
  (0.6, 1.2),
]

# Task-level randomization summary values.
OLD_ROBOT_RANGE = (0.3, 1.25)
TERRAIN_FRICTION = 1.0


def _box_on_plane(robot_mu: float, terrain_mu: float,
                  terrain_priority: int | None = None):
  """Flat wide box resting on a plane (tips only above mu ~ 10, far beyond
  the task's friction range, so sliding is the only escape mode)."""
  import mujoco

  xml = f"""
  <mujoco>
    <option timestep="0.001" gravity="0 0 -9.81"/>
    <worldbody>
      <geom name="plane" type="plane" size="5 5 0.1"
            friction="{terrain_mu} 0 0" condim="3"
            {'priority="' + str(terrain_priority) + '"' if terrain_priority is not None else ''}/>
      <body name="box" pos="0 0 0.03">
        <freejoint/>
        <geom name="box_geom" type="box" size="0.3 0.3 0.03" mass="1"
              friction="{robot_mu} 0 0" condim="3"/>
      </body>
    </worldbody>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  return model, data


def contact_friction_combined(robot_mu: float, terrain_mu: float,
                              terrain_priority: int | None = None) -> float:
  """The engine's own combined contact friction (authoritative).

  Reads ``data.contact.friction[0]`` from a pressed-in box-plane contact:
  this is the exact tangential friction MuJoCo's solver uses, i.e. the
  combination rule itself, with no measurement noise.
  """
  import numpy as np

  model, data = _box_on_plane(robot_mu, terrain_mu, terrain_priority)
  data.qpos[2] -= 0.001  # press into contact
  import mujoco

  mujoco.mj_forward(model, data)
  if data.ncon == 0:
    return float("nan")
  return float(np.asarray(data.contact.friction)[0, 0])


def pull_test_mu(robot_mu: float, terrain_mu: float,
                 terrain_priority: int | None = None) -> float:
  """Measure the effective friction coefficient of a box-on-plane contact.

  Dynamics-level check: the settled box is pulled horizontally with force
  F; the friction cone boundary (box starts sliding) is found by bisection
  on F, so mu_eff = F_threshold / (m g).  Confirms the solver actually
  enforces the combined value reported by ``contact_friction_combined``.
  """
  import mujoco

  model, data = _box_on_plane(robot_mu, terrain_mu, terrain_priority)
  box_id = model.body("box").id
  m, g = 1.0, 9.81

  # Settle the box on the plane (no pull) and snapshot the resting state.
  for _ in range(300):
    mujoco.mj_step(model, data)
  qpos0 = data.qpos.copy()
  qvel0 = data.qvel.copy()

  def slides(fx: float) -> bool:
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    data.xfrc_applied[:] = 0.0
    data.xfrc_applied[box_id, 0] = fx
    mujoco.mj_forward(model, data)
    for _ in range(500):  # 0.5 s of pull
      mujoco.mj_step(model, data)
    return float(data.qpos[0] - qpos0[0]) > 0.01  # >1 cm drift = sliding

  lo, hi = 0.0, 2.5 * m * g  # bracket (max pair friction 1.25)
  if not slides(hi):
    return float("nan")
  for _ in range(18):  # ~1e-4 resolution on mu
    mid = 0.5 * (lo + hi)
    if slides(mid):
      hi = mid
    else:
      lo = mid
  return float(0.5 * (lo + hi) / (m * g))


def main() -> int:
  rows = []
  for robot_mu, terrain_mu in PAIRS:
    avg = (robot_mu + terrain_mu) / 2.0  # Isaac Gym / PhysX
    mx = max(robot_mu, terrain_mu)       # MuJoCo rule (equal priorities)
    combined = contact_friction_combined(robot_mu, terrain_mu)
    measured = pull_test_mu(robot_mu, terrain_mu)
    rows.append({
      "robot_mu": robot_mu,
      "terrain_mu": terrain_mu,
      "isaac_avg": round(avg, 4),
      "mujoco_max_rule": round(mx, 4),
      "mujoco_contact_friction": round(combined, 4),
      "pull_test_measured": round(measured, 4),
      "combined_err_vs_max_rule": round(abs(combined - mx), 4),
    })

  # Priority override check: terrain priority 1 -> contact uses terrain
  # friction entirely, even when the box friction is higher.
  prio_combined = contact_friction_combined(1.25, 0.5, terrain_priority=1)
  prio_pull = pull_test_mu(1.25, 0.5, terrain_priority=1)
  prio_row = {
    "robot_mu": 1.25, "terrain_mu": 0.5, "terrain_priority": 1,
    "mujoco_contact_friction": round(prio_combined, 4),
    "pull_test_measured": round(prio_pull, 4),
    "note": "higher-priority geom wins entirely (expect ~= terrain_mu 0.5)",
  }

  # Task-level effective ranges.
  old_lo = (OLD_ROBOT_RANGE[0] + TERRAIN_FRICTION) / 2
  old_hi = (OLD_ROBOT_RANGE[1] + TERRAIN_FRICTION) / 2
  new_lo = max(OLD_ROBOT_RANGE[0], TERRAIN_FRICTION)
  new_hi = max(OLD_ROBOT_RANGE[1], TERRAIN_FRICTION)
  task_summary = {
    "old_isaac_avg_range": [round(old_lo, 4), round(old_hi, 4)],
    "new_mujoco_max_range": [round(new_lo, 4), round(new_hi, 4)],
    "calibration_options": [
      "Accept max-mix: new stack trains/tests at higher effective friction "
      f"[{new_lo:.2f}, {new_hi:.2f}] vs old [{old_lo:.3f}, {old_hi:.3f}]; "
      "note it in baseline comparison (feet grip more, less slip).",
      "Reproduce old range: give the terrain geom priority=1 and randomize "
      f"TERRAIN friction over the old effective range [{old_lo:.3f}, "
      f"{old_hi:.3f}] per env (priority makes MuJoCo use the terrain value "
      "alone; verified by the priority probe row). Robot DR can then be "
      "dropped or kept (max/priority ignores the lower-priority side).",
    ],
  }

  max_err = max(r["combined_err_vs_max_rule"] for r in rows)
  max_pull_err = max(abs(r["pull_test_measured"] - r["mujoco_max_rule"])
                     for r in rows)
  results = {
    "probe": "friction_probe",
    "engine_rules": {
      "isaacgym_physx": "pairwise AVERAGE of the two shapes' friction "
                        "(old repo config comment; terrain friction 1.0)",
      "mujoco": "element-wise MAX of geom friction vectors at equal "
                "priority (verified: engine-combined data.contact.friction "
                f"matches max within {max_err:.4f}; dynamics pull test "
                f"within {max_pull_err:.4f}); higher geom priority wins "
                "entirely",
    },
    "pairs": rows,
    "priority_probe": prio_row,
    "task_summary": task_summary,
  }

  OUT_DIR.mkdir(parents=True, exist_ok=True)
  (OUT_DIR / "friction_semantics.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False)
  )

  # Markdown table.
  md = [
    "# M5 friction-combination semantics (Isaac Gym vs MuJoCo)",
    "",
    "Robot geom friction (old stack DR: U(0.3, 1.25) per env, one sample for",
    "all robot geoms) meets terrain geom friction (1.0 in both stacks) with",
    "different combination rules:",
    "",
    "* **Isaac Gym / PhysX**: pairwise average — `(robot + terrain) / 2`",
    "  (documented in the old `el_4090_spider_config.py`: *\"on ground planes",
    "  the friction combination mode is averaging, i.e total friction =",
    "  (foot_friction + 1.)/2.\"*).",
    "* **MuJoCo**: element-wise max of the two geom friction vectors at",
    "  equal priority (default 0); with different priorities the",
    "  higher-priority geom's friction is used entirely.",
    "",
    "MuJoCo rule verified empirically two ways on CPU mujoco (box friction",
    "`robot_mu` vs plane friction `terrain_mu`, condim 3, sliding friction",
    "only):",
    "",
    "1. **engine-combined value** — `data.contact.friction[0]`, the exact",
    "   friction MuJoCo's solver applies at the contact;",
    "2. **dynamics pull test** — settled box pulled horizontally, friction",
    "   cone boundary found by force bisection (`mu_eff = F*/(m g)`).",
    "",
    "| robot mu | terrain mu | Isaac avg | MuJoCo max (rule) | "
    "MuJoCo contact.friction | pull test |",
    "|---|---|---|---|---|---|",
  ]
  for r in rows:
    md.append(
      f"| {r['robot_mu']:.3f} | {r['terrain_mu']:.3f} | {r['isaac_avg']:.4f} "
      f"| {r['mujoco_max_rule']:.4f} | {r['mujoco_contact_friction']:.4f} "
      f"| {r['pull_test_measured']:.4f} |"
    )
  md += [
    "",
    "Priority probe: robot 1.25 / terrain 0.5 with terrain `priority=1` ->",
    f"contact.friction = {prio_row['mujoco_contact_friction']:.4f}, "
    f"pull test = {prio_row['pull_test_measured']:.4f} (terrain value wins",
    "entirely).",
    "",
    "## Task-level consequence (old stack DR U(0.3, 1.25) vs terrain 1.0)",
    "",
    f"* old (Isaac avg): effective friction ~ U over [{old_lo:.3f}, {old_hi:.3f}]",
    f"* new (MuJoCo max): effective friction ~ [{new_lo:.2f}, {new_hi:.2f}] "
    "(lower tail floored by terrain 1.0)",
    "",
    "## M5 calibration options",
    "",
  ]
  md += [f"{i + 1}. {opt}" for i, opt in enumerate(task_summary["calibration_options"])]
  md += [
    "",
    "(MIGRATION.md M3 deviation 2 records the unscaled max-mix behavior;",
    "this table quantifies it for the M5 baseline-comparison note.)",
    "",
  ]
  (OUT_DIR / "friction_semantics.md").write_text("\n".join(md))

  print("\n".join(md))
  print(f"-> {OUT_DIR / 'friction_semantics.json'}")
  print(f"-> {OUT_DIR / 'friction_semantics.md'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
