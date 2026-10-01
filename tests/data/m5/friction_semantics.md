# M5 friction-combination semantics (Isaac Gym vs MuJoCo)

Robot geom friction (old stack DR: U(0.3, 1.25) per env, one sample for
all robot geoms) meets terrain geom friction (1.0 in both stacks) with
different combination rules:

* **Isaac Gym / PhysX**: pairwise average — `(robot + terrain) / 2`
  (documented in the old `el4090_spider_config.py`: *"on ground planes
  the friction combination mode is averaging, i.e total friction =
  (foot_friction + 1.)/2."*).
* **MuJoCo**: element-wise max of the two geom friction vectors at
  equal priority (default 0); with different priorities the
  higher-priority geom's friction is used entirely.

MuJoCo rule verified empirically two ways on CPU mujoco (box friction
`robot_mu` vs plane friction `terrain_mu`, condim 3, sliding friction
only):

1. **engine-combined value** — `data.contact.friction[0]`, the exact
   friction MuJoCo's solver applies at the contact;
2. **dynamics pull test** — settled box pulled horizontally, friction
   cone boundary found by force bisection (`mu_eff = F*/(m g)`).

| robot mu | terrain mu | Isaac avg | MuJoCo max (rule) | MuJoCo contact.friction | pull test |
|---|---|---|---|---|---|
| 0.300 | 1.000 | 0.6500 | 1.0000 | 1.0000 | 1.0033 |
| 0.400 | 1.000 | 0.7000 | 1.0000 | 1.0000 | 1.0033 |
| 0.650 | 1.000 | 0.8250 | 1.0000 | 1.0000 | 1.0033 |
| 1.000 | 1.000 | 1.0000 | 1.0000 | 1.0000 | 1.0033 |
| 1.125 | 1.000 | 1.0625 | 1.1250 | 1.1250 | 1.1261 |
| 1.250 | 1.000 | 1.1250 | 1.2500 | 1.2500 | 1.2213 |
| 0.400 | 0.800 | 0.6000 | 0.8000 | 0.8000 | 0.8057 |
| 1.250 | 0.800 | 1.0250 | 1.2500 | 1.2500 | 1.2213 |
| 0.600 | 1.200 | 0.9000 | 1.2000 | 1.2000 | 1.1951 |

Priority probe: robot 1.25 / terrain 0.5 with terrain `priority=1` ->
contact.friction = 0.5000, pull test = 0.5074 (terrain value wins
entirely).

## Task-level consequence (old stack DR U(0.3, 1.25) vs terrain 1.0)

* old (Isaac avg): effective friction ~ U over [0.650, 1.125]
* new (MuJoCo max): effective friction ~ [1.00, 1.25] (lower tail floored by terrain 1.0)

## M5 calibration options

1. Accept max-mix: new stack trains/tests at higher effective friction [1.00, 1.25] vs old [0.650, 1.125]; note it in baseline comparison (feet grip more, less slip).
2. Reproduce old range: give the terrain geom priority=1 and randomize TERRAIN friction over the old effective range [0.650, 1.125] per env (priority makes MuJoCo use the terrain value alone; verified by the priority probe row). Robot DR can then be dropped or kept (max/priority ignores the lower-priority side).

(MIGRATION.md M3 deviation 2 records the unscaled max-mix behavior;
this table quantifies it for the M5 baseline-comparison note.)
