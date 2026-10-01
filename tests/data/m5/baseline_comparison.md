# M5 baseline comparison (new mjlab stack vs old Isaac Gym stack)

* new run: `logs/m5/baseline_new/20260929_191030` (32 envs, 200 iters, seed 1)
* old run: `logs/m5/baseline_old/20260929_182659` (32 envs, 200 iters, seed 1)
* env steps: new 153,600 / old 153,600 (iters x 24 steps/env x num_envs)

> **Fair-comparison note**: Fair only if both runs used the SAME num_envs, iterations and seed (see run_meta.json in each dir). Hardware: both launched on this laptop (RTX 4050 6 GB) unless stated otherwise in run_meta. Engine semantics still differ by design (MuJoCo vs PhysX contact/friction combination — see tests/data/m5/friction_semantics.md; new-stack effective foot friction [1.00, 1.25] vs old [0.65, 1.125]); reward scales and obs/action contracts are ported 1:1 (MIGRATION.md M3).

| scalar | new final | new last10% | old final | old last10% |
|---|---|---|---|---|
| mean_reward | -260.193115 | -260.61726 | -74.931519 | -74.915407 |
| mean_episode_length | 843.530029 | 853.056326 | 1002.0 | 1002.0 |
| loss_value_function | 12.618593 | 11.779428 | 426.77594 | 788.423352 |
| loss_surrogate | -0.009559 | -0.008511 | -0.007806 | -0.004012 |
| loss_learning_rate | 1e-05 | 2.7e-05 | 1e-05 | 1.5e-05 |
| loss_symmetry | 0.001075 | 0.001646 | 0.001871 | 0.002109 |
| loss_aux | 0.14512 | 0.299228 | 0.370723 | 0.245031 |
| policy_noise_std | 1.00898 | 1.008818 | 1.03258 | 1.030621 |
| perf_total_fps | 116.0 | 113.8 | 140.0 | 138.85 |
| perf_collection_time | 1.461153 | 1.528526 | 1.041273 | 1.054643 |
| perf_learning_time | 5.104336 | 5.20724 | 4.430134 | 4.455327 |
