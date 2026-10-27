"""Independent el4090_tripod2_low recipe using the frozen Flat Baseline policy.

Registration is owned by ``hexapod_mjlab.tasks.confined_locomotion_2`` so the
whole v2 family has one registration entry point. Importing this package still
executes the parent package first, which registers this task exactly once.
"""
