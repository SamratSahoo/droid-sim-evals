# tools/

Standalone, manually-run utilities that are **not** part of the automated data-gen / eval / publish
pipeline (nothing in `slurm/`, `run_datagen.sh`, or the main README invokes them). They live here to
keep the top level to the actual pipeline + eval entrypoints.

Run them from the `droid-sim-evals` directory, e.g. `python tools/<script>.py ...` (each adds the
folders it needs — the repo root for `from src...`, `data/` for `import tamp_data_gen` — to `sys.path`).

| Script | Purpose |
| --- | --- |
| `tamp_plan_graph.py` | Generate a cuTAMP plan factor-graph (JSON/DOT) for the toys scene |
| `replay_json_traj.py` | Replay a serialized TiPToP plan JSON into a video (debug/visualization) |
| `save_h5_obs.py` | Save an initial DROID-sim observation to an H5 file |
| `yam_mjcf_to_urdf.py` | Bimanual-YAM assets, step 1: MolmoAct2 MJCF -> URDF + the two cuRobo per-arm configs (writes into `../cuTAMP/cutamp/robots/assets/`). Needs `mujoco`; run it with any interpreter that has one |
| `build_yam_usd.py` | Bimanual-YAM assets, step 2: that URDF -> `assets/yam/bimanual_yam.usd` for Isaac. Must run under `.venv/bin/python` (needs the Isaac app) |
| `replay_dual_plan.py` | Execute a SIMULTANEOUS dual-arm cuTAMP plan (`cutamp.scripts.dual_arm_demo --save-plan`) in Isaac and record it. `--scene 14` = the parallel two-cube demo, `--scene 15` = the handover. Object names come from the plan, so one driver covers both |
| `recover_dataset_merges.py` | One-off recovery: tag already-uploaded LeRobot datasets + rebuild the d100 merges. Stale — superseded by the parquet-concat merge (`../data/merge_d100_toys.py`); kept for reference, safe to delete once no longer needed |
