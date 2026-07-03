# Slurm jobs

Batch scripts for the DROID-sim TAMP pipeline. Each script's runtime knobs live in a YAML file
under [`../cfg/`](../cfg) so you don't edit the script to change a run.

## Usage

```bash
# Use the defaults in cfg/<name>.yml:
sbatch slurm/datagen_8gpu_neuronic.slurm

# Overlay a user config (deep-merged over the default — only set the keys you change):
sbatch slurm/datagen_8gpu_neuronic.slurm cfg/my_run.yml
```

The optional first argument is a YAML config path. Cluster **resource** requests (`--gres`,
`--mem`, `--time`, `--exclusive`) are `#SBATCH` directives parsed before the script runs, so change
those with `sbatch --mem=... -t ...` or by editing the header — not via the YAML.

## How config loading works

[`lib/load_config.py`](lib/load_config.py) reads `cfg/<name>.yml` (optionally deep-merged with the
user file), and emits `export UPPER_KEY=value` lines the script `eval`s. Scalars become environment
variables; the reserved `tamp_overrides` mapping is written to a JSON sidecar and its path exported
as `TAMP_OVERRIDES_JSON`.

## TAMP parameters (data-gen)

`datagen_8gpu_neuronic.slurm` passes `tamp_overrides` to every tiptop planning
server via `--curobo-overrides`, so an entire data-gen run plans with a chosen cuRobo cost regime
(manifold pull, RND novelty, smoothness, time dilation, …). Keys match
`tiptop.motion_planning.apply_cost_overrides`. Leave `tamp_overrides: {}` for stock behavior. See
[`../cfg/datagen_8gpu_neuronic.yml`](../cfg/datagen_8gpu_neuronic.yml) for the full annotated list.

## Scripts

| Script | Config | Purpose |
| --- | --- | --- |
| `datagen_8gpu_neuronic.slurm` | `cfg/datagen_8gpu_neuronic.yml` | 8-GPU scene-6 TAMP data-gen (tamp params configurable) |
| `build_pure.slurm` | `cfg/build_pure.yml` | Build + push one pure toys LeRobot dataset |
| `merge_one.slurm` | `cfg/merge_one.yml` | Merge d100 + a toys set → out repo |
| `fetch_d100.slurm` | `cfg/fetch_d100.yml` | Pre-cache a dataset into the LeRobot cache |
| `build_publish.slurm` | — | Run the full publish pipeline (no configurable knobs) |
