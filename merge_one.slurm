#!/usr/bin/env bash
#SBATCH --account=seas
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --exclude=neu306
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Merge d100 (local cache) + a toys set (streamed from Hub) -> out repo, push, self-delete build.
# Args: <toys_repo> <out_repo>.  Requires d100 already cached (dependency on the d100-fetch job).
set -uo pipefail
set +u; source "$HOME/.bashrc"; set -u
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /n/fs/tamp-vla/tamp-vla/droid-sim-evals
echo "=== $(hostname) $(date) | merge $1 -> $2 (job ${SLURM_JOB_ID:-}) ==="
exec /n/fs/tamp-vla/tamp-vla/openpi/.venv/bin/python -u merge_concat_d100_toys.py --toys-repo "$1" --out-repo "$2" --push
