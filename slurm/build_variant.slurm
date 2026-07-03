#!/usr/bin/env bash
#SBATCH --job-name=build_variant
#SBATCH --account=seas
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --exclude=neu306
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Build + push ONE pure LeRobot dataset from a single raw datagen job dir (gpu_*/ep_*).
# CPU-only; the 300-ep LeRobot build needs ~80 GB RAM (hence --mem=200G). Meant to run as an
# afterok dependency of a datagen job (raw dir = runs/tamp_data/job_<datagen_jobid>).
#
# Usage:  sbatch slurm/build_variant.slurm <raw_dir> <repo_id> [max_episodes]
set -uo pipefail
set +u; source "$HOME/.bashrc"; set -u   # HF_HOME/token, caches on project FS
export HF_HUB_ENABLE_HF_TRANSFER=1       # high-speed HF uploads
DSE_DIR=/n/fs/tamp-vla/tamp-vla/droid-sim-evals
cd "$DSE_DIR"
RAW="${1:?raw_dir required}"
REPO="${2:?repo_id required}"
NMAX="${3:-all}"
echo "=== $(hostname) $(date) | build_variant $REPO from $RAW ($NMAX) job ${SLURM_JOB_ID:-} ==="
df -h /n/fs/tamp-vla | tail -1
exec /n/fs/tamp-vla/tamp-vla/openpi/.venv/bin/python -u data/build_from_raw.py "$RAW" "$REPO" "$NMAX"
