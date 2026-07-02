#!/usr/bin/env bash
#SBATCH --job-name=toys_publish
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
# Build + publish toys{20,100,300}_sim and d100_toys{20,100,300}_sim from the 300 raw episodes.
# CPU-only (no GPU): the 300-ep LeRobot build needs ~80 GB RAM (hence --mem=200G). Storage-conscious:
# publish_pipeline.py deletes each local build after push and prunes caches. hf_transfer enabled.

set -uo pipefail
set +u; source "$HOME/.bashrc"; set -u   # HF_HOME/token, caches on project FS
export HF_HUB_ENABLE_HF_TRANSFER=1       # high-speed HF uploads/downloads

cd /n/fs/tamp-vla/tamp-vla/droid-sim-evals
echo "=== $(hostname) $(date) | publish pipeline (job ${SLURM_JOB_ID:-manual}) ==="
df -h /n/fs/tamp-vla | tail -1
exec /n/fs/tamp-vla/tamp-vla/openpi/.venv/bin/python -u publish_pipeline.py
