#!/usr/bin/env bash
#SBATCH --account=seas
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --exclude=neu306
#SBATCH --cpus-per-task=12
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Build+push one pure toys dataset. Args: <repo_id> <max_episodes|all>. Override --mem/-J at submit.
set -uo pipefail
set +u; source "$HOME/.bashrc"; set -u
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /n/fs/tamp-vla/tamp-vla/droid-sim-evals
echo "=== $(hostname) $(date) | build_pure $1 $2 (job ${SLURM_JOB_ID:-}) ==="
exec /n/fs/tamp-vla/tamp-vla/openpi/.venv/bin/python -u build_one.py "$1" "$2"
