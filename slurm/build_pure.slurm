#!/usr/bin/env bash
#SBATCH --account=seas
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --exclude=neu306
#SBATCH --cpus-per-task=12
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Build+push one pure toys dataset. Config in cfg/build_pure.yml (repo_id, max_episodes). Override
# cluster --mem/-J at submit.  Override config:  sbatch slurm/build_pure.slurm cfg/my_build.yml
set -uo pipefail
set +u; source "$HOME/.bashrc"; set -u
export HF_HUB_ENABLE_HF_TRANSFER=1
DSE_DIR=/n/fs/tamp-vla/tamp-vla/droid-sim-evals
cd "$DSE_DIR"
CFG_USER="${1:-}"
eval "$("$DSE_DIR/.venv/bin/python" "$DSE_DIR/slurm/lib/load_config.py" \
        --default "$DSE_DIR/cfg/build_pure.yml" ${CFG_USER:+--config "$CFG_USER"})"
REPO_ID="${REPO_ID:?repo_id not set in config}"
MAX_EPISODES="${MAX_EPISODES:-all}"
echo "=== $(hostname) $(date) | build_pure $REPO_ID $MAX_EPISODES (job ${SLURM_JOB_ID:-}) ==="
exec /n/fs/tamp-vla/tamp-vla/openpi/.venv/bin/python -u data/build_dataset.py "$REPO_ID" "$MAX_EPISODES"
