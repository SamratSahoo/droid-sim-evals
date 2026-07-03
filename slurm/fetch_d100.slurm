#!/usr/bin/env bash
#SBATCH --job-name=pub_d100
#SBATCH --account=seas
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --exclude=neu306
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Fetch a dataset into the LeRobot cache once, so the (parallel) merge jobs share it without a
# download race. Config in cfg/fetch_d100.yml (repo).  Override:  sbatch slurm/fetch_d100.slurm cfg/my.yml
set -uo pipefail
set +u; source "$HOME/.bashrc"; set -u
export HF_HUB_ENABLE_HF_TRANSFER=1
DSE_DIR=/n/fs/tamp-vla/tamp-vla/droid-sim-evals
cd "$DSE_DIR"
CFG_USER="${1:-}"
eval "$("$DSE_DIR/.venv/bin/python" "$DSE_DIR/slurm/lib/load_config.py" \
        --default "$DSE_DIR/cfg/fetch_d100.yml" ${CFG_USER:+--config "$CFG_USER"})"
export REPO="${REPO:?repo not set in config}"
echo "=== $(hostname) $(date) | fetch $REPO ==="
exec /n/fs/tamp-vla/tamp-vla/openpi/.venv/bin/python -u -c "import os; from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; r=os.environ['REPO']; d=LeRobotDataset(r); print('cached:', r, d.num_episodes, 'episodes')"
