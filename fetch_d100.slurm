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
# Fetch d100 into the LeRobot cache once, so the (parallel) merge jobs share it without a download race.
set -uo pipefail
set +u; source "$HOME/.bashrc"; set -u
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /n/fs/tamp-vla/tamp-vla/droid-sim-evals
echo "=== $(hostname) $(date) | fetch d100 ==="
exec /n/fs/tamp-vla/tamp-vla/openpi/.venv/bin/python -u -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; d=LeRobotDataset('SamratSahoo/d100'); print('d100 cached:', d.num_episodes, 'episodes')"
