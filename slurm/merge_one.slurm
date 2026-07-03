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
# Config in cfg/merge_one.yml (toys_repo, out_repo). Requires d100 already cached (dependency on the
# d100-fetch job).  Override config:  sbatch slurm/merge_one.slurm cfg/my_merge.yml
set -uo pipefail
set +u; source "$HOME/.bashrc"; set -u
export HF_HUB_ENABLE_HF_TRANSFER=1
DSE_DIR=/n/fs/tamp-vla/tamp-vla/droid-sim-evals
cd "$DSE_DIR"
CFG_USER="${1:-}"
eval "$("$DSE_DIR/.venv/bin/python" "$DSE_DIR/slurm/lib/load_config.py" \
        --default "$DSE_DIR/cfg/merge_one.yml" ${CFG_USER:+--config "$CFG_USER"})"
TOYS_REPO="${TOYS_REPO:?toys_repo not set in config}"
OUT_REPO="${OUT_REPO:?out_repo not set in config}"
echo "=== $(hostname) $(date) | merge $TOYS_REPO -> $OUT_REPO (job ${SLURM_JOB_ID:-}) ==="
exec /n/fs/tamp-vla/tamp-vla/openpi/.venv/bin/python -u data/merge_d100_toys.py --toys-repo "$TOYS_REPO" --out-repo "$OUT_REPO" --push
