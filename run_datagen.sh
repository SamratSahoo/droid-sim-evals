#!/usr/bin/env bash
# Start or RESUME the toys100_sim data-gen (K=1/N=16, crash-safe). Safe to re-run after a crash:
# the worker counts already-collected episodes under runs/tamp_data/full and continues from there.
set -uo pipefail
cd "$(dirname "$0")"
OUT="$PWD/runs/tamp_data/full"; LOG="$PWD/runs/tamp_data/full_run.log"
mkdir -p runs/tamp_data
systemctl --user reset-failed tamp-datagen.service 2>/dev/null || true
systemctl --user stop tamp-datagen.service 2>/dev/null || true
systemd-run --user --unit=tamp-datagen --collect --working-directory="$PWD" \
  --setenv=GEMINI_API_KEY="${GEMINI_API_KEY:?set GEMINI_API_KEY (it is in ~/.bashrc)}" \
  --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=USER="$USER" \
  --setenv=XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" --setenv=LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  bash -c ".venv/bin/python -u tamp_data_gen.py --num-envs 16 --num-tiptop-servers 1 --max-attempts 800 --out-dir '$OUT' >> '$LOG' 2>&1"
echo "launched/resumed tamp-datagen.service. Watch: tail -f $LOG | grep -E 'OK [0-9]+/100|Datasets'"
