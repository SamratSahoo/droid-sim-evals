#!/usr/bin/env bash
#SBATCH --job-name=tamp_smoke1
#SBATCH --account=seas
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# 1-GPU end-to-end smoke of the datagen pipeline on a COMPUTE node: M2T2 + 1 tiptop + 1 worker
# (4 envs, 1 episode). Confirms Isaac Sim 4.5 RTX rendering works on the compute nodes' driver
# before committing the full 8-GPU job.

set -uo pipefail
set +u; source "$HOME/.bashrc"; set -u
export PATH="$HOME/.pixi/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set}"

REPO=/n/fs/tamp-vla/tamp-vla
OUT="$REPO/droid-sim-evals/runs/tamp_data/smoke_${SLURM_JOB_ID:-manual}"
LOG="$OUT/logs"; mkdir -p "$LOG"
M2T2_PORT=8123; TIP_PORT=8765

echo "=== $(hostname) $(date) | Isaac smoke on compute node ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1

SERVER_PIDS=()
cleanup(){ for p in "${SERVER_PIDS[@]:-}"; do [ -n "${p:-}" ]||continue; kill -INT "-$p" 2>/dev/null||kill -INT "$p" 2>/dev/null||true; done; sleep 8; for p in "${SERVER_PIDS[@]:-}"; do [ -n "${p:-}" ]||continue; kill -KILL "-$p" 2>/dev/null||true; done; }
trap cleanup EXIT TERM INT

echo "=== M2T2 on GPU0:$M2T2_PORT ==="
setsid bash -c "cd $REPO/M2T2 && exec env CUDA_VISIBLE_DEVICES=0 pixi run python server.py --host 127.0.0.1 --port $M2T2_PORT" > "$LOG/m2t2.log" 2>&1 & SERVER_PIDS+=($!)
echo "=== tiptop on GPU0:$TIP_PORT ==="
setsid bash -c "cd $REPO/tiptop && exec env CUDA_VISIBLE_DEVICES=0 pixi run python -m tiptop.tiptop_websocket_server --port $TIP_PORT" > "$LOG/tiptop.log" 2>&1 & SERVER_PIDS+=($!)

echo "=== wait M2T2 health ==="; ok=0
for i in $(seq 1 400); do curl -sf "http://127.0.0.1:$M2T2_PORT/health" 2>/dev/null | grep -qi healthy && { ok=1; break; }; sleep 3; done
[ $ok = 1 ] && echo "M2T2 READY" || { echo "M2T2 FAILED"; tail -30 "$LOG/m2t2.log"; exit 1; }
echo "=== wait tiptop TCP ==="; ok=0
for i in $(seq 1 600); do (exec 3<>"/dev/tcp/127.0.0.1/$TIP_PORT") 2>/dev/null && { exec 3>&-; ok=1; break; }; sleep 3; done
[ $ok = 1 ] && echo "TIPTOP READY" || { echo "TIPTOP FAILED"; tail -30 "$LOG/tiptop.log"; exit 1; }

echo "=== worker: 4 envs, 1 episode, max_attempts 8 ==="
cd "$REPO/droid-sim-evals"
CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES timeout 1500 .venv/bin/python -u tamp_data_gen.py --worker \
  --scene 6 --variant 0 --tiptop-host 127.0.0.1 --tiptop-port $TIP_PORT --num-tiptop-servers 1 \
  --num-envs 4 --num-episodes 1 --max-attempts 8 --seed 0 --out-dir "$OUT" 2>&1 | tee "$LOG/worker.log"
rc=${PIPESTATUS[0]}
echo "WORKER_RC=$rc"
echo "episodes collected: $(find "$OUT" -maxdepth 1 -name 'ep_*' -type d 2>/dev/null | wc -l)"
echo "SMOKE_DONE"
