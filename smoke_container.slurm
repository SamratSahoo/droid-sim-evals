#!/usr/bin/env bash
#SBATCH --job-name=tamp_smokec
#SBATCH --account=seas
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --exclude=neu306
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Containerized smoke: native M2T2 + tiptop (pixi on host) + Isaac worker inside the Isaac Sim 5.0
# Apptainer container. Confirms the worker runs on 5.0 and measures the scene-6 success rate.

set -uo pipefail
set +u; source "$HOME/.bashrc"; set -u
export PATH="$HOME/.pixi/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set}"

REPO=/n/fs/tamp-vla/tamp-vla
SIF=/n/fs/tamp-vla/.cache/apptainer/isaac_lab_220.sif
DEPS=/n/fs/tamp-vla/.cache/container_deps
OPC=$REPO/openpi/packages/openpi-client
CHOME=/n/fs/tamp-vla/.cache/container_home; mkdir -p "$CHOME"
# Isaac Sim needs writable Kit dirs, but the SIF is read-only -> bind empty host dirs over them
# (they're absent/empty in the image; Kit populates them at runtime).
KITW=/n/fs/tamp-vla/.cache/kit_smoke; mkdir -p "$KITW/data" "$KITW/cache" "$KITW/logs"
OUT=$REPO/droid-sim-evals/runs/tamp_data/smokec_${SLURM_JOB_ID:-manual}
LOG=$OUT/logs; mkdir -p "$LOG"
M2T2_PORT=8123; TIP_PORT=8765
NUM_ENVS="${NUM_ENVS:-16}"; MAX_ATTEMPTS="${MAX_ATTEMPTS:-64}"

echo "=== $(hostname) $(date) | containerized smoke (Isaac Sim 5.0) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1

SERVER_PIDS=()
cleanup(){ for p in "${SERVER_PIDS[@]:-}"; do [ -n "${p:-}" ]||continue; kill -INT "-$p" 2>/dev/null||kill -INT "$p" 2>/dev/null||true; done; sleep 8; for p in "${SERVER_PIDS[@]:-}"; do [ -n "${p:-}" ]||continue; kill -KILL "-$p" 2>/dev/null||true; done; }
trap cleanup EXIT TERM INT

echo "=== native M2T2 on GPU0:$M2T2_PORT ==="
setsid bash -c "cd $REPO/M2T2 && exec env CUDA_VISIBLE_DEVICES=0 pixi run python server.py --host 127.0.0.1 --port $M2T2_PORT" > "$LOG/m2t2.log" 2>&1 & SERVER_PIDS+=($!)
echo "=== native tiptop on GPU0:$TIP_PORT ==="
setsid bash -c "cd $REPO/tiptop && exec env CUDA_VISIBLE_DEVICES=0 TIPTOP_M2T2_PORT=$M2T2_PORT pixi run python -m tiptop.tiptop_websocket_server --port $TIP_PORT" > "$LOG/tiptop.log" 2>&1 & SERVER_PIDS+=($!)

ok=0; for i in $(seq 1 400); do curl -sf "http://127.0.0.1:$M2T2_PORT/health" 2>/dev/null | grep -qi healthy && { ok=1; break; }; sleep 3; done
[ $ok = 1 ] && echo "M2T2 READY" || { echo "M2T2 FAILED"; tail -20 "$LOG/m2t2.log"; exit 1; }
ok=0; for i in $(seq 1 600); do (exec 3<>"/dev/tcp/127.0.0.1/$TIP_PORT") 2>/dev/null && { exec 3>&-; ok=1; break; }; sleep 3; done
[ $ok = 1 ] && echo "TIPTOP READY" || { echo "TIPTOP FAILED"; tail -20 "$LOG/tiptop.log"; exit 1; }

echo "=== worker in container: num_envs=$NUM_ENVS, max_attempts=$MAX_ATTEMPTS (rate probe) ==="
export APPTAINER_CACHEDIR=/n/fs/tamp-vla/.cache/apptainer
apptainer exec --nv --cleanenv --home "$CHOME" \
  --bind /n/fs/tamp-vla,"$KITW/data":/isaac-sim/kit/data,"$KITW/cache":/isaac-sim/kit/cache,"$KITW/logs":/isaac-sim/kit/logs \
  --pwd "$REPO/droid-sim-evals" \
  --env PYTHONPATH="$DEPS:$OPC",OMNI_KIT_ACCEPT_EULA=YES,CUDA_VISIBLE_DEVICES=0 \
  "$SIF" /isaac-sim/python.sh -u tamp_data_gen.py --worker \
    --scene 6 --variant 0 --tiptop-host 127.0.0.1 --tiptop-port $TIP_PORT --num-tiptop-servers 1 \
    --num-envs "$NUM_ENVS" --num-episodes 100 --max-attempts "$MAX_ATTEMPTS" --seed 0 --out-dir "$OUT" \
  2>&1 | tee "$LOG/worker.log"
rc=${PIPESTATUS[0]}
echo "WORKER_RC=$rc"
echo "episodes: $(find "$OUT" -maxdepth 1 -name 'ep_*' -type d 2>/dev/null | wc -l)"
grep -oE "rate [0-9]+/[0-9]+" "$LOG/worker.log" | tail -1
echo "SMOKE_DONE"
