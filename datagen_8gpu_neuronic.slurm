#!/usr/bin/env bash
#SBATCH --job-name=tamp_datagen_300
#SBATCH --account=seas
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --exclude=neu306
#SBATCH --exclusive
#SBATCH --gres=gpu:l40:8
#SBATCH --cpus-per-task=104
#SBATCH --mem=490G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Generate 300 scene-6 TAMP trajectories across 8 L40 GPUs (~38 each) on ONE node.
# One full data-gen instance per GPU. Node layout:
#   * 1 shared M2T2 grasp server on GPU 0 :8123   (tiptop's perception.m2t2.url is hardcoded to
#     localhost:8123, and M2T2 is a stateless grasp predictor, so all tiptop servers share it)
#   * 8 tiptop planning servers, server i pinned to GPU i on port 8765+i
#   * 8 Isaac workers (tamp_data_gen.py --worker), worker i pinned to GPU i, writing gpu_i/
# GPU isolation is via CUDA_VISIBLE_DEVICES (every component hardcodes device 0, so each instance
# sees its assigned GPU as cuda:0). Servers are torn down on exit. Raw episodes only (no HF push);
# the LeRobot build/merge command is printed at the end.
#
# Submit:   sbatch droid-sim-evals/datagen_8gpu_neuronic.slurm
# Override: TOTAL_EPISODES=300 NUM_ENVS=64 sbatch ... (via --export or env)

set -uo pipefail

# ---- config (override via environment) ------------------------------------
TOTAL_EPISODES="${TOTAL_EPISODES:-300}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_ENVS="${NUM_ENVS:-128}"         # parallel Isaac envs per instance. 128 matches the proven 5090
                                    # single-GPU config and fits the 46GB L40 (trio worker+tiptop+M2T2,
                                    # same as the 5090 did on 32GB). Was 64 (conservative).
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2000}" # cap on randomized attempts per instance. Scene-6 success rate is
                                     # low (run_datagen.sh budgeted ~3600 attempts/100 successes), so give
                                     # plenty of headroom; the worker stops early once it hits its N, and
                                     # is resume-safe (recounts ep_* and continues) if walltime is hit.
SCENE="${SCENE:-6}"
VARIANT="${VARIANT:-0}"
M2T2_PORT="${M2T2_PORT:-8123}"      # must be 8123: tiptop's m2t2.url is hardcoded to localhost:8123
TIPTOP_PORT_BASE="${TIPTOP_PORT_BASE:-8765}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"  # per-server readiness wait (cuRobo warmup can take minutes)
WORKER_STAGGER="${WORKER_STAGGER:-15}"    # seconds between worker starts (avoid Isaac-Kit init races)

REPO_ROOT="/n/fs/tamp-vla/tamp-vla"
DSE_DIR="$REPO_ROOT/droid-sim-evals"
M2T2_DIR="$REPO_ROOT/M2T2"
TIPTOP_DIR="$REPO_ROOT/tiptop"
OUT_ROOT="${OUT_ROOT:-$DSE_DIR/runs/tamp_data/job_${SLURM_JOB_ID:-manual}}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

# ---- environment ----------------------------------------------------------
# shellcheck disable=SC1090
set +u; source "$HOME/.bashrc"; set -u   # (/etc/bashrc trips `set -u`); brings GEMINI_API_KEY, HF_*, PIXI_CACHE_DIR, pixi PATH
export PATH="$HOME/.pixi/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES     # Isaac Sim EULA (headless: no TTY to answer the prompt)
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set (tiptop perception needs it; it is in ~/.bashrc)}"
# Thread discipline: up to ~16 CPU-heavy procs (8 tiptop + 8 workers) share this 104-core node.
# Without caps each grabs ~104 BLAS/OMP threads -> massive oversubscription, which stalled cuRobo
# warmup and tripped "Intel oneMKL ERROR ... SLASWP". Cap per-proc threads and use the GNU MKL
# threading layer (avoids the Intel-OMP conflict when many procs call MKL concurrently).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export MKL_THREADING_LAYER=GNU

echo "=== host $(hostname) | $(date) | job ${SLURM_JOB_ID:-manual} ==="
echo "=== $TOTAL_EPISODES episodes / $NUM_GPUS GPUs | num_envs=$NUM_ENVS max_attempts=$MAX_ATTEMPTS ==="
echo "=== out: $OUT_ROOT ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# ---- server bookkeeping + cleanup -----------------------------------------
SERVER_PIDS=()
cleanup() {
  echo "=== cleanup: stopping $((${#SERVER_PIDS[@]})) servers ==="
  for pid in "${SERVER_PIDS[@]:-}"; do [ -n "${pid:-}" ] || continue
    kill -INT "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true; done
  sleep 10
  for pid in "${SERVER_PIDS[@]:-}"; do [ -n "${pid:-}" ] || continue
    kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

wait_tcp() {   # host port timeout name
  local host=$1 port=$2 timeout=$3 name=$4
  local deadline=$((SECONDS + timeout))
  while [ $SECONDS -lt $deadline ]; do
    if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then exec 3>&- 2>/dev/null; echo "  $name ready ($host:$port)"; return 0; fi
    sleep 3
  done
  echo "  ERROR: $name not ready at $host:$port within ${timeout}s" >&2; return 1
}
wait_http_health() {  # url timeout name
  local url=$1 timeout=$2 name=$3
  local deadline=$((SECONDS + timeout))
  while [ $SECONDS -lt $deadline ]; do
    if curl -sf "$url" 2>/dev/null | grep -qiE "healthy|ok"; then echo "  $name healthy ($url)"; return 0; fi
    sleep 3
  done
  echo "  ERROR: $name not healthy at $url within ${timeout}s" >&2; return 1
}

# ---- 1) M2T2 grasp servers, one per GPU -----------------------------------
# One M2T2 per GPU (co-located with that GPU's tiptop + worker), each tiptop pointed at its own via
# TIPTOP_M2T2_PORT (tiptop.yml m2t2.url is env-interpolated). A single SHARED M2T2 saturates: 8
# tiptop servers x concurrent grasp calls -> 30s POST timeouts -> ~73% of plans failed. M2T2 is
# light, so launch all 8 concurrently then health-check each.
echo "=== launching $NUM_GPUS M2T2 servers (one per GPU, ports $M2T2_PORT..$((M2T2_PORT + NUM_GPUS - 1))) ==="
for ((i = 0; i < NUM_GPUS; i++)); do
  mport=$((M2T2_PORT + i))
  setsid bash -c "cd '$M2T2_DIR' && exec env CUDA_VISIBLE_DEVICES=$i pixi run python server.py --host 127.0.0.1 --port $mport" \
    > "$LOG_DIR/m2t2_$i.log" 2>&1 &
  SERVER_PIDS+=("$!")
done
for ((i = 0; i < NUM_GPUS; i++)); do
  wait_http_health "http://127.0.0.1:$((M2T2_PORT + i))/health" "$SERVER_TIMEOUT" "M2T2_$i" || { echo "M2T2_$i failed; see $LOG_DIR/m2t2_$i.log"; tail -20 "$LOG_DIR/m2t2_$i.log"; exit 1; }
done

# ---- 2) tiptop planning servers, one per GPU (STAGGERED) -------------------
# Launch each server and wait for its cuRobo warmup to finish before starting the next, so warmups
# never overlap (concurrent warmups oversubscribe CPU/MKL and stall -- mirrors run_orchestrator).
for ((i = 0; i < NUM_GPUS; i++)); do
  port=$((TIPTOP_PORT_BASE + i))
  echo "=== launching tiptop server $i on GPU$i :$port (M2T2 :$((M2T2_PORT + i)), staggered) ==="
  setsid bash -c "cd '$TIPTOP_DIR' && exec env CUDA_VISIBLE_DEVICES=$i TIPTOP_M2T2_PORT=$((M2T2_PORT + i)) pixi run python -m tiptop.tiptop_websocket_server --port $port" \
    > "$LOG_DIR/tiptop_$i.log" 2>&1 &
  SERVER_PIDS+=("$!")
  wait_tcp 127.0.0.1 "$port" "$SERVER_TIMEOUT" "tiptop_$i" || { echo "tiptop_$i failed; see $LOG_DIR/tiptop_$i.log"; tail -20 "$LOG_DIR/tiptop_$i.log"; exit 1; }
done

# ---- 3) Isaac workers in Apptainer (Isaac Sim 5.0), one per GPU ------------
# The worker runs INSIDE the Isaac Sim 5.0 container (native Isaac 4.5 gives ~0.3% scene-6 success;
# 5.0 restores ~10-15%). Servers stay native; the worker reaches them over localhost (apptainer
# shares host net). Each worker gets its OWN writable Kit dirs + container HOME (concurrent writers
# must not share). Missing worker deps are on PYTHONPATH (see container_deps).
export APPTAINER_CACHEDIR=/n/fs/tamp-vla/.cache/apptainer
SIF=/n/fs/tamp-vla/.cache/apptainer/isaac_lab_220.sif
DEPS=/n/fs/tamp-vla/.cache/container_deps
OPC=$REPO_ROOT/openpi/packages/openpi-client
CROOT="$OUT_ROOT/container"
base=$((TOTAL_EPISODES / NUM_GPUS)); rem=$((TOTAL_EPISODES % NUM_GPUS))
declare -a WORKER_PIDS WORKER_N
echo "=== launching $NUM_GPUS containerized workers (Isaac Sim 5.0; split: first $rem get $((base + 1)), rest $base) ==="
for ((i = 0; i < NUM_GPUS; i++)); do
  n=$base; [ $i -lt $rem ] && n=$((base + 1)); WORKER_N[$i]=$n
  port=$((TIPTOP_PORT_BASE + i))
  out="$OUT_ROOT/gpu_$i"; mkdir -p "$out"
  kd="$CROOT/w$i"; mkdir -p "$kd/data" "$kd/cache" "$kd/logs" "$kd/home"
  echo "  worker $i -> GPU$i, $n episodes, tiptop :$port, out gpu_$i (container)"
  apptainer exec --nv --cleanenv --home "$kd/home" \
      --bind /n/fs/tamp-vla,"$kd/data":/isaac-sim/kit/data,"$kd/cache":/isaac-sim/kit/cache,"$kd/logs":/isaac-sim/kit/logs \
      --pwd "$DSE_DIR" \
      --env PYTHONPATH="$DEPS:$OPC",OMNI_KIT_ACCEPT_EULA=YES,CUDA_VISIBLE_DEVICES=$i,OMP_NUM_THREADS=8,MKL_NUM_THREADS=8 \
      "$SIF" /isaac-sim/python.sh -u tamp_data_gen.py --worker \
        --scene "$SCENE" --variant "$VARIANT" \
        --tiptop-host 127.0.0.1 --tiptop-port "$port" --num-tiptop-servers 1 \
        --num-envs "$NUM_ENVS" --num-episodes "$n" --max-attempts "$MAX_ATTEMPTS" \
        --seed $((i * 100000)) --out-dir "$out" \
    > "$LOG_DIR/worker_$i.log" 2>&1 &
  WORKER_PIDS[$i]=$!
  sleep "$WORKER_STAGGER"
done

# ---- 4) wait for workers, collect exit codes ------------------------------
rc_total=0
for ((i = 0; i < NUM_GPUS; i++)); do
  if wait "${WORKER_PIDS[$i]}"; then rc=0; else rc=$?; fi
  got=$(find "$OUT_ROOT/gpu_$i" -maxdepth 1 -name 'ep_*' -type d 2>/dev/null | wc -l)
  echo "=== worker $i exited rc=$rc | collected $got/${WORKER_N[$i]} episodes ==="
  [ "$rc" -ne 0 ] && rc_total=1
done

# ---- 5) summary -----------------------------------------------------------
TOTAL_GOT=$(find "$OUT_ROOT"/gpu_* -maxdepth 1 -name 'ep_*' -type d 2>/dev/null | wc -l)
echo "=================================================================="
echo "=== DONE: collected $TOTAL_GOT / $TOTAL_EPISODES trajectories ==="
echo "=== raw episodes under: $OUT_ROOT/gpu_{0..$((NUM_GPUS - 1))} ==="
echo "=== To build+push a single LeRobot dataset (openpi venv), consolidate then build, e.g.:"
echo "===   python droid-sim-evals/merge_concat_d100_toys300.py   # or build_lerobot over the gathered eps"
echo "=================================================================="
exit $rc_total
