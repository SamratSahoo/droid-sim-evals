"""Build + push ONE pure LeRobot dataset from a single raw datagen job dir (gpu_*/ep_*).

Usage (openpi venv):  build_from_raw.py <raw_dir> <repo_id> [max_episodes|all]

Unlike build_dataset.py (hardcoded to runs/tamp_data/all300), this is parameterized by the raw job
dir, so each variant run (vae / rnd / vae_rnd) builds from its OWN episodes into its own repo. It
consolidates <raw_dir>/gpu_*/ep_* into <raw_dir>/consolidated as flat ep_* symlinks (interleaved
across GPUs), then builds a DROID-schema LeRobot v3.0 dataset (build_lerobot_dataset now emits
action.joint_position from cmd_joint_position) and pushes it public via hf_transfer.

STALE-BUILD TRAP: delete_repo + rmtree the local build FIRST so the "complete local build exists ->
upload only" shortcut can't re-push old data.
"""
import os
import shutil
import sys
from pathlib import Path

RAW = Path(sys.argv[1]).resolve()
REPO = sys.argv[2]
NMAX = None if len(sys.argv) < 4 or sys.argv[3] in ("all", "None", "") else int(sys.argv[3])

os.chdir(Path(__file__).resolve().parents[1])  # droid-sim-evals root
assert RAW.is_dir(), f"raw dir missing: {RAW}"

# --- consolidate gpu_*/ep_* -> flat ep_* symlinks, interleaved across GPUs ---
gpu_dirs = sorted(RAW.glob("gpu_*"))
per_gpu = [sorted(e for e in g.glob("ep_*") if (e / "tiptop_plan.json").is_file()) for g in gpu_dirs]
CONS = RAW / "consolidated"
if CONS.exists():
    for p in CONS.iterdir():
        p.unlink() if p.is_symlink() else shutil.rmtree(p, ignore_errors=True)
CONS.mkdir(parents=True, exist_ok=True)
idx = 0
for j in range(max((len(eps) for eps in per_gpu), default=0)):
    for eps in per_gpu:
        if j < len(eps):
            (CONS / f"ep_{idx:03d}").symlink_to(eps[j])
            idx += 1
print(f"consolidated {idx} episodes into {CONS} (interleaved from {len(gpu_dirs)} gpu dirs)", flush=True)
assert idx > 0, f"no episodes with tiptop_plan.json found under {RAW}/gpu_*/"

from huggingface_hub import HfApi  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME  # noqa: E402
from tamp_data_gen import INSTRUCTION, build_lerobot_dataset  # noqa: E402

print(f"=== BUILD+PUSH {REPO} from {RAW} (n={idx}, max_episodes={NMAX}) ===", flush=True)
HfApi().delete_repo(REPO, repo_type="dataset", missing_ok=True)  # fresh Hub repo, no stale files
shutil.rmtree(HF_LEROBOT_HOME / REPO, ignore_errors=True)         # kill stale local build (trap)
written = build_lerobot_dataset(repo_id=REPO, out_dir=str(CONS), instruction=INSTRUCTION,
                                push=True, private=False, max_episodes=NMAX)
if not written:
    sys.exit(f"BUILD FAILED {REPO}")
print(f"=== PUSHED {REPO}: {written} eps (with action.joint_position) ===", flush=True)
shutil.rmtree(HF_LEROBOT_HOME / REPO, ignore_errors=True)  # free disk immediately
t, u, f = shutil.disk_usage("/n/fs/tamp-vla")
print(f"=== removed local {REPO}; disk free={f // 2**30}G ===", flush=True)
