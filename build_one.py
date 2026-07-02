"""Build + push ONE pure toys LeRobot dataset from the shared all300 dir, then delete the local build.
Usage (openpi venv):  build_one.py <repo_id> <max_episodes|all>
"""
import os
import shutil
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
repo = sys.argv[1]
nmax = None if sys.argv[2] in ("all", "None", "") else int(sys.argv[2])

from huggingface_hub import HfApi  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME  # noqa: E402
from tamp_data_gen import INSTRUCTION, build_lerobot_dataset  # noqa: E402

ALL = Path("runs/tamp_data/all300")
assert (ALL).is_dir(), f"consolidated dir missing: {ALL}"

print(f"=== BUILD+PUSH {repo} (max_episodes={nmax}) ===", flush=True)
HfApi().delete_repo(repo, repo_type="dataset", missing_ok=True)  # fresh Hub repo, no stale files
# CRITICAL: also delete any stale LOCAL build, else build_lerobot_dataset's "complete local build
# exists -> skip rebuild, upload only" shortcut re-uploads old (possibly binary-gripper) data.
shutil.rmtree(HF_LEROBOT_HOME / repo, ignore_errors=True)
written = build_lerobot_dataset(repo_id=repo, out_dir=str(ALL), instruction=INSTRUCTION,
                                push=True, private=False, max_episodes=nmax)
if not written:
    sys.exit(f"BUILD FAILED {repo}")
print(f"=== PUSHED {repo}: {written} eps ===", flush=True)
shutil.rmtree(HF_LEROBOT_HOME / repo, ignore_errors=True)  # free disk immediately
t, u, f = shutil.disk_usage("/n/fs/tamp-vla")
print(f"=== removed local {repo}; disk free={f // 2**30}G ===", flush=True)
