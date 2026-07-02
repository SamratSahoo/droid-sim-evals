"""Build + publish the toys sim datasets from the 300 raw episodes (job 3478916), then merge with
d100. Storage-conscious: each local build is deleted right after its push; d100 is fetched once for
the merges and removed at the end; the HF hub blob cache is pruned as we go. Run in the openpi venv.

Produces (all SamratSahoo/*):
  pure   : toys20_sim (20), toys100_sim (100), toys300_sim (300)
  merged : d100_toys20_sim, d100_toys100_sim, d100_toys300_sim   (d100 + the corresponding toys set)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
RAW = HERE / "runs" / "tamp_data" / "job_3478916"
ALL = HERE / "runs" / "tamp_data" / "all300"
D100 = "SamratSahoo/d100"


def free(tag=""):
    t, u, f = shutil.disk_usage("/n/fs/tamp-vla")
    print(f"[disk {tag}] free={f // 2**30}G used={u // 2**30}G", flush=True)


def prune_hub_cache(substr):
    """Delete hub blob cache dirs for a repo to reclaim space (safe: re-downloadable)."""
    hub = Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    for d in hub.glob(f"datasets--*{substr}*"):
        shutil.rmtree(d, ignore_errors=True)


# --- 1) consolidate the 300 raw episodes into one dir (interleaved across GPUs for diverse subsets) ---
if ALL.exists():
    for p in ALL.iterdir():
        p.unlink() if p.is_symlink() else shutil.rmtree(p, ignore_errors=True)
ALL.mkdir(parents=True, exist_ok=True)
idx = 0
for j in range(64):
    for i in range(8):
        src = RAW / f"gpu_{i}" / f"ep_{j:03d}"
        if not (src / "tiptop_plan.json").is_file():
            continue
        (ALL / f"ep_{idx:03d}").symlink_to(src)
        idx += 1
print(f"consolidated {idx} episodes into {ALL} (interleaved)", flush=True)
assert idx == 300, f"expected 300 episodes, found {idx}"

from tamp_data_gen import build_lerobot_dataset, INSTRUCTION  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset  # noqa: E402

api = HfApi()

# --- 2) build + push the pure toys datasets; delete each local build right after ---
for repo, n in [("SamratSahoo/toys20_sim", 20), ("SamratSahoo/toys100_sim", 100), ("SamratSahoo/toys300_sim", None)]:
    print(f"\n===== BUILD+PUSH {repo} (max_episodes={n}) =====", flush=True)
    api.delete_repo(repo, repo_type="dataset", missing_ok=True)  # fresh, no stale files
    written = build_lerobot_dataset(repo_id=repo, out_dir=str(ALL), instruction=INSTRUCTION,
                                    push=True, private=False, max_episodes=n)
    if not written:
        sys.exit(f"BUILD FAILED {repo}")
    print(f"===== {repo}: pushed {written} eps =====", flush=True)
    shutil.rmtree(HF_LEROBOT_HOME / repo, ignore_errors=True)  # free disk
    free(f"after {repo}")

# --- 3) fetch d100 once (merge needs it local) ---
print(f"\n===== fetching {D100} (for merges) =====", flush=True)
LeRobotDataset(D100)
free("after d100 fetch")

# --- 4) merge d100 + each toys set (streams toys from the Hub); each push self-deletes its build ---
for pure, out in [("SamratSahoo/toys20_sim", "SamratSahoo/d100_toys20_sim"),
                  ("SamratSahoo/toys100_sim", "SamratSahoo/d100_toys100_sim"),
                  ("SamratSahoo/toys300_sim", "SamratSahoo/d100_toys300_sim")]:
    print(f"\n===== MERGE+PUSH {out} (d100 + {pure}) =====", flush=True)
    rc = subprocess.run([sys.executable, "merge_concat_d100_toys.py",
                         "--toys-repo", pure, "--out-repo", out, "--push"]).returncode
    if rc != 0:
        sys.exit(f"MERGE FAILED {out}")
    prune_hub_cache("toys")  # drop streamed toys blobs
    free(f"after {out}")

# --- 5) cleanup d100 + hub caches ---
shutil.rmtree(HF_LEROBOT_HOME / D100, ignore_errors=True)
prune_hub_cache("d100")
shutil.rmtree(ALL, ignore_errors=True)
print("\n===== ALL DONE: 6 datasets published (toys{20,100,300}_sim + d100_toys{20,100,300}_sim) =====", flush=True)
free("final")
