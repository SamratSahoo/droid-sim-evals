"""Recovery: tag the already-uploaded LeRobot datasets (so they're loadable), then build the
two d100 merges. Run in the openpi venv. Idempotent / re-runnable."""
import json, shutil, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
import tamp_data_gen as T

api = HfApi()

def tag_repo(repo_id: str):
    local = HF_LEROBOT_HOME / repo_id / "meta" / "info.json"
    if local.is_file():
        ver = json.loads(local.read_text()).get("codebase_version")
    else:
        p = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
        ver = json.loads(Path(p).read_text()).get("codebase_version")
    if not ver:
        print(f"  WARN no codebase_version for {repo_id}"); return None
    api.create_tag(repo_id, tag=ver, repo_type="dataset", exist_ok=True)
    print(f"  tagged {repo_id} -> {ver}", flush=True)
    return ver

print("=== tagging existing datasets ===", flush=True)
for r in ["SamratSahoo/toys100_sim", "SamratSahoo/toys20_sim", "SamratSahoo/d100"]:
    tag_repo(r)

print("=== merge 1: d100 + toys100 -> d100_toys100_sim ===", flush=True)
T.merge_datasets(sources=["SamratSahoo/d100", "SamratSahoo/toys100_sim"],
                 repo_id="SamratSahoo/d100_toys100_sim", push=True, private=False)
# free big local copies before merge 2
shutil.rmtree(HF_LEROBOT_HOME / "SamratSahoo/d100_toys100_sim", ignore_errors=True)
shutil.rmtree(HF_LEROBOT_HOME / "SamratSahoo/toys100_sim", ignore_errors=True)
print("=== merge 2: d100 + toys20 -> d100_toys20_sim ===", flush=True)
T.merge_datasets(sources=["SamratSahoo/d100", "SamratSahoo/toys20_sim"],
                 repo_id="SamratSahoo/d100_toys20_sim", push=True, private=False)
print("ALL DONE: 4 datasets on HF", flush=True)
