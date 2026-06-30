"""Low-memory merge of d100 (local) + toys300_sim (Hub) -> d100_toys300_sim (v2.1).

Why this exists: the add_frame-based merge_datasets re-encodes every image and its memory
footprint grows with total frames; at 400 episodes it exceeds this 30G host (systemd-oomd /
kernel OOM kill it). This builds the merged dataset by CONCATENATING parquet files instead:
  * d100's 100 episode parquets are copied VERBATIM (their episode/task/global-index values are
    already correct in the merged dataset, since d100 comes first and keeps task_index 0..46).
  * toys300's 300 parquets are streamed from the Hub one at a time; only 3 scalar int columns are
    rewritten (episode_index += 100, global `index` continues the counter, task_index remapped),
    using pyarrow so the inline-image struct column is carried over untouched (no decode/encode).
Peak memory is ~one episode parquet (~300MB), so it cannot OOM. Per-episode stats are copied
(they're independent), tasks unioned, info.json totals recomputed.

Builds locally and VERIFIES; does NOT push (pass --push to upload after verification passes).
Run under the openpi venv:  ../openpi/.venv/bin/python merge_concat_d100_toys300.py [--push]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME  # noqa: E402

D100_REPO = "SamratSahoo/d100"
TOYS_REPO = "SamratSahoo/toys300_sim"
OUT_REPO = "SamratSahoo/d100_toys300_sim"
D100_DIR = HF_LEROBOT_HOME / D100_REPO
OUT_DIR = HF_LEROBOT_HOME / OUT_REPO
SCRATCH = Path(__file__).resolve().parent / "runs" / "tamp_data" / "_concat_scratch"


def _read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _write_jsonl(p, rows):
    Path(p).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _hub_meta(repo, rel):
    return hf_hub_download(repo, rel, repo_type="dataset", local_dir=str(SCRATCH / "meta"))


def _rewrite_cols(table, *, episode_index, index_start, task_remap):
    """Return a copy of `table` with episode_index/index/task_index replaced; other cols untouched."""
    n = table.num_rows
    cols = {name: table.column(name) for name in table.column_names}
    cols["episode_index"] = pa.array([episode_index] * n, type=table.schema.field("episode_index").type)
    cols["index"] = pa.array(list(range(index_start, index_start + n)), type=table.schema.field("index").type)
    old_ti = table.column("task_index").to_pylist()
    cols["task_index"] = pa.array([task_remap[t] for t in old_ti], type=table.schema.field("task_index").type)
    return pa.table([cols[name] for name in table.column_names], schema=table.schema)


def build():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "meta").mkdir(parents=True)
    (OUT_DIR / "data" / "chunk-000").mkdir(parents=True)
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    os.environ["HF_HUB_CACHE"] = str(SCRATCH / "hub")

    d100_info = json.loads((D100_DIR / "meta" / "info.json").read_text())
    d100_eps = _read_jsonl(D100_DIR / "meta" / "episodes.jsonl")
    d100_tasks = _read_jsonl(D100_DIR / "meta" / "tasks.jsonl")
    d100_estats = {e["episode_index"]: e for e in _read_jsonl(D100_DIR / "meta" / "episodes_stats.jsonl")}
    n_d100 = d100_info["total_episodes"]

    toys_info = json.loads(Path(_hub_meta(TOYS_REPO, "meta/info.json")).read_text())
    toys_eps = {e["episode_index"]: e for e in _read_jsonl(_hub_meta(TOYS_REPO, "meta/episodes.jsonl"))}
    toys_tasks = _read_jsonl(_hub_meta(TOYS_REPO, "meta/tasks.jsonl"))
    toys_estats = {e["episode_index"]: e for e in _read_jsonl(_hub_meta(TOYS_REPO, "meta/episodes_stats.jsonl"))}
    n_toys = toys_info["total_episodes"]
    data_path = toys_info["data_path"]

    # Merged tasks: keep d100 task_index values; append toys tasks not already present.
    task_to_idx = {t["task"]: t["task_index"] for t in d100_tasks}
    merged_tasks = list(d100_tasks)
    nxt = max(t["task_index"] for t in d100_tasks) + 1
    toys_remap = {}
    for t in toys_tasks:
        if t["task"] not in task_to_idx:
            task_to_idx[t["task"]] = nxt
            merged_tasks.append({"task_index": nxt, "task": t["task"]})
            nxt += 1
        toys_remap[t["task_index"]] = task_to_idx[t["task"]]

    out_eps, out_estats = [], []
    gidx = 0  # running global frame index

    # d100: copy parquets verbatim (already correct), carry meta through.
    for ep in range(n_d100):
        src = D100_DIR / f"data/chunk-000/episode_{ep:06d}.parquet"
        dst = OUT_DIR / f"data/chunk-000/episode_{ep:06d}.parquet"
        shutil.copy(src, dst)
        gidx += d100_eps[ep]["length"]
        out_eps.append(d100_eps[ep])
        out_estats.append(d100_estats[ep])
    print(f"copied {n_d100} d100 episodes verbatim; global index now {gidx}", flush=True)

    # toys300: stream each parquet from Hub, rewrite scalar cols, write; per-episode memory only.
    for ep in range(n_toys):
        rel = data_path.format(episode_chunk=0, episode_index=ep)
        local = hf_hub_download(TOYS_REPO, rel, repo_type="dataset", local_dir=str(SCRATCH / "ep"))
        table = pq.read_table(local)
        new_ep = n_d100 + ep
        table = _rewrite_cols(table, episode_index=new_ep, index_start=gidx, task_remap=toys_remap)
        pq.write_table(table, OUT_DIR / f"data/chunk-000/episode_{new_ep:06d}.parquet")
        gidx += table.num_rows
        e = toys_eps[ep]
        out_eps.append({"episode_index": new_ep, "tasks": e["tasks"], "length": e["length"]})
        es = dict(toys_estats[ep])
        es["episode_index"] = new_ep
        out_estats.append(es)
        shutil.rmtree(SCRATCH / "ep", ignore_errors=True)
        shutil.rmtree(SCRATCH / "hub", ignore_errors=True)
        if (ep + 1) % 25 == 0 or ep + 1 == n_toys:
            print(f"  merged {ep + 1}/{n_toys} toys300 episodes (global frames {gidx})", flush=True)

    # Meta
    _write_jsonl(OUT_DIR / "meta" / "episodes.jsonl", out_eps)
    _write_jsonl(OUT_DIR / "meta" / "episodes_stats.jsonl", out_estats)
    _write_jsonl(OUT_DIR / "meta" / "tasks.jsonl", merged_tasks)
    info = dict(d100_info)
    info.update(total_episodes=n_d100 + n_toys, total_frames=gidx, total_tasks=len(merged_tasks),
                total_chunks=1, total_videos=0, splits={"train": f"0:{n_d100 + n_toys}"})
    (OUT_DIR / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f"BUILT {OUT_REPO}: {info['total_episodes']} eps, {info['total_frames']} frames, "
          f"{info['total_tasks']} tasks", flush=True)
    return info


def verify(info):
    import numpy as np
    from PIL import Image
    import io

    parqs = sorted((OUT_DIR / "data" / "chunk-000").glob("episode_*.parquet"))
    eps = _read_jsonl(OUT_DIR / "meta" / "episodes.jsonl")
    estats = _read_jsonl(OUT_DIR / "meta" / "episodes_stats.jsonl")
    tasks = _read_jsonl(OUT_DIR / "meta" / "tasks.jsonl")
    checks = []
    checks.append(("400 parquet files", len(parqs) == 400))
    checks.append(("400 episodes.jsonl", len(eps) == 400))
    checks.append(("400 episodes_stats", len(estats) == 400))
    checks.append(("info total_episodes==400", info["total_episodes"] == 400))
    checks.append(("episode_index 0..399 contiguous", [e["episode_index"] for e in eps] == list(range(400))))
    checks.append(("total_frames == sum(lengths)", info["total_frames"] == sum(e["length"] for e in eps)))
    # frame-level: decode an image from a d100 episode and a toys300 episode
    img_ok = True
    img_keys = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")
    for ep in (0, 399):
        t = pq.read_table(OUT_DIR / f"data/chunk-000/episode_{ep:06d}.parquet").to_pylist()
        row = t[0]
        if int(row["episode_index"]) != ep:
            img_ok = False
        for k in img_keys:
            cell = row[k]
            b = cell["bytes"] if isinstance(cell, dict) else cell
            arr = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
            if arr.shape != (180, 320, 3):
                img_ok = False
    checks.append(("sample frames decode (ep 0 + 399, 180x320x3)", img_ok))
    # global index contiguous across the whole dataset
    idx_ok = True
    g = 0
    for ep in range(400):
        t = pq.read_table(OUT_DIR / f"data/chunk-000/episode_{ep:06d}.parquet")
        col = t.column("index").to_pylist()
        if col != list(range(g, g + len(col))):
            idx_ok = False
            break
        g += len(col)
    checks.append(("global index contiguous 0..N", idx_ok))
    checks.append(("task_index in range", all(0 <= ti < len(tasks) for ep in (399,)
                   for ti in pq.read_table(OUT_DIR / f"data/chunk-000/episode_{ep:06d}.parquet").column("task_index").to_pylist())))
    print("\n=== VERIFICATION ===", flush=True)
    allok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        allok = allok and ok
    print(f"=== {'ALL CHECKS PASSED' if allok else 'SOME CHECKS FAILED'} ===", flush=True)
    return allok


def push():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tamp_data_gen import _upload_dataset
    from huggingface_hub import HfApi
    try:
        HfApi().delete_repo(OUT_REPO, repo_type="dataset", missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
    _upload_dataset(OUT_DIR, OUT_REPO, False, ["droid", "panda", "tamp-vla", "merged"])
    print(f"PUSHED {OUT_REPO}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="upload after verification passes")
    args = ap.parse_args()
    info = build()
    ok = verify(info)
    if not ok:
        print("VERIFICATION FAILED -> NOT pushing.", flush=True)
        sys.exit(1)
    if args.push:
        push()
    else:
        print("Build verified locally. Re-run with --push to upload (or it will be pushed on confirmation).", flush=True)
