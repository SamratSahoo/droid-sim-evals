"""Low-memory merge of d100 (local) + a toys{N}_sim dataset (streamed from the Hub) -> d100_toys{N}_sim.

Parameterized generalization of merge_concat_d100_toys300.py. The add_frame-based merge re-encodes
every image and OOM-kills at ~400 episodes on this 30G host; this CONCATENATES parquet files instead:
  * d100's episode parquets are copied VERBATIM (their episode/task/global-index values are already
    correct, since d100 comes first and keeps task_index 0..N).
  * the toys set's parquets are streamed from the Hub one at a time; only 3 scalar int columns are
    rewritten (episode_index += n_d100, global `index` continues, task_index remapped) via pyarrow so
    the inline-image struct column is carried over untouched (no decode/encode). Peak RSS ~one parquet.

Usage (openpi venv):
  ../openpi/.venv/bin/python merge_concat_d100_toys.py --toys-repo SamratSahoo/toys20_sim \
      --out-repo SamratSahoo/pi05droid_d100_toys20_sim [--push]
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME  # noqa: E402

D100_REPO = "SamratSahoo/d100"
SCRATCH = Path(__file__).resolve().parent / "runs" / "tamp_data" / "_concat_scratch"


def _read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _write_jsonl(p, rows):
    Path(p).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _hub_meta(repo, rel):
    return hf_hub_download(repo, rel, repo_type="dataset", local_dir=str(SCRATCH / "meta"))


def _hub_dl(repo, rel, tries=8):
    """hf_hub_download with backoff for the 1000-req/5-min rate limit (429)."""
    for i in range(tries):
        try:
            return hf_hub_download(repo, rel, repo_type="dataset", local_dir=str(SCRATCH / "ep"))
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) or "Too Many" in str(e) or "rate" in str(e).lower():
                w = 20 * (i + 1)
                print(f"  429/backoff on {rel} -> sleep {w}s", flush=True)
                time.sleep(w)
            else:
                print(f"  ERR {type(e).__name__} {str(e)[:140]} (try {i+1})", flush=True)
                time.sleep(5)
    raise RuntimeError(f"failed to download {rel}")


def _rewrite_cols(table, *, episode_index, index_start, task_remap):
    n = table.num_rows
    cols = {name: table.column(name) for name in table.column_names}
    cols["episode_index"] = pa.array([episode_index] * n, type=table.schema.field("episode_index").type)
    cols["index"] = pa.array(list(range(index_start, index_start + n)), type=table.schema.field("index").type)
    old_ti = table.column("task_index").to_pylist()
    cols["task_index"] = pa.array([task_remap[t] for t in old_ti], type=table.schema.field("task_index").type)
    return pa.table([cols[name] for name in table.column_names], schema=table.schema)


def build(toys_repo, out_repo):
    d100_dir = HF_LEROBOT_HOME / D100_REPO
    out_dir = HF_LEROBOT_HOME / out_repo
    if not (d100_dir / "meta" / "info.json").is_file():
        raise RuntimeError(f"d100 not local at {d100_dir}; fetch it first (LeRobotDataset('{D100_REPO}')).")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "meta").mkdir(parents=True)
    (out_dir / "data" / "chunk-000").mkdir(parents=True)
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    os.environ["HF_HUB_CACHE"] = str(SCRATCH / "hub")

    d100_info = json.loads((d100_dir / "meta" / "info.json").read_text())
    d100_eps = _read_jsonl(d100_dir / "meta" / "episodes.jsonl")
    d100_tasks = _read_jsonl(d100_dir / "meta" / "tasks.jsonl")
    d100_estats = {e["episode_index"]: e for e in _read_jsonl(d100_dir / "meta" / "episodes_stats.jsonl")}
    n_d100 = d100_info["total_episodes"]

    toys_info = json.loads(Path(_hub_meta(toys_repo, "meta/info.json")).read_text())
    toys_eps = {e["episode_index"]: e for e in _read_jsonl(_hub_meta(toys_repo, "meta/episodes.jsonl"))}
    toys_tasks = _read_jsonl(_hub_meta(toys_repo, "meta/tasks.jsonl"))
    toys_estats = {e["episode_index"]: e for e in _read_jsonl(_hub_meta(toys_repo, "meta/episodes_stats.jsonl"))}
    n_toys = toys_info["total_episodes"]
    data_path = toys_info["data_path"]

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
    gidx = 0
    for ep in range(n_d100):
        shutil.copy(d100_dir / f"data/chunk-000/episode_{ep:06d}.parquet",
                    out_dir / f"data/chunk-000/episode_{ep:06d}.parquet")
        gidx += d100_eps[ep]["length"]
        out_eps.append(d100_eps[ep])
        out_estats.append(d100_estats[ep])
    print(f"copied {n_d100} d100 episodes verbatim; global index now {gidx}", flush=True)

    for ep in range(n_toys):
        rel = data_path.format(episode_chunk=0, episode_index=ep)
        local = _hub_dl(toys_repo, rel)
        table = pq.read_table(local)
        new_ep = n_d100 + ep
        table = _rewrite_cols(table, episode_index=new_ep, index_start=gidx, task_remap=toys_remap)
        pq.write_table(table, out_dir / f"data/chunk-000/episode_{new_ep:06d}.parquet")
        gidx += table.num_rows
        e = toys_eps[ep]
        out_eps.append({"episode_index": new_ep, "tasks": e["tasks"], "length": e["length"]})
        es = dict(toys_estats[ep]); es["episode_index"] = new_ep
        out_estats.append(es)
        shutil.rmtree(SCRATCH / "ep", ignore_errors=True)
        shutil.rmtree(SCRATCH / "hub", ignore_errors=True)
        if (ep + 1) % 25 == 0 or ep + 1 == n_toys:
            print(f"  merged {ep + 1}/{n_toys} toys episodes (global frames {gidx})", flush=True)

    _write_jsonl(out_dir / "meta" / "episodes.jsonl", out_eps)
    _write_jsonl(out_dir / "meta" / "episodes_stats.jsonl", out_estats)
    _write_jsonl(out_dir / "meta" / "tasks.jsonl", merged_tasks)
    info = dict(d100_info)
    info.update(total_episodes=n_d100 + n_toys, total_frames=gidx, total_tasks=len(merged_tasks),
                total_chunks=1, total_videos=0, splits={"train": f"0:{n_d100 + n_toys}"})
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f"BUILT {out_repo}: {info['total_episodes']} eps, {info['total_frames']} frames, "
          f"{info['total_tasks']} tasks", flush=True)
    return info, n_d100, n_toys


def verify(out_repo, info, n_d100, n_toys):
    import numpy as np
    from PIL import Image
    import io
    out_dir = HF_LEROBOT_HOME / out_repo
    total = n_d100 + n_toys
    parqs = sorted((out_dir / "data" / "chunk-000").glob("episode_*.parquet"))
    eps = _read_jsonl(out_dir / "meta" / "episodes.jsonl")
    estats = _read_jsonl(out_dir / "meta" / "episodes_stats.jsonl")
    tasks = _read_jsonl(out_dir / "meta" / "tasks.jsonl")
    checks = [
        (f"{total} parquet files", len(parqs) == total),
        (f"{total} episodes.jsonl", len(eps) == total),
        (f"{total} episodes_stats", len(estats) == total),
        (f"info total_episodes=={total}", info["total_episodes"] == total),
        ("episode_index contiguous", [e["episode_index"] for e in eps] == list(range(total))),
        ("total_frames==sum(lengths)", info["total_frames"] == sum(e["length"] for e in eps)),
    ]
    img_ok = True
    for ep in (0, total - 1):  # one d100 episode + one toys episode
        row = pq.read_table(out_dir / f"data/chunk-000/episode_{ep:06d}.parquet").to_pylist()[0]
        if int(row["episode_index"]) != ep:
            img_ok = False
        for k in ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"):
            cell = row[k]; b = cell["bytes"] if isinstance(cell, dict) else cell
            if np.asarray(Image.open(io.BytesIO(b)).convert("RGB")).shape != (180, 320, 3):
                img_ok = False
    checks.append(("sample frames decode (ep 0 + last, 180x320x3)", img_ok))
    idx_ok = True; g = 0
    for ep in range(total):
        col = pq.read_table(out_dir / f"data/chunk-000/episode_{ep:06d}.parquet").column("index").to_pylist()
        if col != list(range(g, g + len(col))):
            idx_ok = False; break
        g += len(col)
    checks.append(("global index contiguous 0..N", idx_ok))
    print("\n=== VERIFICATION ===", flush=True)
    allok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True); allok = allok and ok
    print(f"=== {'ALL CHECKS PASSED' if allok else 'SOME CHECKS FAILED'} ===", flush=True)
    return allok


def push(out_repo):
    from tamp_data_gen import _upload_dataset
    from huggingface_hub import HfApi
    out_dir = HF_LEROBOT_HOME / out_repo
    try:
        HfApi().delete_repo(out_repo, repo_type="dataset", missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
    _upload_dataset(out_dir, out_repo, False, ["droid", "panda", "tamp-vla", "merged"])
    print(f"PUSHED {out_repo}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toys-repo", required=True)
    ap.add_argument("--out-repo", required=True)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--keep-local", action="store_true", help="don't delete the local merged build after push")
    args = ap.parse_args()
    info, n_d100, n_toys = build(args.toys_repo, args.out_repo)
    if not verify(args.out_repo, info, n_d100, n_toys):
        print("VERIFICATION FAILED -> NOT pushing.", flush=True)
        sys.exit(1)
    if args.push:
        push(args.out_repo)
        if not args.keep_local:
            shutil.rmtree(HF_LEROBOT_HOME / args.out_repo, ignore_errors=True)
            print(f"removed local build {args.out_repo} to free disk", flush=True)
    else:
        print("Build verified locally. Re-run with --push to upload.", flush=True)
