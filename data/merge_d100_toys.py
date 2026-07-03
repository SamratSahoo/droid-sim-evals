"""Merge d100 + a toys{N}_sim dataset -> d100_toys{N}_sim, as LeRobot **v3.0** (video, DROID schema).

Both sources are STREAMED from the Hub episode-by-episode (``iter_episodes`` handles a v2.1 inline-image
source like an un-converted d100 AND v3.0 video sources) and re-emitted through the incremental v3.0
writer, so the merge needs no local source download and is memory-bounded (the old v2.1 parquet-concat
existed only to dodge lerobot's add_frame OOM; the v3.0 writer streams to disk and cannot OOM).

Usage (openpi venv):
  ../openpi/.venv/bin/python merge_d100_toys.py --toys-repo SamratSahoo/toys20_sim \
      --out-repo SamratSahoo/d100_toys20_sim [--push]
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME  # noqa: E402
from tamp_data_gen import _upload_dataset, merge_datasets  # noqa: E402

D100_REPO = "SamratSahoo/d100"


def verify(out_repo: str) -> bool:
    """Structural checks on the local v3.0 build before pushing."""
    import av
    import pyarrow.parquet as pq

    root = HF_LEROBOT_HOME / out_repo
    info = json.loads((root / "meta" / "info.json").read_text())
    ep = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pydict()
    n = info["total_episodes"]
    lengths = list(ep["length"])
    # Decode episode 0's exterior_1 video range and check it aligns with its recorded length.
    vp = root / "videos" / "observation.images.exterior_1_left" / "chunk-000" / "file-000.mp4"
    container = av.open(str(vp))
    stream = container.streams.video[0]
    ep0_frames = 0
    for frame in container.decode(stream):
        if float(frame.pts * stream.time_base) >= ep["videos/observation.images.exterior_1_left/to_timestamp"][0]:
            break
        ep0_frames += 1
    container.close()
    checks = [
        ("codebase_version v3.0", info["codebase_version"] == "v3.0"),
        (f"{n} episodes in meta/episodes", len(ep["episode_index"]) == n),
        ("episode_index contiguous 0..N", list(ep["episode_index"]) == list(range(n))),
        ("total_frames == sum(lengths)", info["total_frames"] == sum(lengths)),
        ("episode 0 video frames == length[0]", ep0_frames >= lengths[0]),
    ]
    print("\n=== VERIFICATION ===", flush=True)
    allok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        allok = allok and ok
    print(f"=== {'ALL CHECKS PASSED' if allok else 'SOME CHECKS FAILED'} ===", flush=True)
    return allok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toys-repo", required=True)
    ap.add_argument("--out-repo", required=True)
    ap.add_argument("--push", action="store_true")
    ap.add_argument(
        "--keep-local",
        action="store_true",
        help="don't delete the local merged build after push",
    )
    args = ap.parse_args()

    # Build locally (push=False), verify, then push -- so a bad merge never lands on the Hub.
    total = merge_datasets(
        sources=[D100_REPO, args.toys_repo],
        repo_id=args.out_repo,
        push=False,
        private=False,
    )
    if not total or not verify(args.out_repo):
        sys.exit("VERIFICATION FAILED -> NOT pushing.")
    print(f"BUILT {args.out_repo}: {total} episodes", flush=True)
    if args.push:
        from huggingface_hub import HfApi

        HfApi().delete_repo(args.out_repo, repo_type="dataset", missing_ok=True)
        _upload_dataset(
            HF_LEROBOT_HOME / args.out_repo,
            args.out_repo,
            False,
            ["droid", "panda", "tamp-vla", "merged"],
        )
        print(f"PUSHED {args.out_repo}", flush=True)
        if not args.keep_local:
            shutil.rmtree(HF_LEROBOT_HOME / args.out_repo, ignore_errors=True)
            print(f"removed local build {args.out_repo} to free disk", flush=True)
    else:
        print("Build verified locally. Re-run with --push to upload.", flush=True)


if __name__ == "__main__":
    main()
