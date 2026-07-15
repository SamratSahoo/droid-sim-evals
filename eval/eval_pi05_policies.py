#!/usr/bin/env python3
"""Driver: evaluate Pi-0.5 policies across the sim manipulation tasks (scenes 6-12).

For each policy, one at a time (limited disk + shared GPU), this script:
  1. purges any stale HF cache + downloads ONLY the inference files (``params/``, ``assets/``,
     ``_CHECKPOINT_METADATA`` -- skips the ~6 GB ``train_state/`` optimizer state),
  2. launches the openpi pi-0.5 policy server for that checkpoint (via full_eval's server helpers)
     and waits for it,
  3. runs the parallel-env eval worker (``eval/eval_worker.py``) once per selected scene: each
     worker spins up ``--num-rollouts`` (default 8) parallel Isaac envs with randomized object
     layouts, drives Pi-0.5, and scores every env against that scene's success criteria
     (``eval/scene_specs.py``), writing ``results.json`` (+ optional tiled ``video.mp4``),
  4. kills the server, deletes the checkpoint + HF cache to free disk.

Two metrics per (policy, scene) are aggregated into ``runs/pi05-eval-v2/summary.{json,txt}``:
  * DENSE  -- mean over envs of the fraction of that scene's success criteria met (partial credit),
  * SPARSE -- fraction of envs meeting ALL criteria.

Scenes: 6 toys->plate, 8 PanClean (sponge->pan), 9 BlockStackKitchen (cubes->tray, stacked),
10 push button, 11 open cabinet, 12 blocks->plate (scenes 8 & 9 ported from PolaRiS). The base
Pi-0.5 DROID policy is served straight from gs://.

Run with the eval venv python (cu128 torch), from the droid-sim-evals dir:
    .venv/bin/python eval/eval_pi05_policies.py                       # all policies, all scenes, 8 rollouts
    .venv/bin/python eval/eval_pi05_policies.py --policies pi05_droid_base --scenes 8 10 11 --record-video
    .venv/bin/python eval/eval_pi05_policies.py --num-rollouts 4 --scenes 6 7
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # avoid the ~9 GB duplicate xet chunk cache
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pi05_eval")
for _noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_EVAL_DIR = Path(__file__).resolve().parent          # droid-sim-evals/eval
_DSE_DIR = _EVAL_DIR.parent                          # droid-sim-evals/
_REPO_ROOT = _DSE_DIR.parent                         # tamp-vla/
_VENV_PY = _DSE_DIR / ".venv" / "bin" / "python"
_CKPT_ROOT = _REPO_ROOT / "openpi" / "checkpoints"
_HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
_OUT_ROOT = _DSE_DIR / "runs" / "pi05-eval-v2"
_WORKER = _EVAL_DIR / "eval_worker.py"

sys.path[:0] = [str(_DSE_DIR), str(_EVAL_DIR)]
from full_eval import _server_spec, start_server, wait_for_server, stop_server  # server lifecycle helpers
from scene_specs import get_scene

# scene id -> short output folder name (instructions come from scene_specs, the single source of truth).
# Scenes 8 & 9 are the PolaRiS-ported DROID-PanClean / DROID-BlockStackKitchen tasks (scene 7 removed).
SCENE_TASKS = {6: "toys", 8: "pan_clean", 9: "block_stack", 10: "button", 11: "cabinet", 12: "blocks_plate"}

# (output_name, hf_repo_or_None, serve_config, gs_checkpoint_or_None, control_mode)
#   hf_repo None => served straight from the gs:// path (no download / no cache purge).
#   control_mode "velocity" (7 arm outputs are joint velocities the env integrates) or
#                "position" (absolute joint targets, e.g. the jointpos polaris checkpoint).
POLICIES = [
    ("pi05droid_d100_toys100_sim", "SamratSahoo/pi05droid_d100_toys100_sim", "pi05droid-full-d100+toys100sim", None, "velocity"),
    ("pi05droid_d100_toys20_sim",  "SamratSahoo/pi05droid_d100_toys20_sim",  "pi05droid-full-d100+toys20sim",  None, "velocity"),
    ("pi05droid_toys100_sim",      "SamratSahoo/pi05droid_toys100_sim",      "pi05droid-toys100sim",           None, "velocity"),
    ("pi05droid_toys20_sim",       "SamratSahoo/pi05droid_toys20_sim",       "pi05droid-toys20sim",            None, "velocity"),
    ("pi05base_d100",              "SamratSahoo/pi05base_d100",              "pi05base-full-d100",             None, "velocity"),
    ("pi05base_d100_toys100_sim",  "SamratSahoo/pi05base_d100_toys100_sim",  "pi05base-full-d100+toys100sim",  None, "velocity"),
    ("pi05base_d100_toys20_sim",   "SamratSahoo/pi05base_d100_toys20_sim",   "pi05base-full-d100+toys20sim",   None, "velocity"),
    # Full-DROID (streamed) + toys300 sim finetunes with the VAE / RND / VAE+RND manifold-cost planners.
    ("pi05droid_droidfull_toys300_vae_sim",    "SamratSahoo/pi05droid_droidfull_toys300_vae_sim",    "pi05droid-droid+toys300vaesim-stream",    None, "velocity"),
    ("pi05droid_droidfull_toys300_rnd_sim",    "SamratSahoo/pi05droid_droidfull_toys300_rnd_sim",    "pi05droid-droid+toys300rndsim-stream",    None, "velocity"),
    ("pi05droid_droidfull_toys300_vaernd_sim", "SamratSahoo/pi05droid_droidfull_toys300_vaernd_sim", "pi05droid-droid+toys300vaerndsim-stream", None, "velocity"),
    ("pi05_droid_base",            None, "pi05_droid",
     "gs://openpi-assets/checkpoints/pi05_droid", "velocity"),
    ("pi05_droid_jointpos_polaris", None, "pi05_droid_jointpos_polaris",
     "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris", "position"),
    # Polaris (joint-position) full-DROID (streamed) + toys sim finetunes: toys100 / toys300 plain / VAE / RND / VAE+RND manifold-cost.
    ("pi05polaris_droidfull_toys100_sim",     "SamratSahoo/pi05polaris_droidfull_toys100_sim",     "pi05polaris-droid+toys100sim-jointpos-stream",     None, "position"),
    ("pi05polaris_droidfull_toys300_sim",     "SamratSahoo/pi05polaris_droidfull_toys300_sim",     "pi05polaris-droid+toys300sim-jointpos-stream",     None, "position"),
    ("pi05polaris_droidfull_toys300_vae_sim", "SamratSahoo/pi05polaris_droidfull_toys300_vae_sim", "pi05polaris-droid+toys300vaesim-jointpos-stream", None, "position"),
    ("pi05polaris_droidfull_toys300_rnd_sim", "SamratSahoo/pi05polaris_droidfull_toys300_rnd_sim", "pi05polaris-droid+toys300rndsim-jointpos-stream", None, "position"),
    ("pi05polaris_droidfull_toys300_vaernd_sim", "SamratSahoo/pi05polaris_droidfull_toys300_vaernd_sim", "pi05polaris-droid+toys300vaerndsim-jointpos-stream", None, "position"),
    # TDF-cost and VAE(35k)+TDF-cost polaris (joint-position) full-DROID (streamed) + toys300 sim finetunes.
    ("pi05polaris_droidfull_toys300_tdf_sim",      "SamratSahoo/pi05polaris_droidfull_toys300_tdf_sim",      "pi05polaris-droid+toys300tdfsim-jointpos-stream",      None, "position"),
    ("pi05polaris_droidfull_toys300_vae35ktdf_sim", "SamratSahoo/pi05polaris_droidfull_toys300_vae35ktdf_sim", "pi05polaris-droid+toys300vae35ktdfsim-jointpos-stream", None, "position"),
]

DOWNLOAD_PATTERNS = ["params/**", "assets/**", "_CHECKPOINT_METADATA"]
_PI05_PORT = 8000


def _free_disk_gb() -> float:
    return shutil.disk_usage(str(_REPO_ROOT)).free / 1e9


def _purge_hf_cache(repo: str) -> None:
    cache_dir = _HF_HUB / ("models--" + repo.replace("/", "--"))
    if cache_dir.exists():
        log.info(f"purging stale HF cache: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)


def _download(repo: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    meta = dest / "params" / "_METADATA"
    incomplete = list((dest / ".cache").rglob("*.incomplete")) if (dest / ".cache").exists() else []
    if meta.exists() and not incomplete:
        log.info(f"checkpoint already present at {dest}; skipping download")
        return
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    log.info(f"downloading {repo} (params+assets only) -> {dest}")
    t0 = time.time()
    snapshot_download(repo_id=repo, local_dir=str(dest), allow_patterns=DOWNLOAD_PATTERNS)
    size_gb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e9
    log.info(f"downloaded {size_gb:.1f} GB in {time.time() - t0:.0f}s")


# Some uploaded checkpoints dropped assets/<asset_id>/norm_stats.json. These polaris/droid finetunes
# all reuse the SAME base DROID norm stats (byte-identical across sibling repos), so when a downloaded
# checkpoint lacks the file we backfill it from a known-good sibling. Serving fails without it.
_NORM_STATS_FALLBACK_REPO = "SamratSahoo/pi05polaris_droidfull_toys300_vae_sim"


def _ensure_norm_stats(ckpt_dir: Path) -> None:
    """Backfill assets/droid/norm_stats.json if the downloaded checkpoint is missing it."""
    if list((ckpt_dir / "assets").glob("*/norm_stats.json")):
        return
    from huggingface_hub import hf_hub_download

    log.warning(f"{ckpt_dir.name}: assets/*/norm_stats.json missing; "
                f"backfilling from {_NORM_STATS_FALLBACK_REPO}")
    dst = ckpt_dir / "assets" / "droid"
    dst.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(repo_id=_NORM_STATS_FALLBACK_REPO, filename="assets/droid/norm_stats.json")
    shutil.copyfile(src, dst / "norm_stats.json")
    log.info(f"backfilled norm_stats.json -> {dst / 'norm_stats.json'}")


def _launch_server(config: str, checkpoint: str, xla_mem_fraction: float = 0.5) -> "subprocess.Popen":
    """Launch the openpi pi-0.5 server for a checkpoint and wait until it accepts connections.

    ``xla_mem_fraction`` is the share of the GPU the JAX policy server preallocates; the rest is left
    to the Isaac worker. The 0.5 default suits ~8 parallel envs -- large ``--num-rollouts`` (e.g. 64)
    need a smaller fraction so the worker can fit its scene replicas + per-env camera render targets.
    """
    spec = _server_spec(
        "pi05",
        openpi_dir=str(_REPO_ROOT / "openpi"),
        tiptop_dir=str(_REPO_ROOT / "tiptop"),          # unused for pi05
        pi05_config=config,
        pi05_checkpoint=checkpoint,
        pi05_host="localhost",
        pi05_port=_PI05_PORT,
        tiptop_host="localhost",
        tiptop_port=8765,
        tiptop_server_module="tiptop.tiptop_websocket_server",
        xla_mem_fraction=xla_mem_fraction,              # share the GPU with the Isaac worker
    )
    proc = start_server("pi05", spec)
    if not wait_for_server(proc, "localhost", _PI05_PORT, timeout=1800):
        stop_server(proc)
        raise RuntimeError("pi05 server failed to start")
    return proc


def _run_worker(scene: int, control_mode: str, out_dir: Path, num_rollouts: int, record_video: bool,
                max_steps: int, seed: int) -> dict:
    """Run the parallel-env worker for one scene; return its parsed results.json ({} on failure)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(_VENV_PY), str(_WORKER),
        "--scene", str(scene),
        "--num-rollouts", str(num_rollouts),
        "--out-dir", str(out_dir),
        "--pi05-port", str(_PI05_PORT),
        "--control-mode", control_mode,
        "--max-steps", str(max_steps),
        "--seed", str(seed),
    ]
    if record_video:
        cmd.append("--record-video")
    env = dict(os.environ)
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    log.info(f"  worker: scene {scene} ({SCENE_TASKS[scene]})")
    rc = subprocess.run(cmd, cwd=str(_DSE_DIR), env=env).returncode
    res_path = out_dir / "results.json"
    if rc != 0 or not res_path.exists():
        log.error(f"  worker scene {scene} failed (rc={rc}, results={res_path.exists()})")
        return {}
    return json.loads(res_path.read_text())


def _fmt_summary(summary: dict, scenes: list) -> str:
    """Render a policy x scene table of dense/sparse scores."""
    cols = [SCENE_TASKS[s] for s in scenes]
    w = max(14, max((len(c) for c in cols), default=0) + 2)
    head = f"{'policy':32s} | " + " | ".join(f"{c:>{w}s}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, per_scene in summary.items():
        cells = []
        for s in scenes:
            r = per_scene.get(str(s)) or per_scene.get(s)
            cells.append(f"{r.get('mean_points', r['dense']):.1f}/{r.get('max_points','?')} {r['sparse']:.2f}"
                         if r else "  --/-- ")
        lines.append(f"{name:32s} | " + " | ".join(f"{c:>{w}s}" for c in cells))
    lines.append("\n(cells are MEAN_POINTS/MAX_POINTS SUCCESS_RATE: mean_points = avg # of whole-point "
                 "boolean sub-goals met per rollout; success_rate = frac envs meeting ALL criteria)")
    return "\n".join(lines)


def main(policies_filter=None, scenes_filter=None, num_rollouts=8, record_video=False,
         max_steps=1800, seed=0, fallback_rollouts=None, out_root=None, xla_mem_fraction=0.5) -> None:
    out_root = Path(out_root) if out_root is not None else _OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    scenes = sorted(scenes_filter) if scenes_filter else sorted(SCENE_TASKS)
    policies = POLICIES if not policies_filter else [p for p in POLICIES if p[0] in policies_filter]

    log.info(f"output root: {out_root} | free disk: {_free_disk_gb():.0f} GB")
    log.info(f"scenes: {scenes} ({', '.join(SCENE_TASKS[s] for s in scenes)})")
    log.info(f"policies: {[p[0] for p in policies]} | rollouts/scene: {num_rollouts} | record_video: {record_video}")

    summary: dict = {}
    summary_path = out_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    for name, repo, config, gs_checkpoint, control_mode in policies:
        log.info("#" * 72)
        log.info(f"### {name}  (config={config}, control_mode={control_mode})")
        # Resume: skip scenes whose results already exist; only run the missing ones.
        todo = [s for s in scenes if not (out_root / name / SCENE_TASKS[s] / "results.json").exists()]
        if not todo:
            log.info(f"[{name}] all selected scenes already scored; loading + skipping")
            summary.setdefault(name, {})
            for s in scenes:
                rp = out_root / name / SCENE_TASKS[s] / "results.json"
                if rp.exists():
                    r = json.loads(rp.read_text())
                    summary[name][str(s)] = {"dense": r["dense_score"], "sparse": r["sparse_score"],
                                             "mean_points": r.get("mean_points"), "max_points": r.get("max_points")}
            if repo is not None:
                shutil.rmtree(_CKPT_ROOT / name, ignore_errors=True)
            continue

        ckpt_dir = _CKPT_ROOT / name
        server = None
        try:
            if repo is None:
                checkpoint = gs_checkpoint
            else:
                _purge_hf_cache(repo)
                _download(repo, ckpt_dir)
                _ensure_norm_stats(ckpt_dir)
                checkpoint = str(ckpt_dir)
                log.info(f"free disk after download: {_free_disk_gb():.0f} GB")
            server = _launch_server(config, checkpoint, xla_mem_fraction)

            summary.setdefault(name, {})
            for s in todo:
                out_dir = out_root / name / SCENE_TASKS[s]
                res = _run_worker(s, control_mode, out_dir, num_rollouts, record_video, max_steps, seed)
                if not res and fallback_rollouts and fallback_rollouts < num_rollouts:
                    # A crashed worker (e.g. VRAM OOM on scene load) leaves no results.json; retry smaller.
                    log.warning(f"  [{name}/{SCENE_TASKS[s]}] {num_rollouts}-env run failed; "
                                f"retrying at {fallback_rollouts} rollouts")
                    res = _run_worker(s, control_mode, out_dir, fallback_rollouts, record_video, max_steps, seed)
                if res:
                    summary[name][str(s)] = {"dense": res["dense_score"], "sparse": res["sparse_score"],
                                             "mean_points": res.get("mean_points"), "max_points": res.get("max_points")}
                    log.info(f"  [{name}/{SCENE_TASKS[s]}] dense={res['dense_score']:.3f} "
                             f"sparse={res['sparse_score']:.3f} rates={res['criterion_rates']}")
                summary_path.write_text(json.dumps(summary, indent=2))  # checkpoint after each scene
        except Exception:  # noqa: BLE001 - keep going to the next policy
            log.exception(f"[{name}] failed; continuing to next policy")
        finally:
            if server is not None:
                stop_server(server)
            if repo is not None:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                _purge_hf_cache(repo)
                shutil.rmtree(_HF_HUB.parent / "xet", ignore_errors=True)
                log.info(f"deleted checkpoint + cache; free disk: {_free_disk_gb():.0f} GB")

    summary_path.write_text(json.dumps(summary, indent=2))
    table = _fmt_summary(summary, scenes)
    (out_root / "summary.txt").write_text(table)
    log.info("=" * 72)
    log.info(f"DONE. Results under {out_root}\n\n{table}\n")


def _parse_args():
    ap = argparse.ArgumentParser(description="pi05 sim eval driver (scenes 6-12, parallel rollouts + success scoring)")
    ap.add_argument("--policies", nargs="*", default=None, metavar="NAME",
                    help=f"subset of policy names (default: all). Choices: {', '.join(p[0] for p in POLICIES)}")
    ap.add_argument("--scenes", nargs="*", type=int, default=None, metavar="ID",
                    help=f"subset of scene ids (default: all: {sorted(SCENE_TASKS)})")
    ap.add_argument("--num-rollouts", type=int, default=8, help="parallel randomized envs per scene (default 8)")
    ap.add_argument("--rollout-fallback", type=int, default=None,
                    help="if a scene's worker crashes (e.g. VRAM OOM), retry it at this many rollouts")
    ap.add_argument("--record-video", action="store_true", help="write a tiled multi-env video.mp4 per scene (default off)")
    ap.add_argument("--max-steps", type=int, default=1800, help="env steps per rollout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=str, default=None,
                    help="output root dir (default runs/pi05-eval-v2). Use a separate dir for e.g. video "
                         "capture so it doesn't collide with / get skipped by the scored eval's results.")
    ap.add_argument("--xla-mem-fraction", type=float, default=0.5,
                    help="share of the GPU preallocated by the JAX policy server; the rest is left to the "
                         "Isaac worker. Lower it (e.g. 0.3) for large --num-rollouts (default 0.5)")
    args = ap.parse_args()
    if args.policies:
        known = {p[0] for p in POLICIES}
        bad = [n for n in args.policies if n not in known]
        if bad:
            ap.error(f"unknown --policies {bad}; choices: {sorted(known)}")
    if args.scenes:
        bad = [s for s in args.scenes if s not in SCENE_TASKS]
        if bad:
            ap.error(f"unknown --scenes {bad}; choices: {sorted(SCENE_TASKS)}")
    return args


if __name__ == "__main__":
    a = _parse_args()
    sys.exit(main(policies_filter=a.policies, scenes_filter=a.scenes, num_rollouts=a.num_rollouts,
                  record_video=a.record_video, max_steps=a.max_steps, seed=a.seed,
                  fallback_rollouts=a.rollout_fallback, out_root=a.out_root,
                  xla_mem_fraction=a.xla_mem_fraction))
