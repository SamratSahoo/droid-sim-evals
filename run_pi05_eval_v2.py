#!/usr/bin/env python3
"""Driver: evaluate 5 Pi-0.5 policies on the toys + cubes sim tasks (pi05-eval-v2).

For each policy this script, one at a time to respect limited disk:
  1. purges any stale HF cache for the model (user recently re-pushed),
  2. downloads ONLY the inference-relevant files (``params/``, ``assets/``,
     ``_CHECKPOINT_METADATA`` -- skips the ~6 GB ``train_state/`` optimizer state),
  3. runs ``full_eval.py`` (pi05 only) on BOTH tasks in one server lifetime with
     ``PI05_DEBUG_DUMP=1`` so the client records proprioception + every action,
  4. reorganizes the outputs into
        runs/pi05-eval-v2/<checkpoint_name>/<task>/{video.mp4, data.npy}
     where <task> is ``toys`` (scene 6) or ``cubes`` (scene 7), and data.npy is a
     pickled dict of per-step proprioception + actions (no images),
  5. deletes the downloaded checkpoint and HF cache to free disk.

The base Pi-0.5 DROID policy is served straight from gs:// (openpi caches it; not
purged, not in this host's limited disk budget).

Scene 6 is the plate + toys scene; scene 7 is the "push the yellow cube next to the red
cube" task -- two separated cubes built the same way scene 6 is (an empty base USD plus a
``assets/scene7_0.json`` sidecar; see droid_environment.py::_add_sidecar_objects).

Run with the eval venv python (cu128 torch), from the droid-sim-evals dir:
    PI05_DEBUG_DUMP=1 .venv/bin/python run_pi05_eval_v2.py
Optionally restrict to a subset of policies and/or scenes (e.g. re-run only the cubes task
for three policies after changing scene 7, leaving the toys outputs untouched):
    PI05_DEBUG_DUMP=1 .venv/bin/python run_pi05_eval_v2.py \
        --policies pi05_droid_jointpos_polaris pi05_droid_base pi05droid_toys100_sim --scenes 7
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Disable HF's "xet" transfer layer. It keeps a content-addressed chunk cache
# (~/.cache/huggingface/xet) IN ADDITION to the local_dir copy -- ~9 GB of duplicate,
# persistent disk per model, which is exactly what this storage-limited host can't
# afford. Plain HTTP downloads only the local_dir copy (which we delete each round).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pi05_eval_v2")
# Quiet the per-file HTTP request spam that otherwise floods the driver log.
for _noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_SCRIPT_DIR = Path(__file__).resolve().parent          # droid-sim-evals/
_REPO_ROOT = _SCRIPT_DIR.parent                        # tamp-vla/
_VENV_PY = _SCRIPT_DIR / ".venv" / "bin" / "python"
_CKPT_ROOT = _REPO_ROOT / "openpi" / "checkpoints"     # local download target
_HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
_OUT_ROOT = _SCRIPT_DIR / "runs" / "pi05-eval-v2"

# scene id -> output task folder name + instruction (matches full_eval DEFAULT_INSTRUCTIONS)
TASKS = {
    6: ("toys", "Place the toys on the plate with no collisions"),
    7: ("cubes", "Push the yellow cube next to the red cube"),
}

# (output_name, hf_repo_or_None, serve_config, gs_checkpoint_or_None, control_mode)
#   hf_repo None => served straight from the gs:// path in field 4 (no download / no cache purge).
#   control_mode => "velocity" (7 arm outputs are joint velocities the env integrates) or
#                   "position" (7 arm outputs are ABSOLUTE joint targets, e.g. the jointpos
#                   polaris checkpoint). Wired to full_eval's --pi05-control-mode.
POLICIES = [
    ("pi05droid_d100_toys100_sim", "SamratSahoo/pi05droid_d100_toys100_sim", "pi05droid-full-d100+toys100sim", None, "velocity"),
    ("pi05droid_d100_toys20_sim",  "SamratSahoo/pi05droid_d100_toys20_sim",  "pi05droid-full-d100+toys20sim",  None, "velocity"),
    ("pi05droid_toys100_sim",      "SamratSahoo/pi05droid_toys100_sim",      "pi05droid-toys100sim",           None, "velocity"),
    ("pi05droid_toys20_sim",       "SamratSahoo/pi05droid_toys20_sim",       "pi05droid-toys20sim",            None, "velocity"),
    # Full finetune of pi05_BASE on DROID-100 (own norm stats, bundled at
    # assets/SamratSahoo/d100/norm_stats.json -- NOT the DROID norm stats the pi05droid_* configs reuse).
    ("pi05base_d100",              "SamratSahoo/pi05base_d100",              "pi05base-full-d100",             None, "velocity"),
    ("pi05_droid_base",            None, "pi05_droid",
     "gs://openpi-assets/checkpoints/pi05_droid", "velocity"),
    ("pi05_droid_jointpos_polaris", None, "pi05_droid_jointpos_polaris",
     "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris", "position"),
]

# Only these are needed to *serve* a checkpoint; train_state/ is optimizer state (~6 GB).
DOWNLOAD_PATTERNS = ["params/**", "assets/**", "_CHECKPOINT_METADATA"]
# Big image arrays in the debug npz we don't want in the proprio+action npy.
_NPY_DROP_KEYS = {"query_req_exterior_image", "query_req_wrist_image"}


def _free_disk_gb() -> float:
    return shutil.disk_usage(str(_REPO_ROOT)).free / 1e9


def _purge_hf_cache(repo: str) -> None:
    # repo "SamratSahoo/pi05droid_toys20_sim" -> models--SamratSahoo--pi05droid_toys20_sim
    cache_dir = _HF_HUB / ("models--" + repo.replace("/", "--"))
    if cache_dir.exists():
        log.info(f"purging stale HF cache: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)


def _download(repo: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download
    # Resume-friendly: a previous run may have already pulled this checkpoint. A present
    # params/_METADATA + a clean (no *.incomplete) .cache means the snapshot finished.
    meta = dest / "params" / "_METADATA"
    incomplete = list((dest / ".cache").rglob("*.incomplete")) if (dest / ".cache").exists() else []
    if meta.exists() and not incomplete:
        size_gb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e9
        log.info(f"checkpoint already present ({size_gb:.1f} GB) at {dest}; skipping download")
        return
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    log.info(f"downloading {repo} (params+assets only) -> {dest}")
    t0 = time.time()
    snapshot_download(
        repo_id=repo,
        local_dir=str(dest),
        allow_patterns=DOWNLOAD_PATTERNS,
    )
    size_gb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e9
    log.info(f"downloaded {size_gb:.1f} GB in {time.time() - t0:.0f}s")


def _run_full_eval(config: str, checkpoint: str, staging: Path, control_mode: str, scenes) -> int:
    """Run full_eval.py for pi05 only on the given task scenes; returns exit code."""
    staging.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(_VENV_PY), "full_eval.py",
        "--policies", "pi05",
        "--mode", "alternating",
        "--no-launch-perception",
        "--scenes", *[str(s) for s in scenes],
        "--out-dir", str(staging),
        "--pi05-config", config,
        "--pi05-checkpoint", checkpoint,
        "--pi05-control-mode", control_mode,
    ]
    env = dict(os.environ)
    env["PI05_DEBUG_DUMP"] = "1"
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    log.info("running full_eval: " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(_SCRIPT_DIR), env=env).returncode


def _npz_to_npy(npz_path: Path, npy_path: Path, instruction: str, control_mode: str) -> bool:
    """Distill the pi05 debug npz into a proprioception + actions dict saved as .npy."""
    if not npz_path.exists():
        log.warning(f"no debug npz at {npz_path}; skipping npy")
        return False
    data = np.load(npz_path, allow_pickle=True)
    out = {k: data[k] for k in data.files if k not in _NPY_DROP_KEYS}
    out["instruction"] = instruction
    out["control_mode"] = control_mode
    # The 7 arm action dims are joint velocities (rad/s) in velocity mode, or absolute
    # joint-position targets (rad) in position mode -- state which so the npy is unambiguous.
    arm_desc = "7 joint vel rad/s" if control_mode == "velocity" else "7 joint pos rad (abs targets)"
    # Field guide for the consumer of this npy.
    out["_field_notes"] = (
        f"control_mode={control_mode}. "
        "Per-step (len=num_steps): joint_position[7]=arm proprio (rad), "
        f"gripper_position[1]=gripper proprio, action[8]=action sent to env "
        f"({arm_desc} + binarized gripper), chunk_action[8]=raw served row, "
        "gripper_raw=pre-binarization gripper, queried=server hit that step. "
        "Per-query (len=num_queries): query_action_chunk[H,8]=full predicted chunk."
    )
    np.save(npy_path, out, allow_pickle=True)
    log.info(f"wrote {npy_path} ({int(data['num_steps']) if 'num_steps' in data.files else '?'} steps)")
    return True


def _collect(name: str, staging: Path, control_mode: str, tasks: dict) -> None:
    """Move per-scene mp4/npz from staging into <name>/<task>/{video.mp4,data.npy}."""
    for scene, (task, instruction) in tasks.items():
        dest = _OUT_ROOT / name / task
        dest.mkdir(parents=True, exist_ok=True)
        mp4 = staging / f"pi05_scene{scene}.mp4"
        npz = staging / f"pi05_scene{scene}_debug.npz"
        if mp4.exists():
            shutil.move(str(mp4), str(dest / "video.mp4"))
            log.info(f"video -> {dest / 'video.mp4'}")
        else:
            log.warning(f"[{name}/{task}] no video produced (scene {scene})")
        _npz_to_npy(npz, dest / "data.npy", instruction, control_mode)


def main(policies_filter=None, scenes_filter=None) -> None:
    _OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Restrict to a subset of task scenes (default: all in TASKS) and/or policies (default:
    # all in POLICIES) -- e.g. to re-run only scene 7 (cubes) for a few policies after
    # changing that scene's geometry, without recomputing the (unchanged) scene 6 outputs.
    tasks = TASKS if not scenes_filter else {s: TASKS[s] for s in scenes_filter}
    policies = POLICIES if not policies_filter else [p for p in POLICIES if p[0] in policies_filter]

    log.info(f"output root: {_OUT_ROOT}")
    log.info(f"free disk: {_free_disk_gb():.0f} GB")
    log.info(f"scenes: {sorted(tasks)} ({', '.join(t for t, _ in tasks.values())})")
    log.info(f"policies: {[p[0] for p in policies]}")

    for name, repo, config, gs_checkpoint, control_mode in policies:
        log.info("#" * 70)
        log.info(f"### {name}  (config={config}, control_mode={control_mode})")
        log.info("#" * 70)
        # Resume: if every selected task's output already exists, this policy is done -- skip
        # it (and make sure no stale checkpoint lingers on this storage-limited host).
        done = all((_OUT_ROOT / name / task / "data.npy").exists() for task, _ in tasks.values())
        if done:
            log.info(f"[{name}] outputs already present for selected task(s); skipping")
            if repo is not None:
                shutil.rmtree(_CKPT_ROOT / name, ignore_errors=True)
            continue
        staging = _OUT_ROOT / name / "_staging"
        ckpt_dir = _CKPT_ROOT / name
        try:
            if repo is None:
                checkpoint = gs_checkpoint
            else:
                _purge_hf_cache(repo)
                _download(repo, ckpt_dir)
                checkpoint = str(ckpt_dir)
                log.info(f"free disk after download: {_free_disk_gb():.0f} GB")

            rc = _run_full_eval(config, checkpoint, staging, control_mode, sorted(tasks))
            log.info(f"full_eval exit code: {rc}")
            _collect(name, staging, control_mode, tasks)
        except Exception:  # noqa: BLE001 - keep going to the next policy
            log.exception(f"[{name}] failed; continuing to next policy")
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if repo is not None:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                _purge_hf_cache(repo)
                # Defense-in-depth: xet should be off, but if anything repopulated the
                # chunk cache, drop it so disk usage stays flat across the 4 downloads.
                shutil.rmtree(_HF_HUB.parent / "xet", ignore_errors=True)
                log.info(f"deleted checkpoint + cache; free disk: {_free_disk_gb():.0f} GB")

    log.info("=" * 70)
    log.info(f"DONE. Results under {_OUT_ROOT}")


def _parse_args():
    ap = argparse.ArgumentParser(description="pi05-eval-v2 driver (toys + cubes sim tasks)")
    ap.add_argument("--policies", nargs="*", default=None, metavar="NAME",
                    help="subset of policy output names to run (default: all). "
                         f"Choices: {', '.join(p[0] for p in POLICIES)}")
    ap.add_argument("--scenes", nargs="*", type=int, default=None, metavar="ID",
                    help=f"subset of task scene ids to run (default: all: {sorted(TASKS)})")
    args = ap.parse_args()
    # Validate up front so a typo fails loudly instead of silently running nothing.
    if args.policies:
        known = {p[0] for p in POLICIES}
        bad = [n for n in args.policies if n not in known]
        if bad:
            ap.error(f"unknown --policies {bad}; choices: {sorted(known)}")
    if args.scenes:
        bad = [s for s in args.scenes if s not in TASKS]
        if bad:
            ap.error(f"unknown --scenes {bad}; choices: {sorted(TASKS)}")
    return args


if __name__ == "__main__":
    _args = _parse_args()
    sys.exit(main(policies_filter=_args.policies, scenes_filter=_args.scenes))
