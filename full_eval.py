"""Full evaluation: Pi-0.5 vs tiptop, side-by-side comparison videos.

Runs the existing DROID sim evaluation for BOTH policies across a set of scenes,
records a video per policy per scene, and stitches each scene's two runs into a
single labeled side-by-side comparison video (Pi-0.5 | tiptop).

This script owns the policy-server lifecycle. It launches:
  * the openpi pi-0.5 server  (``scripts/serve_policy.py``, port 8000)
  * the tiptop planning server (``tiptop.tiptop_websocket_server``, port 8765)

and picks how to run them based on available VRAM:
  * ``concurrent``  - both servers up at once, both policies per scene (5 Isaac launches)
  * ``alternating`` - one server at a time: run all scenes for pi-0.5, kill it, then
                      all scenes for tiptop, kill it (10 Isaac launches, lower peak VRAM)
  * ``auto`` (default) - concurrent only if a single GPU has >= --concurrent-min-vram-gb
                      (or multiple GPUs are present), else alternating.

Usage (run from the droid-sim-evals directory):
    uv run python full_eval.py                          # all 5 scenes, auto mode
    uv run python full_eval.py --mode alternating --scenes 1
    uv run python full_eval.py --no-launch-servers      # servers already running
    uv run python full_eval.py --stitch-only --out-dir runs/2026-06-22/12-00-00

Internal: the script re-execs itself with ``--worker-scene N`` to run one scene in
its own Isaac Sim process (matching the one-scene-per-process pattern of the
existing eval scripts). Do not pass the ``--worker-*`` flags manually.
"""

import argparse
import atexit
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("full_eval")

import cv2
import mediapy
import numpy as np
import tyro

from src.visual_utils import add_label_bar, add_top_padding, overlay_timer_ms

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
DEFAULT_OPENPI_DIR = str(_REPO_ROOT / "openpi")
DEFAULT_TIPTOP_DIR = str(_REPO_ROOT / "tiptop")
# Perception model servers the tiptop planning server depends on (HTTP).
DEFAULT_M2T2_DIR = str(_REPO_ROOT / "M2T2")
DEFAULT_FS_DIR = str(_REPO_ROOT / "FoundationStereo")

# Default task instruction per scene (from README.md).
DEFAULT_INSTRUCTIONS = {
    1: "Put the Rubik's cube in the bowl.",
    2: "Put the can in the mug.",
    3: "Put the banana in the bin.",
    4: "Put the cube on the mug and the cans in the bowl.",
    5: "Put 3 blocks in the bowl.",
    6: "Place the toys on the plate with no collisions",
}

KNOWN_POLICIES = ("pi05", "tiptop")
VIDEO_FPS = 15


# --------------------------------------------------------------------------- #
# Memory-safe mp4 writer (streams frames; never holds the whole video in RAM)  #
# --------------------------------------------------------------------------- #
class _Mp4Writer:
    """Lazily-opened streaming mp4 writer. Accepts uint8/float RGB frames.

    Prefers ``mediapy.VideoWriter`` (same encoder as the existing eval scripts);
    falls back to ``cv2.VideoWriter`` if unavailable.
    """

    def __init__(self, path, fps: float = VIDEO_FPS) -> None:
        self._path = str(path)
        self._fps = float(fps)
        self._mp: Optional[object] = None
        self._cv: Optional[cv2.VideoWriter] = None

    def _open(self, h: int, w: int) -> None:
        if hasattr(mediapy, "VideoWriter"):
            try:
                self._mp = mediapy.VideoWriter(self._path, (h, w), fps=self._fps)
                self._mp.__enter__()
                return
            except Exception as e:  # noqa: BLE001 - fall back to cv2
                logger.warning(f"mediapy.VideoWriter unavailable ({e}); using cv2 fallback")
                self._mp = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv = cv2.VideoWriter(self._path, fourcc, self._fps, (w, h))

    def add(self, frame: np.ndarray) -> None:
        # Camera obs are unnormalized uint8 [0,255]; coerce any stray float frame
        # (e.g. _safe_image's black fallback) so the fixed-dtype encoder stays happy.
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        if self._mp is None and self._cv is None:
            self._open(frame.shape[0], frame.shape[1])
        if self._mp is not None:
            self._mp.add_image(frame)
        else:
            self._cv.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._mp is not None:
            self._mp.__exit__(None, None, None)
            self._mp = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None


# --------------------------------------------------------------------------- #
# Rollout (runs inside the worker process)                                     #
# --------------------------------------------------------------------------- #
def run_rollout(env, obs, client, instruction: str, max_steps: int, is_tiptop: bool, out_path: Path) -> int:
    """Step a policy through one episode, writing frames to ``out_path``.

    Mirrors the loop in ``tiptop_eval.py``: frame = exterior|wrist + timer overlay.
    Returns the number of frames written (0 => rollout/planning failed, no file).
    """
    import torch
    from tqdm import tqdm

    writer = _Mp4Writer(out_path, VIDEO_FPS)
    frame_idx = 0
    desc = f"{'tiptop' if is_tiptop else 'pi05'} scene rollout"
    try:
        for _ in tqdm(range(int(max_steps)), desc=desc):
            try:
                ret = client.infer(obs, instruction)
            except Exception as e:  # noqa: BLE001 - planning/inference failure ends the episode
                logger.error(f"infer failed: {e}")
                break

            if is_tiptop and getattr(client, "plan_done", False):
                logger.info(f"tiptop plan fully executed at frame {frame_idx}")
                break

            viz = np.concatenate([ret["right_image"], ret["wrist_image"]], axis=1)
            viz = add_top_padding(viz, pad_px=40)
            overlay_timer_ms(viz, int(frame_idx * 1000 / VIDEO_FPS))
            writer.add(viz)
            frame_idx += 1

            action = torch.tensor(ret["action"])[None]
            obs, _, term, trunc, _ = env.step(action)
            if bool(term) or bool(trunc):
                break
    finally:
        writer.close()

    if frame_idx == 0 and out_path.exists():
        # No frames but a (likely empty) file got created; remove it so stitching skips cleanly.
        try:
            out_path.unlink()
        except OSError:
            pass
    return frame_idx


def run_worker(
    *,
    worker_scene: int,
    worker_policies: str,
    variant: int,
    instruction: str,
    out_dir: str,
    episode_length_s: float,
    open_loop_horizon: int,
    headless: bool,
    pi05_host: str,
    pi05_port: int,
    tiptop_host: str,
    tiptop_port: int,
) -> None:
    """Run one scene (one Isaac process) for the given policies; write per-policy mp4s."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="full_eval worker")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import src.sim_evals.environments  # noqa: F401  (registers the "DROID" gym env)
    from isaaclab_tasks.utils import parse_env_cfg
    from src.sim_evals.sim_utils import settle_sim

    # IMPORTANT: tiptop_websocket.py calls msgpack_numpy.patch() at import time (PyPI
    # lebedov msgpack-numpy), which globally reassigns msgpack.Packer/unpackb to an
    # ndarray encoding the openpi server can't decode. openpi_client.msgpack_numpy
    # captures the ORIGINAL msgpack functions at its own import, so we import it (via
    # the pi05 client) FIRST to make pi05 serialization immune to that patch. Import
    # order here is load-bearing -- do not reorder.
    from src.sim_evals.inference.pi05_websocket import Pi05WebsocketClient
    from src.sim_evals.inference.tiptop_websocket import TiptopWebsocketClient
    from src.sim_evals.environments.droid_environment import set_arm_control_mode, set_camera_resolution

    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.set_scene(str(worker_scene), variant)
    # Eval renders at full sensor resolution; data generation defaults to 180x320 for render speed.
    set_camera_resolution(env_cfg, 720, 1280)
    env_cfg.episode_length_s = episode_length_s
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()  # second render cycle for correctly loaded materials

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    max_steps = env.env.max_episode_length

    policies = [p for p in worker_policies.split(",") if p]
    with torch.no_grad():
        for policy in policies:
            if policy == "tiptop":
                logger.info(f"[scene {worker_scene}] connecting tiptop @ {tiptop_host}:{tiptop_port}")
                client = TiptopWebsocketClient(host=tiptop_host, port=tiptop_port)
                is_tiptop = True
                # tiptop emits absolute joint-position waypoints from the cuRobo plan.
                rollout_mode = "position"
            elif policy == "pi05":
                logger.info(f"[scene {worker_scene}] connecting pi-0.5 @ {pi05_host}:{pi05_port}")
                client = Pi05WebsocketClient(host=pi05_host, port=pi05_port, open_loop_horizon=open_loop_horizon)
                is_tiptop = False
                # pi05_droid (and velocity-trained finetunes) emit joint velocities; the env
                # integrates them onto the current joint position (see set_arm_control_mode).
                rollout_mode = "velocity"
            else:
                logger.warning(f"Unknown policy '{policy}', skipping")
                continue

            # settle_sim holds the pose by commanding the measured joint POSITIONS, so it must run
            # in position mode regardless of the policy; switch to the rollout mode only afterwards
            # (in velocity mode those positions would be misread as velocities and drift the arm).
            set_arm_control_mode("position")
            obs, _ = env.reset()
            obs = settle_sim(env, obs, reset_episode_buf=True)
            set_arm_control_mode(rollout_mode)
            out_path = out / f"{policy}_scene{worker_scene}.mp4"
            n = run_rollout(env, obs, client, instruction, max_steps, is_tiptop, out_path)
            # Dump the full inference trace (proprioception, requests, action chunks,
            # post-processed actions) for offline debugging when enabled. No-op unless
            # PI05_DEBUG_DUMP is set and the client recorded steps.
            if hasattr(client, "dump_debug"):
                client.dump_debug(out / f"{policy}_scene{worker_scene}_debug.npz")
            try:
                client.reset()
                client.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
            if n > 0:
                logger.info(f"[scene {worker_scene}] {policy}: wrote {n} frames -> {out_path}")
            else:
                logger.error(f"[scene {worker_scene}] {policy}: produced no frames")

    env.close()
    simulation_app.close()


# --------------------------------------------------------------------------- #
# Stitching (orchestrator) - streaming, memory-safe                            #
# --------------------------------------------------------------------------- #
def _match_height(a: np.ndarray, b: np.ndarray):
    if a.shape[0] == b.shape[0]:
        return a, b
    h = max(a.shape[0], b.shape[0])

    def pad(x):
        if x.shape[0] == h:
            return x
        rows = np.zeros((h - x.shape[0], x.shape[1], 3), dtype=x.dtype)
        return np.concatenate([x, rows], axis=0)

    return pad(a), pad(b)


def stitch_comparison(
    pi05_mp4: Path,
    tiptop_mp4: Path,
    out_mp4: Path,
    pi05_label: str,
    tiptop_label: str,
    fps: float = VIDEO_FPS,
) -> bool:
    """Write a labeled side-by-side comparison video.

    Streams both inputs frame-by-frame (cv2.VideoCapture); the shorter clip is
    frozen on its last frame until the longer clip ends. Each panel gets a title bar.
    """
    cap_l = cv2.VideoCapture(str(pi05_mp4))
    cap_r = cv2.VideoCapture(str(tiptop_mp4))
    if not cap_l.isOpened() or not cap_r.isOpened():
        logger.error(f"Could not open inputs for stitching: {pi05_mp4}, {tiptop_mp4}")
        cap_l.release()
        cap_r.release()
        return False

    writer = _Mp4Writer(out_mp4, fps)
    last_l = last_r = None
    n = 0
    try:
        while True:
            ok_l, fl = cap_l.read()
            ok_r, fr = cap_r.read()
            if not ok_l and not ok_r:
                break
            if ok_l:
                last_l = fl
            if ok_r:
                last_r = fr
            fl = last_l
            fr = last_r
            if fl is None or fr is None:
                # One stream never produced a frame; cannot compare.
                break
            left = cv2.cvtColor(fl, cv2.COLOR_BGR2RGB)
            right = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            left = add_label_bar(left, pi05_label)
            right = add_label_bar(right, tiptop_label)
            left, right = _match_height(left, right)
            divider = np.zeros((left.shape[0], 8, 3), dtype=left.dtype)
            writer.add(np.concatenate([left, divider, right], axis=1))
            n += 1
    finally:
        writer.close()
        cap_l.release()
        cap_r.release()

    if n == 0:
        logger.error(f"Stitch produced 0 frames for scene video {out_mp4}")
        return False
    logger.info(f"Wrote comparison ({n} frames) -> {out_mp4}")
    return True


# --------------------------------------------------------------------------- #
# Policy-server lifecycle (orchestrator)                                       #
# --------------------------------------------------------------------------- #
_LAUNCHED: "list[subprocess.Popen]" = []


def _server_spec(
    policy: str,
    *,
    openpi_dir: str,
    tiptop_dir: str,
    pi05_config: str,
    pi05_checkpoint: str,
    pi05_host: str,
    pi05_port: int,
    tiptop_host: str,
    tiptop_port: int,
    tiptop_server_module: str,
    xla_mem_fraction: float,
):
    """Return (cmd, cwd, env, host, port) for launching the given policy server."""
    if policy == "pi05":
        # serve_policy.py uses tyro subcommands: top-level options (--port) must precede
        # the `policy:checkpoint` subcommand; its --policy.* options come after it.
        cmd = [
            "uv", "run", "scripts/serve_policy.py",
            "--port", str(pi05_port),
            "policy:checkpoint",
            f"--policy.config={pi05_config}",
            f"--policy.dir={pi05_checkpoint}",
        ]
        env = dict(os.environ)
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(xla_mem_fraction)
        return cmd, str(Path(openpi_dir).expanduser()), env, pi05_host, pi05_port
    if policy == "tiptop":
        cmd = ["pixi", "run", "python", "-m", tiptop_server_module, "--port", str(tiptop_port)]
        return cmd, str(Path(tiptop_dir).expanduser()), dict(os.environ), tiptop_host, tiptop_port
    raise ValueError(f"Unknown policy: {policy}")


def start_server(policy: str, spec) -> subprocess.Popen:
    cmd, cwd, env, _host, _port = spec
    logger.info(f"Launching {policy} server (cwd={cwd}): {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, start_new_session=True)
    _LAUNCHED.append(proc)
    return proc


def wait_for_server(proc: subprocess.Popen, host: str, port: int, timeout: float) -> bool:
    """Poll until the server accepts a TCP connection, the process dies, or timeout."""
    logger.info(f"Waiting for server at {host}:{port} (up to {timeout:.0f}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            logger.error(f"Server exited early (code {proc.returncode}) before becoming ready")
            return False
        try:
            with socket.create_connection((host, port), timeout=5):
                logger.info(f"Server at {host}:{port} is ready")
                return True
        except OSError:
            time.sleep(3)
    logger.error(f"Server at {host}:{port} not ready within {timeout:.0f}s")
    return False


def stop_server(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    # SIGINT first: the tiptop server traps it for clean GPU teardown (os._exit(0)).
    for sig, wait_s in ((signal.SIGINT, 20), (signal.SIGTERM, 8), (signal.SIGKILL, 5)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait_s)
            logger.info(f"Stopped server pid {proc.pid} (signal {sig.name})")
            return
        except subprocess.TimeoutExpired:
            continue


def _stop_all() -> None:
    for proc in _LAUNCHED:
        stop_server(proc)


atexit.register(_stop_all)


# --------------------------------------------------------------------------- #
# Perception model servers (tiptop depends on M2T2; FoundationStereo optional) #
# --------------------------------------------------------------------------- #
def wait_for_http_health(proc: subprocess.Popen, host: str, port: int, timeout: float) -> bool:
    """Poll http://host:port/health until it reports status=healthy, the proc dies, or timeout."""
    url = f"http://{host}:{port}/health"
    logger.info(f"Waiting for {url} (up to {timeout:.0f}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            logger.error(f"Perception server exited early (code {proc.returncode}) before becoming healthy")
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if json.loads(r.read().decode()).get("status") == "healthy":
                    logger.info(f"{url} is healthy")
                    return True
        except Exception:  # noqa: BLE001 - not up yet / not healthy yet
            pass
        time.sleep(3)
    logger.error(f"{url} not healthy within {timeout:.0f}s")
    return False


def start_perception_servers(pcfg: dict, ready_timeout: float):
    """Launch M2T2 (required for tiptop) and optionally FoundationStereo.

    Returns (procs, ok). ok is False only if the REQUIRED M2T2 server failed --
    FoundationStereo is best-effort (the sim path supplies depth and never calls it).
    """
    procs: "list[subprocess.Popen]" = []
    # M2T2 grasp server (required).
    cmd = ["pixi", "run", "python", "server.py", "--host", "0.0.0.0", "--port", str(pcfg["m2t2_port"])]
    cwd = str(Path(pcfg["m2t2_dir"]).expanduser())
    logger.info(f"Launching M2T2 server (cwd={cwd}): {' '.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=cwd, env=dict(os.environ), start_new_session=True)
    _LAUNCHED.append(p)
    procs.append(p)
    if not wait_for_http_health(p, pcfg["m2t2_host"], pcfg["m2t2_port"], ready_timeout):
        logger.error("M2T2 server failed to become healthy -- tiptop perception cannot run")
        return procs, False

    # FoundationStereo depth server (optional; not used by the sim path).
    if pcfg["launch_fs"]:
        cmd = ["pixi", "run", "python", "server.py", "--host", "0.0.0.0", "--port", str(pcfg["fs_port"])]
        cwd = str(Path(pcfg["fs_dir"]).expanduser())
        logger.info(f"Launching FoundationStereo server (cwd={cwd}): {' '.join(cmd)}")
        p = subprocess.Popen(cmd, cwd=cwd, env=dict(os.environ), start_new_session=True)
        _LAUNCHED.append(p)
        procs.append(p)
        if not wait_for_http_health(p, pcfg["fs_host"], pcfg["fs_port"], ready_timeout):
            logger.warning("FoundationStereo not healthy; continuing (sim path does not require it)")
    return procs, True


# --------------------------------------------------------------------------- #
# VRAM probe / mode decision                                                   #
# --------------------------------------------------------------------------- #
def probe_total_vram_gb():
    """Return (max_single_gpu_GiB, num_gpus). (None, 0) if probing fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except Exception as e:  # noqa: BLE001
        logger.warning(f"nvidia-smi probe failed ({e})")
        return None, 0
    totals = [float(x) for x in out.split() if x.strip()]
    if not totals:
        return None, 0
    return max(totals) / 1024.0, len(totals)


def decide_mode(mode: str, threshold_gb: float) -> str:
    if mode in ("concurrent", "alternating"):
        return mode
    max_gb, num = probe_total_vram_gb()
    if num >= 2:
        logger.info(f"auto: {num} GPUs detected -> concurrent")
        return "concurrent"
    if max_gb is None:
        logger.info("auto: VRAM unknown -> alternating (safe default)")
        return "alternating"
    chosen = "concurrent" if max_gb >= threshold_gb else "alternating"
    logger.info(f"auto: max GPU VRAM {max_gb:.1f} GiB vs threshold {threshold_gb:.0f} -> {chosen}")
    return chosen


# --------------------------------------------------------------------------- #
# Worker spawning (orchestrator)                                               #
# --------------------------------------------------------------------------- #
def spawn_worker(scene: int, worker_policies: List[str], instruction: str, out_dir: Path, *, worker_kwargs: dict) -> bool:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker-scene", str(scene),
        "--worker-policies", ",".join(worker_policies),
        "--instruction", instruction,
        "--out-dir", str(out_dir),
        "--variant", str(worker_kwargs["variant"]),
        "--episode-length-s", str(worker_kwargs["episode_length_s"]),
        "--open-loop-horizon", str(worker_kwargs["open_loop_horizon"]),
        "--pi05-host", worker_kwargs["pi05_host"],
        "--pi05-port", str(worker_kwargs["pi05_port"]),
        "--tiptop-host", worker_kwargs["tiptop_host"],
        "--tiptop-port", str(worker_kwargs["tiptop_port"]),
    ]
    if not worker_kwargs["headless"]:
        cmd.append("--no-headless")
    logger.info(f"=== Worker: scene {scene}, policies={worker_policies} ===")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        logger.error(f"Worker for scene {scene} exited with code {rc}")
    return rc == 0


def _run_concurrent(scenes, run_policies, launch_servers, spec_fn, instr_for, out_dir, worker_kwargs, ready_timeout, pcfg) -> bool:
    """Run both policies per scene with both servers up. Returns False to request fallback."""
    procs = {}
    perception_procs: "list[subprocess.Popen]" = []
    try:
        if launch_servers:
            if "tiptop" in run_policies and pcfg["launch_perception"]:
                perception_procs, ok = start_perception_servers(pcfg, pcfg["ready_timeout"])
                if not ok:
                    logger.warning("M2T2 perception server unavailable; falling back to alternating mode")
                    return False
            for pol in run_policies:
                procs[pol] = start_server(pol, spec_fn(pol))
            for pol in run_policies:
                _, _, _, host, port = spec_fn(pol)
                if not wait_for_server(procs[pol], host, port, ready_timeout):
                    logger.warning(f"{pol} server not ready; falling back to alternating mode")
                    return False
        for scene in scenes:
            spawn_worker(scene, run_policies, instr_for(scene), out_dir, worker_kwargs=worker_kwargs)
        return True
    finally:
        for pol in list(procs):
            stop_server(procs[pol])
        for pp in perception_procs:
            stop_server(pp)


def _run_alternating(scenes, run_policies, launch_servers, spec_fn, instr_for, out_dir, worker_kwargs, ready_timeout, pcfg) -> None:
    """Run one policy at a time: launch its server(s), do all scenes, kill them."""
    for pol in run_policies:
        proc = None
        perception_procs: "list[subprocess.Popen]" = []
        try:
            if launch_servers:
                if pol == "tiptop" and pcfg["launch_perception"]:
                    perception_procs, ok = start_perception_servers(pcfg, pcfg["ready_timeout"])
                    if not ok:
                        logger.error("Skipping tiptop rollouts: M2T2 perception server unavailable")
                        continue
                spec = spec_fn(pol)
                proc = start_server(pol, spec)
                _, _, _, host, port = spec
                if not wait_for_server(proc, host, port, ready_timeout):
                    logger.error(f"{pol} server failed to start; skipping its rollouts")
                    continue
            for scene in scenes:
                spawn_worker(scene, [pol], instr_for(scene), out_dir, worker_kwargs=worker_kwargs)
        finally:
            if launch_servers:
                stop_server(proc)
                for pp in perception_procs:
                    stop_server(pp)


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #
def run_orchestrator(
    *,
    scenes,
    variant,
    mode,
    policies,
    episode_length_s,
    open_loop_horizon,
    headless,
    out_dir,
    stitch_only,
    launch_servers,
    instruction_override,
    pi05_host,
    pi05_port,
    tiptop_host,
    tiptop_port,
    pi05_label,
    tiptop_label,
    openpi_dir,
    tiptop_dir,
    pi05_config,
    pi05_checkpoint,
    tiptop_server_module,
    xla_mem_fraction,
    server_ready_timeout_s,
    concurrent_min_vram_gb,
    launch_perception,
    m2t2_dir,
    m2t2_host,
    m2t2_port,
    launch_fs,
    fs_dir,
    fs_host,
    fs_port,
    perception_ready_timeout_s,
) -> None:
    if out_dir:
        out_path = Path(out_dir).resolve()
    else:
        now = datetime.now()
        out_path = (Path("runs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_path}")

    run_policies = [p for p in policies if p in KNOWN_POLICIES]
    if not run_policies:
        logger.error(f"No valid policies in {policies} (known: {KNOWN_POLICIES})")
        return

    def instr_for(scene: int) -> str:
        return instruction_override if instruction_override else DEFAULT_INSTRUCTIONS.get(scene, "")

    worker_kwargs = dict(
        variant=variant,
        episode_length_s=episode_length_s,
        open_loop_horizon=open_loop_horizon,
        headless=headless,
        pi05_host=pi05_host,
        pi05_port=pi05_port,
        tiptop_host=tiptop_host,
        tiptop_port=tiptop_port,
    )

    pcfg = dict(
        launch_perception=launch_perception,
        m2t2_dir=m2t2_dir,
        m2t2_host=m2t2_host,
        m2t2_port=m2t2_port,
        launch_fs=launch_fs,
        fs_dir=fs_dir,
        fs_host=fs_host,
        fs_port=fs_port,
        ready_timeout=perception_ready_timeout_s,
    )

    def spec_fn(pol: str):
        return _server_spec(
            pol,
            openpi_dir=openpi_dir,
            tiptop_dir=tiptop_dir,
            pi05_config=pi05_config,
            pi05_checkpoint=pi05_checkpoint,
            pi05_host=pi05_host,
            pi05_port=pi05_port,
            tiptop_host=tiptop_host,
            tiptop_port=tiptop_port,
            tiptop_server_module=tiptop_server_module,
            xla_mem_fraction=xla_mem_fraction,
        )

    if not stitch_only:
        if launch_servers:
            effective_mode = decide_mode(mode, concurrent_min_vram_gb)
        else:
            effective_mode = "concurrent"
            logger.info("launch_servers=False: assuming both servers already running")

        if effective_mode == "concurrent":
            ok = _run_concurrent(
                scenes, run_policies, launch_servers, spec_fn, instr_for, out_path,
                worker_kwargs, server_ready_timeout_s, pcfg,
            )
            if not ok:
                effective_mode = "alternating"
        if effective_mode == "alternating":
            _run_alternating(
                scenes, run_policies, launch_servers, spec_fn, instr_for, out_path,
                worker_kwargs, server_ready_timeout_s, pcfg,
            )

    # --- stitch ---
    logger.info("Stitching side-by-side comparison videos...")
    made = []
    for scene in scenes:
        pi05_mp4 = out_path / f"pi05_scene{scene}.mp4"
        tiptop_mp4 = out_path / f"tiptop_scene{scene}.mp4"
        out_mp4 = out_path / f"comparison_scene{scene}.mp4"
        if pi05_mp4.exists() and tiptop_mp4.exists():
            if stitch_comparison(pi05_mp4, tiptop_mp4, out_mp4, pi05_label, tiptop_label):
                made.append(out_mp4)
        else:
            logger.warning(
                f"Scene {scene}: missing per-policy video(s) "
                f"(pi05={pi05_mp4.exists()}, tiptop={tiptop_mp4.exists()}); skipping stitch"
            )

    logger.info("=" * 60)
    logger.info(f"Done. Output dir: {out_path}")
    if made:
        logger.info("Comparison videos:")
        for m in made:
            logger.info(f"  {m}")
    else:
        logger.warning("No comparison videos were produced.")


# --------------------------------------------------------------------------- #
# Entry point (dual-mode: orchestrator by default, worker if --worker-scene)   #
# --------------------------------------------------------------------------- #
def main(
    scenes: List[int] = [1, 2, 3, 4, 5],
    variant: int = 0,
    mode: str = "auto",
    policies: List[str] = ["pi05", "tiptop"],
    episode_length_s: float = 90.0,
    open_loop_horizon: int = 8,
    headless: bool = True,
    out_dir: Optional[str] = None,
    stitch_only: bool = False,
    launch_servers: bool = True,
    instruction: Optional[str] = None,
    pi05_host: str = "localhost",
    pi05_port: int = 8000,
    tiptop_host: str = "localhost",
    tiptop_port: int = 8765,
    pi05_label: str = "Pi-0.5",
    tiptop_label: str = "tiptop",
    openpi_dir: str = DEFAULT_OPENPI_DIR,
    tiptop_dir: str = DEFAULT_TIPTOP_DIR,
    pi05_config: str = "pi05_droid",
    pi05_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_droid",
    tiptop_server_module: str = "tiptop.tiptop_websocket_server",
    xla_mem_fraction: float = 0.5,
    server_ready_timeout_s: float = 1200.0,
    concurrent_min_vram_gb: float = 48.0,
    launch_perception: bool = True,
    m2t2_dir: str = DEFAULT_M2T2_DIR,
    m2t2_host: str = "localhost",
    m2t2_port: int = 8123,
    launch_fs: bool = False,
    fs_dir: str = DEFAULT_FS_DIR,
    fs_host: str = "localhost",
    fs_port: int = 8124,
    perception_ready_timeout_s: float = 600.0,
    worker_scene: Optional[int] = None,
    worker_policies: Optional[str] = None,
):
    """Run Pi-0.5 and tiptop across scenes and produce side-by-side comparison videos.

    Args:
        scenes: Scene numbers to evaluate (each yields one comparison video).
        variant: Scene variant index (object configuration).
        mode: Server scheduling: ``auto`` | ``concurrent`` | ``alternating``.
        policies: Which policies to run (``pi05``, ``tiptop``). Stitching needs both.
        episode_length_s: Episode length; pi-0.5 runs to this (it has no done signal).
        open_loop_horizon: pi-0.5 actions executed per server query before re-querying.
        headless: Run Isaac Sim without the GUI.
        out_dir: Output directory (default ``runs/<date>/<time>``).
        stitch_only: Skip rollouts; just (re)build comparison videos from existing mp4s.
        launch_servers: Launch the policy servers (False => assume already running).
        instruction: Override the per-scene instruction for all scenes.
        server_ready_timeout_s: Max wait for a server to accept connections.
        concurrent_min_vram_gb: In ``auto`` mode, single-GPU VRAM at/above this => concurrent.
        worker_scene: Internal (re-exec). Do not set manually.
        worker_policies: Internal (re-exec). Comma-separated policy list.
    """
    if worker_scene is not None:
        run_worker(
            worker_scene=worker_scene,
            worker_policies=worker_policies or ",".join(policies),
            variant=variant,
            instruction=instruction or DEFAULT_INSTRUCTIONS.get(worker_scene, ""),
            out_dir=out_dir or ".",
            episode_length_s=episode_length_s,
            open_loop_horizon=open_loop_horizon,
            headless=headless,
            pi05_host=pi05_host,
            pi05_port=pi05_port,
            tiptop_host=tiptop_host,
            tiptop_port=tiptop_port,
        )
        return

    run_orchestrator(
        scenes=scenes,
        variant=variant,
        mode=mode,
        policies=policies,
        episode_length_s=episode_length_s,
        open_loop_horizon=open_loop_horizon,
        headless=headless,
        out_dir=out_dir,
        stitch_only=stitch_only,
        launch_servers=launch_servers,
        instruction_override=instruction,
        pi05_host=pi05_host,
        pi05_port=pi05_port,
        tiptop_host=tiptop_host,
        tiptop_port=tiptop_port,
        pi05_label=pi05_label,
        tiptop_label=tiptop_label,
        openpi_dir=openpi_dir,
        tiptop_dir=tiptop_dir,
        pi05_config=pi05_config,
        pi05_checkpoint=pi05_checkpoint,
        tiptop_server_module=tiptop_server_module,
        xla_mem_fraction=xla_mem_fraction,
        server_ready_timeout_s=server_ready_timeout_s,
        concurrent_min_vram_gb=concurrent_min_vram_gb,
        launch_perception=launch_perception,
        m2t2_dir=m2t2_dir,
        m2t2_host=m2t2_host,
        m2t2_port=m2t2_port,
        launch_fs=launch_fs,
        fs_dir=fs_dir,
        fs_host=fs_host,
        fs_port=fs_port,
        perception_ready_timeout_s=perception_ready_timeout_s,
    )


if __name__ == "__main__":
    tyro.cli(main)
