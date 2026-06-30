"""Generate a TAMP (tiptop) LeRobot dataset from Isaac-sim scene 6 and push it to HuggingFace.

This runs the tiptop TAMP policy inside the DROID sim's **scene 6** ("place 3 toys on a
plate, no collisions"), randomizing the plate + toy layout each episode, keeping only the
episodes that succeed (judged from the simulation), and writing the result as a DROID-schema
LeRobot dataset (the same schema as ``SamratSahoo/d100`` / ``toys20``) uploaded to
``SamratSahoo/toys100_sim``. It loads directly through openpi's ``LeRobotDROIDDataConfig``.

Data representation (see plan / README):
  * Images            -> the ACTUAL sim-rendered camera frames captured during execution.
  * joint_position,
    gripper_position  -> the MEASURED sim proprioception captured each frame during execution
                         (the real articulation joint states, read from the sim every step -- NOT
                         the cuRobo plan). The gripper is the continuous finger opening in [0,1]
                         (0 open, 1 closed), so it ramps smoothly (~0.8 s open<->close) instead of
                         snapping. This fixes the binary/bang-bang gripper that made fine-tuned
                         policies chatter the gripper open/closed.
  * actions (8)       -> [7 joint velocities | 1 gripper position], on the SAME 15 Hz recorded-frame
                         timeline as the proprioception. The 7 joint velocities are the per-frame forward
                         difference of the COMMANDED joint positions (the cuRobo waypoint the client
                         issued each step) -- i.e. the velocity actually executed: ~zero while the arm
                         holds during a gripper open/close, the planned velocity during motion. (Finite-
                         differencing the *measured* position-controlled joints is *not* used -- it
                         yields velocities 15-70x over physical limits; the *commanded* waypoints are
                         smooth, so differencing them is fine, and unlike the raw plan velocities it
                         stays aligned with the measured state across gripper-hold frames.) The gripper
                         action is the MEASURED next-frame finger opening -- a continuous gripper-position
                         command that leads the state by one 15 Hz step, matching DROID's continuous
                         gripper-position action (and so is no longer an exact copy of the proprioception).
  * Success gating    -> measured sim object poses (sim-truth).

The script spans TWO venvs (the Isaac venv has no lerobot; the openpi venv has it). It has
three modes, dispatched from ``main``:
  * orchestrator (default) - launch M2T2 + tiptop servers, spawn the Isaac worker, then run
                             the LeRobot build under the openpi venv.
  * ``--worker``           - one persistent Isaac process; loops episodes until N successes,
                             writing per-episode raw data (3 camera mp4s + tiptop_plan.json).
  * ``--build-lerobot``    - (openpi venv) assemble + push the single LeRobot dataset.
  * ``--selftest``         - pure-numpy quaternion checks (no Isaac/lerobot needed).

Top-level imports are stdlib + numpy + tyro only (present in both venvs); everything heavy is
lazy-imported inside the mode that needs it, so each mode imports cleanly under its own venv.

Usage (run from the droid-sim-evals directory):
  uv run python tamp_data_gen.py                       # full run: 100 successes -> build -> push
  uv run python tamp_data_gen.py --num-episodes 3 --max-attempts 12 --no-push   # smoke run
  uv run python tamp_data_gen.py --no-launch-servers   # servers already running
  <openpi>/.venv/bin/python tamp_data_gen.py --build-lerobot --out-dir <dir> --no-push  # build only

Prereqs: GEMINI_API_KEY (tiptop task parsing) and HF_TOKEN (push) exported.
"""

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tyro

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tamp_data_gen")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
DEFAULT_OPENPI_DIR = str(_REPO_ROOT / "openpi")
DEFAULT_TIPTOP_DIR = str(_REPO_ROOT / "tiptop")
DEFAULT_M2T2_DIR = str(_REPO_ROOT / "M2T2")
DEFAULT_FS_DIR = str(_REPO_ROOT / "FoundationStereo")

FPS = 15
PLAN_DT = 0.02  # tiptop/cuRobo trajectories are time-parameterized at 50 Hz.
INSTRUCTION = "Place the toys on the plate with no collisions"  # scene 6 (from full_eval.py)
TOY_NAMES = ["blue_toy", "brown_toy", "pink_toy"]
SCENE_OBJECTS = ["plate"] + TOY_NAMES
_IMG_HW = (180, 320)  # DROID LeRobot image size (H, W)

# Canonical "resting flat" toy orientation (wxyz) from assets/scene6_0.json: 90 deg about X.
TOY_BASE_QUAT = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)

# DROID joint schema -- must match openpi's LeRobotDROIDDataConfig features.
_DROID_FEATURES = {
    "exterior_image_1_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
    "exterior_image_2_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
    "wrist_image_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
    "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
    "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
    # Joint *velocity* (7) + gripper position (1); the action space pi05-DROID was pretrained on.
    "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
}

# sim observation camera key -> (LeRobot feature, per-episode mp4 filename)
_CAMERAS = {
    "external_cam": ("exterior_image_1_left", "external_cam.mp4"),
    "external_cam_2": ("exterior_image_2_left", "external_cam_2.mp4"),
    "wrist_cam": ("wrist_image_left", "wrist_cam.mp4"),
}


@dataclass
class Geom:
    """Geometry constants (metres) for randomization + success checking.

    Object/plate sizes come from assets/custom/plate_toys/processed/metadata.json. All
    success z-tests are relative to the *measured* settled plate, so the JSON z-frame
    ambiguity (sidecar says table-top z~=0.395 but object z~=0.05) never matters.
    """

    plate_radius: float = 0.1125  # plate disc radius (extents 0.225 / 2)
    toy_radius: float = 0.045  # circumscribed toy radius (longest extent 0.09 / 2)
    # randomization XY boxes ((xmin,xmax),(ymin,ymax)) -- conservative, near the baseline layout
    plate_box: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.40, 0.55), (-0.05, 0.20))
    toy_box: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.30, 0.60), (-0.10, 0.22))
    off_plate_margin: float = 0.02  # toys must start clear of the plate by this much
    pair_min: float = 0.05  # min toy-pair center distance at spawn (avoid PhysX interpenetration)
    # success tolerances
    plate_xy_tol: float = 0.03  # plate must not translate more than this
    plate_z_tol: float = 0.02
    plate_ang_tol_deg: float = 12.0
    # "on plate": toy center over the disc AND lifted off the table onto it (plate is ~0.0356 thick).
    # The lift is measured against the toy's OWN settled start z, so it's immune to the table z-frame.
    on_plate_xy: float = 0.11  # toy center within this of plate center (plate radius 0.1125)
    lift_thresh: float = 0.015  # toy must rise this far above its start to count as on the plate
    floor_drop: float = 0.05  # toy must not drop more than this below its start (else it fell off)
    collision_thresh: float = 0.045  # toy-pair center distance must exceed this


# --------------------------------------------------------------------------- #
# Quaternion helpers (wxyz, Isaac convention) -- pure numpy                     #
# --------------------------------------------------------------------------- #
def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (X) q2, both wxyz."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def yaw_on_baseline(theta: float, q_base: np.ndarray = TOY_BASE_QUAT) -> np.ndarray:
    """Compose a world-Z yaw (theta) onto the resting-flat baseline: q_yaw (X) q_base.

    Order matters: applying q_base first (rest flat) then a *world*-frame Z yaw means
    left-multiplying by q_yaw. The reverse would yaw about the toy's tilted body axis and
    lift it off the table.
    """
    q_yaw = np.array([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)], dtype=np.float64)
    q = quat_mul_wxyz(q_yaw, q_base)
    return q / np.linalg.norm(q)


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Smallest rotation angle (radians) between two unit wxyz quaternions."""
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


# --------------------------------------------------------------------------- #
# Randomization sampler                                                         #
# --------------------------------------------------------------------------- #
def sample_scene_poses(rng: np.random.Generator, geom: Geom, base: dict) -> dict:
    """Sample randomized poses for the plate + 3 toys.

    Returns {name: [x, y, z, qw, qx, qy, qz]}. ``base`` holds each object's measured settled
    default pose ({name: {"pos","quat"}}) -- we keep each object's default z (and the plate's
    flat orientation), randomize XY, and give each toy a random world-Z yaw. Toys are
    rejection-sampled to start clearly off the plate and not interpenetrating each other.
    """
    (px0, px1), (py0, py1) = geom.plate_box
    plate_xy = np.array([rng.uniform(px0, px1), rng.uniform(py0, py1)])
    plate_z = float(base["plate"]["pos"][2])
    plate_quat = np.asarray(base["plate"]["quat"], dtype=np.float64)
    poses = {"plate": [plate_xy[0], plate_xy[1], plate_z, *plate_quat.tolist()]}

    (tx0, tx1), (ty0, ty1) = geom.toy_box
    placed: List[np.ndarray] = []
    for name in TOY_NAMES:
        z = float(base[name]["pos"][2])
        xy = None
        for _ in range(400):
            cand = np.array([rng.uniform(tx0, tx1), rng.uniform(ty0, ty1)])
            if np.linalg.norm(cand - plate_xy) < geom.plate_radius + geom.toy_radius + geom.off_plate_margin:
                continue  # too close to (or on) the plate
            if any(np.linalg.norm(cand - o) < geom.pair_min for o in placed):
                continue  # would interpenetrate an already-placed toy
            xy = cand
            break
        if xy is None:
            raise RuntimeError(f"sampler could not place {name} after 400 tries")
        placed.append(xy)
        q = yaw_on_baseline(rng.uniform(-math.pi, math.pi))
        poses[name] = [xy[0], xy[1], z, *q.tolist()]
    return poses


# --------------------------------------------------------------------------- #
# Object pose read/write (Isaac Lab RigidObject API) -- worker only             #
# --------------------------------------------------------------------------- #
def _get_rigid(scene, name: str):
    try:
        return scene.rigid_objects[name]
    except (KeyError, AttributeError):
        return scene[name]


def read_object_poses_vec(env) -> List[dict]:
    """Per-env [{name: {"pos": np(3), "quat": np(4 wxyz)}}] (world frame) for plate + toys.

    Success checks are all relative (final - initial) within an env, so the env-origin offset
    in world-frame positions cancels out -- no need to convert to env-local here.
    """
    scene = env.unwrapped.scene
    n = int(env.unwrapped.num_envs)
    out = [dict() for _ in range(n)]
    for name in SCENE_OBJECTS:
        obj = _get_rigid(scene, name)
        pos = obj.data.root_pos_w.detach().cpu().numpy().astype(np.float64)  # (N, 3)
        quat = obj.data.root_quat_w.detach().cpu().numpy().astype(np.float64)  # (N, 4)
        for i in range(n):
            out[i][name] = {"pos": pos[i], "quat": quat[i]}
    return out


def write_poses_vec(env, poses_per_env: List[dict], env_origins) -> None:
    """Write per-env randomized root poses (+ zero velocity). Call AFTER env.reset().

    ``poses_per_env[i][name]`` is a 7-vector [x,y,z,qw,qx,qy,qz] in env-LOCAL (robot) frame;
    we add ``env_origins[i]`` to the position so each env's objects land next to its own robot.
    """
    import torch

    scene = env.unwrapped.scene
    n = int(env.unwrapped.num_envs)
    for name in SCENE_OBJECTS:
        obj = _get_rigid(scene, name)
        dev = obj.data.root_pos_w.device
        pose = torch.zeros((n, 7), dtype=torch.float32, device=dev)
        for i in range(n):
            p = torch.tensor(poses_per_env[i][name], dtype=torch.float32, device=dev)
            pose[i, :3] = p[:3] + env_origins[i]
            pose[i, 3:] = p[3:]
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros((n, 6), dtype=torch.float32, device=dev))


def slice_obs(obs: dict, i: int, env_origin) -> dict:
    """Single-env view of a batched obs for env ``i``, shifted into that env's LOCAL frame.

    The tiptop planner effectively assumes the robot is at the world origin (true for a single
    env). For env ``i`` we subtract ``env_origin`` from the wrist-cam world position so the
    perceived point cloud lands in robot-base coordinates; everything else (images, depth,
    intrinsics, joints, quaternion) is already env-independent. Keeps the leading (1, ...) dim
    the client expects via ``[i:i+1]``.
    """
    policy = obs["policy"]
    out = {k: v[i : i + 1] for k, v in policy.items()}
    out["wrist_cam_pos_w"] = out["wrist_cam_pos_w"] - env_origin  # (1,3) - (3,) broadcast
    return {"policy": out}


# --------------------------------------------------------------------------- #
# Success evaluation -- all relative to the measured settled poses              #
# --------------------------------------------------------------------------- #
def evaluate_success(init: dict, final: dict, geom: Geom) -> Tuple[bool, dict]:
    """Check the 4 task criteria from measured sim poses. Returns (success, detail)."""
    p0 = init["plate"]["pos"]
    q0 = init["plate"]["quat"]
    pf = final["plate"]["pos"]
    qf = final["plate"]["quat"]

    plate_dxy = float(np.linalg.norm(pf[:2] - p0[:2]))
    plate_dz = float(abs(pf[2] - p0[2]))
    plate_ang = math.degrees(quat_angle(q0, qf))
    crit_plate = plate_dxy < geom.plate_xy_tol and plate_dz < geom.plate_z_tol and plate_ang < geom.plate_ang_tol_deg

    toy_xy = []
    on_plate = {}
    on_table = {}
    plate_dist = {}  # toy center -> plate center, xy
    lift = {}  # toy z rise from its settled start (≈ plate thickness when placed on the plate)
    for name in TOY_NAMES:
        t0 = init[name]["pos"]
        tf = final[name]["pos"]
        toy_xy.append(tf[:2])
        d_xy = float(np.linalg.norm(tf[:2] - pf[:2]))
        lz = float(tf[2] - t0[2])
        plate_dist[name] = round(d_xy, 4)
        lift[name] = round(lz, 4)
        on_plate[name] = d_xy < geom.on_plate_xy and lz > geom.lift_thresh
        on_table[name] = lz > -geom.floor_drop  # didn't fall to the floor / off the table
    crit_on_plate = all(on_plate.values())
    crit_table = all(on_table.values())

    pair_dists = {}
    crit_collision = True
    for i in range(len(TOY_NAMES)):
        for j in range(i + 1, len(TOY_NAMES)):
            d = float(np.linalg.norm(toy_xy[i] - toy_xy[j]))
            pair_dists[f"{TOY_NAMES[i]}-{TOY_NAMES[j]}"] = round(d, 4)
            if d <= geom.collision_thresh:
                crit_collision = False

    success = crit_plate and crit_on_plate and crit_collision and crit_table
    detail = {
        "success": success,
        "plate_unmoved": crit_plate,
        "toys_on_plate": crit_on_plate,
        "no_collision": crit_collision,
        "toys_on_table": crit_table,
        "plate_dxy": round(plate_dxy, 4),
        "plate_ang_deg": round(plate_ang, 2),
        "plate_dist": plate_dist,
        "lift": lift,
        "on_plate": on_plate,
        "pair_dists": pair_dists,
    }
    return success, detail


# --------------------------------------------------------------------------- #
# Plan serialization (worker) -- dump client._plan to a JSON the build reads    #
# --------------------------------------------------------------------------- #
def serialize_plan(steps: list) -> Tuple[list, bool]:
    """Convert a captured tiptop plan (client._plan) into a JSON-safe ``steps`` list.

    Returns (steps, has_velocities). ``has_velocities`` is False if any trajectory step is
    missing velocities -- such an episode cannot produce plan-velocity actions and must be
    rejected by the caller.
    """
    out = []
    has_velocities = True
    for s in steps:
        stype = s.get("type")
        if stype == "trajectory":
            pos = np.asarray(s["positions"], dtype=np.float32)
            step = {"type": "trajectory", "positions": pos.tolist()}
            if s.get("velocities") is not None:
                step["velocities"] = np.asarray(s["velocities"], dtype=np.float32).tolist()
            else:
                has_velocities = False
            if "dt" in s:
                step["dt"] = float(s["dt"])
            if "label" in s:
                step["label"] = s["label"]
            out.append(step)
        elif stype == "gripper":
            step = {"type": "gripper", "action": s.get("action")}
            if "label" in s:
                step["label"] = s["label"]
            out.append(step)
        # skip metadata / unknown step types
    return out, has_velocities


# --------------------------------------------------------------------------- #
# Worker: run TAMP in sim, capture sim frames + plan, gate on success           #
# --------------------------------------------------------------------------- #
def run_batch_rollout(env, obs, clients, instruction: str, max_steps: int, ep_dirs, env_origins):
    """Drive N envs through their TAMP plans in lockstep, streaming each env's 3 cameras to mp4s.

    Each env has its own tiptop client + plan; the batch runs until every env reaches plan_done
    (or fails / hits max_steps). Per env we record frames only until that env's plan_done.
    Returns per-env lists: (ok, plan_steps, n_frames) plus the final obs.

    Only each client's FIRST infer() hits its tiptop server (the ~9s plan); later per-step
    infers just walk the cached plan locally. So we issue all N first-infers CONCURRENTLY in a
    thread pool -- with K tiptop servers (clients round-robined across them) up to K plans
    compute at once, cutting the planning phase from ~N*9s to ~(N/K)*9s. After that the per-step
    loop is sequential (cheap, no server round-trip).
    """
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from full_eval import _Mp4Writer
    # Module-level obs functions: the real measured arm joints (7) and gripper opening (1, rescaled
    # to [0,1]). We call them directly (not via the ObservationManager) so we log the clean measured
    # articulation state -- the env's Gaussian observation noise is a policy-input augmentation, not
    # part of the recorded ground-truth proprioception.
    from src.sim_evals.environments.droid_environment import arm_joint_pos as _read_arm_joints
    from src.sim_evals.environments.droid_environment import gripper_pos as _read_gripper

    n = len(clients)
    dev = env.unwrapped.device
    writers = [{key: _Mp4Writer(ep_dirs[i] / fname, FPS) for key, (_, fname) in _CAMERAS.items()} for i in range(n)]
    plan_steps = [None] * n
    frame_count = [0] * n
    done = [False] * n  # plan fully executed
    failed = [False] * n  # planning/inference error
    ok = [False] * n
    # Per-env sim data, appended in lockstep with each recorded camera frame.
    sim_joint: list = [[] for _ in range(n)]  # MEASURED arm joint positions (7,) per frame (proprioception)
    sim_grip: list = [[] for _ in range(n)]  # MEASURED gripper opening in [0,1] per frame (continuous)
    sim_cmd: list = [[] for _ in range(n)]  # COMMANDED arm joint positions (7,) per frame (-> velocity action)

    # Consistency guard: arm_joint_pos must enumerate panda_joint1..7 in the SAME order as the cuRobo
    # plan / action term (which is what ret["action"][:7] is), else joint_position[k] (measured) and the
    # velocity action[k] (forward-diff of commanded positions, plan order) would refer to different joints.
    # Franka lists them in numeric order; fail loudly if a future asset reorders the DOFs.
    _panda_order = [nm for nm in env.unwrapped.scene["robot"].data.joint_names if nm.startswith("panda_joint")]
    assert _panda_order == [f"panda_joint{k}" for k in range(1, 8)], (
        f"arm joint order {_panda_order} != panda_joint1..7; proprio/velocity columns would mismatch"
    )

    def _hold(i):
        return torch.cat([obs["policy"]["arm_joint_pos"][i], obs["policy"]["gripper_pos"][i]])

    # --- Planning phase: concurrent first-infer for every env (the only server round-trip). ---
    def _plan_one(i):
        try:
            ret = clients[i].infer(slice_obs(obs, i, env_origins[i]), instruction)
            return i, ret, (list(clients[i]._plan) if clients[i]._plan else []), False
        except Exception as e:  # noqa: BLE001 - planning failure ends this env's episode
            return i, None, None, True

    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(_plan_one, range(n)))
    first_ret = [None] * n
    for i, ret, ps, fl in results:
        first_ret[i], plan_steps[i], failed[i] = ret, ps, fl
        if fl:
            logger.warning(f"{ep_dirs[i].name}: planning failed; discarding")
    logger.info(f"batch: {sum(not f for f in failed)}/{n} plans received concurrently; executing (up to {max_steps} steps)")

    try:
        for step in range(int(max_steps)):
            actions = torch.zeros((n, 8), dtype=torch.float32, device=dev)
            # Measured sim proprioception for this step, read BEFORE env.step so it is in lockstep
            # with the camera frame captured below (same sim instant). These are the true articulation
            # states (not the plan): arm joints (n,7) and the continuous gripper opening (n,) in [0,1].
            arm_now = _read_arm_joints(env.unwrapped).detach().cpu().numpy()
            grip_now = _read_gripper(env.unwrapped).detach().cpu().numpy().reshape(n)
            for i in range(n):
                if done[i] or failed[i]:
                    actions[i] = _hold(i)  # hold the robot where it is
                    continue
                if step == 0:
                    ret = first_ret[i]  # already fetched concurrently above
                else:
                    try:
                        ret = clients[i].infer(slice_obs(obs, i, env_origins[i]), instruction)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"{ep_dirs[i].name}: infer failed mid-rollout ({e}); discarding")
                        failed[i] = True
                        actions[i] = _hold(i)
                        continue
                for key, (_, _fname) in _CAMERAS.items():
                    writers[i][key].add(obs["policy"][key][i].detach().cpu().numpy())
                sim_joint[i].append(arm_now[i].astype(np.float32))
                sim_grip[i].append(np.float32(grip_now[i]))
                sim_cmd[i].append(np.asarray(ret["action"], dtype=np.float32)[:7])  # commanded arm joints
                frame_count[i] += 1
                actions[i] = torch.as_tensor(ret["action"], dtype=torch.float32, device=dev)

            obs, _, _, _, _ = env.step(actions)

            for i in range(n):
                if not done[i] and not failed[i] and clients[i].plan_done:
                    done[i] = True
                    ok[i] = True
            if all(done[i] or failed[i] for i in range(n)):
                break
    finally:
        for w_set in writers:
            for w in w_set.values():
                w.close()
        # Persist the measured sim proprioception next to the camera mp4s so the LeRobot build uses
        # true sim state (continuous gripper) instead of the plan. On success the prov dir is renamed
        # to ep_NNN, carrying this file along; on failure the whole prov dir (incl. this file) is dropped.
        for i in range(n):
            np.savez(
                ep_dirs[i] / "sim_state.npz",
                joint_position=np.asarray(sim_joint[i], dtype=np.float32).reshape(-1, 7),
                gripper_position=np.asarray(sim_grip[i], dtype=np.float32).reshape(-1),
                cmd_joint_position=np.asarray(sim_cmd[i], dtype=np.float32).reshape(-1, 7),
            )
    return ok, plan_steps, frame_count, obs


def run_worker(
    *,
    out_dir: str,
    num_episodes: int,
    scene: str,
    variant: int,
    instruction: str,
    seed: int,
    headless: bool,
    tiptop_host: str,
    tiptop_port: int,
    num_tiptop_servers: int,
    num_envs: int,
    max_steps_per_episode: int,
    max_attempts: int,
    settle_steps: int,
    post_plan_settle_steps: int,
) -> None:
    """Persistent Isaac process running ``num_envs`` parallel scene-6 episodes per batch.

    Each batch randomizes all envs, runs their TAMP plans in lockstep (one tiptop client per
    env), then success-gates each env independently. Loops batches until N successes. Works for
    num_envs=1 (then env_origins is the world origin and it reduces to the serial case).
    """
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="tamp_data_gen worker")
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

    from src.sim_evals.inference.tiptop_websocket import TiptopWebsocketClient
    from src.sim_evals.sim_utils import settle_sim

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    prov_root = out_path / "_prov"
    if prov_root.exists():
        shutil.rmtree(prov_root, ignore_errors=True)  # clear stale provisional dirs from a prior crash
    geom = Geom()
    n = int(num_envs)

    # Resume-on-restart: count already-collected episodes and continue from there, so a machine
    # crash mid-run never loses progress. The seed is offset by the resume count so re-runs explore
    # fresh layouts instead of repeating the same ones.
    resume_count = len(sorted(p for p in out_path.glob("ep_*") if (p / "tiptop_plan.json").is_file()))
    resume_base = resume_count * 100003  # large stride -> distinct per-episode seeds across resumes
    if resume_count:
        logger.info(f"Resuming: {resume_count} episodes already in {out_path}; continuing toward {num_episodes}")

    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=n, use_fabric=True)
    env_cfg.set_scene(str(scene), variant)
    # Episode must be long enough that time_out never truncates a valid rollout + settle.
    env_cfg.episode_length_s = max(120.0, (max_steps_per_episode + settle_steps + post_plan_settle_steps) / FPS + 10.0)
    env = gym.make("DROID", cfg=env_cfg)

    max_steps = min(int(max_steps_per_episode), int(env.unwrapped.max_episode_length))
    env_origins = env.unwrapped.scene.env_origins  # (N,3) tensor on device
    # Round-robin the N clients across the K tiptop servers (ports tiptop_port .. +K-1) so up
    # to K plans compute concurrently instead of all N serializing on one server.
    clients = [TiptopWebsocketClient(host=tiptop_host, port=tiptop_port + (i % max(1, num_tiptop_servers))) for i in range(n)]
    logger.info(f"Worker up: {n} parallel envs, max_steps={max_steps}, target={num_episodes} successes")

    successes = resume_count
    attempts = 0
    batch = 0
    try:
        with torch.no_grad():
            # Baseline: default settled poses give each object's resting z (+ plate quat),
            # which are env-independent (env origins differ only in XY).
            obs, _ = env.reset()
            obs, _ = env.reset()  # second render cycle for correct materials
            # Single global dome for ambient (SceneCfg.global_dome); deactivate the per-env scene-USD
            # DomeLight clones. N cloned infinite domes otherwise STACK their direct illumination and
            # over-expose every env (brightness grew with num_envs, blowing out highlights), so a
            # 128-env data frame no longer matches the single-env eval. With this, each env renders with
            # one dome's worth of ambient -- num_envs-invariant and matching full_eval.
            from src.sim_evals.environments.droid_environment import collapse_dome_lights
            collapse_dome_lights()
            obs = settle_sim(env, obs, steps=settle_steps, reset_episode_buf=True)
            # Convert env 0's settled poses to env-LOCAL frame (the multi-env grid may not put
            # env 0 at the world origin). The sampler only reads z + plate quat (origin-free), but
            # the sampler-failure fallback reuses full positions, so keep base in the local frame.
            base_world = read_object_poses_vec(env)[0]
            origin0 = env_origins[0].detach().cpu().numpy()
            base = {k: {"pos": base_world[k]["pos"] - origin0, "quat": base_world[k]["quat"]} for k in SCENE_OBJECTS}

            while successes < num_episodes and attempts < max_attempts:
                batch += 1
                batch_start = attempts

                # sample a per-env layout (env-local frame); fall back to baseline on sampler failure
                poses_per_env = []
                valid = [True] * n
                for i in range(n):
                    rng = np.random.default_rng(seed + resume_base + batch_start + i)
                    try:
                        poses_per_env.append(sample_scene_poses(rng, geom, base))
                    except RuntimeError as e:  # noqa: BLE001
                        logger.warning(f"batch {batch} env {i}: {e}; baseline layout, will discard")
                        poses_per_env.append({k: [*base[k]["pos"], *base[k]["quat"]] for k in SCENE_OBJECTS})
                        valid[i] = False

                obs, _ = env.reset()
                obs, _ = env.reset()
                write_poses_vec(env, poses_per_env, env_origins)
                obs = settle_sim(env, obs, steps=settle_steps, reset_episode_buf=True)
                init = read_object_poses_vec(env)

                prov_dirs = [prov_root / f"b{batch:03d}_e{i}" for i in range(n)]
                for d in prov_dirs:
                    if d.exists():
                        shutil.rmtree(d)
                    d.mkdir(parents=True)

                ok, plan_steps, n_frames, obs = run_batch_rollout(
                    env, obs, clients, instruction, max_steps, prov_dirs, env_origins
                )
                obs = settle_sim(env, obs, steps=post_plan_settle_steps, reset_episode_buf=False)
                final = read_object_poses_vec(env)

                for i in range(n):
                    attempts += 1
                    ep_seed = seed + resume_base + batch_start + i
                    success, detail, plan_json = False, {}, None
                    if valid[i] and ok[i] and plan_steps[i]:
                        serial, has_vel = serialize_plan(plan_steps[i])
                        n_traj = sum(1 for s in serial if s.get("type") == "trajectory")
                        if has_vel and n_traj > 0:
                            success, detail = evaluate_success(init[i], final[i], geom)
                            plan_json = {"steps": serial}

                    if success and plan_json is not None and successes < num_episodes:
                        ep_dir = out_path / f"ep_{successes:03d}"
                        if ep_dir.exists():
                            shutil.rmtree(ep_dir)
                        prov_dirs[i].rename(ep_dir)
                        (ep_dir / "tiptop_plan.json").write_text(json.dumps(plan_json))
                        meta = {
                            "instruction": instruction,
                            "fps": FPS,
                            "n_frames": n_frames[i],
                            "scene": str(scene),
                            "variant": variant,
                            "seed": int(ep_seed),
                            "cameras": {feat: fname for (feat, fname) in _CAMERAS.values()},
                            "success_detail": detail,
                            "init_poses": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in init[i].items()},
                            "final_poses": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in final[i].items()},
                        }
                        (ep_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
                        successes += 1
                        logger.info(
                            f"[OK {successes}/{num_episodes}] batch {batch} env {i} ({n_frames[i]} frames) -> {ep_dir.name} "
                            f"| rate {successes}/{attempts} = {successes / attempts:.0%}"
                        )
                    else:
                        shutil.rmtree(prov_dirs[i], ignore_errors=True)
                        reason = detail if detail else {"valid": valid[i], "ok": ok[i], "had_plan": bool(plan_steps[i])}
                        logger.info(f"[fail] batch {batch} env {i}: {reason} | rate {successes}/{attempts}")

                for c in clients:
                    c.reset()  # reconnect -> fresh server-side cuTAMP state for the next batch

            if successes < num_episodes:
                logger.warning(
                    f"Stopped at {successes}/{num_episodes} successes after {attempts} attempts (max_attempts reached)."
                )
            else:
                logger.info(f"Collected {successes} successful episodes in {attempts} attempts -> {out_path}")
    finally:
        for c in clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(prov_root, ignore_errors=True)
        env.close()
        simulation_app.close()


# --------------------------------------------------------------------------- #
# Build: assemble + push a single LeRobot dataset (openpi venv)                 #
# --------------------------------------------------------------------------- #
def _dense_plan(plan: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a tiptop plan into dense 50 Hz arrays (positions[M,7], velocities[M,7], gripper[M]).

    Copied from openpi/examples/droid/convert_combined_droid_toys_to_lerobot.py: trajectory
    steps contribute their per-row positions/velocities; gripper steps are open/close events
    that set the gripper channel (0 open, 1 closed) for all following rows. Episodes start open.
    """
    positions, velocities, gripper = [], [], []
    grip_state = 0.0
    for step in plan["steps"]:
        if step.get("type") == "gripper":
            grip_state = 1.0 if step.get("action") == "close" else 0.0
        elif step.get("type") == "trajectory" and step.get("positions") is not None:
            pos = np.asarray(step["positions"], dtype=np.float32)
            vel = np.asarray(step["velocities"], dtype=np.float32)
            positions.append(pos)
            velocities.append(vel)
            gripper.append(np.full(len(pos), grip_state, dtype=np.float32))
    return np.concatenate(positions), np.concatenate(velocities), np.concatenate(gripper)


def _resample_indices(n_src: int, n_dst: int) -> np.ndarray:
    """Indices that evenly sample n_dst points from a sequence of length n_src (nearest)."""
    if n_dst <= 1 or n_src <= 1:
        return np.zeros(max(n_dst, 0), dtype=int)
    return np.clip(np.round(np.arange(n_dst) * (n_src - 1) / (n_dst - 1)), 0, n_src - 1).astype(int)


def _decode_resized(path: str, hw: Tuple[int, int]) -> list:
    """Decode an mp4 into an ordered list of HWC uint8 RGB frames, resized to (H, W)."""
    import av
    import cv2

    h, w = hw
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        rgb = frame.to_ndarray(format="rgb24")
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    container.close()
    return frames


def _upload_dataset(dataset_root, repo_id: str, private: bool, tags: list) -> None:
    """Stream a finished local LeRobot dataset folder to the HF Hub with low RAM.

    LeRobotDataset.push_to_hub() materializes the whole (image-inline) dataset in memory and
    OOM-kills on large datasets / low-RAM machines. HfApi.upload_folder reads files off disk and
    uploads them, so RAM stays flat. The local folder already IS the LeRobot format (data/ +
    meta/), so the uploaded repo loads back through LeRobotDataset / LeRobotDROIDDataConfig.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    logger.info(f"Uploading {dataset_root} -> HF dataset {repo_id} (streaming, low-RAM)")
    api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(dataset_root),
        commit_message="Add LeRobot dataset (streamed)",
    )
    # LeRobotDataset(repo_id) requires a git tag matching info.json's codebase_version, else it
    # raises RevisionNotFoundError. upload_folder doesn't create it, so add it explicitly.
    version = json.loads((Path(dataset_root) / "meta" / "info.json").read_text()).get("codebase_version")
    if version:
        api.create_tag(repo_id, tag=version, repo_type="dataset", exist_ok=True)
    logger.info(f"Uploaded + tagged {repo_id} ({version}) to the Hub.")


def build_lerobot_dataset(
    *,
    repo_id: str,
    out_dir: str,
    instruction: str,
    push: bool,
    private: bool,
    max_episodes: Optional[int] = None,
) -> int:
    """Assemble successful episodes under ``out_dir`` into one DROID-schema LeRobot dataset.

    ``max_episodes`` caps the number of (sorted, first-N) episode dirs used -- the way the
    20-trajectory subset (toys20_sim) is built from the same raw episodes. Returns the
    number of episodes written.
    """
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    out_path = Path(out_dir)
    ep_dirs = sorted(p for p in out_path.glob("ep_*") if (p / "tiptop_plan.json").is_file())
    if max_episodes is not None:
        ep_dirs = ep_dirs[:max_episodes]
    if not ep_dirs:
        logger.error(f"No episodes with tiptop_plan.json under {out_path}")
        return 0

    dataset_root = HF_LEROBOT_HOME / repo_id
    # If a complete local build already exists (e.g. a prior run wrote all episodes but OOM'd on
    # the push), skip the expensive rebuild and just (re)upload it -- makes the build resumable.
    info_path = dataset_root / "meta" / "info.json"
    if info_path.is_file():
        try:
            n_existing = json.loads(info_path.read_text()).get("total_episodes", -1)
        except Exception:  # noqa: BLE001
            n_existing = -1
        if n_existing == len(ep_dirs):
            logger.info(f"{repo_id}: complete local build exists ({n_existing} eps) -> skip rebuild, upload only")
            if push:
                _upload_dataset(dataset_root, repo_id, private, ["droid", "panda", "sim", "tamp", "toys"])
            return n_existing
        logger.info(f"{repo_id}: local build incomplete ({n_existing} vs {len(ep_dirs)}) -> rebuilding")
    if dataset_root.exists():
        logger.info(f"Removing existing local dataset at {dataset_root}")
        shutil.rmtree(dataset_root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=FPS,
        features=_DROID_FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_written = 0
    for ep in ep_dirs:
        plan = json.loads((ep / "tiptop_plan.json").read_text())
        task = instruction
        meta_path = ep / "_meta.json"
        if meta_path.is_file():
            task = json.loads(meta_path.read_text()).get("instruction", instruction)

        # State + action are taken from the MEASURED sim trajectory (sim_state.npz) on its NATIVE 15 Hz
        # recorded-frame timeline (n = F frames). joint_position/gripper_position pass through 1:1 (true
        # measured proprioception; the gripper is the continuous finger opening). The joint-velocity
        # action is the per-frame forward difference of the COMMANDED joint positions (* FPS) -- i.e.
        # the velocity actually executed each step: ~ZERO while the arm holds during a gripper open/close
        # (the client commands the arm to hold its current pose, so the forward-diff is negligible) and
        # the planned velocity during motion. This keeps the velocity action on the SAME timeline as the
        # measured state (the plan timeline has no rows during gripper holds, so plan velocities would
        # drift ahead of the state there). Finite-differencing the smooth *commanded* waypoints is safe;
        # the 15-70x over-limit problem is only from differencing the *measured* (jittery) joints. The
        # gripper action is the next-frame opening (continuous, leads the state by one 15 Hz step),
        # matching DROID. Legacy raw episodes without sim_state.npz fall back to the old plan-derived
        # (binary gripper, plan timeline) build so they still assemble, with a warning.
        sim_path = ep / "sim_state.npz"
        joint_pos = g_state = vel = None
        from_sim_state = False
        n = 0
        if sim_path.is_file():
            sd = np.load(sim_path)
            sj = sd["joint_position"].astype(np.float32)
            sg = sd["gripper_position"].astype(np.float32)
            cj = sd["cmd_joint_position"].astype(np.float32) if "cmd_joint_position" in sd.files else None
            if len(sj) >= 2 and cj is not None and len(cj) == len(sj):
                n = len(sj)
                joint_pos = sj
                g_state = np.clip(sg, 0.0, 1.0)
                dvel = (cj[1:] - cj[:-1]) * FPS  # commanded joint velocity at 15 Hz (~0 during holds)
                vel = np.concatenate([dvel, np.zeros((1, 7), np.float32)], axis=0).astype(np.float32)
                from_sim_state = True
            else:
                logger.warning(f"{ep.name}: sim_state.npz unusable (len={len(sj)}, cmd={None if cj is None else len(cj)}); falling back to plan state")
        else:
            logger.warning(f"{ep.name}: no sim_state.npz (legacy raw episode); falling back to plan-derived binary gripper")
        if joint_pos is None:  # fallback: legacy plan-derived state (binary gripper) on the plan timeline
            pos50, vel50, grip50 = _dense_plan(plan)
            m = len(pos50)
            n = max(2, int(round(m * PLAN_DT * FPS)))  # 50 Hz -> 15 Hz
            sel = _resample_indices(m, n)
            joint_pos = pos50[sel].astype(np.float32)
            g_state = grip50[sel].astype(np.float32)
            vel = vel50[sel].astype(np.float32)
        # gripper position *command* = next-frame opening (continuous, leads the state by one frame)
        g_action = np.concatenate([g_state[1:], g_state[-1:]]).astype(np.float32)

        decoded = {}
        missing = False
        for _key, (feat, fname) in _CAMERAS.items():
            mp4 = ep / fname
            if not mp4.is_file():
                logger.warning(f"{ep.name}: missing {fname}; skipping episode")
                missing = True
                break
            decoded[feat] = _decode_resized(str(mp4), _IMG_HW)
        if missing:
            continue
        # On the sim_state path, camera frames are recorded in lockstep with the measured state (n = F),
        # so len(frames) should equal n; warn on any divergence (e.g. an mp4 encoder drop) since it would
        # desync images from the measured state. (Skipped on the legacy fallback, where n is plan-derived
        # and differing from F is expected and handled by the resample.)
        if from_sim_state:
            for feat, frames in decoded.items():
                if abs(len(frames) - n) > 2:
                    logger.warning(f"{ep.name}: {feat} has {len(frames)} frames vs {n} state frames; resampling (possible image/state desync)")
        img_sel = {feat: _resample_indices(len(frames), n) for feat, frames in decoded.items()}

        for i in range(n):
            dataset.add_frame(
                {
                    "exterior_image_1_left": decoded["exterior_image_1_left"][img_sel["exterior_image_1_left"][i]],
                    "exterior_image_2_left": decoded["exterior_image_2_left"][img_sel["exterior_image_2_left"][i]],
                    "wrist_image_left": decoded["wrist_image_left"][img_sel["wrist_image_left"][i]],
                    "joint_position": joint_pos[i],
                    "gripper_position": g_state[i : i + 1],
                    "actions": np.concatenate([vel[i], g_action[i : i + 1]]).astype(np.float32),
                    "task": task,
                }
            )
        dataset.save_episode()
        n_written += 1
        logger.info(f"  [{n_written}/{len(ep_dirs)}] {ep.name}: {n} frames | task: {task!r}")

    logger.info(f"Wrote {n_written} episodes to {dataset_root}")
    dataset.stop_image_writer()  # release the writer pool + finalize files before uploading
    if push and n_written:
        _upload_dataset(dataset_root, repo_id, private, ["droid", "panda", "sim", "tamp", "toys"])
    return n_written


# --------------------------------------------------------------------------- #
# Merge: combine finished DROID-joint LeRobot datasets (openpi venv)            #
# --------------------------------------------------------------------------- #
_IMG_KEYS = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")


def _to_hwc_uint8(img) -> np.ndarray:
    """A LeRobot image frame (CHW float[0,1] tensor or HWC array) -> HWC uint8 for add_frame."""
    arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    return arr


def merge_datasets(*, sources: List[str], repo_id: str, push: bool, private: bool) -> int:
    """Merge >=2 finished DROID-joint LeRobot datasets (local cache or Hub) into one.

    Mirrors openpi/examples/droid/merge_lerobot_datasets.py: episodes are copied verbatim
    (per-episode 15 Hz timeline rebuilt by add_frame), instructions carried over. Returns the
    number of episodes written.
    """
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    dataset_root = HF_LEROBOT_HOME / repo_id
    if dataset_root.exists():
        logger.info(f"Removing existing local dataset at {dataset_root}")
        shutil.rmtree(dataset_root)

    # Synchronous, single-process image writing (no async worker pool): merging also loads a large
    # source dataset into this process, so we keep the writer's RAM footprint minimal to leave
    # headroom (the multi-process writer otherwise buffers images across 5 procs).
    out = LeRobotDataset.create(
        repo_id=repo_id, robot_type="panda", fps=FPS, features=_DROID_FEATURES,
        image_writer_threads=2, image_writer_processes=0,
    )
    total = 0
    for src_id in sources:
        src = LeRobotDataset(src_id)  # local cache if present, else pulled from the Hub
        missing = [k for k in _DROID_FEATURES if k not in src.features]
        if missing:
            raise ValueError(f"Source '{src_id}' missing {missing}; not a DROID-joint LeRobot dataset.")
        logger.info(f"Merging {src.num_episodes} episodes from {src_id}")
        for ep in range(src.num_episodes):
            start = int(src.episode_data_index["from"][ep])
            end = int(src.episode_data_index["to"][ep])
            task = "do something"
            for i in range(start, end):
                f = src[i]
                task = f["task"]
                frame = {
                    "joint_position": np.asarray(f["joint_position"], dtype=np.float32).reshape(7),
                    "gripper_position": np.asarray(f["gripper_position"], dtype=np.float32).reshape(1),
                    "actions": np.asarray(f["actions"], dtype=np.float32).reshape(8),
                    "task": task,
                }
                for k in _IMG_KEYS:
                    frame[k] = _to_hwc_uint8(f[k])
                out.add_frame(frame)
            out.save_episode()
            total += 1
    logger.info(f"Wrote {total} merged episodes from {len(sources)} sources to {dataset_root}")
    out.stop_image_writer()  # release the writer pool + finalize files before uploading
    if push and total:
        _upload_dataset(dataset_root, repo_id, private, ["droid", "panda", "tamp-vla", "merged"])
    return total


def build_all_datasets(
    *,
    out_dir: str,
    repo_id: str,
    instruction: str,
    push: bool,
    private: bool,
    do_subset: bool,
    subset_repo_id: str,
    subset_n: int,
    do_merges: bool,
    d100_repo: str,
    merge_repo_id: str,
    subset_merge_repo_id: str,
) -> None:
    """Full LeRobot-side pipeline (openpi venv): main dataset + 20-subset + the two d100 merges."""
    n_main = build_lerobot_dataset(repo_id=repo_id, out_dir=out_dir, instruction=instruction,
                                   push=push, private=private)
    if n_main == 0:
        logger.error("Main dataset is empty; skipping subset/merges.")
        return

    if do_subset:
        n_sub = build_lerobot_dataset(repo_id=subset_repo_id, out_dir=out_dir, instruction=instruction,
                                      push=push, private=private, max_episodes=subset_n)
        logger.info(f"Subset {subset_repo_id}: {n_sub} episodes.")

    if do_merges:
        merge_datasets(sources=[d100_repo, repo_id], repo_id=merge_repo_id, push=push, private=private)
        if do_subset:
            merge_datasets(sources=[d100_repo, subset_repo_id], repo_id=subset_merge_repo_id,
                           push=push, private=private)


# --------------------------------------------------------------------------- #
# Orchestrator: launch servers, spawn worker, run build                         #
# --------------------------------------------------------------------------- #
def spawn_worker(*, out_dir: Path, worker_kwargs: dict) -> int:
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--out-dir", str(out_dir),
        "--num-episodes", str(worker_kwargs["num_episodes"]),
        "--scene", str(worker_kwargs["scene"]),
        "--variant", str(worker_kwargs["variant"]),
        "--instruction", worker_kwargs["instruction"],
        "--seed", str(worker_kwargs["seed"]),
        "--tiptop-host", worker_kwargs["tiptop_host"],
        "--tiptop-port", str(worker_kwargs["tiptop_port"]),
        "--num-tiptop-servers", str(worker_kwargs["num_tiptop_servers"]),
        "--num-envs", str(worker_kwargs["num_envs"]),
        "--max-steps-per-episode", str(worker_kwargs["max_steps_per_episode"]),
        "--max-attempts", str(worker_kwargs["max_attempts"]),
        "--settle-steps", str(worker_kwargs["settle_steps"]),
        "--post-plan-settle-steps", str(worker_kwargs["post_plan_settle_steps"]),
        "--headless" if worker_kwargs["headless"] else "--no-headless",
    ]
    logger.info(f"=== Spawning Isaac worker: {worker_kwargs['num_episodes']} successes ===")
    return subprocess.run(cmd).returncode


def run_build_subprocess(*, openpi_python: str, openpi_dir: str, repo_id: str, out_dir: Path,
                         instruction: str, push: bool, private: bool, do_subset: bool,
                         subset_repo_id: str, subset_n: int, do_merges: bool, d100_repo: str,
                         merge_repo_id: str, subset_merge_repo_id: str) -> int:
    cmd = [
        openpi_python, str(Path(__file__).resolve()), "--build-lerobot",
        "--repo-id", repo_id,
        "--out-dir", str(out_dir),
        "--instruction", instruction,
        "--push" if push else "--no-push",
        "--private" if private else "--no-private",
        "--do-subset" if do_subset else "--no-do-subset",
        "--subset-repo-id", subset_repo_id,
        "--subset-n", str(subset_n),
        "--do-merges" if do_merges else "--no-do-merges",
        "--d100-repo", d100_repo,
        "--merge-repo-id", merge_repo_id,
        "--subset-merge-repo-id", subset_merge_repo_id,
    ]
    logger.info(f"=== Building LeRobot datasets under openpi venv: {' '.join(cmd)} ===")
    return subprocess.run(cmd, cwd=str(Path(openpi_dir).expanduser()), env={**os.environ}).returncode


def run_orchestrator(
    *,
    repo_id: str,
    out_dir: Optional[str],
    launch_servers: bool,
    push: bool,
    private: bool,
    build: bool,
    do_subset: bool,
    subset_repo_id: str,
    subset_n: int,
    do_merges: bool,
    d100_repo: str,
    merge_repo_id: str,
    subset_merge_repo_id: str,
    openpi_dir: str,
    openpi_venv_python: Optional[str],
    tiptop_dir: str,
    tiptop_host: str,
    tiptop_port: int,
    num_tiptop_servers: int,
    tiptop_server_module: str,
    m2t2_dir: str,
    m2t2_host: str,
    m2t2_port: int,
    server_ready_timeout_s: float,
    perception_ready_timeout_s: float,
    xla_mem_fraction: float,
    worker_kwargs: dict,
) -> None:
    from full_eval import _server_spec, start_perception_servers, start_server, stop_server, wait_for_server

    if out_dir:
        out_path = Path(out_dir).resolve()
    else:
        now = datetime.now()
        out_path = (Path("runs") / "tamp_data" / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_path}")

    tiptop_procs: List = []
    perception_procs: List = []
    worker_rc = 1
    try:
        if launch_servers:
            pcfg = dict(
                m2t2_dir=m2t2_dir, m2t2_host=m2t2_host, m2t2_port=m2t2_port,
                launch_fs=False, fs_dir=DEFAULT_FS_DIR, fs_host="localhost", fs_port=8124,
            )
            perception_procs, ok = start_perception_servers(pcfg, perception_ready_timeout_s)
            if not ok:
                logger.error("M2T2 perception server failed to start; aborting (tiptop needs it).")
                return
            # Launch K independent tiptop servers (each its own cuRobo instance) on consecutive
            # ports; the worker round-robins its N clients across them so up to K plans run at once.
            for k in range(num_tiptop_servers):
                cmd, cwd, env, host, _port = _server_spec(
                    "tiptop",
                    openpi_dir=openpi_dir, tiptop_dir=tiptop_dir,
                    pi05_config="unused", pi05_checkpoint="unused", pi05_host="localhost", pi05_port=8000,
                    tiptop_host=tiptop_host, tiptop_port=tiptop_port + k,
                    tiptop_server_module=tiptop_server_module, xla_mem_fraction=xla_mem_fraction,
                )
                proc = start_server("tiptop", (cmd, cwd, env, host, tiptop_port + k))
                tiptop_procs.append(proc)
                # Stagger: wait for each server's cuRobo warmup to finish before launching the next,
                # so warmups never overlap (extra safety + avoids the peak-memory spike of K at once).
                if not wait_for_server(proc, tiptop_host, tiptop_port + k, server_ready_timeout_s):
                    logger.error(f"tiptop server {k} (port {tiptop_port + k}) not ready; aborting.")
                    return
        else:
            logger.info(f"launch_servers=False: assuming M2T2 + {num_tiptop_servers} tiptop server(s) already running.")

        worker_rc = spawn_worker(out_dir=out_path, worker_kwargs=worker_kwargs)
        if worker_rc != 0:
            logger.error(f"Worker exited with code {worker_rc}; skipping build.")
            return
    finally:
        for p in tiptop_procs:
            stop_server(p)
        for p in perception_procs:
            stop_server(p)

    if build and worker_rc == 0:
        openpi_python = openpi_venv_python or str(Path(openpi_dir).expanduser() / ".venv" / "bin" / "python")
        rc = run_build_subprocess(
            openpi_python=openpi_python, openpi_dir=openpi_dir, repo_id=repo_id,
            out_dir=out_path, instruction=worker_kwargs["instruction"], push=push, private=private,
            do_subset=do_subset, subset_repo_id=subset_repo_id, subset_n=subset_n,
            do_merges=do_merges, d100_repo=d100_repo, merge_repo_id=merge_repo_id,
            subset_merge_repo_id=subset_merge_repo_id,
        )
        if rc != 0:
            logger.error(f"Build step exited with code {rc}.")
        else:
            logger.info(f"Done. Datasets assembled{' and pushed' if push else ''}: {repo_id}")


# --------------------------------------------------------------------------- #
# Self-test (pure numpy; runs in either venv)                                   #
# --------------------------------------------------------------------------- #
def run_selftest() -> None:
    # yaw 0 returns the baseline.
    assert np.allclose(yaw_on_baseline(0.0), TOY_BASE_QUAT, atol=1e-6), "yaw(0) != baseline"
    # The toy's body up-axis must stay world-flat (small |z|) across yaws -- i.e. the toy
    # only spins about world Z, never tilts. Rotate the mesh's local +Z by q_final.
    from scipy.spatial.transform import Rotation

    base_up = Rotation.from_quat(np.r_[TOY_BASE_QUAT[1:], TOY_BASE_QUAT[0]]).apply([0, 0, 1])
    for theta in np.linspace(-math.pi, math.pi, 9):
        q = yaw_on_baseline(theta)
        up = Rotation.from_quat(np.r_[q[1:], q[0]]).apply([0, 0, 1])
        assert abs(up[2] - base_up[2]) < 1e-6, f"toy tilted at theta={theta}: up={up}"
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6, "non-unit quaternion"
    # quat_angle sanity
    assert abs(math.degrees(quat_angle(TOY_BASE_QUAT, yaw_on_baseline(math.pi / 2))) - 90.0) < 1e-3
    logger.info("selftest OK: yaw_on_baseline keeps toys flat, spins about world Z, unit quats.")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #
def main(
    repo_id: str = "SamratSahoo/toys100_sim",
    num_episodes: int = 100,
    scene: str = "6",
    variant: int = 0,
    instruction: str = INSTRUCTION,
    out_dir: Optional[str] = None,
    seed: int = 0,
    headless: bool = True,
    launch_servers: bool = True,
    push: bool = True,
    private: bool = False,
    build: bool = True,
    do_subset: bool = True,
    subset_repo_id: str = "SamratSahoo/toys20_sim",
    subset_n: int = 20,
    do_merges: bool = True,
    d100_repo: str = "SamratSahoo/d100",
    merge_repo_id: str = "SamratSahoo/d100_toys100_sim",
    subset_merge_repo_id: str = "SamratSahoo/d100_toys20_sim",
    num_envs: int = 128,  # parallel envs/batch — amortizes the dominant ~150s render. 256 OOM'd the 32GB 5090 during scene load (crashed the host); 128 leaves headroom alongside the tiptop + M2T2 servers. With the perception-overlap server, planning no longer serializes badly at high N (watch VRAM + Gemini rate limits).
    max_steps_per_episode: int = 2400,
    max_attempts: int = 400,
    settle_steps: int = 120,
    post_plan_settle_steps: int = 80,
    openpi_dir: str = DEFAULT_OPENPI_DIR,
    openpi_venv_python: Optional[str] = None,
    tiptop_dir: str = DEFAULT_TIPTOP_DIR,
    tiptop_host: str = "localhost",
    tiptop_port: int = 8765,
    num_tiptop_servers: int = 1,
    tiptop_server_module: str = "tiptop.tiptop_websocket_server",
    m2t2_dir: str = DEFAULT_M2T2_DIR,
    m2t2_host: str = "localhost",
    m2t2_port: int = 8123,
    server_ready_timeout_s: float = 1200.0,
    perception_ready_timeout_s: float = 600.0,
    xla_mem_fraction: float = 0.5,
    # internal mode flags (do not set manually)
    worker: bool = False,
    build_lerobot: bool = False,
    selftest: bool = False,
) -> None:
    """Generate a scene-6 TAMP LeRobot dataset and push it to HuggingFace.

    Args:
        repo_id: HuggingFace dataset repo to create/push (default SamratSahoo/toys100_sim).
        num_episodes: Number of *successful* trajectories to collect.
        instruction: Scene-6 task string (also the LeRobot ``task`` field).
        out_dir: Raw per-episode output dir (default runs/tamp_data/<date>/<time>).
        seed: Base RNG seed (episode seed = seed + attempt index).
        launch_servers: Launch tiptop + M2T2 servers (False => assume already running).
        push: Push the assembled dataset to the Hub.
        build: Run the LeRobot build step after the worker (False => raw episodes only).
        max_attempts: Cap on randomized attempts while chasing ``num_episodes`` successes.
        worker / build_lerobot / selftest: Internal mode selectors (do not set manually).
    """
    if selftest:
        run_selftest()
        return
    if worker:
        run_worker(
            out_dir=out_dir or ".",
            num_episodes=num_episodes,
            scene=scene,
            variant=variant,
            instruction=instruction,
            seed=seed,
            headless=headless,
            tiptop_host=tiptop_host,
            tiptop_port=tiptop_port,
            num_tiptop_servers=num_tiptop_servers,
            num_envs=num_envs,
            max_steps_per_episode=max_steps_per_episode,
            max_attempts=max_attempts,
            settle_steps=settle_steps,
            post_plan_settle_steps=post_plan_settle_steps,
        )
        return
    if build_lerobot:
        build_all_datasets(
            out_dir=out_dir or ".",
            repo_id=repo_id,
            instruction=instruction,
            push=push,
            private=private,
            do_subset=do_subset,
            subset_repo_id=subset_repo_id,
            subset_n=subset_n,
            do_merges=do_merges,
            d100_repo=d100_repo,
            merge_repo_id=merge_repo_id,
            subset_merge_repo_id=subset_merge_repo_id,
        )
        return

    worker_kwargs = dict(
        num_episodes=num_episodes,
        scene=scene,
        variant=variant,
        instruction=instruction,
        seed=seed,
        headless=headless,
        tiptop_host=tiptop_host,
        tiptop_port=tiptop_port,
        num_tiptop_servers=num_tiptop_servers,
        num_envs=num_envs,
        max_steps_per_episode=max_steps_per_episode,
        max_attempts=max_attempts,
        settle_steps=settle_steps,
        post_plan_settle_steps=post_plan_settle_steps,
    )
    run_orchestrator(
        repo_id=repo_id,
        out_dir=out_dir,
        launch_servers=launch_servers,
        push=push,
        private=private,
        build=build,
        do_subset=do_subset,
        subset_repo_id=subset_repo_id,
        subset_n=subset_n,
        do_merges=do_merges,
        d100_repo=d100_repo,
        merge_repo_id=merge_repo_id,
        subset_merge_repo_id=subset_merge_repo_id,
        openpi_dir=openpi_dir,
        openpi_venv_python=openpi_venv_python,
        tiptop_dir=tiptop_dir,
        tiptop_host=tiptop_host,
        tiptop_port=tiptop_port,
        num_tiptop_servers=num_tiptop_servers,
        tiptop_server_module=tiptop_server_module,
        m2t2_dir=m2t2_dir,
        m2t2_host=m2t2_host,
        m2t2_port=m2t2_port,
        server_ready_timeout_s=server_ready_timeout_s,
        perception_ready_timeout_s=perception_ready_timeout_s,
        xla_mem_fraction=xla_mem_fraction,
        worker_kwargs=worker_kwargs,
    )


if __name__ == "__main__":
    tyro.cli(main)
