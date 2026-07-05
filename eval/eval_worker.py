#!/usr/bin/env python3
"""Parallel-env Pi-0.5 evaluation worker: one scene, one policy, N randomized rollouts.

Runs ``num_envs`` (= ``--num-rollouts``) parallel Isaac envs of a single scene (6-12) for a single
Pi-0.5 policy, each env with an independently randomized object layout, and scores every env against
that scene's success criteria (``eval/scene_specs.py``). Reuses the exact multi-env + per-env
randomization + ``collapse_dome_lights`` pattern proven in ``data/tamp_data_gen.py`` (so lighting is
num_envs-invariant), but drives Pi-0.5 (one websocket client per env, batched via a thread pool)
instead of the TAMP planner, and computes generic per-scene success instead of gating a dataset.

Reports, to ``<out-dir>/results.json``:
  * ``dense_score``  -- mean over envs of the *fraction* of that scene's criteria met (partial credit),
  * ``sparse_score`` -- fraction of envs meeting ALL criteria,
  * ``criterion_rates`` + full per-env breakdown.
Optionally (``--record-video``) writes a tiled multi-env ``video.mp4`` (exterior|wrist per env).

This launches its own Isaac process (like the tamp_data_gen worker); the driver
(``eval/eval_pi05_policies.py``) spawns one per (policy, scene) against an already-running pi05 server.

Usage (eval .venv python, from droid-sim-evals; pi05 server already up on --pi05-port):
    .venv/bin/python eval/eval_worker.py --scene 8 --num-rollouts 8 --out-dir /tmp/s8 \
        --pi05-port 8000 --control-mode velocity --record-video
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_worker")

_EVAL_DIR = Path(__file__).resolve().parent          # droid-sim-evals/eval
_DSE_DIR = _EVAL_DIR.parent                          # droid-sim-evals/
sys.path[:0] = [str(_DSE_DIR), str(_EVAL_DIR)]

FPS = 15


# --------------------------------------------------------------------------- #
# Object state read / randomized-pose write (Isaac Lab RigidObject/Articulation)#
# --------------------------------------------------------------------------- #
def read_states(env, rigid_names, art_names) -> list:
    """Per-env [{name: {"pos": np(3), "quat": np(4 wxyz), ["joint": np(J)]}}] in WORLD frame.

    All success checks are relative (final/traj vs init) within one env, so the world-frame
    env-origin offset cancels -- no env-local conversion needed for scoring.
    """
    scene = env.unwrapped.scene
    n = int(env.unwrapped.num_envs)
    out = [dict() for _ in range(n)]
    for name in list(rigid_names) + list(art_names):
        obj = scene[name]
        pos = obj.data.root_pos_w.detach().cpu().numpy().astype(np.float64)
        quat = obj.data.root_quat_w.detach().cpu().numpy().astype(np.float64)
        joints = obj.data.joint_pos.detach().cpu().numpy().astype(np.float64) if name in art_names else None
        for i in range(n):
            d = {"pos": pos[i], "quat": quat[i]}
            if joints is not None:
                d["joint"] = joints[i]
            out[i][name] = d
    return out


def write_layout(env, poses_per_env, rigid_names, art_names, env_origins) -> None:
    """Write each env's randomized root pose (+ zero velocity); reset articulation joints to default.

    ``poses_per_env[i][name]`` is [x,y,z,qw,qx,qy,qz] in env-LOCAL (robot) frame; we add
    ``env_origins[i]`` so each env's objects land next to its own robot. Call AFTER env.reset().
    """
    import torch

    scene = env.unwrapped.scene
    n = int(env.unwrapped.num_envs)
    for name in list(rigid_names) + list(art_names):
        obj = scene[name]
        dev = obj.data.root_pos_w.device
        pose = torch.zeros((n, 7), dtype=torch.float32, device=dev)
        for i in range(n):
            p = torch.tensor(poses_per_env[i][name], dtype=torch.float32, device=dev)
            pose[i, :3] = p[:3] + env_origins[i]
            pose[i, 3:] = p[3:]
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros((n, 6), dtype=torch.float32, device=dev))
        if name in art_names:
            obj.write_joint_state_to_sim(obj.data.default_joint_pos.clone(), torch.zeros_like(obj.data.default_joint_vel))


def measure_base(env, rigid_names, art_names, env_origins) -> dict:
    """Env-0 settled pose per object, in env-LOCAL frame -- the sampler's baseline (z + orientation)."""
    world0 = read_states(env, rigid_names, art_names)[0]
    o0 = env_origins[0].detach().cpu().numpy()
    return {k: {"pos": world0[k]["pos"] - o0, "quat": world0[k]["quat"]} for k in world0}


# --------------------------------------------------------------------------- #
# Tiled multi-env video                                                         #
# --------------------------------------------------------------------------- #
def tile_grid(panels: list) -> np.ndarray:
    """Arrange N (H, W, 3) uint8 panels into a near-square grid, padding missing cells black."""
    import math

    n = len(panels)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    h, w = panels[0].shape[:2]
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    grid_rows = []
    for r in range(rows):
        cells = [panels[r * cols + c] if r * cols + c < n else blank for c in range(cols)]
        grid_rows.append(np.concatenate(cells, axis=1))
    return np.concatenate(grid_rows, axis=0)


def fit_width(img: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale ``img`` (area-interp, crisp) so its width <= max_width; force even dims for the encoder."""
    import cv2

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        h, w = int(round(h * scale)), max_width
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    h2, w2 = h - (h % 2), w - (w % 2)  # even dims (mp4 encoders require it)
    return img[:h2, :w2]


# --------------------------------------------------------------------------- #
# Worker                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="parallel-env pi05 eval worker (one scene, one policy)")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--num-rollouts", type=int, default=8, help="parallel envs (randomized layouts)")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--pi05-host", type=str, default="localhost")
    ap.add_argument("--pi05-port", type=int, default=8000)
    ap.add_argument("--control-mode", choices=["velocity", "position"], default="velocity")
    ap.add_argument("--open-loop-horizon", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=1800, help="env steps per rollout (pi05 has no done signal)")
    ap.add_argument("--settle-steps", type=int, default=30)
    ap.add_argument("--post-settle-steps", type=int, default=30)
    ap.add_argument("--track-every", type=int, default=2, help="record object state every k steps (temporal criteria)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record-video", action="store_true")
    ap.add_argument("--video-every", type=int, default=1, help="record a video frame every k steps (1 = smoothest)")
    # Dedicated HIGH-RES video cameras (separate from the 180x320 policy obs cams, so the policy input --
    # and thus eval behaviour -- is unchanged; the sim-finetuned policies were trained at 180x320).
    ap.add_argument("--render-height", type=int, default=720, help="video camera render height (per view)")
    ap.add_argument("--render-width", type=int, default=1280, help="video camera render width (per view)")
    ap.add_argument("--video-max-width", type=int, default=2560,
                    help="downscale the tiled grid to at most this width (keeps multi-env files sane)")
    ap.add_argument("--dummy-policy", action="store_true",
                    help="skip the pi05 server; hold the arm each step (validates scene/scoring/video only)")
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.set_defaults(headless=True)
    args = ap.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": args.headless, "enable_cameras": True})
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import src.sim_evals.environments  # noqa: F401  (registers the "DROID" gym env)
    from isaaclab_tasks.utils import parse_env_cfg

    from src.sim_evals.inference.pi05_batched import Pi05BatchedClient
    from src.sim_evals.environments.droid_environment import collapse_dome_lights, set_arm_control_mode
    from src.sim_evals.sim_utils import settle_sim
    from scene_specs import get_scene, sample_layout

    spec = get_scene(args.scene)
    n = int(args.num_rollouts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"scene {args.scene} '{spec.instruction}' | {n} envs | control={args.control_mode}")

    device = f"cuda:{app_launcher.device_id}"
    env_cfg = parse_env_cfg("DROID", device=device, num_envs=n, use_fabric=True)
    env_cfg.set_scene(str(args.scene), 0)
    env_cfg.episode_length_s = max(120.0, (args.max_steps + args.settle_steps + args.post_settle_steps) / FPS + 10.0)

    # Speed opt 2: pi05 only consumes external_cam + wrist_cam, so shrink the unused 2nd exterior camera
    # to ~nothing -> it no longer costs a full render each step (kept at 16x16, not removed, so its
    # observation term still resolves; the tiny image is simply never read).
    env_cfg.scene.external_cam_2.height = 16
    env_cfg.scene.external_cam_2.width = 16

    # Speed opt 1: render the cameras only every open_loop_horizon control steps instead of every step --
    # the policy reads images once per query (every open_loop_horizon steps), so ~7/8 of the per-step
    # renders are wasted. We keep Isaac's AUTO render on (fully disabling it hangs the RTX-sensor pipeline)
    # but slow it to the query cadence, and phase-align it below so each query still gets a FRESH image
    # (verified). Disabled when recording video (that needs a frame every step).
    fast_render = not args.record_video
    if fast_render:
        env_cfg.sim.render_interval = env_cfg.decimation * args.open_loop_horizon

    if args.record_video:
        # Add high-res video-only clones of the third-person + wrist cameras (deep-copied from the
        # policy cams so they share pose/intrinsics, but at --render-height x --render-width and NOT in
        # the observation group -> the policy still sees the original 180x320 cams. InteractiveScene
        # picks up any TiledCameraCfg attribute set on the scene cfg (same mechanism as sidecar objects).
        import copy

        for src, dst in (("external_cam", "video_ext"), ("wrist_cam", "video_wrist")):
            cam = copy.deepcopy(getattr(env_cfg.scene, src))
            cam.height, cam.width = args.render_height, args.render_width
            cam.data_types = ["rgb"]
            cam.prim_path = cam.prim_path.replace(src, dst)
            setattr(env_cfg.scene, dst, cam)
    env = gym.make("DROID", cfg=env_cfg)
    max_steps = min(int(args.max_steps), int(env.unwrapped.max_episode_length))
    env_origins = env.unwrapped.scene.env_origins  # (N,3) on device

    rigid, art = spec.rigid, spec.articulated
    results = {}
    with torch.no_grad():
        obs, _ = env.reset()
        obs, _ = env.reset()  # second render cycle -> correct materials
        collapse_dome_lights()

        # Baseline settled poses (env-local) drive the sampler (each object's rest z + orientation).
        set_arm_control_mode("position")
        obs = settle_sim(env, obs, steps=args.settle_steps, reset_episode_buf=True)
        base = measure_base(env, rigid, art, env_origins)

        # Sample one randomized layout per env, apply, settle, measure the true start poses.
        poses_per_env = []
        for i in range(n):
            rng = np.random.default_rng(args.seed + i)
            try:
                poses_per_env.append(sample_layout(rng, spec, base))
            except RuntimeError as e:
                logger.warning(f"env {i}: {e}; using baseline layout")
                poses_per_env.append({k: [*base[k]["pos"], *base[k]["quat"]] for k in base})
        obs, _ = env.reset()
        obs, _ = env.reset()
        write_layout(env, poses_per_env, rigid, art, env_origins)
        obs = settle_sim(env, obs, steps=args.settle_steps, reset_episode_buf=True)
        init = read_states(env, rigid, art)

        # --- rollout: ONE batched pi05 client drives all N envs per query step (single batch-N forward
        # pass instead of N serial infers, which were ~67% of the eval's wall clock). ---
        client = None if args.dummy_policy else Pi05BatchedClient(
            host=args.pi05_host, port=args.pi05_port, num_envs=n, open_loop_horizon=args.open_loop_horizon)
        set_arm_control_mode("position" if args.dummy_policy else args.control_mode)
        traj = [[] for _ in range(n)]
        writer = None
        if args.record_video:
            from full_eval import _Mp4Writer
            writer = _Mp4Writer(out_dir / "video.mp4", FPS // max(1, args.video_every))

        def video_panels():
            # per-env [third-person | wrist] from the HIGH-RES video cams (not the policy obs cams).
            scene = env.unwrapped.scene
            ext = scene["video_ext"].data.output["rgb"][..., :3].detach().cpu().numpy()
            wri = scene["video_wrist"].data.output["rgb"][..., :3].detach().cpu().numpy()
            return [np.clip(np.concatenate([ext[i], wri[i]], axis=1), 0, 255).astype(np.uint8) for i in range(n)]

        effective_mode = "position" if args.dummy_policy else args.control_mode

        def hold_all():
            # (N,8) hold: zero arm velocity (velocity mode) or the current joint pos (position mode).
            grip = obs["policy"]["gripper_pos"].detach().cpu().numpy().reshape(n, 1)
            arm = (np.zeros((n, 7), np.float32) if effective_mode == "velocity"
                   else obs["policy"]["arm_joint_pos"].detach().cpu().numpy())
            return np.concatenate([arm, grip], axis=1).astype(np.float32)

        if fast_render:
            # Align the render phase: a render fires when _sim_step_counter hits a multiple of
            # render_interval (= decimation * open_loop_horizon), so zeroing it here makes a render land on
            # the env.step just before each query step -> query-step images stay fresh (verified).
            env.unwrapped._sim_step_counter = 0

        held = False   # once a batched query errors, hold the arm for the rest of the rollout
        from tqdm import tqdm
        for step in tqdm(range(max_steps), desc=f"scene{args.scene}"):
            if args.dummy_policy or held:
                actions = hold_all()
            else:
                try:
                    actions = client.infer(obs, spec.instruction)   # (N, 8)
                except Exception as e:  # noqa: BLE001 - a server error ends the rollout
                    logger.warning(f"batched infer failed ({e}); holding for the rest of the rollout")
                    held, actions = True, hold_all()
                if step == 0 and fast_render and client.chunk_len and client.chunk_len < args.open_loop_horizon:
                    logger.warning(f"pi05 chunk length {client.chunk_len} < open_loop_horizon "
                                   f"{args.open_loop_horizon}: policy re-queries faster than the render cadence "
                                   f"-> some query images will be stale. Re-run with --open-loop-horizon "
                                   f"{client.chunk_len}.")
            if step % args.track_every == 0:
                for i, s in enumerate(read_states(env, rigid, art)):
                    traj[i].append(s)
            if writer is not None and step % args.video_every == 0:
                writer.add(fit_width(tile_grid(video_panels()), args.video_max_width))
            obs, _, _, _, _ = env.step(torch.as_tensor(actions, dtype=torch.float32, device=env.unwrapped.device))
        if writer is not None:
            writer.close()
        if client is not None:
            client.close()

        # settle_sim holds by commanding measured joint POSITIONS, so it must run in position mode;
        # in velocity mode those positions would be read as velocities and flail the arm, corrupting
        # the final object poses. Switch back before the post-rollout settle.
        set_arm_control_mode("position")
        obs = settle_sim(env, obs, steps=args.post_settle_steps, reset_episode_buf=False)
        final = read_states(env, rigid, art)

        # --- score every env (on-table gates are multiplicative; ordered criteria credited in sequence) ---
        from scene_specs import resolve_gates, score_env

        per_env = [spec.evaluate(init[i], final[i], traj[i], spec) for i in range(n)]
        criteria = list(per_env[0].keys())
        scored = [score_env(e, spec) for e in per_env]
        dense = float(np.mean([s["dense"] for s in scored]))
        sparse = float(np.mean([s["sparse"] for s in scored]))
        raw_rates = {c: float(np.mean([e[c] for e in per_env])) for c in criteria}          # did it physically happen
        credited_rates = {c: float(np.mean([s["credited"][c] for s in scored])) for c in criteria}  # counted after ordering
        results = {
            "scene": args.scene, "instruction": spec.instruction, "num_envs": n,
            "criteria": criteria, "gates": resolve_gates(spec, criteria), "sequence": spec.sequence,
            "per_env": per_env,
            "dense_score": dense, "sparse_score": sparse,
            "criterion_rates": raw_rates, "credited_rates": credited_rates,
        }
        logger.info(f"scene {args.scene}: dense={dense:.3f} sparse={sparse:.3f} "
                    f"raw={raw_rates} credited={credited_rates}")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    logger.info(f"wrote {out_dir / 'results.json'}")
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
