#!/usr/bin/env python3
"""Execute a SIMULTANEOUS dual-arm cuTAMP plan in Isaac Sim and record a video.

cuTAMP and Isaac live in different Python environments, so the plan is handed over as JSON:

    # tiptop env (cuRobo + cuTAMP)
    cd ../tiptop && pixi run python -m cutamp.scripts.dual_arm_demo \
        --arm-mode dual --save-plan /tmp/dual_plan.json

    # this repo's Isaac venv
    .venv/bin/python tools/replay_dual_plan.py --plan /tmp/dual_plan.json

The plan's trajectory steps carry FULL 12-DOF configurations (left arm's six joints then the right
arm's), and its gripper steps carry an ``"arm"`` -- which is the whole point: at the grasp
configuration a ``close`` is emitted for each hand, so both jaws shut on their own object at the same
instant rather than one arm waiting for the other.

Scene 14 mirrors ``cutamp/scripts/dual_arm_demo.py``'s environment exactly (two cubes, two trays, one
pair per side), so what executes here is the same problem cuTAMP solved.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

import cv2
import mediapy
import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visual_utils import add_top_padding, overlay_timer_ms

VIDEO_FPS = 15  # the env's control rate (decimation 8 x sim.dt 1/120)
GRIPPER_HOLD_STEPS = 20  # control steps to hold while a jaw travels; the finger needs ~0.5 s
# Control steps to hold the final waypoint at the end of each trajectory. The plan is resampled from
# cuRobo's ~50 Hz down to the env's 15 Hz, and the stiff position controller lags a fast 12-DOF
# joint-space move, so without this the jaws start closing while the arms are still converging on the
# grasp configuration and the objects squirt out.
SETTLE_STEPS = 12


def main(
    plan: Path,
    scene: int = 14,
    variant: int = 0,
    headless: bool = True,
    camera_height: int = 720,
    camera_width: int = 1280,
    episode_length_s: float = 240.0,
):
    """Replay a dual-arm plan JSON in Isaac and write an MP4.

    Args:
        plan: JSON written by ``cutamp.scripts.dual_arm_demo --save-plan``.
        scene: Scene id whose sidecar matches the plan's objects (14 = the dual-arm demo scene).
        variant: Scene variant.
        headless: If True, no Isaac viewport; only the video file is written.
        camera_height: Render height for all cameras.
        camera_width: Render width for all cameras.
        episode_length_s: Sim-time budget; the plan is several long joint-space segments.
    """
    spec = json.loads(plan.read_text())
    if spec.get("arm_mode") != "dual":
        raise ValueError(f"{plan} is not a dual-arm plan (arm_mode={spec.get('arm_mode')!r})")
    steps = spec["steps"]
    logger.info(f"Loaded {len(steps)} plan steps from {plan}")

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Dual-arm plan replay")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    simulation_app = AppLauncher(args_cli).app

    import gymnasium as gym

    import src.sim_evals.environments  # noqa: F401  (registers the "YAM" gym id)
    from isaaclab_tasks.utils import parse_env_cfg
    from src.sim_evals.environments.bimanual_yam import ARM_JOINTS, set_contact_friction
    from src.sim_evals.environments.droid_environment import collapse_dome_lights
    from src.sim_evals.environments.yam_environment import (
        ACTION_SLICES,
        hold_action,
        set_camera_resolution,
        settle_sim,
        verify_action_mapping,
    )

    env_cfg = parse_env_cfg("YAM", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.set_scene(str(scene), variant)
    set_camera_resolution(env_cfg, camera_height, camera_width)
    set_contact_friction(env_cfg)
    env_cfg.episode_length_s = episode_length_s
    env = gym.make("YAM", cfg=env_cfg)
    collapse_dome_lights()

    obs, _ = env.reset()
    obs, _ = env.reset()
    obs = settle_sim(env, obs, reset_episode_buf=True)

    # Isaac orders its DOFs breadth-first, interleaving the two arms, while a plan row is
    # [left_joint1..6, right_joint1..6]. Name resolution inside each action term undoes that, but the
    # rest posture is symmetric enough to hide a swap, so assert the mapping rather than trust it.
    verify_action_mapping(env)

    device = env.unwrapped.device
    # cuTAMP's configuration order is [left arm x6, right arm x6]; ACTION_SLICES maps each arm's
    # 7 numbers (6 joints + 1 gripper) into the env's 14-wide action vector.
    plan_slices = {"left": slice(0, 6), "right": slice(6, 12)}
    grip = {"left": 0.0, "right": 0.0}  # 0 = open, 1 = closed
    video: list[np.ndarray] = []
    frame = 0

    def step(arm_positions: dict, label: str):
        """One control step: drive each arm to its target and hold both grippers where they are."""
        nonlocal obs, frame
        action = hold_action(obs)[0].clone()
        for arm, q in arm_positions.items():
            sl = ACTION_SLICES[arm]
            action[sl.start:sl.start + 6] = torch.as_tensor(q, dtype=action.dtype, device=device)
            action[sl.start + 6] = grip[arm]
        obs, _, term, trunc, _ = env.step(action[None])

        pol = obs["policy"]
        viz = np.concatenate([pol["external_cam"][0].cpu().numpy().astype(np.uint8),
                              pol["top_cam"][0].cpu().numpy().astype(np.uint8)], axis=1)
        viz = add_top_padding(viz, pad_px=40)
        overlay_timer_ms(viz, int(frame * 1000 / VIDEO_FPS))
        cv2.putText(viz, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        video.append(viz)
        frame += 1
        return term or trunc

    def current_targets() -> dict:
        pol = obs["policy"]
        return {arm: pol[f"{arm}_arm_joint_pos"][0].cpu().numpy() for arm in ("left", "right")}

    for st in steps:
        if st["type"] == "trajectory":
            positions = np.asarray(st["positions"], dtype=np.float32)
            # cuRobo interpolates at ~50 Hz; the env runs at 15 Hz. Keep every stride-th waypoint and
            # always the last one, so the arms actually arrive at the planned configuration.
            stride = max(1, int(round((1.0 / VIDEO_FPS) / float(st["dt"]))))
            idxs = list(range(0, len(positions), stride))
            if idxs[-1] != len(positions) - 1:
                idxs.append(len(positions) - 1)
            label = st["label"].split("(")[0]
            for i in idxs:
                if step({a: positions[i][plan_slices[a]] for a in ("left", "right")}, label):
                    break
            final = {a: positions[-1][plan_slices[a]] for a in ("left", "right")}
            for _ in range(SETTLE_STEPS):
                if step(final, label):
                    break
        else:
            # A gripper step names every hand that acts at that instant, so a lockstep PickBoth
            # actuates both jaws within ONE control step and holds once. Emitting a step per hand
            # instead would look like -- and be indistinguishable from -- a deliberate sequence, which
            # is what a handover genuinely needs (taker closes, THEN giver opens).
            step_arms = st.get("arms") or [st["arm"]]
            for arm in step_arms:
                grip[arm] = 1.0 if st["action"] == "close" else 0.0
            label = f"{st['action']} {' + '.join(step_arms)} gripper"
            logger.info(f"  {label}")
            for _ in range(GRIPPER_HOLD_STEPS):
                if step(current_targets(), label):
                    break

    out_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"yam_{spec.get('dual_task', 'parallel')}_scene{scene}_{variant}.mp4"
    mediapy.write_video(path, video, fps=VIDEO_FPS)
    logger.info(f"Saved video to {path} ({len(video)} frames)")

    # The plan names its own objects, so the same driver replays the parallel scene and the handover
    # scene without knowing which is which.
    for name in spec.get("objects", {}):
        if name == "table":
            continue
        try:
            pos = env.unwrapped.scene[name].data.root_pos_w[0].cpu().numpy()
        except KeyError:
            continue
        logger.info(f"  final {name:8s} {np.round(pos, 4)}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
