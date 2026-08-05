#!/usr/bin/env python3
"""Execute a TIPTOP-PRODUCED dual-arm/handover plan (bimanual_yam_dual) in Isaac Sim and record a video.

Sim-validation counterpart to ``tools/replay_dual_plan.py``, adapted to consume a
PERCEPTION-GROUNDED plan produced by tiptop's own pipeline (``tiptop.planning.serialize_plan``,
schema >= 1.4.0) instead of ``cutamp.scripts.dual_arm_demo``'s disconnected, hand-built scene. The
step dispatch (trajectory resampling, gripper hold/settle) is UNCHANGED from ``replay_dual_plan.py``
-- the two schemas agree on everything replay actually reads (trajectory ``positions``/``dt``,
gripper ``action``/``arm``/``arms``/``label``). What differs is the loader (tiptop's plan has no
``arm_mode``/``objects`` keys -- ``q_init`` width is the dual-plan signal instead) and an added,
FAILING check for the one invariant a real handover cannot survive without: a taker-close must be
immediately followed, with no trajectory step between, by the giver's open (see
``cuTAMP/cutamp/motion_solver.py``'s ``Handover`` branch).

Usage, as the mandatory sim-validation gate before any real-hardware handover attempt (see the
project plan): drive tiptop's dual/handover pipeline end to end via a websocket server against
``tiptop_yam_dual.yml`` and ``eval/yam_tiptop_eval.py --handover`` (which writes ``tiptop_plan.json``
next to its run outputs using the SAME ``serialize_plan`` tiptop's real-hardware path uses), then
replay that exact file here:

    # tiptop env, in one terminal
    cd ../tiptop && TIPTOP_CONFIG=tiptop_yam_dual.yml \
        pixi run python -m tiptop.tiptop_websocket_server --port 8767

    # this repo's Isaac venv, in another
    .venv/bin/python eval/yam_tiptop_eval.py --handover --port 8767
    .venv/bin/python tools/replay_tiptop_dual_plan.py \
        --plan tiptop_server_outputs/<timestamp>/tiptop_plan.json --scene 15
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
SETTLE_STEPS = 12  # control steps to hold the final waypoint at the end of each trajectory
DUAL_DOF = 12


def _step_arms(step: dict) -> list[str]:
    """Which hand(s) a gripper step names (``arms`` plural, else ``arm`` singular, else none)."""
    arms = step.get("arms")
    if arms:
        return list(arms)
    return [step["arm"]] if step.get("arm") else []


def _assert_handover_ordering(steps: list[dict], plan_path: Path) -> None:
    """Fail loudly unless at least one taker-close is immediately followed by the giver's open.

    This is the automated ordering assertion the project's sim-validation safety gate requires
    (see the plan doc): every ``Handover`` in ``handover_tamp_operators`` emits its close and open
    as two adjacent gripper steps with no trajectory in between (``motion_solver.py:791,796``) --
    that shape is unique to the handover exchange in this domain (a plain ``PickGiver`` close is
    always followed by a trajectory, ``MoveHoldingGiver``, never directly by another gripper step),
    so "adjacent single-arm close then single-arm open on the OTHER arm" is a precise detector, not
    a heuristic. Raises if the plan does not contain at least one such pair, which would mean either
    this isn't actually a handover plan or the ordering invariant was violated upstream -- either
    way, this must not silently pass the gate.
    """
    pairs = []
    for i, st in enumerate(steps):
        if st.get("type") != "gripper" or st.get("action") != "close":
            continue
        arms = _step_arms(st)
        if len(arms) != 1:
            continue  # a simultaneous close (PickBoth) has no handover ordering to check
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        if nxt is None or nxt.get("type") != "gripper" or nxt.get("action") != "open":
            continue
        nxt_arms = _step_arms(nxt)
        if len(nxt_arms) == 1 and nxt_arms != arms:
            pairs.append((i, arms[0], nxt_arms[0]))

    if not pairs:
        raise ValueError(
            f"{plan_path}: no taker-close -> giver-open pair found adjacent anywhere in the plan. "
            "Either this is not a handover plan, or the safety-critical close/open ordering "
            "invariant was violated upstream (goal building or serialization) -- refusing to "
            "replay. See tiptop.execute_plan.execute_cutamp_dual_plan's hard-sequencing and "
            "tiptop.planning.serialize_plan's arm/arms passthrough."
        )
    for i, taker, giver in pairs:
        logger.info(f"Handover ordering OK: step {i} close({taker}) -> immediately open({giver})")


def main(
    plan: Path,
    scene: int = 15,
    variant: int = 0,
    headless: bool = True,
    camera_height: int = 720,
    camera_width: int = 1280,
    episode_length_s: float = 240.0,
):
    """Replay a tiptop dual/handover plan JSON in Isaac and write an MP4.

    Args:
        plan: ``tiptop_plan.json`` written by ``tiptop.planning.serialize_plan`` (schema >= 1.4.0)
            during a ``bimanual_yam_dual`` session (real websocket run or otherwise).
        scene: Scene id whose sidecar matches the plan's objects (15 = the handover demo scene).
        variant: Scene variant.
        headless: If True, no Isaac viewport; only the video file is written.
        camera_height / camera_width: Render size for all cameras.
        episode_length_s: Sim-time budget; a handover plan is several long joint-space segments.
    """
    spec = json.loads(plan.read_text())
    steps = spec["steps"]
    q_init = spec.get("q_init") or []
    if len(q_init) != DUAL_DOF:
        raise ValueError(
            f"{plan} does not look like a bimanual_yam_dual plan: q_init has {len(q_init)} entries, "
            f"expected {DUAL_DOF}. Was this produced against a `robot.arm_mode: dual` config?"
        )
    logger.info(f"Loaded {len(steps)} plan steps from {plan}")
    _assert_handover_ordering(steps, plan)

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="tiptop dual-arm plan replay")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    simulation_app = AppLauncher(args_cli).app

    import gymnasium as gym

    import src.sim_evals.environments  # noqa: F401  (registers the "YAM" gym id)
    from isaaclab_tasks.utils import parse_env_cfg
    from src.sim_evals.environments.bimanual_yam import set_contact_friction
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
            step_arms = _step_arms(st) or list(plan_slices)
            for arm in step_arms:
                grip[arm] = 1.0 if st["action"] == "close" else 0.0
            label = f"{st['action']} {' + '.join(step_arms)} gripper"
            logger.info(f"  {label}")
            for _ in range(GRIPPER_HOLD_STEPS):
                if step(current_targets(), label):
                    break

    out_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tiptop_dual_scene{scene}_{variant}.mp4"
    mediapy.write_video(path, video, fps=VIDEO_FPS)
    logger.info(f"Saved video to {path} ({len(video)} frames)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
