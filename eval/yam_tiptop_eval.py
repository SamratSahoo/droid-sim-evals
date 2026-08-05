"""Run TiPToP on the bimanual YAM in Isaac Sim and record a video.

The YAM counterpart of ``tiptop_eval.py``. Same shape -- connect to a running TiPToP websocket
server, send one observation, get a plan back, execute it -- with two differences that matter:

* **Perception runs off the FIXED third-person camera** on the robot's base column (the MolmoAct2
  ``top_cam``: RealSense D435i optics, 69.4 deg HFOV, 0.80 m above the base looking 80 deg down), not
  a wrist camera. That is the camera whose RGB + depth + intrinsics + extrinsics go on the wire, and
  it is the right pane of every recorded video.

* **Both arms can be used.** cuTAMP plans ONE kinematic chain, so a single plan drives a single arm
  (the other is locked in its cuRobo config and collision-checked as an obstacle). ``--bimanual``
  therefore runs the arms in sequence: split the objects between them, ask each arm's server for a
  plan covering its own objects, and execute those plans back to back in one continuous rollout.
  This is sequential bimanual, not simultaneous dual-arm TAMP -- see ``_split_by_arm``.

* **``--handover`` is the third mode: SIMULTANEOUS dual-arm TAMP.** One server, configured
  ``robot.type: bimanual_yam_dual`` / ``arm_mode: dual`` (``tiptop_yam_dual.yml``), perceives and
  plans the whole scene as one 12-DOF chain -- there is no per-arm split, and the plan itself may
  hand an object between the hands (``cuTAMP``'s ``Handover`` operator). This is the
  perception-grounded counterpart to ``cuTAMP/cutamp/scripts/dual_arm_demo.py --task handover``,
  and the plan it produces is what ``tools/replay_tiptop_dual_plan.py`` validates offline (see that
  file's docstring for the mandatory sim-validation gate this feeds).

Usage -- single arm on scene 6 variant 0::

    # M2T2 grasp server (TiPToP needs it; FoundationStereo is unused -- the sim renders true depth)
    (cd ../M2T2 && pixi run python server.py --port 8123 &)
    # TiPToP planning server for the left arm
    (cd ../tiptop && TIPTOP_CONFIG=tiptop_yam.yml pixi run python -m tiptop.tiptop_websocket_server --port 8765 &)
    .venv/bin/python eval/yam_tiptop_eval.py

Usage -- bimanual on scene 6 variant 1, where the toys are spread too wide for one arm::

    (cd ../tiptop && TIPTOP_CONFIG=tiptop_yam.yml       pixi run python -m tiptop.tiptop_websocket_server --port 8765 &)
    (cd ../tiptop && TIPTOP_CONFIG=tiptop_yam_right.yml pixi run python -m tiptop.tiptop_websocket_server --port 8766 &)
    .venv/bin/python eval/yam_tiptop_eval.py --bimanual --variant 1

Usage -- handover on scene 15 (bar on one side, tray on the other, out of that arm's reach)::

    (cd ../tiptop && TIPTOP_CONFIG=tiptop_yam_dual.yml pixi run python -m tiptop.tiptop_websocket_server --port 8767 &)
    .venv/bin/python eval/yam_tiptop_eval.py --handover --scene 15 --ws-port-dual 8767

Needs ``GEMINI_API_KEY`` in the *server* process's environment. Videos land in ``runs/<date>/<time>/``.
The server writes its own ``tiptop_plan.json`` under ``tiptop_server_outputs/<timestamp>/`` for every
query (see ``tiptop_websocket_server.py::_run_pipeline``) -- that is the file
``tools/replay_tiptop_dual_plan.py`` and the sim-validation gate consume, not anything this script
saves separately.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

import cv2
import imageio.v3 as iio
import mediapy
import msgpack_numpy
import numpy as np
import torch
import tyro
import websockets.sync.client
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path for `src`
from eval.scene_specs import get_scene
from src.sim_evals.inference.yam_tiptop_websocket import YamTiptopWebsocketClient
from src.visual_utils import add_top_padding, depth_to_turbo, overlay_timer_ms

VIDEO_FPS = 15  # the env's control rate: decimation 8 x sim.dt 1/120

# --handover step timing (matches tools/replay_tiptop_dual_plan.py -- both drive the SAME kind of
# tiptop dual/handover plan against this env, and must not diverge on gripper hold/settle timing).
GRIPPER_HOLD_STEPS = 20  # control steps to hold while a jaw travels; the finger needs ~0.5 s
SETTLE_STEPS = 12  # control steps to hold the final waypoint at the end of each trajectory

# How each tracked object is referred to in a natural-language sub-task. Only needed when splitting a
# scene between the arms: each arm's server gets an instruction naming just its own objects, and the
# phrasing has to be something the VLM will ground to the same object it detects in the image (colour
# words are what it reliably keys on -- across runs it has named these "blue_toy" and
# "blue_elephant_toy", but always with the colour).
OBJECT_PHRASE = {
    "blue_toy": "blue toy", "brown_toy": "brown toy", "pink_toy": "pink toy",
    "block_a": "first block", "block_b": "second block", "block_c": "third block",
}


def _split_by_arm(obj_positions: dict[str, np.ndarray], base_y: float) -> dict[str, list[str]]:
    """Assign each object to the arm on its side of the robot's midline.

    This is a HARNESS-level split, not part of TAMP: cuTAMP plans one chain at a time and has no
    operator for choosing an arm, so something outside it has to decide who picks what. The rule is
    the geometry the reachability sweep measured -- with the base resting on the table at
    (0.24, 0, 0.0551) the left arm (mounted at +0.24 in y) covers y >= -0.17 and the right covers
    y <= +0.17, so splitting at the midline gives each arm objects it can definitely reach and puts
    the ambiguous middle ones on whichever side they lean. Object poses come from the simulator,
    which the harness already reads for scoring; the planner still does all the perception, grasp
    selection and motion planning itself.
    """
    assignment: dict[str, list[str]] = {"left": [], "right": []}
    for name, pos in obj_positions.items():
        assignment["left" if float(pos[1]) >= base_y else "right"].append(name)
    return assignment


def _instruction_for(objects: list[str], target_phrase: str) -> str:
    phrases = [OBJECT_PHRASE.get(name, name.replace("_", " ")) for name in objects]
    if len(phrases) == 1:
        listed = phrases[0]
    else:
        listed = ", ".join(phrases[:-1]) + f" and {phrases[-1]}"
    return f"Place the {listed} on the {target_phrase}"


def _dual_step_arms(step: dict) -> list[str]:
    """Which hand(s) a gripper step names (``arms`` plural, else ``arm`` singular, else both)."""
    arms = step.get("arms")
    if arms:
        return list(arms)
    return [step["arm"]] if step.get("arm") else ["left", "right"]


def _query_dual_server(host: str, port: int, obs: dict, instruction: str) -> dict:
    """One request/response against a ``bimanual_yam_dual`` tiptop server: raw wire protocol.

    Unlike :class:`YamTiptopWebsocketClient`, there is no per-tick client-side plan stepping here --
    the caller (``main``'s ``--handover`` branch) drives the returned plan directly against the live
    env, with step logic mirroring ``tools/replay_tiptop_dual_plan.py``'s so the two paths cannot
    silently diverge on how a handover's gripper ordering is driven. ``q_init`` is the FULL 12-DOF
    configuration (left arm's 6 joints, then right's), matching cuTAMP's own ``bimanual_yam_dual``
    joint-column convention.
    """
    policy = obs["policy"]
    q_init = np.concatenate([
        policy["left_arm_joint_pos"][0].cpu().numpy(),
        policy["right_arm_joint_pos"][0].cpu().numpy(),
    ]).astype(np.float32)
    rgb = policy["top_cam"][0].cpu().numpy().astype(np.uint8)
    depth = np.nan_to_num(policy["top_depth"][0].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    intrinsics = policy["top_intrinsics"][0].cpu().numpy().astype(np.float32)

    from scipy.spatial.transform import Rotation

    # Isaac reports quat_w (wxyz); TiPToP's extrinsic wants a plain xyz+3x3 rotation matrix. Same
    # ROS-optical-frame convention YamTiptopWebsocketClient already relies on -- no frame nudge
    # needed (the YAM's fingers are prismatic, unlike the Robotiq tool_from_ee offset).
    pos = policy["top_cam_pos_w"][0].cpu().numpy()
    quat_xyzw = policy["top_cam_quat_w"][0].cpu().numpy()[[1, 2, 3, 0]]
    world_from_cam = np.eye(4, dtype=np.float32)
    world_from_cam[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    world_from_cam[:3, 3] = pos

    request = {
        "rgb": rgb,
        "depth": depth.astype(np.float32),
        "intrinsics": intrinsics,
        "world_from_cam": world_from_cam,
        "task": instruction,
        "q_init": q_init,
    }
    uri = f"ws://{host}:{port}"
    logger.info(f"Connecting to the dual-arm tiptop server at {uri}...")
    ws = websockets.sync.client.connect(uri, compression=None, max_size=None)
    try:
        ws.recv()  # server metadata, unused here
        packer = msgpack_numpy.Packer()
        ws.send(packer.pack(request))
        logger.info(f"Querying tiptop server for handover task: '{instruction}' (this may take a while)...")
        start = time.time()
        response = json.loads(ws.recv())
        logger.info(f"Got response in {time.time() - start:.1f}s")
    finally:
        ws.close()

    if not response["success"]:
        raise RuntimeError(f"Dual-arm tiptop server returned an error: {response.get('error', 'unknown')}")
    steps = response["plan"]["steps"]
    for step in steps:
        if step["type"] == "trajectory":
            step["positions"] = np.array(step["positions"], dtype=np.float32)
    return steps


def main(
    instruction: str | None = None,
    episodes: int = 1,
    headless: bool = True,
    scene: int = 6,
    variant: int = 0,
    bimanual: bool = False,
    handover: bool = False,
    ws_host: str = "localhost",
    ws_port: int = 8765,
    ws_port_right: int = 8766,
    ws_port_dual: int = 8767,
    camera_height: int = 720,
    camera_width: int = 1280,
    episode_length_s: float = 180.0,
):
    """Plan with TiPToP and execute on the bimanual YAM, saving one video per episode.

    Args:
        instruction: Natural-language task. Defaults to the scene's own instruction from
            ``eval/scene_specs.py`` (ignored in ``--bimanual`` mode, which builds one sub-task per
            arm instead). In ``--handover`` mode there is no ``SceneSpec`` to default from, so this
            is used directly, defaulting to ``"Place the bar on the tray"`` if unset.
        episodes: Number of episodes to run.
        headless: If True (default), no Isaac viewport; only the video file is written.
        scene: Scene number. 6 (toys on a plate) is what the YAM mount pose was tuned for; 15 is the
            handover demo scene (a bar on one side, a tray on the other, out of that arm's reach).
        variant: Scene variant. 0 is the stock layout (reachable by the left arm alone); 1 spreads the
            toys too wide for either arm on its own and is the layout to use with ``--bimanual``.
        bimanual: Split the objects between the two arms and run one TiPToP plan per arm, in
            sequence, within a single rollout. Requires a second server for the right arm.
            Mutually exclusive with ``handover``.
        handover: Run ONE simultaneous dual-arm plan against a ``bimanual_yam_dual`` server (no
            per-arm split -- see the module docstring). Mutually exclusive with ``bimanual``.
        ws_host: TiPToP websocket server host.
        ws_port: Port of the LEFT arm's server (``robot.type: bimanual_yam_left``).
        ws_port_right: Port of the RIGHT arm's server, used only with ``--bimanual``.
        ws_port_dual: Port of the dual-arm server (``robot.type: bimanual_yam_dual``), used only
            with ``--handover``.
        camera_height: Render height for all cameras. The third-person camera feeds TiPToP's point
            cloud, so the default is the full sensor resolution rather than the 180x320 the data-gen
            path uses.
        camera_width: Render width for all cameras.
        episode_length_s: Sim-time budget per episode. A bimanual rollout runs two full plans, so
            this is generous.
    """
    if bimanual and handover:
        raise ValueError("--bimanual and --handover are mutually exclusive (sequential vs simultaneous dual-arm)")

    # Scene 15 (the handover demo scene, assets/scene15_0.json) has no eval/scene_specs.py entry --
    # it carries no per-scene scoring/instruction metadata the way scenes 6/9/12 etc. do, since the
    # goal here comes from the natural-language instruction (grounded by perception), not from a
    # SceneSpec's tracked object list. Skip the registry lookup entirely in handover mode.
    spec = None if handover else get_scene(scene)
    if instruction is None:
        # Matches build_handover_env()'s bar/tray naming in cutamp/scripts/dual_arm_demo.py, so the
        # default instruction actually grounds to the objects the scene sidecar places.
        instruction = "Place the bar on the tray" if handover else spec.instruction

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="TiPToP bimanual-YAM evaluation")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import src.sim_evals.environments  # noqa: F401  (registers the "YAM" gym id)
    from isaaclab_tasks.utils import parse_env_cfg
    from src.sim_evals.environments.bimanual_yam import MOUNT_XYZ, set_contact_friction
    from src.sim_evals.environments.droid_environment import collapse_dome_lights
    from src.sim_evals.environments.yam_environment import (
        ACTION_SLICES,
        hold_action,
        set_camera_resolution,
        settle_sim,
    )

    env_cfg = parse_env_cfg("YAM", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.set_scene(str(scene), variant)
    set_camera_resolution(env_cfg, camera_height, camera_width)
    set_contact_friction(env_cfg)
    env_cfg.episode_length_s = episode_length_s
    env = gym.make("YAM", cfg=env_cfg)
    # The scene USD embeds a DomeLight that InteractiveScene clones per env; with one env it is
    # merely redundant with the config-time global dome, but zeroing it keeps exposure identical to
    # every other runner in this repo.
    collapse_dome_lights()

    obs, _ = env.reset()
    obs, _ = env.reset()  # second render cycle so materials are loaded before the planner sees them

    arms = ["left", "right"] if bimanual else ["left"]
    clients = {}
    if not handover:
        for arm in arms:
            port = ws_port if arm == "left" else ws_port_right
            logger.info(f"Connecting the {arm} arm to ws://{ws_host}:{port}...")
            clients[arm] = YamTiptopWebsocketClient(host=ws_host, port=port, arm=arm)

    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    max_steps = env.env.max_episode_length
    saved: list[Path] = []

    def save_perception_frame(obs: dict, tag: str) -> None:
        """Write the exact third-person RGB + depth that is about to go to the planner.

        The depth is the same array the request carries (metres, 0 = invalid), rendered with turbo
        so it can be eyeballed against the RGB. The server keeps its own copy under
        tiptop_server_outputs/, but having it beside the video means a rollout is self-contained.
        """
        p = obs["policy"]
        rgb = p["top_cam"][0].cpu().numpy().astype(np.uint8)
        depth = p["top_depth"][0].cpu().numpy()
        iio.imwrite(video_dir / f"{tag}_top_rgb.png", rgb)
        iio.imwrite(video_dir / f"{tag}_top_depth_turbo.png", depth_to_turbo(depth))
        d = depth[np.isfinite(depth) & (depth > 0)]
        logger.info(f"{tag}: saved third-person rgb + turbo depth "
                    f"({d.size}/{depth.size} valid, {d.min():.3f}-{d.max():.3f} m)")

    def object_positions() -> dict[str, np.ndarray]:
        return {name: env.unwrapped.scene[name].data.root_pos_w[0].cpu().numpy() for name in spec.rigid}

    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset()
            obs = settle_sim(env, obs, reset_episode_buf=True)
            video: list[np.ndarray] = []
            frame_idx = 0

            if handover:
                # SIMULTANEOUS dual-arm: one query, one plan, driven directly against the live env.
                # The step logic below (trajectory resampling, gripper hold/settle, independent
                # per-arm gripper channels) mirrors tools/replay_tiptop_dual_plan.py's -- both drive
                # the same kind of plan against this same env and must not diverge on it, especially
                # not on a handover's taker-close/giver-open ordering.
                save_perception_frame(obs, f"ep{ep}_dual")
                try:
                    steps = _query_dual_server(ws_host, ws_port_dual, obs, instruction)
                except Exception as e:
                    logger.error(f"handover query failed: {e}")
                    continue

                plan_slices = {"left": slice(0, 6), "right": slice(6, 12)}
                grip = {"left": 0.0, "right": 0.0}
                terminated = False

                def dual_step(arm_positions: dict, label: str) -> bool:
                    nonlocal obs, frame_idx, terminated
                    action = hold_action(obs)[0].clone()
                    for arm, q in arm_positions.items():
                        sl = ACTION_SLICES[arm]
                        action[sl.start:sl.start + 6] = torch.as_tensor(
                            q, dtype=action.dtype, device=action.device)
                        action[sl.start + 6] = grip[arm]
                    obs, _, term, trunc, _ = env.step(action[None])
                    pol = obs["policy"]
                    viz = np.concatenate([pol["external_cam"][0].cpu().numpy().astype(np.uint8),
                                          pol["top_cam"][0].cpu().numpy().astype(np.uint8)], axis=1)
                    viz = add_top_padding(viz, pad_px=40)
                    overlay_timer_ms(viz, int(frame_idx * 1000 / VIDEO_FPS))
                    cv2.putText(viz, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    video.append(viz)
                    frame_idx += 1
                    terminated = bool(term or trunc)
                    return terminated

                def dual_current_targets() -> dict:
                    pol = obs["policy"]
                    return {arm: pol[f"{arm}_arm_joint_pos"][0].cpu().numpy() for arm in ("left", "right")}

                for st in steps:
                    if terminated or frame_idx >= max_steps:
                        break
                    if st["type"] == "trajectory":
                        positions = st["positions"]
                        stride = max(1, int(round((1.0 / VIDEO_FPS) / float(st["dt"]))))
                        idxs = list(range(0, len(positions), stride))
                        if idxs[-1] != len(positions) - 1:
                            idxs.append(len(positions) - 1)
                        label = st["label"].split("(")[0]
                        for i in idxs:
                            if dual_step({a: positions[i][plan_slices[a]] for a in ("left", "right")}, label):
                                break
                        if not terminated:
                            final_q = {a: positions[-1][plan_slices[a]] for a in ("left", "right")}
                            for _ in range(SETTLE_STEPS):
                                if dual_step(final_q, label):
                                    break
                    else:
                        step_arms = _dual_step_arms(st)
                        for arm in step_arms:
                            grip[arm] = 1.0 if st["action"] == "close" else 0.0
                        label = f"{st['action']} {' + '.join(step_arms)} gripper"
                        logger.info(f"  {label}")
                        for _ in range(GRIPPER_HOLD_STEPS):
                            if dual_step(dual_current_targets(), label):
                                break

                if not video:
                    continue
                path = video_dir / f"yam_tiptop_handover_scene{scene}_{variant}_ep{ep}.mp4"
                mediapy.write_video(path, video, fps=VIDEO_FPS)
                saved.append(path)
                logger.info(f"Saved video to {path}")
                for name in ("bar", "tray"):  # assets/scene15_0.json's tracked objects
                    try:
                        pos = env.unwrapped.scene[name].data.root_pos_w[0].cpu().numpy()
                        logger.info(f"  final {name:8s} {np.round(pos, 4)}")
                    except KeyError:
                        pass
                continue

            if bimanual:
                positions = object_positions()
                target = spec.meta["plate"]
                items = {n: p for n, p in positions.items() if n != target}
                split = _split_by_arm(items, base_y=MOUNT_XYZ[1])
                tasks = [(arm, _instruction_for(split[arm], OBJECT_PHRASE.get(target, target.replace("_", " "))))
                         for arm in arms if split[arm]]
                for arm, sub in tasks:
                    logger.info(f"  {arm} arm  <- {sub!r}   (objects: {', '.join(split[arm])})")
            else:
                tasks = [("left", instruction)]

            for arm, sub_instruction in tasks:
                client = clients[arm]
                save_perception_frame(obs, f"ep{ep}_{arm}")
                try:
                    while frame_idx < max_steps:
                        ret = client.infer(obs, sub_instruction)
                        if client.plan_done:
                            logger.info(f"{arm} arm: plan fully executed at step {frame_idx}")
                            break

                        # Left pane: the side view. Right pane: the third-person camera TiPToP
                        # planned from -- the same pixels that went over the wire.
                        viz = np.concatenate([ret["right_image"], ret["wrist_image"]], axis=1)
                        viz = add_top_padding(viz, pad_px=40)
                        overlay_timer_ms(viz, int(frame_idx * 1000 / VIDEO_FPS))
                        cv2.putText(viz, f"{arm} arm", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (255, 255, 255), 2, cv2.LINE_AA)
                        if not headless:
                            try:
                                cv2.imshow("Camera View", cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                                cv2.waitKey(1)
                            except cv2.error:
                                pass
                        video.append(viz)
                        frame_idx += 1

                        # This arm's 7 numbers go in its slot; the other arm holds where it is, so a
                        # handoff never jerks the parked arm (and it stays where cuRobo's lock_joints
                        # believes it is).
                        action = hold_action(obs)[0].clone()
                        action[ACTION_SLICES[arm]] = torch.as_tensor(
                            ret["action"], dtype=action.dtype, device=action.device)
                        obs, _, term, trunc, _ = env.step(action[None])
                        if term or trunc:
                            break
                except Exception as e:
                    logger.error(f"{arm} arm failed: {e}")
                finally:
                    client.reset()

            if not video:
                continue
            path = video_dir / f"yam_tiptop_scene{scene}_{variant}_ep{ep}.mp4"
            mediapy.write_video(path, video, fps=VIDEO_FPS)
            saved.append(path)
            logger.info(f"Saved video to {path}")

            final = object_positions()
            plate_xy = final[spec.meta["plate"]][:2]
            on_plate = [n for n in spec.meta["items"]
                        if float(np.linalg.norm(final[n][:2] - plate_xy)) < spec.meta["on_plate_xy"]]
            logger.info(f"Episode {ep}: {len(on_plate)}/{len(spec.meta['items'])} on the plate"
                        + (f" ({', '.join(on_plate)})" if on_plate else ""))

    for client in clients.values():
        client.close()
    env.close()
    simulation_app.close()

    robot_desc = (
        "both arms, simultaneous (handover)" if handover
        else "both arms, in sequence" if bimanual
        else "left arm"
    )
    logger.info("Run complete\n" + "\n".join([
        f"  Robot       : bimanual YAM ({robot_desc})",
        "  Perception  : fixed third-person camera on the robot base (MolmoAct2 top_cam)",
        f"  Scene       : {scene}_{variant}",
        f"  Episodes    : {episodes} ({len(saved)} recorded)",
        f"  Output dir  : {video_dir.resolve()}",
        *(f"  Video       : {p.resolve()}" for p in saved),
    ]))


if __name__ == "__main__":
    tyro.cli(main)
