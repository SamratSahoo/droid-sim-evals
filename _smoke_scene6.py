"""Smoke test for scene 6 (plate + 3 deformable toys) -- no policy servers.

Launches the DROID env on scene 6, resets/settles, and verifies:
  * the rigid plate spawned and rests on the table,
  * the three toys cooked into PhysX deformable bodies and settled (no NaN/explosion),
  * the cameras render the scene.
Saves frames + a short clip to runs/smoke_scene6/.
"""
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_scene6")

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import cv2

import src.sim_evals.environments  # noqa: F401  (registers "DROID")
from isaaclab_tasks.utils import parse_env_cfg
from src.sim_evals.sim_utils import settle_sim

OUT = Path("runs/smoke_scene6")
OUT.mkdir(parents=True, exist_ok=True)


def held_action(obs):
    aj = obs["policy"]["arm_joint_pos"]
    gp = obs["policy"]["gripper_pos"]
    return torch.cat([aj, gp], dim=1)


def to_img(t):
    a = t[0].detach().cpu().numpy()
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def main():
    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.set_scene("6", 0)
    env_cfg.episode_length_s = 30.0
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()  # second render cycle for correctly loaded materials

    scene = env.unwrapped.scene
    logger.info(f"rigid_objects   : {list(scene.rigid_objects.keys())}")
    logger.info(f"deformable_objs : {list(scene.deformable_objects.keys())}")

    # Measure the table-top collider world bbox so object placement can be validated.
    from pxr import Usd, UsdGeom
    top = scene.stage.GetPrimAtPath("/World/envs/env_0/scene/table/table_01/top")
    if top and top.IsValid():
        rng = UsdGeom.Imageable(top).ComputeWorldBound(Usd.TimeCode.Default(), "default").ComputeAlignedBox()
        lo, hi = rng.GetMin(), rng.GetMax()
        logger.info(f"TABLE TOP world bbox: x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}] z[{lo[2]:.3f},{hi[2]:.3f}]")

    # Settle, then report resting states.
    obs = settle_sim(env, obs, steps=150, reset_episode_buf=True)

    for name, ro in scene.rigid_objects.items():
        p = ro.data.root_pos_w[0].cpu().numpy()
        logger.info(f"rigid '{name}': pos={np.round(p, 4)} finite={np.isfinite(p).all()}")
    for name, do in scene.deformable_objects.items():
        nodal = do.data.nodal_pos_w[0].cpu().numpy()
        logger.info(
            f"deformable '{name}': center={np.round(nodal.mean(0), 4)} "
            f"z[{nodal[:,2].min():.4f},{nodal[:,2].max():.4f}] finite={np.isfinite(nodal).all()}"
        )

    # Save still frames.
    ext = to_img(obs["policy"]["external_cam"])
    ext2 = to_img(obs["policy"]["external_cam_2"])
    wrist = to_img(obs["policy"]["wrist_cam"])
    for nm, im in (("external", ext), ("external2", ext2), ("wrist", wrist)):
        cv2.imwrite(str(OUT / f"scene6_{nm}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        logger.info(f"saved {OUT / f'scene6_{nm}.png'}  shape={im.shape}  mean={im.mean():.1f} std={im.std():.1f}")

    # Short clip from external cam while holding pose (confirms physics stays stable).
    fps = 15
    h, w = ext.shape[:2]
    vw = cv2.VideoWriter(str(OUT / "scene6_smoke.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    with torch.no_grad():
        for _ in range(45):
            obs, _, term, trunc, _ = env.step(held_action(obs))
            vw.write(cv2.cvtColor(to_img(obs["policy"]["external_cam"]), cv2.COLOR_RGB2BGR))
            if bool(term) or bool(trunc):
                break
    vw.release()
    logger.info(f"saved {OUT / 'scene6_smoke.mp4'}")

    env.close()
    simulation_app.close()
    logger.info("SMOKE TEST DONE")


if __name__ == "__main__":
    main()
