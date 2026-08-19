import sys
print("PYVER", sys.version.split()[0], flush=True)
from isaaclab.app import AppLauncher
import argparse
p=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a,_=p.parse_known_args([]); a.headless=True; a.enable_cameras=True
app=AppLauncher(a).app
print("STEP: app_ok", flush=True)
import gymnasium as gym, torch
import src.sim_evals.environments  # registers DROID + goes through mesh_assets patched import
from isaaclab_tasks.utils import parse_env_cfg
print("STEP: imports_ok", flush=True)
cfg=parse_env_cfg("DROID", device="cuda:0", num_envs=4, use_fabric=True)
print("STEP: parse_env_cfg_ok", flush=True)
cfg.set_scene("6", 0)          # scene6_0.json sidecar -> soft-body toys via patched mesh path
cfg.episode_length_s=120.0
print("STEP: set_scene_ok (scene6 dynamic-mesh path)", flush=True)
env=gym.make("DROID", cfg=cfg)
print("STEP: gym_make_ok", flush=True)
obs,_=env.reset()
u=env.unwrapped
print("STEP: reset_ok num_envs=%d device=%s" % (u.num_envs, u.device), flush=True)
# Step several times so RTX actually renders (the librtx.scenedb.plugin segfault path on drv610).
act=torch.zeros((u.num_envs,8),device=u.device)
for i in range(6):
    env.step(act)
print("STEP: stepped_6x_ok", flush=True)
# RAW camera readout — bypass _safe_image (which masks RTX failure with zeros).
ok_rtx=True
for name in ("external_cam","external_cam_2","wrist_cam"):
    out=u.scene[name].data.output["rgb"]
    mx=float(out.float().abs().max()); mean=float(out.float().mean())
    nz = mx > 0
    ok_rtx = ok_rtx and nz
    print("RTX %-14s shape=%s max=%.3f mean=%.3f %s" % (name, tuple(out.shape), mx, mean, "NONZERO" if nz else "ALL-ZERO!!"), flush=True)
if ok_rtx:
    print("ENVBUILD_SUCCESS: env builds, steps, AND RTX renders non-zero frames on Isaac Sim 6.0 / drv610", flush=True)
else:
    print("ENVBUILD_FAIL_RTX: env builds but RTX returned all-zero frames (renderer not working on drv610)", flush=True)
app.close()
