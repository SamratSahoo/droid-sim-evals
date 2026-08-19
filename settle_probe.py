import sys, traceback
print("PYVER", sys.version.split()[0], flush=True)
from isaaclab.app import AppLauncher
import argparse
p=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a,_=p.parse_known_args([]); a.headless=True; a.enable_cameras=True
app=AppLauncher(a).app
print("STEP: app_ok", flush=True)
import numpy as np, torch, gymnasium as gym
import src.sim_evals.environments
from isaaclab_tasks.utils import parse_env_cfg
from src.sim_evals.sim_utils import settle_sim
from src.sim_evals.environments.droid_environment import collapse_dome_lights
# worker helpers (tamp_data_gen.py is a script run with cwd=droid-sim-evals; add data/ to path)
sys.path.insert(0, "data")
from tamp_data_gen import (Geom, SCENE_OBJECTS, sample_scene_poses,
                           read_object_poses_vec, write_poses_vec)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
SETTLE = int(sys.argv[2]) if len(sys.argv) > 2 else 120
print(f"STEP: imports_ok  N={N} settle={SETTLE}", flush=True)
try:
    geom = Geom()
    cfg = parse_env_cfg("DROID", device="cuda:0", num_envs=N, use_fabric=True)
    cfg.set_scene("6", 0)
    cfg.episode_length_s = max(120.0, (2400 + SETTLE + 80) / 15.0 + 10.0)
    env = gym.make("DROID", cfg=cfg)
    env_origins = env.unwrapped.scene.env_origins
    print("STEP: env_built", flush=True)
    with torch.no_grad():
        obs, _ = env.reset(); obs, _ = env.reset()
        print("STEP: reset_x2_ok", flush=True)
        collapse_dome_lights()
        print("STEP: collapse_ok", flush=True)
        obs = settle_sim(env, obs, steps=SETTLE, reset_episode_buf=True)
        print("STEP: baseline_settle_ok  <-- this is where the worker died", flush=True)
        base_world = read_object_poses_vec(env)[0]
        origin0 = env_origins[0].detach().cpu().numpy()
        base = {k: {"pos": base_world[k]["pos"] - origin0, "quat": base_world[k]["quat"]} for k in SCENE_OBJECTS}
        print("STEP: read_poses_ok", flush=True)
        # one batch of sampling + write + settle (loop body pre-planning)
        poses = []
        for i in range(N):
            rng = np.random.default_rng(1234 + i)
            try: poses.append(sample_scene_poses(rng, geom, base))
            except RuntimeError: poses.append({k:[*base[k]["pos"],*base[k]["quat"]] for k in SCENE_OBJECTS})
        obs,_=env.reset(); obs,_=env.reset()
        write_poses_vec(env, poses, env_origins)
        print("STEP: write_poses_ok", flush=True)
        obs = settle_sim(env, obs, steps=SETTLE, reset_episode_buf=True)
        print("STEP: batch_settle_ok", flush=True)
    print("SETTLE_PROBE_SUCCESS: full pre-planning sequence ran at N=%d without crash" % N, flush=True)
except BaseException as e:
    print("SETTLE_PROBE_EXCEPTION: %r" % (e,), flush=True)
    traceback.print_exc()
finally:
    try: app.close()
    except Exception: pass
