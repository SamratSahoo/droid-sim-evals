import sys, numpy as np, traceback
from isaaclab.app import AppLauncher
import argparse
p=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a,_=p.parse_known_args([]); a.headless=True; a.enable_cameras=True
app=AppLauncher(a).app
import torch, gymnasium as gym
import src.sim_evals.environments
from isaaclab_tasks.utils import parse_env_cfg
from src.sim_evals.sim_utils import settle_sim
from src.sim_evals.environments.droid_environment import collapse_dome_lights
print("STEP: imports_ok", flush=True)
def dump(tag, arr):
    a=np.asarray(arr, dtype=np.float64)
    fin=np.isfinite(a); 
    print(f"[{tag}] shape={a.shape} dtype={np.asarray(arr).dtype} "
          f"finite%={100*fin.mean():.1f} inf%={100*np.isinf(a).mean():.1f} nan%={100*np.isnan(a).mean():.1f} "
          f"le0%={100*(a<=0).mean():.1f}", flush=True)
    if fin.any():
        v=a[fin]; print(f"       min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f} median={np.median(v):.4f}", flush=True)
try:
    cfg=parse_env_cfg("DROID", device="cuda:0", num_envs=4, use_fabric=True)
    cfg.set_scene("6",0); cfg.episode_length_s=200.0
    env=gym.make("DROID", cfg=cfg); u=env.unwrapped
    with torch.no_grad():
        obs,_=env.reset(); obs,_=env.reset(); collapse_dome_lights()
        obs=settle_sim(env,obs,steps=120,reset_episode_buf=True)
    pol=obs["policy"]
    print("STEP: obs_keys", list(pol.keys()), flush=True)
    # --- replicate client extraction for env 0 ---
    depth = pol["wrist_depth"][0]
    depth = depth.cpu().numpy() if hasattr(depth,"cpu") else np.asarray(depth)
    if depth.ndim==3 and depth.shape[-1]==1: depth=depth.squeeze(-1)
    dump("wrist_depth", depth)
    K = pol["wrist_intrinsics"][0]; K = K.cpu().numpy() if hasattr(K,"cpu") else np.asarray(K)
    print("[intrinsics] shape=%s\n%s" % (np.asarray(K).shape, np.array2string(np.asarray(K), precision=3)), flush=True)
    pos = pol["wrist_cam_pos_w"][0]; pos = pos.cpu().numpy() if hasattr(pos,"cpu") else np.asarray(pos)
    quat= pol["wrist_cam_quat_w"][0]; quat= quat.cpu().numpy() if hasattr(quat,"cpu") else np.asarray(quat)
    print("[extrinsics] pos_w=%s quat_w=%s" % (np.round(pos,4).tolist(), np.round(quat,4).tolist()), flush=True)
    # --- reconstruct camera-frame point cloud (same deprojection M2T2 uses) ---
    K=np.asarray(K,dtype=np.float64)
    if K.shape==(3,3):
        fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
        H,W=depth.shape
        print(f"[img] HxW={H}x{W}  fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  (expect cx~{W/2:.0f} cy~{H/2:.0f})", flush=True)
        vv,uu=np.mgrid[0:H,0:W]
        d=depth.astype(np.float64); m=np.isfinite(d)&(d>0)&(d<5)
        X=(uu-cx)/fx*d; Y=(vv-cy)/fy*d; Z=d
        pts=np.stack([X[m],Y[m],Z[m]],axis=1)
        print(f"[cloud] n_valid_pts={pts.shape[0]} (of {H*W})", flush=True)
        if pts.shape[0]:
            bb_min=pts.min(0); bb_max=pts.max(0); ext=bb_max-bb_min
            print(f"        bbox_min={np.round(bb_min,3).tolist()} bbox_max={np.round(bb_max,3).tolist()}", flush=True)
            print(f"        extent(m)={np.round(ext,3).tolist()} centroid={np.round(pts.mean(0),3).tolist()}", flush=True)
    else:
        print("[intrinsics] NOT 3x3 -> this is the bug", flush=True)

    # save full arrays for offline M2T2 reproduction
    try:
        wrgb = pol["wrist_cam"][0]; wrgb = wrgb.cpu().numpy() if hasattr(wrgb,"cpu") else np.asarray(wrgb)
        eo = u.scene.env_origins[0].cpu().numpy()
        np.savez("/n/fs/tamp-vla/tamp-vla/droid-sim-evals/percep_capture.npz",
                 rgb=wrgb, depth=depth, K=np.asarray(K), pos=np.asarray(pos), quat=np.asarray(quat), env_origin=eo)
        print("SAVED_NPZ rgb_shape=%s rgb_dtype=%s rgb_min=%.1f rgb_max=%.1f" % (wrgb.shape, wrgb.dtype, float(wrgb.min()), float(wrgb.max())), flush=True)
    except Exception as e:
        print("save failed:", repr(e), flush=True)

    try:
        from PIL import Image as _Img
        for _nm in ("external_cam","external_cam_2","wrist_cam"):
            _im = pol[_nm][0]; _im = _im.cpu().numpy() if hasattr(_im,"cpu") else np.asarray(_im)
            _im = np.clip(_im,0,255).astype(np.uint8)
            _Img.fromarray(_im).save("/n/fs/tamp-vla/tamp-vla/droid-sim-evals/isaac6_%s.png" % _nm)
            print("saved isaac6_%s.png mean=%.0f" % (_nm, float(_im.mean())), flush=True)
    except Exception as _e:
        print("cam save failed", repr(_e), flush=True)
    print("PERCEPTION_PROBE_DONE", flush=True)
except BaseException as e:
    print("PERCEPTION_PROBE_EXCEPTION: %r"%(e,), flush=True); traceback.print_exc()
finally:
    try: app.close()
    except Exception: pass
