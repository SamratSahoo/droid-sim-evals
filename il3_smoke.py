import sys
print("PYVER", sys.version.split()[0])
# Launch Isaac headless via the app launcher (as the worker does) so omni-backed imports resolve
from isaaclab.app import AppLauncher
import argparse
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args,_ = p.parse_known_args([]); args.headless=True; args.enable_cameras=True
app = AppLauncher(args).app
print("APP_LAUNCH_OK")
import isaaclab
print("ISAACLAB_VERSION", getattr(isaaclab,"__version__","?"))
tests = [
 "import isaaclab.sim as sim_utils",
 "from isaaclab.utils import configclass, noise",
 "from isaaclab.managers import SceneEntityCfg, ObservationTermCfg, ObservationGroupCfg, EventTermCfg, TerminationTermCfg",
 "from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg",
 "import isaaclab.envs.mdp as mdp",
 "from isaaclab.envs.mdp.actions.joint_actions import JointAction, JointPositionAction",
 "from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction",
 "from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg",
 "from isaaclab.actuators import ImplicitActuatorCfg",
 "from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg, DeformableObjectCfg",
 "from isaaclab.sensors import CameraCfg, TiledCameraCfg, ContactSensorCfg",
 "from isaaclab.scene import InteractiveSceneCfg",
 "from isaaclab.sim.utils import clone, get_all_matching_child_prims",
 "from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh",
 "from isaaclab.sim.spawners.meshes import meshes_cfg",
 "from isaaclab.sim.spawners.from_files import from_files_cfg",
 "from isaaclab.sim import schemas",
 "import isaacsim.core.utils.prims as prim_utils",
 "from isaaclab_tasks.utils import parse_env_cfg",
]
nfail=0
for t in tests:
    try:
        exec(t); print("PASS ::", t)
    except Exception as e:
        nfail+=1; print("FAIL ::", t, "->", type(e).__name__, str(e)[:90])
print("SMOKE_DONE fails=%d/%d" % (nfail,len(tests)))
app.close()
