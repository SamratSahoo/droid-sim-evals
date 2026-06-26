import json
import logging
import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import numpy as np

from typing import List
from pathlib import Path
from pxr import Usd, UsdPhysics

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointAction, JointPositionAction
from isaaclab.utils import configclass, noise
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg, DeformableObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg

from .nvidia_droid import NVIDIA_DROID
from .mesh_assets import FileMeshCfg, UsdRigidCfg

DATA_PATH = Path(__file__).parent / "../../../assets/"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Arm control mode (runtime-switchable; one Isaac env is shared by several      #
# policies within a worker process, so the arm term's interpretation of the     #
# policy's 7 arm outputs is flipped at runtime -- see full_eval.run_worker).     #
#                                                                                #
#   "position" (default): outputs are ABSOLUTE joint-position targets. Used by   #
#       tiptop (cuRobo plan waypoints) and the pi05_droid_jointpos checkpoint.   #
#       Behaves exactly like the original mdp.JointPositionActionCfg, so data-   #
#       gen / tiptop eval are unaffected.                                        #
#   "velocity": outputs are JOINT VELOCITIES (rad/s) from a velocity-action      #
#       policy (stock pi05_droid, or a velocity-trained finetune). They are      #
#       integrated onto the current measured joint position each control step    #
#       (q_target = q_meas + v * step_dt) and fed to the SAME stiff position     #
#       controller -- so no actuator-gain retuning is needed and re-reading      #
#       q_meas every step prevents open-loop integration drift.                  #
# --------------------------------------------------------------------------- #
_ARM_CONTROL = {"mode": "position"}


def set_arm_control_mode(mode: str) -> None:
    """Select how the arm action term interprets the policy's 7 arm outputs.

    ``"position"`` (default) = absolute joint-position targets; ``"velocity"`` =
    joint velocities integrated onto the current joint position. Call before
    stepping a given policy (the setting is process-global, one env per worker).
    """
    if mode not in ("position", "velocity"):
        raise ValueError(f"arm control mode must be 'position' or 'velocity', got {mode!r}")
    _ARM_CONTROL["mode"] = mode


def set_camera_resolution(env_cfg, height: int, width: int) -> None:
    """Override the render resolution of all DROID cameras on a parsed env cfg.

    Cameras default to 180x320 (the LeRobot/DROID image size) so data generation renders only the
    pixels it keeps; full_eval.py overrides to the full 720x1280 sensor resolution for higher-
    fidelity comparison videos. Call after ``parse_env_cfg`` and before ``gym.make`` (the cfg is
    read when the scene is built).
    """
    for cam_name in ("external_cam", "external_cam_2", "wrist_cam"):
        cam = getattr(env_cfg.scene, cam_name)
        cam.height = height
        cam.width = width


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    sphere_light = AssetBaseCfg(
        prim_path="/World/spehre",
        spawn=sim_utils.SphereLightCfg(intensity=5000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )

    robot = NVIDIA_DROID

    # TiledCamera = one tiled render pass per camera across ALL envs, so vision scales to many
    # parallel envs without exhausting the RTX renderer's per-viewport descriptor pool (plain per-env
    # Camera sensors hit "Unable to allocate descriptor sets" at ~32 envs x 3 cams; tiled also renders
    # faster). Same CameraData interface (.data.output / intrinsics / pos_w), so the obs funcs below
    # are unchanged. Default 180x320 (the LeRobot/DROID image size) so data gen renders only the pixels
    # it keeps; full_eval.py overrides to 720x1280 via set_camera_resolution() for eval.
    external_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        height=180,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.66), rot=(-0.393, -0.195, 0.399, 0.805), convention="opengl"
        ),
    )

    external_cam_2 = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam_2",
        height=180,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, -0.57, 0.66), rot=(0.805, 0.399, -0.195, -0.393), convention="opengl"
        ),
    )

    wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        height=180,
        width=320,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409), convention="opengl"
        ),
    )

    def dynamic_scene(self, scene_name: str):
        environment_path = DATA_PATH / f"scene{scene_name}.usd"
        scene = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/scene",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=str(environment_path),
                    ),
                )
        self.scene = scene

        stage = Usd.Stage.Open(
            str(environment_path)
        )
        scene_prim = stage.GetPrimAtPath("/World")
        children = scene_prim.GetChildren()

        _NON_RIGID_PRIMS = {"DomeLight", "Environment", "table", "sphere_light"}

        for child in children:
            if child.GetName() in _NON_RIGID_PRIMS:
                continue
            # Register prims with RigidBodyAPI directly, or payloaded prims
            # whose referenced asset defines a rigid body internally
            if not UsdPhysics.RigidBodyAPI(child) and not child.HasPayload():
                continue

            name = child.GetName()
            logger.debug(f"Found rigid body: {name}")
            pos = child.GetAttribute("xformOp:translate").Get()
            rot = child.GetAttribute("xformOp:orient").Get()
            rot = (rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])
            asset = RigidObjectCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                        spawn=None,
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=pos,
                            rot=rot,
                        ),
                    )
            setattr(self, name, asset)

        # Programmatically-spawned objects (rigid + deformable) described by an optional
        # `scene<name>.json` sidecar next to the USD. Used by scene 6 (plate + 3 deformable
        # toys) whose meshes are scanned assets without baked physics; see mesh_assets.py.
        self._add_sidecar_objects(scene_name)

    def _add_sidecar_objects(self, scene_name: str):
        """Spawn rigid/deformable objects from preprocessed meshes listed in a sidecar JSON.

        Each entry spawns via :class:`FileMeshCfg` (the Isaac-native mesh spawn path):
        ``kind: "rigid"`` -> a :class:`RigidObjectCfg` with a convex collider; ``kind:
        "deformable"`` -> a :class:`DeformableObjectCfg` (PhysX FEM soft body, tetrahedralized
        at spawn from the watertight mesh). Poses are in world meters.
        """
        spec_path = (DATA_PATH / f"scene{scene_name}.json").resolve()
        if not spec_path.exists():
            return
        spec = json.loads(spec_path.read_text())
        for obj in spec.get("objects", []):
            name = obj["name"]
            pos = tuple(obj.get("pos", (0.0, 0.0, 0.0)))
            rot = tuple(obj.get("rot", (1.0, 0.0, 0.0, 0.0)))
            kind = obj.get("kind", "rigid")

            if kind == "usd_rigid":
                # Reference a textured USD/USDZ asset directly (keeps its scanned material/
                # textures) as a rigid body with a convex collider; scale/pose from the sidecar.
                s = float(obj.get("scale", 1.0))
                spawn = UsdRigidCfg(
                    usd_path=str((DATA_PATH / obj["usd"]).resolve()),
                    scale=(s, s, s),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=obj.get("mass", 0.03)),
                    collision_approximation=obj.get("collision", "convexHull"),
                )
                setattr(self, name, RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{name}",
                    spawn=spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                ))
                continue

            mesh_path = str((DATA_PATH / obj["mesh"]).resolve())
            visual = sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(obj.get("color", (0.7, 0.7, 0.7))),
                roughness=0.6,
                metallic=0.0,
            )
            if kind == "deformable":
                dp = obj.get("deformable", {})
                spawn = FileMeshCfg(
                    mesh_path=mesh_path,
                    deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                        rest_offset=dp.get("rest_offset", 0.001),
                        contact_offset=dp.get("contact_offset", 0.002),
                        solver_position_iteration_count=dp.get("solver_iters", 20),
                        simulation_hexahedral_resolution=dp.get("hex_res", 10),
                        self_collision=dp.get("self_collision", False),
                    ),
                    physics_material=sim_utils.DeformableBodyMaterialCfg(
                        youngs_modulus=dp.get("youngs", 5e6),
                        poissons_ratio=dp.get("poisson", 0.4),
                        dynamic_friction=dp.get("friction", 0.6),
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=obj.get("mass", 0.03)),
                    visual_material=visual,
                )
                asset = DeformableObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{name}",
                    spawn=spawn,
                    init_state=DeformableObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                )
            else:  # rigid
                rp = obj.get("rigid", {})
                spawn = FileMeshCfg(
                    mesh_path=mesh_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=obj.get("kinematic", False),
                        # A large flat convex disc resting on the table can blow up via
                        # explosive depenetration; cap it and add solver iterations.
                        max_depenetration_velocity=rp.get("max_depenetration_velocity", 1.0),
                        solver_position_iteration_count=rp.get("solver_iters", 32),
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    collision_approximation=obj.get("collision"),
                    mass_props=sim_utils.MassPropertiesCfg(mass=obj.get("mass", 0.4)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=rp.get("static_friction", 1.0),
                        dynamic_friction=rp.get("dynamic_friction", 1.0),
                        restitution=rp.get("restitution", 0.0),
                    ),
                    visual_material=visual,
                )
                asset = RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{name}",
                    spawn=spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                )
            setattr(self, name, asset)


class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type = BinaryJointPositionZeroToOneAction


class ArmJointAction(JointPositionAction):
    """Arm action term supporting absolute-position and integrated-velocity control.

    In ``"position"`` mode this is identical to :class:`JointPositionAction` (the 7 arm
    outputs are absolute joint-position targets). In ``"velocity"`` mode the outputs are
    treated as joint velocities (rad/s) and integrated onto the *current measured* joint
    position over one control step before being applied as a position target -- letting a
    velocity-action policy (e.g. pi05_droid) drive the same stiff position-controlled
    Franka without retuning actuators. The mode is read from :data:`_ARM_CONTROL` every
    step, so a single shared env can switch policies at runtime.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Velocity feed-forward target (rad/s); zero except in velocity mode.
        self._vel_target = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        if _ARM_CONTROL["mode"] == "velocity":
            self._raw_actions[:] = actions
            # Treat outputs as joint velocities (rad/s). Set the position target one control step
            # ahead (q_meas + v*step_dt; step_dt = decimation*sim.dt = 1/15 s) and stash the velocity
            # feed-forward; apply_actions commands BOTH so the stiff PD's damping term (kd) drives the
            # joint toward v instead of opposing it (a position-only target tracks at only ~kp*dt/(kd+..)
            # of v). Re-reading q_meas each step makes this a closed-loop follower with no drift.
            self._vel_target = self._raw_actions * self._scale
            q_cur = self._asset.data.joint_pos[:, self._joint_ids]
            self._processed_actions = q_cur + self._vel_target * self._env.step_dt
            if self.cfg.clip is not None:
                self._processed_actions = torch.clamp(
                    self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
                )
        else:
            super().process_actions(actions)
            self._vel_target = torch.zeros_like(self._processed_actions)

    def apply_actions(self):
        # Always command the position target; in velocity mode also command the velocity feed-forward.
        # In position mode _vel_target is zero, so this is identical to JointPositionAction (the joint
        # velocity target defaults to zero there too).
        self._asset.set_joint_velocity_target(self._vel_target, joint_ids=self._joint_ids)
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


@configclass
class ArmJointActionCfg(JointPositionActionCfg):
    """Config for :class:`ArmJointAction` (dual-mode position/velocity arm control)."""

    class_type = ArmJointAction


@configclass
class ActionCfg:
    # Dual-mode arm term: absolute joint positions ("position", default) or integrated
    # joint velocities ("velocity"); switch via set_arm_control_mode(). In "position" mode
    # it is byte-for-byte the original mdp.JointPositionActionCfg behaviour.
    body = ArmJointActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr = {"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )

def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    # get joint inidices
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]
    return joint_pos


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]

    # rescale
    joint_pos = joint_pos / (np.pi / 4)

    return joint_pos


def _safe_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str,
    normalize: bool,
    channels: int,
) -> torch.Tensor:
    """Wrapper around mdp.observations.image that returns zeros if the camera annotator
    has no data yet (e.g. GPU OOM during init, or shape-probing before first sim step)."""
    try:
        return mdp.observations.image(env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=normalize)
    except RuntimeError:
        sensor = env.scene[sensor_cfg.name]
        h, w = sensor.cfg.height, sensor.cfg.width
        return torch.zeros((env.num_envs, h, w, channels), device=env.device, dtype=torch.float32)


def external_cam_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("external_cam"),
    data_type: str = "rgb",
    normalize: bool = False,
) -> torch.Tensor:
    return _safe_image(env, sensor_cfg, data_type, normalize, channels=3)


def external_cam_2_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("external_cam_2"),
    data_type: str = "rgb",
    normalize: bool = False,
) -> torch.Tensor:
    return _safe_image(env, sensor_cfg, data_type, normalize, channels=3)


def wrist_cam_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    data_type: str = "rgb",
    normalize: bool = False,
) -> torch.Tensor:
    return _safe_image(env, sensor_cfg, data_type, normalize, channels=3)


def wrist_cam_depth(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam")
):
    """Get wrist camera depth image."""
    sensor = env.scene[sensor_cfg.name]
    try:
        depth = sensor.data.output["distance_to_image_plane"]
    except RuntimeError:
        h, w = sensor.cfg.height, sensor.cfg.width
        return torch.zeros((env.num_envs, h, w, 1), device=env.device, dtype=torch.float32)
    return depth


def wrist_cam_intrinsics(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam")
):
    """Get wrist camera intrinsic matrix."""
    sensor = env.scene[sensor_cfg.name]
    return sensor.data.intrinsic_matrices


def wrist_cam_pos_w(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam")
):
    """Get wrist camera position in world frame."""
    sensor = env.scene[sensor_cfg.name]
    return sensor.data.pos_w


def wrist_cam_quat_w(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam")
):
    """Get wrist camera quaternion in world frame (w, x, y, z)."""
    sensor = env.scene[sensor_cfg.name]
    return sensor.data.quat_w_ros  # ROS convention: w, x, y, z


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        arm_joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(
            func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
        )
        external_cam = ObsTerm(
                func=external_cam_image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )
        external_cam_2 = ObsTerm(
                func=external_cam_2_image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam_2"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )
        wrist_cam = ObsTerm(
                func=wrist_cam_image,
                params={
                    "sensor_cfg": SceneEntityCfg("wrist_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )
        wrist_depth = ObsTerm(func=wrist_cam_depth)
        wrist_intrinsics = ObsTerm(func=wrist_cam_intrinsics)
        wrist_cam_pos_w = ObsTerm(func=wrist_cam_pos_w)
        wrist_cam_quat_w = ObsTerm(func=wrist_cam_quat_w)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

@configclass
class CommandsCfg:
    """Command terms for the MDP."""


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum configuration."""


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=7.0)

    observations = ObservationCfg()
    actions = ActionCfg()
    rewards = RewardsCfg()

    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = 30

        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 8
        self.sim.dt = 1 / (15 * 8)
        self.sim.render_interval = self.decimation

        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_temp_buffer_capacity = 2**26
        self.sim.physx.gpu_heap_capacity = 2**26
        self.sim.physx.gpu_collision_stack_size = 2**26
        # Headroom for PhysX FEM soft-body (deformable toy) contacts in scene 6.
        self.sim.physx.gpu_max_soft_body_contacts = 2**21
        self.rerender_on_reset = True

    
    def set_scene(self, scene_name: str, variant: int = 0):
        self.scene.dynamic_scene(f"{scene_name}_{variant}")
