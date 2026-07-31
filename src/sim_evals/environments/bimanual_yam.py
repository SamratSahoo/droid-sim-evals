"""Isaac Sim articulation config for the bimanual YAM, plus the fixed third-person camera it uses.

Counterpart of :mod:`nvidia_droid` for the second embodiment. The USD is built from the same URDF
cuRobo plans over (``tools/yam_mjcf_to_urdf.py`` then ``tools/build_yam_usd.py``), so the joint names
here, the joint names in ``bimanual_yam_{left,right}.yml``, and the joint names the eval driver maps
plan waypoints onto are all the same strings by construction.

The base is spawned at the env origin because the mount offset is baked into the URDF's
``world -> bimanual_base`` fixed joint (see ``MOUNT_XYZ`` in the generator). That is what makes
cuRobo's base frame coincide with this repo's world frame, so scene coordinates, camera extrinsics
and plan waypoints all live in one frame.
"""

from pathlib import Path

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg

ASSET_PATH = Path(__file__).parent / "../../../assets"


# Arms are named by side; cuTAMP plans one at a time (see cutamp/robots/bimanual_yam.py). Scene 6's
# objects sit at y in [-0.10, 0.22] and the left arm is the one mounted toward +y, so it is the arm
# the reachability sweep was run for and the default everywhere downstream.
PLANNING_ARM = "left"

# Where `bimanual_base` sits in the world frame. MUST match MOUNT_XYZ in
# droid-sim-evals/tools/yam_mjcf_to_urdf.py, which bakes it into the URDF's world -> bimanual_base
# fixed joint and therefore into the USD; this copy exists only so the eval driver can split objects
# about the robot's midline without parsing the URDF. yam_tiptop_eval asserts the two agree at
# startup, so a regenerated asset cannot silently drift from it.
MOUNT_XYZ = (0.24, 0.0, 0.0551)

ARM_JOINTS = {side: [f"{side}_joint{i}" for i in range(1, 7)] for side in ("left", "right")}
FINGER_JOINTS = {side: [f"{side}_left_finger", f"{side}_right_finger"] for side in ("left", "right")}
FINGER_LINKS = {side: [f"{side}_link_left_finger", f"{side}_link_right_finger"] for side in ("left", "right")}

# Prismatic finger travel, from the MJCF actuator ctrlrange. Negative is open; both fingers of a hand
# move together. Kept in sync with YAM_FINGER_OPEN / YAM_FINGER_CLOSED in cutamp/robots/bimanual_yam.py.
FINGER_OPEN = -0.0475
FINGER_CLOSED = 0.0

# Elbow-up rest posture, matching ARM_RETRACT in the generator and cuRobo's retract_config. The idle
# arm is held here permanently, which is also the configuration cuRobo has it locked at, so the
# planner's collision model of the parked arm matches what the simulator actually shows.
ARM_REST = (0.0, np.pi / 4, np.pi / 2, 0.0, 0.0, 0.0)

BIMANUAL_YAM = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str((ASSET_PATH / "yam" / "bimanual_yam.usd").resolve()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Same as the DROID Franka: gravity off so the stiff position controller does not have to
            # fight it, which keeps tracking of a cuRobo plan tight. The MJCF models the real arm's
            # gravity compensation the same way (gravcomp=1 on every link).
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # The two arms cannot touch each other in any configuration cuRobo will produce (it
            # collision-checks the parked arm), and enabling this makes the finger pads fight their
            # own link_6 colliders.
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # the mount offset lives in the URDF; see the module docstring
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            **{name: float(v) for side in ("left", "right") for name, v in zip(ARM_JOINTS[side], ARM_REST)},
            **{name: FINGER_OPEN for side in ("left", "right") for name in FINGER_JOINTS[side]},
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        # Effort limits are the MJCF actuator forceranges: dm4340 servos on joints 1-3, dm4310 on
        # 4-6. Stiffness/damping are NOT the MJCF's low hardware gains -- like the DROID Franka cfg
        # this is a stiff position controller whose job is to track a planned trajectory exactly.
        "yam_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[r".*_joint[1-3]"],
            effort_limit=28.0,
            velocity_limit=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        "yam_wrist": ImplicitActuatorCfg(
            joint_names_expr=[r".*_joint[4-6]"],
            effort_limit=10.0,
            velocity_limit=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        "yam_gripper": ImplicitActuatorCfg(
            joint_names_expr=[r".*_finger"],
            effort_limit=40.0,
            stiffness=1000.0,
            damping=None,
            # For IMPLICIT actuators PhysX ignores `velocity_limit`; only `velocity_limit_sim` writes
            # set_dof_max_velocities. Without a cap the stiff finger snaps shut in one frame and
            # flicks the light photogrammetry toys away before trapping them -- the same failure the
            # DROID Franka gripper hit (see nvidia_droid.py). 0.1 m/s closes the 0.0475 m of travel
            # in ~0.5 s, roughly 7 frames at 15 Hz.
            velocity_limit_sim=0.1,
        ),
    },
)


# Contact friction for the whole scene. The MJCF gives the YAM no per-geom friction, so its finger
# pads fall back to PhysX's default 0.5 -- not enough to hold scene 6's smooth photogrammetry toys,
# which the gripper closes on and then drops. MolmoAct2 fixes this by binding a high-friction
# material to the four finger links after load; we cannot, because the URDF importer emits the
# collider prims as INSTANCE PROXIES and USD forbids authoring on those. Setting the simulation's
# DEFAULT material reaches them instead, and reaches every other collider that does not carry its own
# (the toys and the table; the plate keeps the 1.0/1.0 from its sidecar entry). The value is chosen
# so the finger-toy pair lands near MolmoAct2's effective grip: they pair 3.0/2.5 pads against
# PhysX-default 0.5 objects, which PhysX averages to 1.75/1.5.
DEFAULT_STATIC_FRICTION = 1.75
DEFAULT_DYNAMIC_FRICTION = 1.5


def set_contact_friction(
    env_cfg,
    static_friction: float = DEFAULT_STATIC_FRICTION,
    dynamic_friction: float = DEFAULT_DYNAMIC_FRICTION,
) -> None:
    """Raise the simulation's default contact friction. Call BEFORE ``gym.make``."""
    env_cfg.sim.physics_material.static_friction = static_friction
    env_cfg.sim.physics_material.dynamic_friction = dynamic_friction


# --------------------------------------------------------------------------- #
# Fixed third-person camera                                                     #
# --------------------------------------------------------------------------- #
# TiPToP's perception runs off ONE camera. On the DROID Franka that is the wrist camera; the bimanual
# YAM has no usable wrist view for this (both wrist cameras are on moving arms, and the real rig
# perceives from a fixed overhead RealSense D435i on the base column), so this is the camera the
# planner sees through.
#
# Mount, orientation and optics are the MolmoAct2 sim-eval `top_cam` verbatim
# (allenai/molmoact2 -> sim_eval/robots/bimanual_yam.py): mounted on `bimanual_base`, offset
# (0.15, 0, 0.80), 69.4 deg horizontal FOV (a RealSense D435i), 640x360 native.
#
# Their pose is a SAPIEN one, whose camera frame is +x forward / +y left / +z up, while Isaac's "ros"
# convention is the OPTICAL frame (+x right, +y down, +z forward) -- the same frame TiPToP assumes
# when it unprojects depth. Converting `q = [0.7660444431189782, 0, 0.6427876096865391, 0]` through
# x_opt = -y_sap, y_opt = -z_sap, z_opt = +x_sap gives the quaternion below; it looks 80 deg below
# horizontal, tilted 10 deg forward in +x, which is what the SAPIEN pose encodes.
TOP_CAM_OFFSET = (0.15, 0.0, 0.80)
TOP_CAM_ROT_ROS = (-0.06162842, 0.70441603, -0.70441603, 0.06162842)
TOP_CAM_HFOV_DEG = 69.4
# Isaac derives fx from `width * focal_length / horizontal_aperture`; the aperture value is the one
# the rest of this repo's cameras use, so only focal_length encodes the FOV.
TOP_CAM_HORIZONTAL_APERTURE = 5.376


def top_camera_cfg(height: int = 720, width: int = 1280) -> CameraCfg:
    """The fixed third-person camera, as a per-env :class:`CameraCfg` mounted on the robot base.

    Not a ``TiledCameraCfg`` (which the DROID env uses to scale vision to many parallel envs): this
    is the camera TiPToP plans from and it needs ``distance_to_image_plane`` depth plus intrinsics
    and a world pose, and the eval runs a single env.
    """
    focal_length = (width / 2.0) / np.tan(np.deg2rad(TOP_CAM_HFOV_DEG) / 2.0) * TOP_CAM_HORIZONTAL_APERTURE / width
    return CameraCfg(
        # Under the base link, so the camera rides with the robot exactly as the real one does.
        prim_path="{ENV_REGEX_NS}/robot/bimanual_base/top_cam",
        height=height,
        width=width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(focal_length),
            focus_distance=28.0,
            horizontal_aperture=TOP_CAM_HORIZONTAL_APERTURE,
            vertical_aperture=TOP_CAM_HORIZONTAL_APERTURE * height / width,
        ),
        offset=CameraCfg.OffsetCfg(pos=TOP_CAM_OFFSET, rot=TOP_CAM_ROT_ROS, convention="ros"),
    )
