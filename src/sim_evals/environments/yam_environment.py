"""Isaac Lab env config for the bimanual YAM, reusing the DROID scenes verbatim.

Deliberately a thin layer over :mod:`droid_environment` rather than a copy: the scene loading
(``dynamic_scene`` / ``_add_sidecar_objects``), the lighting fixes (``global_dome`` +
``collapse_dome_lights``), the physics tuning and ``set_scene`` are all embodiment-neutral and are
inherited. What is Franka-specific, and therefore overridden here, is exactly four things:

  * which articulation is spawned as ``robot``,
  * which joints the action terms drive (6 left-arm revolutes + 2 prismatic fingers, vs 7 + 1),
  * the observation terms that read those joints back, and
  * the camera TiPToP perceives through -- a FIXED third-person camera on the robot base instead of
    the Franka's wrist camera (see :mod:`bimanual_yam`).

Registered as the gym id ``"YAM"`` alongside ``"DROID"``; the two are separate configs for the same
``ManagerBasedRLEnv``, one Isaac process at a time.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass, noise

from .bimanual_yam import (
    ARM_JOINTS,
    BIMANUAL_YAM,
    FINGER_CLOSED,
    FINGER_JOINTS,
    FINGER_OPEN,
    PLANNING_ARM,
    top_camera_cfg,
)
from .droid_environment import (
    ArmJointActionCfg,
    BinaryJointPositionZeroToOneActionCfg,
    EnvCfg,
    EventCfg,
    SceneCfg,
    _safe_image,
    external_cam_2_image,
    external_cam_image,
    wrist_cam_image,
)


# --------------------------------------------------------------------------- #
# Observations                                                                  #
# --------------------------------------------------------------------------- #
def _joint_indices(robot, names) -> list:
    """Indices of ``names`` in the articulation's joint order, in the order given.

    The DROID equivalents build this with a filtering comprehension, which returns the
    articulation's order rather than the requested one. That is harmless for the Franka (the two
    coincide) but would be a subtle joint-permutation bug here, where cuRobo's ordering is what the
    plan waypoints are indexed by.
    """
    order = list(robot.data.joint_names)
    return [order.index(n) for n in names]


def arm_joint_pos(env: ManagerBasedRLEnv, arm: str = PLANNING_ARM,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """One arm's 6 joint positions, in cuRobo/URDF order."""
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, _joint_indices(robot, ARM_JOINTS[arm])]


def gripper_pos(env: ManagerBasedRLEnv, arm: str = PLANNING_ARM,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """One hand's aperture normalized to 0 = fully open .. 1 = closed.

    Both fingers are commanded together and mirror each other, so one finger defines the aperture.
    The sign flip (travel is negative-open) is folded in here, so downstream code -- the eval's
    gripper bookkeeping, scene_specs' ``_grip`` release checks -- reads the same 0..1 convention it
    does for the Franka.
    """
    robot = env.scene[asset_cfg.name]
    idx = _joint_indices(robot, FINGER_JOINTS[arm][:1])
    return 1.0 - robot.data.joint_pos[:, idx] / FINGER_OPEN


def top_cam_image(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("top_cam"),
                  data_type: str = "rgb", normalize: bool = False) -> torch.Tensor:
    return _safe_image(env, sensor_cfg, data_type, normalize, channels=3)


def top_cam_depth(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("top_cam")):
    sensor = env.scene[sensor_cfg.name]
    try:
        return sensor.data.output["distance_to_image_plane"]
    except RuntimeError:
        h, w = sensor.cfg.height, sensor.cfg.width
        return torch.zeros((env.num_envs, h, w, 1), device=env.device, dtype=torch.float32)


def top_cam_intrinsics(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("top_cam")):
    return env.scene[sensor_cfg.name].data.intrinsic_matrices


def top_cam_pos_w(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("top_cam")):
    return env.scene[sensor_cfg.name].data.pos_w


def top_cam_quat_w(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("top_cam")):
    """World orientation in ROS *optical* convention (x right, y down, z forward), wxyz."""
    return env.scene[sensor_cfg.name].data.quat_w_ros


# --------------------------------------------------------------------------- #
# Scene / actions / observations                                                #
# --------------------------------------------------------------------------- #
@configclass
class YamSceneCfg(SceneCfg):
    """DROID scene with the YAM in place of the Franka and a fixed third-person camera added."""

    robot = BIMANUAL_YAM
    top_cam = top_camera_cfg()

    # Re-declared rather than inherited: the DROID path points at a Robotiq link that does not exist
    # here. Keeping the term (instead of dropping it) preserves a wrist view for the eval video.
    # Mount and pose are ported from the MolmoAct2 YAM wrist camera
    # (sim_eval/robots/bimanual_yam.py, `left_cam`): on link_6, 9 cm out along the finger axis and
    # 6 cm forward, looking down the approach direction.
    wrist_cam = TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/robot/{PLANNING_ARM}_link_6/wrist_cam",
        height=180,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.09, 0.06), rot=(0.6124, -0.3536, -0.3536, -0.6124),
                                   convention="ros"),
    )


def _arm_action(arm: str) -> ArmJointActionCfg:
    # Same dual-mode term the DROID env uses; TiPToP drives it in the default "position" mode, where
    # the 6 outputs are absolute joint targets.
    return ArmJointActionCfg(asset_name="robot", joint_names=ARM_JOINTS[arm],
                            preserve_order=True, use_default_offset=False)


def _gripper_action(arm: str) -> BinaryJointPositionZeroToOneActionCfg:
    # Both fingers are driven from one 0/1 command so the jaw stays symmetric, exactly like the
    # Franka's single `finger_joint` term. 0 -> open (negative travel), 1 -> closed.
    return BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=FINGER_JOINTS[arm],
        open_command_expr={name: FINGER_OPEN for name in FINGER_JOINTS[arm]},
        close_command_expr={name: FINGER_CLOSED for name in FINGER_JOINTS[arm]},
    )


@configclass
class YamActionCfg:
    """BOTH arms: ``[left arm (6), left gripper (1), right arm (6), right gripper (1)]`` = 14.

    Declaration order IS the action-vector order (IsaacLab concatenates action terms in the order
    they are declared), and :data:`ACTION_SLICES` below is the single place that layout is written
    down. Both arms are always driven even though TiPToP plans one at a time, so a bimanual rollout
    can hand off between them without rebuilding the env -- the idle arm is simply commanded to hold
    its measured position.
    """

    left_arm = _arm_action("left")
    left_gripper = _gripper_action("left")
    right_arm = _arm_action("right")
    right_gripper = _gripper_action("right")


@configclass
class YamObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Both arms are reported so a bimanual rollout can read whichever it is currently driving and
        # hold the other one where it is.
        left_arm_joint_pos = ObsTerm(func=arm_joint_pos, params={"arm": "left"})
        left_gripper_pos = ObsTerm(func=gripper_pos, params={"arm": "left"},
                                   noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1))
        right_arm_joint_pos = ObsTerm(func=arm_joint_pos, params={"arm": "right"})
        right_gripper_pos = ObsTerm(func=gripper_pos, params={"arm": "right"},
                                    noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1))
        external_cam = ObsTerm(
            func=external_cam_image,
            params={"sensor_cfg": SceneEntityCfg("external_cam"), "data_type": "rgb", "normalize": False},
        )
        external_cam_2 = ObsTerm(
            func=external_cam_2_image,
            params={"sensor_cfg": SceneEntityCfg("external_cam_2"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=wrist_cam_image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        # The perception camera. Same five terms the DROID env exposes for its WRIST camera
        # (rgb / depth / intrinsics / world position / world orientation), which is the complete set
        # the TiPToP websocket request needs -- so the client differs only in which keys it reads.
        top_cam = ObsTerm(
            func=top_cam_image,
            params={"sensor_cfg": SceneEntityCfg("top_cam"), "data_type": "rgb", "normalize": False},
        )
        top_depth = ObsTerm(func=top_cam_depth)
        top_intrinsics = ObsTerm(func=top_cam_intrinsics)
        top_cam_pos_w = ObsTerm(func=top_cam_pos_w)
        top_cam_quat_w = ObsTerm(func=top_cam_quat_w)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class YamEnvCfg(EnvCfg):
    """``EnvCfg`` with the YAM scene, actions and observations; everything else is inherited."""

    scene = YamSceneCfg(num_envs=1, env_spacing=7.0)
    observations = YamObservationCfg()
    actions = YamActionCfg()
    events = EventCfg()


# Where each arm's 7 numbers (6 joints + 1 gripper) sit in the 14-wide action vector. Must match the
# field order of YamActionCfg -- that ordering is the contract between the planner-facing clients,
# which each produce 7 numbers for one arm, and the env.
ACTION_DIM = 14
ACTION_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}


def verify_action_mapping(env) -> dict:
    """Assert that each action slice really drives that arm's joints, in joint1..joint6 order.

    Worth an explicit check because the three stacks that share this robot order its DOFs THREE
    DIFFERENT WAYS, and a mismatch is close to invisible:

        MuJoCo (MJCF)   left_joint1..6, left fingers, right_joint1..6, right fingers   (depth-first)
        cuRobo (URDF)   left_joint1..6, right_joint1..6                (fingers locked; tail if not)
        Isaac (PhysX)   left_joint1, right_joint1, left_joint2, right_joint2, ...     (BREADTH-first)

    Isaac interleaves the two arms one joint per level. Nothing here is broken by that, because
    IsaacLab's ``JointAction`` resolves its configured ``joint_names`` to indices and writes through
    them — the left-arm term comes out as ids [0, 2, 4, 6, 8, 10]. But that safety is a property of
    NAME-based resolution, not of the layout, and it would be silently lost if a term were ever given
    indices directly or the joints renamed. The rest posture cannot catch it either: it is identical
    for both arms with four zeros per arm, so a whole left/right swap moves nothing at all.

    Returns the resolved ids per term. Raises RuntimeError on any disagreement.
    """
    from .bimanual_yam import ARM_JOINTS, FINGER_JOINTS

    robot = env.unwrapped.scene["robot"]
    names = robot.joint_names
    manager = env.unwrapped.action_manager
    expected = {}
    for arm in ("left", "right"):
        expected[f"{arm}_arm"] = list(ARM_JOINTS[arm])
        expected[f"{arm}_gripper"] = list(FINGER_JOINTS[arm])

    resolved, offset, problems = {}, 0, []
    for term_name in manager.active_terms:
        term = manager.get_term(term_name)
        ids = getattr(term, "_joint_ids", None)
        ids = list(range(len(names))) if ids is None or ids is Ellipsis else list(ids)
        got = [names[i] for i in ids]
        resolved[term_name] = {"action": (offset, offset + term.action_dim), "ids": ids, "joints": got}
        if term_name in expected and got != expected[term_name]:
            problems.append(f"{term_name}: drives {got}, expected {expected[term_name]}")
        offset += term.action_dim

    for arm, sl in ACTION_SLICES.items():
        arm_span = resolved.get(f"{arm}_arm", {}).get("action")
        grip_span = resolved.get(f"{arm}_gripper", {}).get("action")
        if arm_span != (sl.start, sl.start + 6) or grip_span != (sl.start + 6, sl.stop):
            problems.append(
                f"ACTION_SLICES[{arm!r}]={sl} disagrees with the action manager "
                f"(arm term at {arm_span}, gripper term at {grip_span})"
            )
    if offset != ACTION_DIM:
        problems.append(f"total action dim {offset} != ACTION_DIM {ACTION_DIM}")
    if problems:
        raise RuntimeError("YAM action mapping is wrong:\n  " + "\n  ".join(problems))
    return resolved


def hold_action(obs: dict) -> "torch.Tensor":
    """A full 14-wide action that commands both arms to stay exactly where they are.

    The arm terms are absolute position targets and the gripper terms threshold at 0.5, so feeding
    the measured state back is a closed-loop "stay put" for both arms at once -- the bimanual
    equivalent of what ``sim_utils.settle_sim`` does for the Franka.
    """
    p = obs["policy"]
    return torch.cat([p["left_arm_joint_pos"], p["left_gripper_pos"],
                      p["right_arm_joint_pos"], p["right_gripper_pos"]], dim=-1)


def settle_sim(env, obs: dict, steps: int = 100, reset_episode_buf: bool = False) -> dict:
    """Hold both arms in place for ``steps`` control steps so the scene settles.

    The YAM equivalent of ``src.sim_evals.sim_utils.settle_sim``, which builds a 7+1 Franka action.
    """
    for _ in range(steps):
        obs, _, _, _, _ = env.step(hold_action(obs))
    if reset_episode_buf:
        env.env.episode_length_buf[:] = 0
    return obs


def set_camera_resolution(env_cfg, height: int, width: int) -> None:
    """Override the render resolution of every YAM camera on a parsed env cfg.

    The third-person camera is resized along with the rest because it is what TiPToP unprojects into
    a point cloud: at the DROID default of 180x320 a 4.5 cm toy covers a handful of pixels and M2T2
    has nothing to fit a grasp to.
    """
    for cam_name in ("external_cam", "external_cam_2", "wrist_cam", "top_cam"):
        cam = getattr(env_cfg.scene, cam_name)
        cam.height = height
        cam.width = width
        if cam.spawn is not None and hasattr(cam.spawn, "vertical_aperture"):
            # Keep the vertical FOV consistent with the new aspect ratio; the horizontal FOV is
            # fixed by focal_length / horizontal_aperture and must not change.
            cam.spawn.vertical_aperture = cam.spawn.horizontal_aperture * height / width
