"""TiPToP websocket client for the bimanual YAM, perceiving through the FIXED third-person camera.

Same server, same wire protocol, same plan state machine as :class:`TiptopWebsocketClient` -- the
plan-stepping logic is embodiment-agnostic, so only the four things that genuinely differ are
overridden here:

  * **which camera the request carries.** The DROID client sends the wrist camera; the YAM has no
    usable wrist view for perception (both wrists are on moving arms, and the real rig perceives from
    a fixed overhead RealSense on the base column), so this sends the ``top_cam`` terms.
  * **no tool-frame nudge.** The DROID client shifts the camera pose down 1.5 cm to cancel the
    Robotiq ``tool_from_ee`` offset (``tiptop_websocket.py``, ``_get_camera_params``). The YAM's
    fingers are prismatic and its ``tool_from_ee`` has no such offset, so the extrinsic is sent as
    measured -- nudging it here would bend the point cloud for no reason.
  * **6 arm joints instead of 7**, which the base class already handles by comparing against the
    measured arm dof.
  * **video framing**, which pairs the third-person view the planner actually saw with a side view.

The extrinsic needs no frame conversion: Isaac reports ``quat_w_ros`` in the ROS *optical* frame
(x right, y down, z forward), which is exactly the convention TiPToP unprojects depth in, and the
camera offset in :mod:`bimanual_yam` is authored in that same frame.
"""

import logging

import numpy as np

from .tiptop_websocket import TiptopWebsocketClient

_log = logging.getLogger(__name__)


class YamTiptopWebsocketClient(TiptopWebsocketClient):
    """Queries the TiPToP server once for a plan, then steps ONE YAM arm through it.

    ``arm`` selects which arm's joints are read back and which server this talks to. A bimanual
    rollout runs one of these per arm against a server configured for the matching
    ``robot.type: bimanual_yam_<arm>``, and the driver splices each client's 7 outputs into the
    env's 14-wide action vector (see ``yam_environment.ACTION_SLICES``).
    """

    def __init__(self, *args, arm: str = "left", **kwargs) -> None:
        if arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        self.arm = arm
        super().__init__(*args, **kwargs)

    def _extract_observation(self, obs_dict: dict) -> dict:
        policy = obs_dict["policy"]
        return {
            # `right_image` / `wrist_image` keep the base class's key names so the shared
            # `_make_result` visualization path works unchanged; what they hold is this
            # embodiment's side view and the third-person view the planner perceives through.
            "right_image": policy["external_cam"][0].clone().detach().cpu().numpy(),
            "wrist_image": policy["top_cam"][0].clone().detach().cpu().numpy(),
            "joint_position": policy[f"{self.arm}_arm_joint_pos"].clone().detach().cpu().numpy(),
            "gripper_position": policy[f"{self.arm}_gripper_pos"].clone().detach().cpu().numpy(),
        }

    def _get_wrist_depth(self, raw_obs: dict) -> np.ndarray:
        policy = raw_obs.get("policy", {})
        if "top_depth" not in policy:
            raise ValueError("Third-person camera depth not found; expected a 'top_depth' observation term.")
        depth = policy["top_depth"][0]
        if hasattr(depth, "cpu"):
            depth = depth.cpu().numpy()
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        # The scene has no ground plane -- the floor is the HDR dome, which renders at infinite
        # depth. The server clamps non-finite and out-of-range samples to 0 (its "invalid" value)
        # anyway; doing it here too keeps the saved request self-consistent.
        return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_camera_params(self, raw_obs: dict) -> tuple[np.ndarray, np.ndarray]:
        policy = raw_obs.get("policy", {})
        for key in ("top_intrinsics", "top_cam_pos_w", "top_cam_quat_w"):
            if key not in policy:
                raise ValueError(f"Third-person camera parameter '{key}' not found in the observation.")

        def _np(x):
            return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

        intrinsics = _np(policy["top_intrinsics"][0])
        world_from_cam = self._pose_to_matrix(_np(policy["top_cam_pos_w"][0]), _np(policy["top_cam_quat_w"][0]))
        return intrinsics, world_from_cam
