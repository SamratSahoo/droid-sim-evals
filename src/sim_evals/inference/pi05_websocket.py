"""Pi-0.5 (openpi) websocket client.

Adapts the openpi ``WebsocketClientPolicy`` to the same interface as
``TiptopWebsocketClient`` so the shared rollout loop in ``full_eval.py`` /
``tiptop_eval.py`` works unchanged:

    ret = client.infer(obs, instruction)
    viz = np.concatenate([ret["right_image"], ret["wrist_image"]], axis=1)
    action = torch.tensor(ret["action"])[None]

Unlike tiptop (which plans once per episode), pi-0.5 is queried repeatedly:
we cache the predicted action chunk and execute ``open_loop_horizon`` actions
from it before re-querying the server.

Server (launch separately, e.g. via full_eval.py):
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_droid_jointpos_polaris \
        --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos --port 8000
"""

import logging
from typing import Optional

import numpy as np

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from .abstract_client import InferenceClient

_log = logging.getLogger(__name__)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Coerce an RGB frame to uint8 in [0, 255] (handles float [0,1] or [0,255])."""
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.floating) and img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


class Pi05WebsocketClient(InferenceClient):
    """Queries an openpi pi-0.5 DROID policy server and steps its action chunk."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        open_loop_horizon: int = 8,
    ) -> None:
        _log.info(f"Connecting to pi-0.5 policy server at ws://{host}:{port}...")
        # WebsocketClientPolicy blocks here, retrying every 5s until the server accepts.
        self._policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        _log.info(f"Connected to pi-0.5 server: {self._policy.get_server_metadata()}")

        self._open_loop_horizon = open_loop_horizon
        self._chunk: Optional[np.ndarray] = None
        self._chunk_idx = 0

    def infer(self, obs: dict, instruction: str) -> dict:
        curr_obs = self._extract_observation(obs)

        # Re-query when we have no chunk, exhausted the open-loop horizon, or ran off
        # the end of a (possibly shorter-than-horizon) returned chunk.
        need_query = (
            self._chunk is None
            or self._chunk_idx >= self._open_loop_horizon
            or self._chunk_idx >= len(self._chunk)
        )
        if need_query:
            request = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    _to_uint8(curr_obs["right_image"]), 224, 224
                ),
                "observation/wrist_image_left": image_tools.resize_with_pad(
                    _to_uint8(curr_obs["wrist_image"]), 224, 224
                ),
                "observation/joint_position": curr_obs["joint_position"].flatten().astype(np.float32),
                "observation/gripper_position": curr_obs["gripper_position"].flatten().astype(np.float32),
                "prompt": instruction,
            }
            self._chunk = np.asarray(self._policy.infer(request)["actions"])
            self._chunk_idx = 0

        chunk_action = self._chunk[self._chunk_idx]
        self._chunk_idx += 1

        # First 7 dims are absolute joint positions (do NOT clip to [-1, 1] -- joint
        # angles exceed that range). Dim 7 is the gripper in [0, 1]; binarize at 0.5 to
        # match the sim's BinaryJointPositionZeroToOneActionCfg (1=close, 0=open).
        gripper = 1.0 if float(chunk_action[7]) > 0.5 else 0.0
        action = np.concatenate([chunk_action[:7], [gripper]]).astype(np.float32)

        return {
            "action": action,
            "right_image": curr_obs["right_image"],
            "wrist_image": curr_obs["wrist_image"],
        }

    def reset(self) -> None:
        """Start a new episode. The pi-0.5 server is stateless, so only clear the chunk."""
        self._chunk = None
        self._chunk_idx = 0
        self._policy.reset()

    def close(self) -> None:
        ws = getattr(self._policy, "_ws", None)
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    def _extract_observation(self, obs_dict: dict) -> dict:
        policy = obs_dict["policy"]
        return {
            "right_image": policy["external_cam"][0].clone().detach().cpu().numpy(),
            "wrist_image": policy["wrist_cam"][0].clone().detach().cpu().numpy(),
            "joint_position": policy["arm_joint_pos"].clone().detach().cpu().numpy(),
            "gripper_position": policy["gripper_pos"].clone().detach().cpu().numpy(),
        }
