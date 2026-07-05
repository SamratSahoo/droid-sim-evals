"""Batched Pi-0.5 client: ONE server round-trip drives ALL parallel envs per query step.

The stock :class:`Pi05WebsocketClient` is one-env-per-connection, so a multi-env eval spins up N of them and
the single policy server runs N inferences SERIALLY each query step (measured: 64 envs = ~3.8 s/query,
~64x a single infer -> ~67% of the whole eval's wall clock). This client instead packs all N observations
into ONE batched request (``{"__batched_observations__": [...]}``) that the server runs as a single batch-N
forward pass (see openpi ``Policy.infer_batch`` + ``websocket_policy_server``), caches per-env action chunks,
and walks them open-loop for ``open_loop_horizon`` steps before the next batched query.

Drop-in for the worker: ``infer(obs, instruction)`` returns an ``(N, 8)`` action array (7 arm outputs +
binarized gripper) every step -- the arm outputs are forwarded raw (joint velocities or absolute joint
targets depending on the checkpoint; the env's arm term interprets them per its control mode).
"""

import logging

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

_log = logging.getLogger(__name__)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.floating) and img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


class Pi05BatchedClient:
    """Queries an openpi pi-0.5 server with ALL N envs' observations in one batched forward pass."""

    def __init__(self, host: str = "localhost", port: int = 8000, num_envs: int = 1,
                 open_loop_horizon: int = 8) -> None:
        _log.info(f"Connecting batched pi-0.5 client ({num_envs} envs) to ws://{host}:{port}...")
        self._policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self._server_metadata = self._policy.get_server_metadata()
        self._n = int(num_envs)
        self._horizon = int(open_loop_horizon)
        self._chunks = None   # (N, H, 8) cached action chunks
        self._idx = 0

    @property
    def chunk_len(self):
        return None if self._chunks is None else int(self._chunks.shape[1])

    def _build_requests(self, obs: dict, instruction: str) -> list:
        # Transfer every env's images/joints ONCE (batched device->host), then build N per-env requests.
        p = obs["policy"]
        ext = p["external_cam"].detach().cpu().numpy()      # (N, H, W, 3)
        wri = p["wrist_cam"].detach().cpu().numpy()          # (N, H, W, 3)
        jp = p["arm_joint_pos"].detach().cpu().numpy()       # (N, 7)
        gp = p["gripper_pos"].detach().cpu().numpy()         # (N, 1)
        return [{
            "observation/exterior_image_1_left": image_tools.resize_with_pad(_to_uint8(ext[i]), 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(_to_uint8(wri[i]), 224, 224),
            "observation/joint_position": jp[i].flatten().astype(np.float32),
            "observation/gripper_position": gp[i].flatten().astype(np.float32),
            "prompt": instruction,
        } for i in range(self._n)]

    def infer(self, obs: dict, instruction: str) -> np.ndarray:
        """Return the (N, 8) action for this step; queries the server (batched) only every open_loop_horizon steps."""
        if self._chunks is None or self._idx >= self._horizon or self._idx >= self._chunks.shape[1]:
            resp = self._policy.infer({"__batched_observations__": self._build_requests(obs, instruction)})
            self._chunks = np.stack([np.asarray(a["actions"], dtype=np.float32) for a in resp["actions_batch"]], axis=0)
            self._idx = 0
        row = self._chunks[:, self._idx, :]     # (N, 8)
        self._idx += 1
        # Arm dims 0-6 forwarded raw; dim 7 gripper binarized at 0.5 (1=close, 0=open) per env.
        gripper = (row[:, 7] > 0.5).astype(np.float32)
        return np.concatenate([row[:, :7], gripper[:, None]], axis=1).astype(np.float32)

    def reset(self) -> None:
        self._chunks = None
        self._idx = 0
        try:
            self._policy.reset()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        ws = getattr(self._policy, "_ws", None)
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass
