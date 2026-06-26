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
        --policy.config=pi05_droid \
        --policy.dir=gs://openpi-assets/checkpoints/pi05_droid --port 8000

Action space: pi05_droid (and DROID-style finetunes) emit JOINT VELOCITIES (7, rad/s) +
gripper. The sim env's arm term runs in "velocity" mode for this client (see
droid_environment.set_arm_control_mode), integrating those velocities onto the current
joint position -- so we forward the raw 7 arm outputs unchanged.
"""

import logging
import os
import time
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
        self._server_metadata = self._policy.get_server_metadata()
        _log.info(f"Connected to pi-0.5 server: {self._server_metadata}")

        self._open_loop_horizon = open_loop_horizon
        self._chunk: Optional[np.ndarray] = None
        self._chunk_idx = 0

        # Debug capture: when PI05_DEBUG_DUMP is set, record every quantity that flows
        # through inference (raw sim proprioception, the exact server request, the full
        # returned action chunk, and the post-processed action) so the inference path can
        # be replayed/audited offline. Disabled (zero overhead) when the env var is unset.
        self._debug = bool(os.environ.get("PI05_DEBUG_DUMP"))
        self._step_records: list = []   # one entry per infer() call
        self._query_records: list = []  # one entry per server query (subset of steps)
        self._step = 0

    def infer(self, obs: dict, instruction: str) -> dict:
        curr_obs = self._extract_observation(obs)

        # Re-query when we have no chunk, exhausted the open-loop horizon, or ran off
        # the end of a (possibly shorter-than-horizon) returned chunk.
        need_query = (
            self._chunk is None
            or self._chunk_idx >= self._open_loop_horizon
            or self._chunk_idx >= len(self._chunk)
        )
        request = None
        infer_latency_s = None
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
            _t0 = time.perf_counter()
            self._chunk = np.asarray(self._policy.infer(request)["actions"])
            infer_latency_s = time.perf_counter() - _t0
            self._chunk_idx = 0

        chunk_row = self._chunk_idx  # row of the chunk consumed this step
        chunk_action = self._chunk[self._chunk_idx]
        self._chunk_idx += 1

        # First 7 dims are JOINT VELOCITIES (rad/s) -- the pi05_droid action space. Forward
        # them unchanged (do NOT clip to [-1, 1]); the sim env's arm term, switched to
        # "velocity" mode for pi05, integrates them onto the current joint position. Dim 7 is
        # the gripper in [0, 1]; binarize at 0.5 to match the sim's
        # BinaryJointPositionZeroToOneActionCfg (1=close, 0=open).
        gripper = 1.0 if float(chunk_action[7]) > 0.5 else 0.0
        action = np.concatenate([chunk_action[:7], [gripper]]).astype(np.float32)

        if self._debug:
            self._record(
                instruction=instruction,
                curr_obs=curr_obs,
                need_query=need_query,
                request=request,
                infer_latency_s=infer_latency_s,
                chunk_row=chunk_row,
                chunk_action=chunk_action,
                gripper_binarized=gripper,
                action=action,
            )
        self._step += 1

        return {
            "action": action,
            "right_image": curr_obs["right_image"],
            "wrist_image": curr_obs["wrist_image"],
        }

    def _record(self, *, instruction, curr_obs, need_query, request, infer_latency_s,
                chunk_row, chunk_action, gripper_binarized, action) -> None:
        """Accumulate per-step (and, on query steps, per-query) debug data."""
        self._step_records.append(
            {
                "step": self._step,
                "queried": bool(need_query),
                "query_index": len(self._query_records) - 1 if not need_query else len(self._query_records),
                "chunk_row": int(chunk_row),
                "joint_position": np.asarray(curr_obs["joint_position"], np.float32).flatten(),
                "gripper_position": np.asarray(curr_obs["gripper_position"], np.float32).flatten(),
                "chunk_action": np.asarray(chunk_action, np.float32),  # 8-dim served row (7 joint vel + gripper)
                "gripper_raw": np.float32(chunk_action[7]),
                "gripper_binarized": np.float32(gripper_binarized),
                "action": np.asarray(action, np.float32),  # 8-dim action sent to env.step
            }
        )
        if need_query:
            self._query_records.append(
                {
                    "step": self._step,
                    "prompt": instruction,
                    "infer_latency_s": np.float32(infer_latency_s if infer_latency_s is not None else np.nan),
                    "req_exterior_image": np.asarray(request["observation/exterior_image_1_left"], np.uint8),
                    "req_wrist_image": np.asarray(request["observation/wrist_image_left"], np.uint8),
                    "req_joint_position": np.asarray(request["observation/joint_position"], np.float32),
                    "req_gripper_position": np.asarray(request["observation/gripper_position"], np.float32),
                    "action_chunk": np.asarray(self._chunk, np.float32),  # full returned chunk (H, 8)
                }
            )

    def dump_debug(self, path) -> None:
        """Write all recorded inference data to a compressed ``.npz`` (no-op if empty)."""
        if not self._debug or not self._step_records:
            _log.info("pi05 debug dump: nothing recorded (PI05_DEBUG_DUMP unset or no steps).")
            return

        def _stack(records, key):
            return np.stack([r[key] for r in records], axis=0)

        out = {
            # --- per-step arrays (length = number of env steps) ---
            "step": _stack(self._step_records, "step"),
            "queried": _stack(self._step_records, "queried"),
            "query_index": _stack(self._step_records, "query_index"),
            "chunk_row": _stack(self._step_records, "chunk_row"),
            "joint_position": _stack(self._step_records, "joint_position"),
            "gripper_position": _stack(self._step_records, "gripper_position"),
            "chunk_action": _stack(self._step_records, "chunk_action"),
            "gripper_raw": _stack(self._step_records, "gripper_raw"),
            "gripper_binarized": _stack(self._step_records, "gripper_binarized"),
            "action": _stack(self._step_records, "action"),
            # --- per-query arrays (length = number of server queries) ---
            "query_step": _stack(self._query_records, "step"),
            "query_infer_latency_s": _stack(self._query_records, "infer_latency_s"),
            "query_req_exterior_image": _stack(self._query_records, "req_exterior_image"),
            "query_req_wrist_image": _stack(self._query_records, "req_wrist_image"),
            "query_req_joint_position": _stack(self._query_records, "req_joint_position"),
            "query_req_gripper_position": _stack(self._query_records, "req_gripper_position"),
            "query_action_chunk": _stack(self._query_records, "action_chunk"),
            "query_prompt": np.asarray([r["prompt"] for r in self._query_records]),
            # --- run-level metadata ---
            "open_loop_horizon": np.int64(self._open_loop_horizon),
            "num_steps": np.int64(len(self._step_records)),
            "num_queries": np.int64(len(self._query_records)),
            "server_metadata": np.asarray(str(self._server_metadata)),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **out)
        _log.info(
            f"pi05 debug dump: {len(self._step_records)} steps / "
            f"{len(self._query_records)} queries -> {path}"
        )

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
