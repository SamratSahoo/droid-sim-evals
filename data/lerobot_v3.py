"""Write LeRobot **v3.0** (video) datasets that match ``lerobot/droid_1.0.1``'s schema.

Used by the TAMP data-collection / conversion scripts (``tamp_data_gen.py``, ``merge_*.py``) to emit
datasets in the same format as DROID 1.0.1: camera frames are stored as separate MP4 **videos** (not
inline in the parquet), the low-dim columns use DROID's v3.0 names, and the training action is joint
velocity + gripper. Datasets written here are consumed by openpi's streaming loader
(``openpi.training.streaming_dataset``, v3.0 path).

Design notes:
  * memory-bounded (important on this 30G host): camera frames are encoded to disk one frame at a time
    and the low-dim parquet is written incrementally (one row group per episode), so peak RAM is ~one
    episode's decoded images -- unlike lerobot's ``add_frame`` writer which buffered across episodes.
  * ``actions`` (joint velocity[7] + gripper[1]) is split into ``action.joint_velocity`` +
    ``action.gripper_position``; state (joint_position[7] + gripper[1]) into ``observation.state.*``.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Common frame-dict image key (as produced by the sim/merge callers) -> DROID v3.0 video column name.
VIDEO_KEY_MAP = {
    "exterior_image_1_left": "observation.images.exterior_1_left",
    "exterior_image_2_left": "observation.images.exterior_2_left",
    "wrist_image_left": "observation.images.wrist_left",
}

# Fixed pyarrow schema for the v3.0 data parquet (DROID low-dim columns). Lists match DROID (variable
# list<float>), which openpi's v3.0 reader flattens per row.
_DATA_SCHEMA = pa.schema(
    [
        ("observation.state.joint_position", pa.list_(pa.float32())),
        ("observation.state.gripper_position", pa.list_(pa.float32())),
        ("observation.state", pa.list_(pa.float32())),
        ("action.joint_velocity", pa.list_(pa.float32())),
        ("action.gripper_position", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ]
)


def _feature_dict(height: int, width: int) -> dict:
    def feat(dtype, shape):
        return {"dtype": dtype, "shape": list(shape), "names": None}

    return {
        **{v: feat("video", [height, width, 3]) for v in VIDEO_KEY_MAP.values()},
        "observation.state.joint_position": feat("float32", [7]),
        "observation.state.gripper_position": feat("float32", [1]),
        "observation.state": feat("float32", [8]),
        "action.joint_velocity": feat("float32", [7]),
        "action.gripper_position": feat("float32", [1]),
        "action": feat("float32", [8]),
        "timestamp": feat("float32", [1]),
        "frame_index": feat("int64", [1]),
        "episode_index": feat("int64", [1]),
        "index": feat("int64", [1]),
        "task_index": feat("int64", [1]),
    }


class V3DatasetWriter:
    """Incrementally assemble a LeRobot v3.0 (video) dataset in ``root`` (DROID schema).

    Call ``add_episode`` once per episode, then ``finalize`` (or use as a context manager).
    """

    def __init__(
        self,
        root: str | Path,
        fps: int,
        *,
        height: int = 180,
        width: int = 320,
        robot_type: str = "panda",
    ):
        self.root = Path(root)
        self.fps = int(fps)
        self.height = int(height)
        self.width = int(width)
        self.robot_type = robot_type
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

        # One MP4 per camera (all episodes concatenated), encoded incrementally.
        self._encoders: dict[str, tuple] = {}
        for droid_key in VIDEO_KEY_MAP.values():
            path = self.root / "videos" / droid_key / "chunk-000" / "file-000.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            container = av.open(str(path), mode="w")
            stream = container.add_stream("libx264", rate=self.fps)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "20"}
            self._encoders[droid_key] = (container, stream)

        self._data_writer = pq.ParquetWriter(str(self.root / "data" / "chunk-000" / "file-000.parquet"), _DATA_SCHEMA)
        self._episode_rows: list[dict] = []
        self._tasks: dict[str, int] = {}
        self._global_index = 0
        self._frame_cursor = 0  # cumulative frames written to each camera MP4
        self._finalized = False

    # -- context manager ----------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self._finalized:
            self.finalize()

    # -- helpers ------------------------------------------------------------------------------
    def _task_index(self, task: str) -> int:
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
        return self._tasks[task]

    def _encode(self, droid_key: str, image: np.ndarray) -> None:
        container, stream = self._encoders[droid_key]
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image.astype(np.uint8)), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # -- main API -----------------------------------------------------------------------------
    def add_episode(
        self,
        *,
        images: dict[str, np.ndarray],
        joint_position: np.ndarray,
        gripper_position: np.ndarray,
        actions: np.ndarray,
        task: str,
    ) -> None:
        """Append one episode.

        Args:
            images: {common image key -> [N, H, W, 3] uint8}. Keys must be in VIDEO_KEY_MAP.
            joint_position: [N, 7]; gripper_position: [N, 1]; actions: [N, 8] (joint_velocity[7] + gripper[1]).
            task: language instruction for this episode.
        """
        joint_position = np.asarray(joint_position, dtype=np.float32).reshape(-1, 7)
        gripper_position = np.asarray(gripper_position, dtype=np.float32).reshape(-1, 1)
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, 8)
        n = len(joint_position)
        if n == 0:
            return
        if not (len(gripper_position) == n and len(actions) == n):
            raise ValueError(f"episode length mismatch: joint={n} gripper={len(gripper_position)} act={len(actions)}")

        episode_index = len(self._episode_rows)
        task_index = self._task_index(task)
        from_ts = self._frame_cursor / self.fps

        # Encode video frames (per camera) for this episode.
        for common_key, droid_key in VIDEO_KEY_MAP.items():
            frames = images[common_key]
            if len(frames) != n:
                raise ValueError(f"{common_key}: {len(frames)} frames vs {n} state rows")
            for i in range(n):
                self._encode(droid_key, np.asarray(frames[i]))
        self._frame_cursor += n
        to_ts = self._frame_cursor / self.fps

        # Low-dim rows -> one parquet row group.
        state = np.concatenate([joint_position, gripper_position], axis=1)  # [N, 8]
        joint_velocity = actions[:, :7]
        gripper_action = actions[:, 7:8]
        start = self._global_index
        table = pa.table(
            {
                "observation.state.joint_position": list(joint_position),
                "observation.state.gripper_position": list(gripper_position),
                "observation.state": list(state),
                "action.joint_velocity": list(joint_velocity),
                "action.gripper_position": list(gripper_action),
                "action": list(actions),
                "timestamp": np.arange(n, dtype=np.float32) / self.fps,
                "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, episode_index, dtype=np.int64),
                "index": np.arange(start, start + n, dtype=np.int64),
                "task_index": np.full(n, task_index, dtype=np.int64),
            },
            schema=_DATA_SCHEMA,
        )
        self._data_writer.write_table(table)
        self._global_index += n

        ep_meta = {
            "episode_index": episode_index,
            "length": n,
            "tasks": [task],
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start,
            "dataset_to_index": start + n,
        }
        for droid_key in VIDEO_KEY_MAP.values():
            ep_meta[f"videos/{droid_key}/chunk_index"] = 0
            ep_meta[f"videos/{droid_key}/file_index"] = 0
            ep_meta[f"videos/{droid_key}/from_timestamp"] = from_ts
            ep_meta[f"videos/{droid_key}/to_timestamp"] = to_ts
        self._episode_rows.append(ep_meta)

    def finalize(self) -> dict:
        """Flush encoders/parquet and write meta/info.json + meta/episodes + meta/tasks.parquet."""
        if self._finalized:
            return json.loads((self.root / "meta" / "info.json").read_text())
        for container, stream in self._encoders.values():
            for packet in stream.encode():  # flush
                container.mux(packet)
            container.close()
        self._data_writer.close()

        # Validate every encoded MP4 opens and has the expected frame count (a truncated / non-finalized
        # video would only fail later, with "Invalid data", at load time).
        for droid_key in VIDEO_KEY_MAP.values():
            vpath = self.root / "videos" / droid_key / "chunk-000" / "file-000.mp4"
            try:
                with av.open(str(vpath)) as container:
                    n_frames = sum(1 for packet in container.demux(video=0) if packet.size)
            except Exception as exc:  # noqa: BLE001 - re-raise with context.
                raise RuntimeError(f"Encoded video {vpath} is unreadable ({exc!r}).") from exc
            if n_frames != self._global_index:
                raise RuntimeError(f"Encoded video {vpath} has {n_frames} frames, expected {self._global_index}.")

        pq.write_table(
            pa.Table.from_pylist(self._episode_rows),
            self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        )
        ordered = sorted(self._tasks.items(), key=lambda kv: kv[1])
        pq.write_table(
            pa.table({"task_index": [i for _, i in ordered], "task": [t for t, _ in ordered]}),
            self.root / "meta" / "tasks.parquet",
        )
        info = {
            "codebase_version": "v3.0",
            "robot_type": self.robot_type,
            "total_episodes": len(self._episode_rows),
            "total_frames": self._global_index,
            "total_tasks": len(self._tasks),
            "fps": self.fps,
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": _feature_dict(self.height, self.width),
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
        self._finalized = True
        return info


# --- Reading (for merges): stream any Hub LeRobot dataset episode-by-episode --------------------

# DROID v3.0 low-dim columns we need + the common frame keys they map to.
_V3_JOINT = "observation.state.joint_position"
_V3_GRIPPER = "observation.state.gripper_position"
_V3_ACT_JV = "action.joint_velocity"
_V3_ACT_GRIP = "action.gripper_position"


def _decode_video(fs, video_path: str, from_ts: float, num_frames: int) -> np.ndarray:
    """Decode ``num_frames`` RGB frames starting at ``from_ts`` from a streamed MP4 (HWC uint8)."""
    handle = fs.open(video_path, "rb")
    frames: list[np.ndarray] = []
    try:
        container = av.open(handle)
        stream = container.streams.video[0]
        container.seek(
            int(from_ts / stream.time_base),
            stream=stream,
            backward=True,
            any_frame=False,
        )
        for frame in container.decode(stream):
            if float(frame.pts * stream.time_base) < from_ts - 1e-3:
                continue
            if len(frames) >= num_frames:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
    finally:
        handle.close()
    while frames and len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames)


def iter_episodes(repo_id: str) -> Iterator[dict]:
    """Stream a Hub LeRobot dataset episode-by-episode, in order, handling BOTH formats.

    Yields per episode ``{"images": {common_key -> [N,H,W,3] uint8}, "joint_position": [N,7],
    "gripper_position": [N,1], "actions": [N,8], "task": str}`` -- the exact inputs ``add_episode``
    expects. Handles v3.0 (video) sources produced by this writer / DROID 1.0.1, and legacy v2.1
    (inline-image) sources (e.g. an un-converted d100). Streams (no full local download).
    """
    import datasets as _ds
    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

    fs = HfFileSystem()
    info = json.loads(Path(hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")).read_text())
    common_keys = list(VIDEO_KEY_MAP.keys())

    if str(info.get("codebase_version", "")).startswith("v3"):
        all_files = HfApi().list_repo_files(repo_id, repo_type="dataset")
        ep_meta_files = sorted(f for f in all_files if f.startswith("meta/episodes/") and f.endswith(".parquet"))
        raw_video_keys = [k for k in VIDEO_KEY_MAP.values() if k in info["features"]]
        cols = ["episode_index", "length", "data/chunk_index", "data/file_index"]
        for rk in raw_video_keys:
            cols += [
                f"videos/{rk}/chunk_index",
                f"videos/{rk}/file_index",
                f"videos/{rk}/from_timestamp",
            ]
        meta = pa.concat_tables(
            [pq.ParquetFile(fs.open(f"datasets/{repo_id}/{m}", "rb")).read(columns=cols) for m in ep_meta_files]
        ).to_pydict()
        order = np.argsort(np.asarray(meta["episode_index"]))
        ep_len = np.asarray(meta["length"])[order]
        video_loc = {
            rk: {
                "chunk": np.asarray(meta[f"videos/{rk}/chunk_index"])[order],
                "file": np.asarray(meta[f"videos/{rk}/file_index"])[order],
                "from_ts": np.asarray(meta[f"videos/{rk}/from_timestamp"])[order],
            }
            for rk in raw_video_keys
        }
        tasks_meta = pq.ParquetFile(fs.open(f"datasets/{repo_id}/meta/tasks.parquet", "rb")).read().to_pydict()
        text_key = next((k for k in ("task", "__index_level_0__") if k in tasks_meta), None) or next(
            k for k in tasks_meta if k != "task_index"
        )
        tasks = dict(
            zip(
                tasks_meta.get("task_index", range(len(tasks_meta[text_key]))),
                tasks_meta[text_key],
            )
        )
        data_files = sorted(
            {
                info["data_path"].format(chunk_index=int(c), file_index=int(f))
                for c, f in zip(meta["data/chunk_index"], meta["data/file_index"])
            }
        )
        lowdim = [
            _V3_JOINT,
            _V3_GRIPPER,
            _V3_ACT_JV,
            _V3_ACT_GRIP,
            "episode_index",
            "task_index",
        ]
        for rel in data_files:
            low = pq.ParquetFile(fs.open(f"datasets/{repo_id}/{rel}", "rb")).read(columns=lowdim).to_pydict()
            ep_col = np.asarray(low["episode_index"])
            bounds = np.nonzero(np.diff(ep_col))[0] + 1
            for s, e in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(ep_col)]])):
                ep, n = int(ep_col[s]), int(e - s)
                raw_to_common = {v: k for k, v in VIDEO_KEY_MAP.items()}
                images = {}
                for rk in raw_video_keys:
                    vp = info["video_path"].format(
                        video_key=rk,
                        chunk_index=int(video_loc[rk]["chunk"][ep]),
                        file_index=int(video_loc[rk]["file"][ep]),
                    )
                    frames = _decode_video(
                        fs,
                        f"datasets/{repo_id}/{vp}",
                        float(video_loc[rk]["from_ts"][ep]),
                        int(ep_len[ep]),
                    )
                    images[raw_to_common[rk]] = frames[:n]
                jv = np.asarray([low[_V3_ACT_JV][i] for i in range(s, e)], dtype=np.float32).reshape(n, 7)
                ga = np.asarray([low[_V3_ACT_GRIP][i] for i in range(s, e)], dtype=np.float32).reshape(n, 1)
                jp = np.asarray([low[_V3_JOINT][i] for i in range(s, e)], dtype=np.float32).reshape(n, 7)
                gp = np.asarray([low[_V3_GRIPPER][i] for i in range(s, e)], dtype=np.float32).reshape(n, 1)
                yield {
                    "images": images,
                    "joint_position": jp,
                    "gripper_position": gp,
                    "actions": np.concatenate([jv, ga], axis=1),
                    "task": str(tasks.get(int(low["task_index"][s]), "")),
                }
    else:  # v2.1: one inline-image parquet per episode.
        tasks = {}
        for raw_line in (
            Path(hf_hub_download(repo_id, "meta/tasks.jsonl", repo_type="dataset")).read_text().splitlines()
        ):
            if raw_line.strip():
                row = json.loads(raw_line)
                tasks[int(row["task_index"])] = str(row["task"])
        data_files = sorted(
            f
            for f in HfApi().list_repo_files(repo_id, repo_type="dataset")
            if f.startswith("data/") and f.endswith(".parquet")
        )
        for rel in data_files:
            rows = list(
                _ds.load_dataset(
                    "parquet",
                    data_files={"train": [f"hf://datasets/{repo_id}/{rel}"]},
                    split="train",
                    streaming=True,
                )
            )
            n = len(rows)
            if n == 0:
                continue
            images = {k: np.stack([np.asarray(r[k]) for r in rows]) for k in common_keys}
            jp = np.stack([np.asarray(r["joint_position"], np.float32).reshape(7) for r in rows])
            gp = np.stack([np.atleast_1d(np.asarray(r["gripper_position"], np.float32)).reshape(1) for r in rows])
            act = np.stack([np.asarray(r["actions"], np.float32).reshape(8) for r in rows])
            yield {
                "images": images,
                "joint_position": jp,
                "gripper_position": gp,
                "actions": act,
                "task": str(tasks.get(int(rows[0]["task_index"]), "")),
            }
