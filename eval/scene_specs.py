"""Declarative per-scene config for the parallel Pi-0.5 sim eval (scenes 6-12).

Each :class:`SceneSpec` bundles everything the eval worker needs that is *not* about spawning
physics (that lives in the ``assets/scene<N>_0.json`` sidecars):

  * ``instruction``     -- the language prompt handed to the policy.
  * ``rigid``/``articulated`` -- object names to read each step (rigid: world pose;
                          articulated: joint positions), matching the sidecar ``name`` fields.
  * ``randomize``       -- per-object start-pose randomization (XY box + optional world-Z yaw),
                          sampled per env each rollout so the 8 rollouts differ.
  * ``evaluate``        -- a pure-numpy success function (init, final, traj, spec) -> ordered
                          ``{criterion: value in [0,1]}`` (0/1 for a boolean check, or the FRACTION of
                          objects satisfying a multi-object check, e.g. 2/3 toys on the plate -> 0.667).
                          Aggregated by ``score_env`` (on-table gates multiply; sequenced criteria are
                          credited in order); dense = gated, sequence-credited mean; sparse = all fully met.

All success checks are *relative* (final/traj vs the measured settled start pose) within an env,
so the world-frame env-origin offset and the table-top z frame cancel out (same trick as
``tamp_data_gen.evaluate_success``). Pure numpy -- imports cleanly under the Isaac venv.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Quaternion helpers (wxyz, Isaac convention)                                   #
# --------------------------------------------------------------------------- #
def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (X) q2, both wxyz."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def yaw_quat(theta: float) -> np.ndarray:
    """World-Z yaw of ``theta`` rad as a wxyz quaternion."""
    return np.array([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)], dtype=np.float64)


def apply_world_yaw(q_base: np.ndarray, theta: float) -> np.ndarray:
    """Compose a *world*-frame Z yaw onto a resting orientation (left-multiply)."""
    q = quat_mul(yaw_quat(theta), np.asarray(q_base, dtype=np.float64))
    return q / np.linalg.norm(q)


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Smallest rotation angle (degrees) between two unit wxyz quaternions."""
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


# --------------------------------------------------------------------------- #
# Randomization                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class ObjRandom:
    """Start-pose randomization for one object (env-LOCAL / robot frame, metres)."""
    name: str
    xy_box: Tuple[Tuple[float, float], Tuple[float, float]]  # ((xmin,xmax),(ymin,ymax))
    yaw: bool = False                       # randomize a world-Z yaw onto the settled orientation
    yaw_range: Tuple[float, float] = (-math.pi, math.pi)
    min_sep: float = 0.0                    # reject if within this XY dist of an earlier-placed object
    avoid: Tuple[Tuple[str, float], ...] = ()  # reject if within `clearance` of object `name`'s XY


def sample_layout(rng: np.random.Generator, spec: "SceneSpec", base: dict) -> dict:
    """Sample one env's randomized layout -> {name: [x,y,z,qw,qx,qy,qz]} (env-local).

    Keeps each object's measured settled ``z`` (and its settled orientation, onto which an optional
    world-Z yaw is composed); randomizes XY inside the object's box with rejection sampling for
    pairwise separation and any ``avoid`` clearances. Non-randomized objects keep their base pose.
    """
    poses = {n: [*base[n]["pos"], *base[n]["quat"]] for n in base}   # default: settled base pose
    placed_xy: Dict[str, np.ndarray] = {}
    for o in spec.randomize:
        (x0, x1), (y0, y1) = o.xy_box
        z = float(base[o.name]["pos"][2])
        q_base = np.asarray(base[o.name]["quat"], dtype=np.float64)
        xy = None
        for _ in range(400):
            cand = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            if any(np.linalg.norm(cand - p) < o.min_sep for p in placed_xy.values()):
                continue
            if any(np.linalg.norm(cand - _avoid_xy(other, placed_xy, base)) < clr for other, clr in o.avoid):
                continue
            xy = cand
            break
        if xy is None:
            raise RuntimeError(f"scene {spec.scene_id}: could not place {o.name} after 400 tries")
        placed_xy[o.name] = xy
        q = apply_world_yaw(q_base, rng.uniform(*o.yaw_range)) if o.yaw else q_base
        poses[o.name] = [float(xy[0]), float(xy[1]), z, *q.tolist()]
    return poses


def _avoid_xy(name: str, placed_xy: dict, base: dict) -> np.ndarray:
    return placed_xy[name] if name in placed_xy else np.asarray(base[name]["pos"][:2], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Success-evaluation helpers (all relative to the settled start pose)           #
# --------------------------------------------------------------------------- #
def _pos(state: dict, name: str) -> np.ndarray:
    return np.asarray(state[name]["pos"], dtype=np.float64)


def _lift_over_traj(traj: List[dict], name: str, z0: float) -> float:
    """Max rise (metres) of ``name`` above its start z across the trajectory."""
    return max((float(_pos(s, name)[2]) - z0 for s in traj), default=0.0)


def _joint_disp_over_traj(traj: List[dict], name: str, j0: np.ndarray) -> float:
    """Max abs per-joint displacement of articulation ``name`` from its start across the traj."""
    best = 0.0
    for s in traj:
        j = np.asarray(s[name]["joint"], dtype=np.float64)
        best = max(best, float(np.max(np.abs(j - j0))))
    return best


# --------------------------------------------------------------------------- #
# Per-scene geometry / thresholds                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Geom:
    # place-on-plate (scenes 6, 12)
    plate_xy_tol: float = 0.03
    plate_z_tol: float = 0.02
    plate_ang_tol_deg: float = 12.0
    on_plate_xy: float = 0.11
    lift_thresh: float = 0.012
    floor_drop: float = 0.05
    collision_thresh: float = 0.045
    # push cubes (scene 7)
    pick_lift_cap: float = 0.03          # neither cube may rise this far (must be pushed, not lifted)
    touch_dist: float = 0.075            # final center-center xy dist to count as "touching" (0.05 cubes)
    mover_move_min: float = 0.05         # the pushed cube must translate at least this
    stayer_move_max: float = 0.045       # the other cube must stay within this
    # wipe (scene 8)
    board_half: Tuple[float, float] = (0.16, 0.11)  # whiteboard XY half-extents (over-footprint band)
    contact_below: float = 0.02          # eraser may dip this far below board center and still "contact"
    contact_above: float = 0.06          # ...and up to this far above (eraser sitting on the board)
    grab_lift: float = 0.02              # eraser must rise this far off the table to count as grabbed
    wipe_len: float = 0.08               # lateral eraser path length while in contact
    board_move_tol: float = 0.10         # whiteboard may drift this much and still be "on the table"
    # stack (scene 9)
    block_size: float = 0.05
    stack_gap: Tuple[float, float] = (0.030, 0.075)  # allowed consecutive z-gap in a stack
    stack_xy: float = 0.035              # consecutive blocks must overlap within this XY
    # articulation (scenes 10, 11)
    press_thresh: float = 0.0012         # button press joint |disp| (m) to count as "pressed"
    drawer_thresh: float = 0.03          # cabinet drawer slide |disp| (m) to count as "opened"
    base_move_tol: float = 0.04          # anchored base may drift this much and still be "on the table"


# --------------------------------------------------------------------------- #
# Evaluators (init, final, traj, spec) -> ordered {criterion: bool}             #
# --------------------------------------------------------------------------- #
def eval_place_on_plate(init, final, traj, spec) -> Dict[str, float]:
    """Scenes 6 & 12: plate stays put, every item ends lifted onto the plate, none colliding, all on table."""
    g = spec.geom
    plate = spec.meta["plate"]
    items = spec.meta["items"]
    p0, pf = _pos(init, plate), _pos(final, plate)
    plate_unmoved = (np.linalg.norm(pf[:2] - p0[:2]) < g.plate_xy_tol
                     and abs(pf[2] - p0[2]) < g.plate_z_tol
                     and quat_angle_deg(init[plate]["quat"], final[plate]["quat"]) < g.plate_ang_tol_deg)
    on_plate, on_table, item_xy = [], [], []
    for it in items:
        t0, tf = _pos(init, it), _pos(final, it)
        item_xy.append(tf[:2])
        d = float(np.linalg.norm(tf[:2] - pf[:2]))
        on_plate.append(d < g.on_plate_xy and (tf[2] - t0[2]) > g.lift_thresh)
        on_table.append((tf[2] - t0[2]) > -g.floor_drop)
    pair_clear = [float(np.linalg.norm(item_xy[i] - item_xy[j]) > g.collision_thresh)
                  for i in range(len(items)) for j in range(i + 1, len(items))]
    return {
        "plate_unmoved": float(plate_unmoved),
        "items_on_plate": float(np.mean(on_plate)),                  # fraction of items lifted onto the plate
        "no_collision": float(np.mean(pair_clear)) if pair_clear else 1.0,  # fraction of item pairs not colliding
        "items_on_table": float(np.mean(on_table)),                  # fraction of items still on the table
    }


def eval_push_cubes(init, final, traj, spec) -> Dict[str, float]:
    """Scene 7: robot PUSHES the yellow cube into the (stationary) red cube; nothing gets lifted."""
    g = spec.geom
    mover, stayer = spec.meta["mover"], spec.meta["stayer"]
    z0_m, z0_s = float(_pos(init, mover)[2]), float(_pos(init, stayer)[2])
    not_picked_up = float(np.mean([_lift_over_traj(traj, mover, z0_m) < g.pick_lift_cap,
                                   _lift_over_traj(traj, stayer, z0_s) < g.pick_lift_cap]))
    touching = float(np.linalg.norm(_pos(final, mover)[:2] - _pos(final, stayer)[:2])) < g.touch_dist
    mover_d = float(np.linalg.norm(_pos(final, mover)[:2] - _pos(init, mover)[:2]))
    stayer_d = float(np.linalg.norm(_pos(final, stayer)[:2] - _pos(init, stayer)[:2]))
    mover_pushed = mover_d > g.mover_move_min and stayer_d < g.stayer_move_max
    on_table = float(np.mean([(_pos(final, mover)[2] - z0_m) > -g.floor_drop,
                              (_pos(final, stayer)[2] - z0_s) > -g.floor_drop]))
    return {
        "not_picked_up": not_picked_up,          # fraction of cubes never lifted
        "cubes_touching": float(touching),
        "yellow_is_mover": float(mover_pushed),
        "cubes_on_table": on_table,              # fraction of cubes still on the table
    }


def eval_wipe(init, final, traj, spec) -> Dict[str, float]:
    """Scene 8: grab the eraser, wipe it across the (flat) whiteboard in contact, leave both settled."""
    g = spec.geom
    board, eraser = spec.meta["board"], spec.meta["eraser"]
    z0_e = float(_pos(init, eraser)[2])
    grabbed = _lift_over_traj(traj, eraser, z0_e) > g.grab_lift

    # Frames where the eraser is horizontally over the board AND at board-surface height ("in contact").
    hx, hy = g.board_half
    contact_xy = []
    for s in traj:
        be, br = _pos(s, eraser), _pos(s, board)
        dz = be[2] - br[2]
        if abs(be[0] - br[0]) < hx and abs(be[1] - br[1]) < hy and -g.contact_below < dz < g.contact_above:
            contact_xy.append(be[:2])
    touches_board = len(contact_xy) >= 1
    wipe_path = sum(float(np.linalg.norm(contact_xy[k + 1] - contact_xy[k])) for k in range(len(contact_xy) - 1))
    wiping = wipe_path > g.wipe_len

    b0, bf = _pos(init, board), _pos(final, board)
    board_on_table = np.linalg.norm(bf[:2] - b0[:2]) < g.board_move_tol and (bf[2] - b0[2]) > -g.floor_drop
    eraser_ok = (_pos(final, eraser)[2] - z0_e) > -g.floor_drop   # ended in gripper / on table / on board
    return {
        "grabbed_eraser": float(grabbed),
        "wiping_motion": float(wiping),
        "eraser_touched_board": float(touches_board),
        "board_on_table": float(board_on_table),
        "eraser_settled": float(eraser_ok),
    }


def eval_stack(init, final, traj, spec) -> Dict[str, float]:
    """Scene 9: all blocks end on the table AND form a vertical stack (partial credit per stacked link)."""
    g = spec.geom
    blocks = spec.meta["blocks"]
    on_table = float(np.mean([(_pos(final, b)[2] - _pos(init, b)[2]) > -g.floor_drop for b in blocks]))
    fp = [_pos(final, b) for b in blocks]
    lo, hi = g.stack_gap
    # Partial credit: a block counts as stacked if some OTHER block sits ~one block-height below it and is
    # xy-aligned. A perfect stack of N has N-1 such blocks (all but the bottom), so normalize by (N-1).
    resting = sum(
        any(i != j and lo < (fp[i][2] - fp[j][2]) < hi and np.linalg.norm(fp[i][:2] - fp[j][:2]) < g.stack_xy
            for j in range(len(fp)))
        for i in range(len(fp))
    )
    stacked = float(resting) / (len(fp) - 1) if len(fp) > 1 else 1.0
    return {"blocks_on_table": on_table, "blocks_stacked": stacked}


def eval_articulation(init, final, traj, spec) -> Dict[str, bool]:
    """Scenes 10 & 11: the articulation moves non-trivially during the episode; its base stays on the table."""
    g = spec.geom
    name = spec.meta["art"]
    thresh = spec.meta["move_thresh"]
    j0 = np.asarray(init[name]["joint"], dtype=np.float64)
    moved = _joint_disp_over_traj(traj, name, j0) > thresh
    b0, bf = _pos(init, name), _pos(final, name)
    on_table = np.linalg.norm(bf[:2] - b0[:2]) < g.base_move_tol and (bf[2] - b0[2]) > -g.floor_drop
    return {spec.meta["move_crit"]: float(moved), spec.meta["table_crit"]: float(on_table)}


# --------------------------------------------------------------------------- #
# Scene registry                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class SceneSpec:
    scene_id: int
    instruction: str
    rigid: List[str]
    articulated: List[str]
    randomize: List[ObjRandom]
    evaluate: Callable
    geom: Geom = field(default_factory=Geom)
    meta: dict = field(default_factory=dict)
    # Scoring structure (see score_env):
    #   gates    -- criteria that act as a MULTIPLIER on the env's dense score (x1 met / x0 not), rather
    #               than an additive term. None -> auto: every criterion whose name ends in "_on_table"
    #               (so knocking an object off the table zeroes that env's score).
    #   sequence -- ordered prerequisite chain: a criterion is credited only if every EARLIER criterion
    #               in the chain was met (e.g. "eraser_settled" counts only after "wiping_motion").
    gates: Optional[List[str]] = None
    sequence: List[str] = field(default_factory=list)

    @property
    def objects(self) -> List[str]:
        return self.rigid + self.articulated

    @property
    def gate_criteria(self) -> Optional[List[str]]:
        return self.gates


def resolve_gates(spec: "SceneSpec", criteria: List[str]) -> List[str]:
    """Which criteria are multiplicative gates (explicit ``spec.gates`` or every ``*_on_table``)."""
    return spec.gates if spec.gates is not None else [c for c in criteria if c.endswith("_on_table")]


def score_env(raw: Dict[str, bool], spec: "SceneSpec") -> dict:
    """Turn one env's raw {criterion: bool} into dense/sparse scores + the credited per-criterion values.

    Each criterion value is in [0, 1] -- 0/1 for a boolean check, or the FRACTION of objects satisfying a
    multi-object check (e.g. 2/3 toys on the plate -> 0.667).

    dense  = (product of gate criteria) * (mean of the CREDITED non-gate criteria). A gate contributes its
             fractional value (all objects on the table -> x1; one of three knocked off -> x2/3; all off -> x0).
             A criterion in ``spec.sequence`` is scaled by the product of its earlier chain members, so a
             later step cannot earn more credit than the prerequisites it depends on (bool chains reduce to
             the usual AND; fractional prerequisites scale downstream credit).
    sparse = 1.0 iff EVERY criterion is fully met (== 1.0) -- a perfect env.
    """
    criteria = list(raw)
    gates = resolve_gates(spec, criteria)
    gate = 1.0
    for g_ in gates:
        gate *= float(raw[g_])
    credited = {c: float(raw[c]) for c in criteria}
    prefix = 1.0
    for c in spec.sequence:                      # scale each chain member by the product of earlier members
        credited[c] = float(raw[c]) * prefix
        prefix *= float(raw[c])
    task = [c for c in criteria if c not in gates]
    dense = gate * (float(np.mean([credited[c] for c in task])) if task else 1.0)
    sparse = 1.0 if all(float(raw[c]) >= 1.0 - 1e-9 for c in criteria) else 0.0
    return {"dense": dense, "sparse": sparse, "credited": credited, "gates": gates}


# Reachable-workspace XY boxes (env-local / robot frame, metres) live in each ObjRandom below.
# They mirror the scene6/7 sidecars' placement region (x~[0.30,0.60], y~[-0.12,0.24]).
_TOY_BOX = ((0.30, 0.60), (-0.10, 0.22))
_PLATE_BOX = ((0.40, 0.55), (-0.05, 0.20))
_R = 0.1125 + 0.045 + 0.02   # plate_radius + toy/block radius + margin (toys must start off the plate)

SCENES: Dict[int, SceneSpec] = {
    6: SceneSpec(
        scene_id=6,
        instruction="Place the toys on the plate with no collisions",
        rigid=["plate", "blue_toy", "brown_toy", "pink_toy"],
        articulated=[],
        randomize=[
            ObjRandom("plate", _PLATE_BOX),
            ObjRandom("blue_toy", _TOY_BOX, yaw=True, min_sep=0.05, avoid=(("plate", _R),)),
            ObjRandom("brown_toy", _TOY_BOX, yaw=True, min_sep=0.05, avoid=(("plate", _R),)),
            ObjRandom("pink_toy", _TOY_BOX, yaw=True, min_sep=0.05, avoid=(("plate", _R),)),
        ],
        evaluate=eval_place_on_plate,
        meta={"plate": "plate", "items": ["blue_toy", "brown_toy", "pink_toy"]},
    ),
    7: SceneSpec(
        scene_id=7,
        instruction="Push the yellow block into the red block.",
        rigid=["red_cube", "yellow_cube"],
        articulated=[],
        randomize=[
            ObjRandom("red_cube", ((0.40, 0.55), (0.10, 0.22)), yaw=True),
            ObjRandom("yellow_cube", ((0.40, 0.55), (-0.12, 0.04)), yaw=True, min_sep=0.10),
        ],
        evaluate=eval_push_cubes,
        meta={"mover": "yellow_cube", "stayer": "red_cube"},
        # A valid push is ordered: don't lift -> move the yellow cube -> end touching. cubes_on_table gates.
        sequence=["not_picked_up", "yellow_is_mover", "cubes_touching"],
    ),
    8: SceneSpec(
        scene_id=8,
        instruction="Use the eraser to wipe the whiteboard.",
        rigid=["whiteboard", "eraser"],
        articulated=[],
        randomize=[
            ObjRandom("whiteboard", ((0.42, 0.52), (0.02, 0.16))),          # no yaw (per design)
            ObjRandom("eraser", ((0.30, 0.40), (-0.12, 0.06)), yaw=True, avoid=(("whiteboard", 0.20),)),
        ],
        evaluate=eval_wipe,
        meta={"board": "whiteboard", "eraser": "eraser"},
        # Ordered wipe: grab -> touch the board -> wipe across it -> leave the eraser settled. board_on_table gates.
        sequence=["grabbed_eraser", "eraser_touched_board", "wiping_motion", "eraser_settled"],
    ),
    9: SceneSpec(
        scene_id=9,
        instruction="Stack the blocks together",
        rigid=["block_a", "block_b", "block_c"],
        articulated=[],
        randomize=[
            ObjRandom("block_a", _TOY_BOX, yaw=True, min_sep=0.09),
            ObjRandom("block_b", _TOY_BOX, yaw=True, min_sep=0.09),
            ObjRandom("block_c", _TOY_BOX, yaw=True, min_sep=0.09),
        ],
        evaluate=eval_stack,
        meta={"blocks": ["block_a", "block_b", "block_c"]},
    ),
    10: SceneSpec(
        scene_id=10,
        instruction="Push the red button",
        rigid=[],
        articulated=["button"],
        randomize=[ObjRandom("button", ((0.42, 0.55), (-0.08, 0.18)), yaw=True)],
        evaluate=eval_articulation,
        meta={"art": "button", "move_thresh": Geom.press_thresh, "move_crit": "button_pressed",
              "table_crit": "button_on_table"},
    ),
    11: SceneSpec(
        scene_id=11,
        instruction="Open the top drawer",
        rigid=[],
        articulated=["cabinet"],
        randomize=[ObjRandom("cabinet", ((0.45, 0.55), (-0.05, 0.12)), yaw=True, yaw_range=(-0.35, 0.35))],
        evaluate=eval_articulation,
        meta={"art": "cabinet", "move_thresh": Geom.drawer_thresh, "move_crit": "drawer_opened",
              "table_crit": "cabinet_on_table"},
    ),
    12: SceneSpec(
        scene_id=12,
        instruction="Place the blocks on the plate with no collisions",
        rigid=["plate", "block_a", "block_b", "block_c"],
        articulated=[],
        randomize=[
            ObjRandom("plate", _PLATE_BOX),
            ObjRandom("block_a", _TOY_BOX, yaw=True, min_sep=0.06, avoid=(("plate", _R),)),
            ObjRandom("block_b", _TOY_BOX, yaw=True, min_sep=0.06, avoid=(("plate", _R),)),
            ObjRandom("block_c", _TOY_BOX, yaw=True, min_sep=0.06, avoid=(("plate", _R),)),
        ],
        evaluate=eval_place_on_plate,
        meta={"plate": "plate", "items": ["block_a", "block_b", "block_c"]},
    ),
}


def get_scene(scene_id: int) -> SceneSpec:
    if scene_id not in SCENES:
        raise KeyError(f"unknown scene id {scene_id}; known: {sorted(SCENES)}")
    return SCENES[scene_id]
