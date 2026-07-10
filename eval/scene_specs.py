"""Declarative per-scene config for the parallel Pi-0.5 sim eval (scenes 6-12).

Each :class:`SceneSpec` bundles everything the eval worker needs that is *not* about spawning
physics (that lives in the ``assets/scene<N>_0.json`` sidecars):

  * ``instruction``     -- the language prompt handed to the policy.
  * ``rigid``/``articulated`` -- object names to read each step (rigid: world pose;
                          articulated: joint positions), matching the sidecar ``name`` fields.
  * ``randomize``       -- per-object start-pose randomization (XY box + optional world-Z yaw),
                          sampled per env each rollout so the 8 rollouts differ.
  * ``evaluate``        -- a pure-numpy success function (init, final, traj, spec) -> ``{criterion: 0.0|1.0}``.
                          Every criterion is an independent BOOLEAN sub-goal worth exactly one whole point --
                          multi-object checks are split into per-object booleans (no fractions), and there is
                          NO gate/dependency/sequence. Every criterion is a start-FALSE achievement (a
                          do-nothing rollout scores 0). ``score_env`` counts them: per-env score = number of
                          criteria met (a whole number 0..N); success (sparse) = 1 iff ALL are met. The worker
                          reports mean_points, max_points (=N), success_rate, and dense = mean_points/max_points.

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


# End-effector + gripper are tracked by the worker under reserved keys ("_ee": wrist-camera world
# position as an EE proxy; "_gripper": normalized finger, 0 open .. 1 closed) for the PolaRiS-ported
# reach / release criteria. Object-only evaluators (scenes 6, 10-12) never read them.
def _ee(state: dict) -> np.ndarray:
    return np.asarray(state["_ee"]["pos"], dtype=np.float64)


def _grip(state: dict) -> float:
    return float(state["_gripper"]["val"])


# --------------------------------------------------------------------------- #
# Per-scene geometry / thresholds                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Geom:
    # place-on-plate (scenes 6, 12)
    plate_xy_tol: float = 0.03
    plate_z_tol: float = 0.02
    plate_ang_tol_deg: float = 12.0
    on_plate_xy: float = 0.11            # default; scenes 6/12 override via meta (plate_radius - item_halfwidth)
    place_lift: float = 0.008            # final-minus-init z rise for a toy/block to count as placed ONTO the plate
    lift_thresh: float = 0.012           # (legacy, unused by the new place rubric)
    floor_drop: float = 0.05
    collision_thresh: float = 0.045
    # reach / grasp / release -- shared by the PolaRiS-ported scenes 8 & 9 (mirrors PolaRiS's
    # reach / lift / is_within_xy checkers, adapted to our trajectory + wrist-cam-EE framework).
    reach_thresh: float = 0.05           # TCP (fingertip midpoint) within this of an object = "reached"
                                         #   (matches PolaRiS reach() default 0.05 m; keyed off the true
                                         #   grasp point, NOT the wrist camera -- see eval_worker._ee)
    grab_lift: float = 0.04              # object must rise this far off its start z = "grabbed" (lifted); kept
                                         #   modest (PolaRiS's lift threshold is small) so it never under-credits
                                         #   a genuine pick-and-place that is a prerequisite for the *_in_target step
    grip_open: float = 0.20              # normalized finger (0 open .. 1 closed) below this = gripper released
    # pan-clean (scene 8 = DROID-PanClean): sponge -> frying pan
    pan_xy_radius: float = 0.14          # sponge center within this of the pan center = "in the pan"
    # block-stack-kitchen (scene 9 = DROID-BlockStackKitchen): two cubes -> green tray, green on wood
    tray_xy_radius: float = 0.12         # cube center within this of the tray center = "in the tray"
    stack_xy: float = 0.045              # the two cubes' centers within this (xy) = stacked/aligned
    stack_dz: float = 0.03              # one cube at least this far above the other = stacked (not side-by-side)
    # articulation (scenes 10, 11)
    press_thresh: float = 0.0012         # button press joint |disp| (m) to count as "pressed"
    drawer_thresh: float = 0.03          # cabinet drawer slide |disp| (m) to count as "opened"
    base_move_tol: float = 0.04          # anchored base may drift this much and still be "on the table"


# --------------------------------------------------------------------------- #
# Evaluators (init, final, traj, spec) -> ordered {criterion: bool}             #
# --------------------------------------------------------------------------- #
def eval_place_on_plate(init, final, traj, spec) -> Dict[str, float]:
    """Scenes 6 & 12: place each item on the plate. Independent boolean whole-point criteria (no gate, no
    partial, no start-satisfiable free points). ALL criteria are MAX-OVER-TRAJECTORY ("ever achieved") -- a
    sub-goal reached at ANY step counts, even if physics later undoes it:

      <item>_on_plate  : at SOME step the item was lifted onto the plate AND fully over the plate (measured vs
                         the plate's pose AT THAT STEP) AND the gripper released --
                         (z(s) - init_z) > place_lift AND ||item_xy(s) - plate_xy(s)|| < on_plate_xy AND
                         _grip(s) < grip_open, for any s in traj+[final]. The release + lift terms mean a toy
                         merely CARRIED over the plate in a closed gripper does not latch (gripper not open).
      <item>_lifted    : ever picked up (max rise over the episode > grab_lift) -- ranks pick-succeeds/place-fails.
      items_spaced_on_plate (optional, spec.meta["score_spacing"]) : at SOME step, >=2 items were placed and
                         every placed pair was > spaced_thresh apart -- honours the 'no collisions' clause as
                         ONE separate boolean (NOT folded into placement, which would non-monotonically zero
                         both of a touching pair).

    No plate_unmoved / on_table / gate: placement is measured against the plate's live pose, so shoving the
    plate earns no credit; a floored item simply never satisfies placement. All checks are frame-relative;
    still 0 for a no-op (nothing is lifted/over-plate/released at any step).
    """
    g = spec.geom
    plate, items = spec.meta["plate"], spec.meta["items"]
    on_plate_xy = spec.meta.get("on_plate_xy", g.on_plate_xy)
    place_lift = spec.meta.get("place_lift", g.place_lift)
    seq = list(traj) + [final]
    z0 = {it: float(_pos(init, it)[2]) for it in items}

    def placed_at(s, it):
        return ((_pos(s, it)[2] - z0[it]) > place_lift
                and float(np.linalg.norm(_pos(s, it)[:2] - _pos(s, plate)[:2])) < on_plate_xy
                and _grip(s) < g.grip_open)

    out: Dict[str, float] = {f"{it}_on_plate": float(any(placed_at(s, it) for s in seq)) for it in items}
    for it in items:
        out[f"{it}_lifted"] = float(_lift_over_traj(seq, it, z0[it]) > g.grab_lift)
    if spec.meta.get("score_spacing"):
        spaced_thresh = spec.meta.get("spaced_thresh", 2.0 * g.collision_thresh)

        def spaced_at(s):
            pl = [_pos(s, it)[:2] for it in items if placed_at(s, it)]
            return len(pl) >= 2 and all(
                float(np.linalg.norm(pl[i] - pl[j])) > spaced_thresh
                for i in range(len(pl)) for j in range(i + 1, len(pl)))
        out["items_spaced_on_plate"] = float(any(spaced_at(s) for s in seq))
    return out


def eval_pan_clean(init, final, traj, spec) -> Dict[str, float]:
    """Scene 8 = DROID-PanClean: pick up the sponge and put it in the frying pan. Boolean whole-point criteria:

      sponge_lifted : sponge ever picked up (max rise over the episode > grab_lift).
      sponge_in_pan : sponge was lifted AND at SOME step was within the pan disk in xy AND not on the floor --
                      MAX-OVER-TRAJECTORY, so a sponge that reaches the pan at any instant counts even if it is
                      later knocked out. The lift prerequisite (ever) folds in so sponge_in_pan implies
                      sponge_lifted; the floor guard re-folds the dropped on_table check.

    No gripper-release clause (unreliable finger angle on a compressible sponge), no pan_unmoved (the pan is
    kinematic -> a start-true free point). All checks frame-relative; 0/2 at t=0.
    NB: PolaRiS's reach() predicate stays dropped (no ee_frame at the grasp point; see eval_worker._ee /
    results["min_tcp_dist"]).
    """
    g = spec.geom
    sponge, pan = spec.meta["item"], spec.meta["target"]
    pan_xy_radius = spec.meta.get("pan_xy_radius", g.pan_xy_radius)
    seq = list(traj) + [final]
    z0 = float(_pos(init, sponge)[2])
    lifted = _lift_over_traj(seq, sponge, z0) > g.grab_lift
    in_pan = lifted and any(
        float(np.linalg.norm(_pos(s, sponge)[:2] - _pos(s, pan)[:2])) < pan_xy_radius
        and (_pos(s, sponge)[2] - z0) > -g.floor_drop
        for s in seq)
    return {"sponge_lifted": float(lifted), "sponge_in_pan": float(in_pan)}


def eval_block_stack_kitchen(init, final, traj, spec) -> Dict[str, float]:
    """Scene 9 = DROID-BlockStackKitchen: place both cubes in the green tray and stack them. Boolean whole-point:

      <cube>_lifted   : cube ever picked up (max rise over the episode > grab_lift).
      <cube>_in_tray  : cube was lifted AND at SOME step was over the tray footprint with the gripper released
                        (lift folded in, so a pushed-never-picked cube does not count).
      blocks_stacked  : at SOME step, one cube above the other (|dz| > stack_dz) AND xy-aligned (< stack_xy)
                        AND the LOWER cube over the tray footprint AND released -- so a stack must be ON the
                        tray, not merely somewhere on the table.

    ALL criteria are MAX-OVER-TRAJECTORY ("ever achieved"): a cube placed in the tray / a stack formed at any
    instant counts even if it is later knocked over. No tray_unmoved (tray is kinematic) / no on_table gate.
    All checks frame-relative; 0/5 at t=0.
    """
    g = spec.geom
    green, wood, tray = spec.meta["green"], spec.meta["wood"], spec.meta["target"]
    tray_r = spec.meta.get("tray_xy_radius", g.tray_xy_radius)
    seq = list(traj) + [final]

    def over_tray(s, name: str) -> bool:
        return float(np.linalg.norm(_pos(s, name)[:2] - _pos(s, tray)[:2])) < tray_r

    green_lifted = _lift_over_traj(seq, green, float(_pos(init, green)[2])) > g.grab_lift
    wood_lifted = _lift_over_traj(seq, wood, float(_pos(init, wood)[2])) > g.grab_lift

    def stacked_at(s) -> bool:
        gs, ws = _pos(s, green), _pos(s, wood)
        lower = green if gs[2] <= ws[2] else wood
        return (abs(gs[2] - ws[2]) > g.stack_dz
                and float(np.linalg.norm(gs[:2] - ws[:2])) < g.stack_xy
                and over_tray(s, lower) and _grip(s) < g.grip_open)

    return {
        "green_lifted": float(green_lifted),
        "wood_lifted": float(wood_lifted),
        "green_in_tray": float(green_lifted and any(over_tray(s, green) and _grip(s) < g.grip_open for s in seq)),
        "wood_in_tray": float(wood_lifted and any(over_tray(s, wood) and _grip(s) < g.grip_open for s in seq)),
        "blocks_stacked": float(any(stacked_at(s) for s in seq)),
    }


def eval_articulation(init, final, traj, spec) -> Dict[str, float]:
    """Scenes 10 & 11: actuate ONE named DOF of a world-anchored articulation. Two nested thresholds give a
    0/1/2 ordinal (contacted/cracked < pressed/opened) so the scene has real partial-credit resolution.

    The DOF is resolved by NAME (``meta["target_joint"]``) from the articulation's joint_names (robust to
    MJCF->USD reordering), and only that DOF is scored -- so opening the WRONG drawer does NOT count. Both the
    button press and the drawer open move the joint in the NEGATIVE direction from its settled rest, so the
    actuation displacement is ``j0 - j`` (positive when actuated); an inward push (opposite sign) never counts.

      mode "traj_excl_final" (scene 10 button, spring-return): MAX (j0 - j) over ``traj`` EXCLUDING final --
                             the spring relaxes the cap to ~0 after release, so a final-state check would miss
                             a valid press-and-release.
      mode "traj_incl_final" (scene 11 drawer, no return spring): MAX (j0 - j) over the WHOLE episode
                             (traj + final) -- MAX-OVER-TRAJECTORY, so a drawer opened at ANY instant counts
                             even if it is later pushed shut.
      mode "final": (j0 - j) at the FINAL settled state only (kept for generality; unused by scenes 10/11).

    No *_on_table criterion (the base is fix_base / world-anchored -> always true -> a start-satisfied free
    point). Frame-relative (an intrinsic joint DOF is independent of env origin, table height, base pose);
    0 at t=0 (disp == 0 when settled).
    """
    m = spec.meta
    art = m["art"]
    names = list(init[art]["joint_names"])
    idx = names.index(m["target_joint"])
    j0 = float(np.asarray(init[art]["joint"], dtype=np.float64)[idx])

    def _disp(s):
        return j0 - float(np.asarray(s[art]["joint"], dtype=np.float64)[idx])

    mode = m.get("mode", "final")
    if mode == "final":
        disp = _disp(final)
    elif mode == "traj_excl_final":
        disp = max((_disp(s) for s in traj), default=0.0)
    else:  # traj_incl_final -- ever actuated over the whole episode (non-spring joints, e.g. the drawer)
        disp = max((_disp(s) for s in list(traj) + [final]), default=0.0)
    return {m["contact_crit"]: float(disp > m["contact_thresh"]),
            m["move_crit"]: float(disp > m["move_thresh"])}


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

    @property
    def objects(self) -> List[str]:
        return self.rigid + self.articulated


def score_env(raw: Dict[str, float], spec: "SceneSpec") -> dict:
    """One env's boolean criteria -> a WHOLE-NUMBER score.

    Every criterion is an independent boolean sub-goal worth exactly one point (no gate, no partial credit,
    no dependency chaining). ``points`` = COUNT of criteria met (an integer 0..N); ``success`` = 1 iff ALL
    criteria are met. The ``>= 0.5`` collapse is a defensive guard so that even if an evaluator ever returned
    a fractional value the per-rollout score stays an integer. Every declared criterion is a start-FALSE
    achievement (a do-nothing rollout scores 0), so the count is not inflated by preserved start conditions.
    """
    met = {c: (1 if float(raw[c]) >= 0.5 else 0) for c in raw}
    points = int(sum(met.values()))
    max_points = len(met)
    return {"points": points, "max_points": max_points,
            "success": 1 if (max_points > 0 and points == max_points) else 0, "met": met}


# Reachable-workspace XY boxes (env-local / robot frame, metres) live in each ObjRandom below.
# They mirror the scene6 sidecar's placement region (x~[0.30,0.60], y~[-0.12,0.24]).
_TOY_BOX = ((0.30, 0.60), (-0.10, 0.22))
_PLATE_BOX = ((0.40, 0.55), (-0.05, 0.20))
_R = 0.1125 + 0.045 + 0.02   # plate_radius + toy/block radius + margin (toys must start off the plate)
# PolaRiS-ported scenes (8, 9). Boxes keep the pick objects off the fixed target and in the FOV;
# see the assets/scene8_0.json / scene9_0.json sidecars for the matching object layout.
_SPONGE_BOX = ((0.31, 0.36), (-0.11, -0.01))    # scene 8: sponge start region (front-left corner, clear of
                                                #   the pan-avoid ring AND the back-left distractors)
_TRAY_BOX = ((0.47, 0.52), (0.06, 0.13))        # scene 9: light tray jitter (kinematic target)
_CUBE_BOX = ((0.31, 0.41), (-0.10, 0.10))       # scene 9: cube start region (left of the tray)
_TRAY_R = 0.15                                   # cubes must start this far from the tray centre (off it)

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
        # on_plate_xy = plate_radius(~0.1125) - toy_halfwidth(~0.045); toys are scale-0.045 rigid.
        # Criteria (max 7): {blue,brown,pink}_toy_on_plate, {..}_lifted, items_spaced_on_plate.
        meta={"plate": "plate", "items": ["blue_toy", "brown_toy", "pink_toy"],
              "on_plate_xy": 0.067, "place_lift": 0.008, "score_spacing": True, "spaced_thresh": 0.09},
    ),
    # Scene 7 (push cubes) was removed. Scenes 8 & 9 are ported from PolaRiS (owhan/PolaRiS-Hub):
    8: SceneSpec(
        scene_id=8,
        instruction="Use the yellow sponge to scrub the blue handle frying pan.",
        rigid=["pan", "sponge"],            # the pan is a fixed kinematic target; distractors (sidecar) are untracked
        articulated=[],
        randomize=[
            # Only the sponge is randomized (PolaRiS pan_clean varies only the sponge pose); it must start
            # off the pan and in the front-left reach region.
            ObjRandom("sponge", _SPONGE_BOX, yaw=True, avoid=(("pan", 0.15),)),
        ],
        evaluate=eval_pan_clean,
        # pan_xy_radius tightened toward the true pan interior (scale 0.35). Criteria: sponge_lifted, sponge_in_pan.
        meta={"item": "sponge", "target": "pan", "pan_xy_radius": 0.12},
    ),
    9: SceneSpec(
        scene_id=9,
        instruction="Place and stack the blocks on top of the green tray.",
        rigid=["tray", "green_cube", "wood_cube"],   # tray = fixed kinematic target; distractors untracked
        articulated=[],
        randomize=[
            ObjRandom("tray", _TRAY_BOX),                                              # light target jitter (placed first)
            ObjRandom("green_cube", _CUBE_BOX, yaw=True, min_sep=0.09, avoid=(("tray", _TRAY_R),)),
            ObjRandom("wood_cube", _CUBE_BOX, yaw=True, min_sep=0.09, avoid=(("tray", _TRAY_R),)),
        ],
        evaluate=eval_block_stack_kitchen,
        # tray_xy_radius shrunk from 0.12 (isotropic) toward the tray's short half-extent so a cube pushed
        # against the short edge onto bare table no longer counts. Criteria: {green,wood}_lifted,
        # {green,wood}_in_tray, blocks_stacked (stack must sit over the tray).
        meta={"green": "green_cube", "wood": "wood_cube", "target": "tray", "tray_xy_radius": 0.085},
    ),
    10: SceneSpec(
        scene_id=10,
        instruction="Push the red button",
        rigid=[],
        articulated=["button"],
        randomize=[ObjRandom("button", ((0.42, 0.55), (-0.08, 0.18)), yaw=True)],
        evaluate=eval_articulation,
        # Single prismatic DOF 'press_joint' (range -8..0 authored x scale ~3mm; press = negative). Spring
        # returns the cap after release -> score MAX over traj EXCLUDING final. Nested: contacted < pressed.
        meta={"art": "button", "target_joint": "press_joint", "mode": "traj_excl_final",
              "move_crit": "button_pressed", "move_thresh": Geom.press_thresh,
              "contact_crit": "button_contacted", "contact_thresh": Geom.press_thresh / 3.0},
    ),
    11: SceneSpec(
        scene_id=11,
        instruction="Open the top drawer",
        rigid=[],
        articulated=["cabinet"],
        randomize=[ObjRandom("cabinet", ((0.45, 0.55), (-0.05, 0.12)), yaw=True, yaw_range=(-0.35, 0.35))],
        evaluate=eval_articulation,
        # ONLY the top drawer's slide DOF 'top_level' is scored (opening middle/bottom must NOT count).
        # Max-over-trajectory (ever opened) -> credit an open at any instant. Open = negative. Nested: cracked < opened.
        meta={"art": "cabinet", "target_joint": "top_level", "mode": "traj_incl_final",
              "move_crit": "top_drawer_opened", "move_thresh": Geom.drawer_thresh,
              "contact_crit": "drawer_cracked", "contact_thresh": Geom.drawer_thresh / 3.0},
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
        # 0.05 m cubes on a convexDecomposition plate (concave rim survives -> lower inner floor than scene 6's
        # convexHull, so place_lift is smaller). on_plate_xy = plate_radius - block_halfwidth(~0.025).
        # Criteria (max 7): {block_a,b,c}_on_plate, {..}_lifted, items_spaced_on_plate.
        meta={"plate": "plate", "items": ["block_a", "block_b", "block_c"],
              "on_plate_xy": 0.088, "place_lift": 0.005, "score_spacing": True, "spaced_thresh": 0.052},
    ),
}


def get_scene(scene_id: int) -> SceneSpec:
    if scene_id not in SCENES:
        raise KeyError(f"unknown scene id {scene_id}; known: {sorted(SCENES)}")
    return SCENES[scene_id]
