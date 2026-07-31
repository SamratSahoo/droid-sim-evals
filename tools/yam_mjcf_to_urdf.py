#!/usr/bin/env python3
"""Convert the bimanual-YAM MuJoCo MJCF into a URDF (plus cuRobo collision spheres).

Why not ``mjcf-urdf-simple-converter``: that package drives ``dm_control`` and emits one URDF link
per *geom* with box/sphere primitives only, dropping the visual meshes and renaming every body. We
need the link/joint names to survive the round-trip, because the SAME names are used by three
consumers that have to agree:

  * cuRobo / cuTAMP  -- reads this URDF for kinematics + the ``*.yml`` collision spheres,
  * Isaac Sim        -- imports this URDF to USD (``tools/build_yam_assets.py``), and the resulting
                        articulation's joint names are what ``JointPositionAction`` matches on,
  * the eval driver  -- maps cuTAMP's 6 arm waypoints onto Isaac joints by name.

So we do the conversion directly. The YAM MJCF is a plain tree (every joint is ``hinge`` or
``slide`` with ``pos="0 0 0"``, every mesh is an OBJ), which makes the mapping exact:

  MJCF body pos/quat (relative to parent body)  ->  URDF <joint><origin>
  MJCF joint axis (child body frame)            ->  URDF <joint><axis>   (same frame)
  MJCF hinge/slide + range                      ->  URDF revolute/prismatic + <limit>

Inertials are read from the *compiled* ``mujoco.MjModel`` rather than the XML text, so bodies whose
``<inertial>`` MuJoCo infers from geom density (``left_link_6`` and the four finger bodies) get the
exact mass/inertia MuJoCo itself uses instead of a guess.

Collision geoms are capsules/boxes/spheres, which is a gift: capsules convert to URDF cylinders
(Isaac's ``replace_cylinders_with_capsules`` turns them back into capsules) AND they sample directly
into the ``[x, y, z, radius]`` spheres that cuRobo wants, with no mesh-approximation step.

Usage (from the droid-sim-evals directory)::

    python tools/yam_mjcf_to_urdf.py --out ../cuTAMP/cutamp/robots/assets/yam_description

Needs ``mujoco`` and ``numpy``; neither is a droid-sim-evals runtime dependency, so run this with any
interpreter that has them (see tools/README.md).
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HF_ASSET_REPO = "TreeePlanter/molmoact2-sim-eval-assets"
MJCF_REL = "assets/yam/yam_mujoco/bimanual_yam_linear_flattened.xml"

# MuJoCo's <default class="*_visual"> sets density=0, so visual geoms contribute no mass; the
# collision capsules/boxes carry MuJoCo's default 1000 kg/m^3. We keep MuJoCo's compiled values.
ARMS = ("left", "right")

# The two arm-base housings are rigidly fixed to the world (via bimanual_base), so collision-checking
# them buys nothing the planner can act on -- and with the robot resting ON the table their capsules,
# which extend 1.7 cm below the mount plane in the MJCF, would intersect the table slab and make EVERY
# configuration invalid. Dropped from the cuRobo collision set for the same reason ur5e_robotiq_2f_85.yml
# omits `base_link`.
FIXED_BASE_LINKS = frozenset(f"{arm}_arm" for arm in ARMS)

# Where the shared base sits in the droid-sim-evals world frame (the frame scene_specs and the
# scene<N>.json sidecars use; the Franka's base is its origin and the table top is at z = 0.0451).
#
# Chosen by sweeping cuRobo IK over the scene-6 object region: a 5x5 grid of grasp sites spanning
# x[0.30,0.60] y[-0.10,0.22] at z=0.065, each with 8 top-down orientations (jaw yaw over [0,pi), the
# full set for a parallel jaw). This pose reaches all 25 sites with the LEFT arm, 57% of the 200
# poses, and never fewer than 3 yaws at any single site. For comparison: mounting at the origin like
# the Franka reaches only 14/25 sites, and neighbouring candidates trade a few percent either way.
# Top-down is the tightest orientation constraint a grasp can have, so this is a lower bound on what
# cuTAMP (which samples full 6-DOF M2T2 grasps) actually gets.
#
# The -0.18 in y centres the left arm, which sits at +0.24 on the base, over the object region's
# y-centre of +0.06. The base ends up 0.205 m above the table top, just past its near edge.
MOUNT_XYZ = (0.24, 0.0, 0.0551)


# --------------------------------------------------------------------------- #
# Small transform helpers (MJCF and URDF are both right-handed, metres, radians) #
# --------------------------------------------------------------------------- #
def quat_to_mat(q: Sequence[float]) -> np.ndarray:
    """wxyz quaternion -> 3x3 rotation matrix (MuJoCo/USD convention)."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """3x3 rotation -> URDF fixed-axis roll/pitch/yaw (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) > 1.0 - 1e-9:  # gimbal lock: fold roll into yaw
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return float(roll), float(pitch), float(yaw)


def quat_to_rpy(q: Sequence[float]) -> Tuple[float, float, float]:
    return mat_to_rpy(quat_to_mat(q))


def fmt(vals: Sequence[float]) -> str:
    return " ".join(f"{float(v):.9g}" for v in vals)


def parse_vec(text: Optional[str], default: Sequence[float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    return np.asarray([float(v) for v in text.split()], dtype=float)


# --------------------------------------------------------------------------- #
# MJCF parsing                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Geom:
    """One MJCF <geom>, resolved through its `class` default."""

    gtype: str                  # "mesh" | "capsule" | "box" | "sphere" | "cylinder"
    pos: np.ndarray
    quat: np.ndarray
    size: np.ndarray
    mesh: Optional[str] = None
    rgba: Optional[Tuple[float, float, float, float]] = None
    is_visual: bool = False


@dataclass
class Body:
    name: str
    parent: Optional[str]
    pos: np.ndarray
    quat: np.ndarray
    joint_name: Optional[str] = None
    joint_type: Optional[str] = None       # "hinge" | "slide"
    joint_axis: Optional[np.ndarray] = None
    joint_range: Optional[np.ndarray] = None
    geoms: List[Geom] = field(default_factory=list)
    sites: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


def _resolve_defaults(root: ET.Element) -> Dict[str, dict]:
    """Flatten <default class=...> into {class_name: {geom-attrs}} (only what we need: geom type)."""
    out: Dict[str, dict] = {}

    def walk(node: ET.Element, inherited: dict) -> None:
        cls = node.get("class")
        merged = dict(inherited)
        geom = node.find("geom")
        if geom is not None:
            merged.update(geom.attrib)
        if cls:
            out[cls] = merged
        for child in node.findall("default"):
            walk(child, merged)

    for d in root.findall("default"):
        walk(d, {})
    return out


def _materials(root: ET.Element) -> Dict[str, Tuple[float, float, float, float]]:
    mats: Dict[str, Tuple[float, float, float, float]] = {}
    for m in root.iterfind("asset/material"):
        rgba = parse_vec(m.get("rgba"), (0.5, 0.5, 0.5, 1.0))
        mats[m.get("name", "")] = tuple(float(v) for v in rgba)  # type: ignore[assignment]
    return mats


def _mesh_files(root: ET.Element) -> Dict[str, str]:
    return {m.get("name", ""): m.get("file", "") for m in root.iterfind("asset/mesh")}


def parse_mjcf(mjcf_path: Path) -> Tuple[Dict[str, Body], List[str], Dict[str, str]]:
    """Parse the MJCF into an ordered body dict, a root-order list, and the mesh-name -> file map."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    defaults = _resolve_defaults(root)
    materials = _materials(root)
    mesh_files = _mesh_files(root)

    bodies: Dict[str, Body] = {}
    order: List[str] = []

    def geom_of(el: ET.Element, childclass: Optional[str]) -> Geom:
        cls = el.get("class") or childclass
        base = dict(defaults.get(cls, {})) if cls else {}
        base.update(el.attrib)
        gtype = base.get("type", "sphere")
        rgba = None
        mat = base.get("material")
        if mat and mat in materials:
            rgba = materials[mat]
        elif base.get("rgba"):
            rgba = tuple(float(v) for v in base["rgba"].split())  # type: ignore[assignment]
        # A geom is "visual" when MuJoCo excludes it from contacts (contype=conaffinity=0).
        is_visual = base.get("contype") == "0" and base.get("conaffinity") == "0"
        return Geom(
            gtype=gtype,
            pos=parse_vec(base.get("pos"), (0, 0, 0)),
            quat=parse_vec(base.get("quat"), (1, 0, 0, 0)),
            size=parse_vec(base.get("size"), (0.01,)),
            mesh=base.get("mesh"),
            rgba=rgba,
            is_visual=is_visual,
        )

    def walk(el: ET.Element, parent: Optional[str], childclass: Optional[str]) -> None:
        name = el.get("name")
        if name is None:
            raise ValueError("every MJCF <body> in this model is expected to be named")
        cc = el.get("childclass") or childclass
        body = Body(
            name=name,
            parent=parent,
            pos=parse_vec(el.get("pos"), (0, 0, 0)),
            quat=parse_vec(el.get("quat"), (1, 0, 0, 0)),
        )
        joints = el.findall("joint")
        if len(joints) > 1:
            raise ValueError(f"body {name!r} has {len(joints)} joints; only 0 or 1 is supported")
        if joints:
            j = joints[0]
            jcls = j.get("class") or cc
            jtype = j.get("type")
            if jtype is None:
                # MJCF default joint type is "hinge"; the finger class overrides it to "slide".
                jtype = "slide" if (jcls and "finger" in jcls) else "hinge"
            if not np.allclose(parse_vec(j.get("pos"), (0, 0, 0)), 0.0):
                raise ValueError(f"joint {j.get('name')!r} has a non-zero pos; URDF needs it at the link origin")
            body.joint_name = j.get("name")
            body.joint_type = jtype
            body.joint_axis = parse_vec(j.get("axis"), (0, 0, 1))
            body.joint_range = parse_vec(j.get("range"), (0.0, 0.0))
        for g in el.findall("geom"):
            body.geoms.append(geom_of(g, cc))
        for s in el.findall("site"):
            body.sites[s.get("name", "")] = (
                parse_vec(s.get("pos"), (0, 0, 0)),
                parse_vec(s.get("quat"), (1, 0, 0, 0)),
            )
        bodies[name] = body
        order.append(name)
        for child in el.findall("body"):
            walk(child, name, cc)

    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"{mjcf_path} has no <worldbody>")
    for top in world.findall("body"):
        walk(top, None, None)
    return bodies, order, mesh_files


# --------------------------------------------------------------------------- #
# Inertials, taken from the compiled MuJoCo model so inferred ones are exact     #
# --------------------------------------------------------------------------- #
def compiled_inertials(mjcf_path: Path) -> Dict[str, Tuple[float, np.ndarray, np.ndarray]]:
    """{body_name: (mass, com_in_body_frame, 3x3 inertia about the com in body axes)}."""
    import mujoco  # optional dependency; only needed to regenerate the URDF

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    out: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name is None:
            continue
        R = quat_to_mat(model.body_iquat[bid])
        # MuJoCo stores the inertia diagonal in the principal frame given by body_iquat; rotate it
        # into body axes so URDF can carry it with a zero-rotation <origin>.
        inertia = R @ np.diag(model.body_inertia[bid]) @ R.T
        out[name] = (float(model.body_mass[bid]), np.asarray(model.body_ipos[bid], dtype=float), inertia)
    return out


def actuator_efforts(mjcf_path: Path) -> Dict[str, float]:
    """{joint_name: |forcerange| (Nm or N)} from the MJCF actuators, for the URDF <limit effort=>."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    out: Dict[str, float] = {}
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        lo, hi = model.actuator_forcerange[aid]
        if name and hi > lo:
            out[name] = float(max(abs(lo), abs(hi)))
    return out


def measure_jaws(mjcf_path: Path, arm: str, finger_open: float) -> Tuple[float, float, float]:
    """Measure one hand at full open, in its ``<arm>_link_6`` frame.

    Returns ``(tip_z, pad_base_z, half_opening)``: the distal fingertip plane, the proximal end of
    the pads, and the half-distance between the two jaws -- all read off the MJCF finger-pad
    colliders under MuJoCo forward kinematics, so they track the model rather than a hand-copied
    constant. Also asserts the jaws separate along the link_6 X axis, which is what makes the
    Robotiq's ``tool_from_ee`` rotation reusable verbatim (see cutamp/robots/bimanual_yam.py).
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    for side in ("left", "right"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{side}_finger")
        data.qpos[model.jnt_qposadr[jid]] = finger_open
    mujoco.mj_kinematics(model, data)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_link_6")
    T6 = np.eye(4)
    T6[:3, :3] = data.xmat[bid].reshape(3, 3)
    T6[:3, 3] = data.xpos[bid]
    inv = np.linalg.inv(T6)

    corners: List[np.ndarray] = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not name or not name.startswith(f"{arm}_") or "tip_collider" not in name:
            continue
        c = (inv @ np.append(data.geom_xpos[gid], 1.0))[:3]
        R = inv[:3, :3] @ data.geom_xmat[gid].reshape(3, 3)
        half = model.geom_size[gid]
        corners += [c + R @ (np.array([sx, sy, sz]) * half)
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    if not corners:
        raise ValueError(f"no {arm}_*_tip_collider geoms found in {mjcf_path}")
    pts = np.asarray(corners)
    half_opening = float(max(abs(pts[:, 0].min()), pts[:, 0].max()))
    if half_opening < abs(pts[:, 1]).max():
        raise ValueError("expected the YAM jaws to separate along the link_6 X axis")
    return float(pts[:, 2].max()), float(pts[:, 2].min()), half_opening


# --------------------------------------------------------------------------- #
# Collision-sphere extraction (feeds the cuRobo yml)                             #
# --------------------------------------------------------------------------- #
def _sweep(
    center: np.ndarray,
    R: np.ndarray,
    axis: int,
    half_len: float,
    cross_radius: float,
    max_samples: int,
) -> List[Tuple[np.ndarray, float]]:
    """Cover a segment-swept cross-section with ``n <= max_samples`` spheres, with NO gaps.

    ``n`` spheres spaced evenly over ``[-half_len, +half_len]`` sit ``step = 2*half_len/(n-1)``
    apart. A sphere of radius ``cross_radius`` at each sample leaves the midpoints uncovered, so
    the radius is grown to ``hypot(cross_radius, step/2)`` -- the exact radius at which consecutive
    spheres meet on the swept surface. ``n`` is chosen so that inflation stays under ~25% and then
    clamped, which trades a slightly fatter model for far fewer spheres (cuRobo cost scales with
    sphere count, and the reference configs use ~60-80 for a whole arm).
    """
    n = int(np.clip(np.ceil(2 * half_len / max(1.5 * cross_radius, 1e-9)) + 1, 2, max_samples))
    step = 2 * half_len / (n - 1)
    radius = float(np.hypot(cross_radius, step / 2))
    out: List[Tuple[np.ndarray, float]] = []
    for t in np.linspace(-half_len, half_len, n):
        offset = np.zeros(3)
        offset[axis] = t
        out.append((center + R @ offset, radius))
    return out


def spheres_for_geom(g: Geom, max_samples: int = 6) -> List[Tuple[np.ndarray, float]]:
    """Approximate one collision geom by spheres in the owning body's frame (never under-covering)."""
    R = quat_to_mat(g.quat)
    if g.gtype in ("capsule", "cylinder"):
        # MJCF capsule/cylinder size is [radius, half_length] along the geom's local +z.
        return _sweep(g.pos, R, 2, float(g.size[1]), float(g.size[0]), max_samples)
    if g.gtype == "box":
        # Sweep along the box's longest axis so a thin flat finger pad is not blown up into one fat
        # sphere; the cross-section radius is the half-diagonal of the other two half-extents.
        half = np.asarray(g.size[:3], dtype=float)
        long_axis = int(np.argmax(half))
        others = [i for i in range(3) if i != long_axis]
        return _sweep(g.pos, R, long_axis, float(half[long_axis]), float(np.linalg.norm(half[others])), max_samples)
    if g.gtype == "sphere":
        return [(g.pos.copy(), float(g.size[0]))]
    raise ValueError(f"unsupported collision geom type {g.gtype!r}")


def body_collision_spheres(body: Body, max_samples: int = 6) -> List[Tuple[np.ndarray, float]]:
    out: List[Tuple[np.ndarray, float]] = []
    for g in body.geoms:
        if g.is_visual or g.gtype == "mesh":
            continue
        out.extend(spheres_for_geom(g, max_samples))
    return out


# --------------------------------------------------------------------------- #
# URDF emission                                                                 #
# --------------------------------------------------------------------------- #
def _origin(parent: ET.Element, pos: Sequence[float], quat: Sequence[float]) -> None:
    ET.SubElement(parent, "origin", xyz=fmt(pos), rpy=fmt(quat_to_rpy(quat)))


def _geometry(parent: ET.Element, g: Geom, mesh_files: Dict[str, str], mesh_dir: str) -> bool:
    """Append a <geometry> for ``g``. Returns False if the geom has no URDF equivalent."""
    geom_el = ET.SubElement(parent, "geometry")
    if g.gtype == "mesh":
        fname = mesh_files.get(g.mesh or "", "")
        if not fname:
            parent.remove(geom_el)
            return False
        ET.SubElement(geom_el, "mesh", filename=f"{mesh_dir}/{fname}")
    elif g.gtype in ("capsule", "cylinder"):
        # URDF has no capsule primitive. A cylinder of the same radius/length is the standard
        # stand-in; Isaac's UrdfConverterCfg(replace_cylinders_with_capsules=True) restores the
        # rounded caps, and cuRobo ignores URDF collision geometry entirely (it uses the yml
        # spheres), so nothing downstream is misled.
        ET.SubElement(geom_el, "cylinder", radius=f"{float(g.size[0]):.9g}", length=f"{2 * float(g.size[1]):.9g}")
    elif g.gtype == "box":
        ET.SubElement(geom_el, "box", size=fmt(2 * np.asarray(g.size[:3], dtype=float)))
    elif g.gtype == "sphere":
        ET.SubElement(geom_el, "sphere", radius=f"{float(g.size[0]):.9g}")
    else:
        parent.remove(geom_el)
        return False
    return True


def build_urdf(
    bodies: Dict[str, Body],
    order: List[str],
    mesh_files: Dict[str, str],
    inertials: Dict[str, Tuple[float, np.ndarray, np.ndarray]],
    *,
    robot_name: str = "bimanual_yam",
    mesh_dir: str = "meshes",
    grasp_frame_z: Optional[Dict[str, float]] = None,
    mount_xyz: Sequence[float] = MOUNT_XYZ,
    efforts: Optional[Dict[str, float]] = None,
) -> ET.ElementTree:
    efforts = efforts or {}
    robot = ET.Element("robot", name=robot_name)
    # XML forbids a double hyphen inside a comment, so no em-dashes here.
    robot.append(ET.Comment(
        " Generated by droid-sim-evals/tools/yam_mjcf_to_urdf.py from the MolmoAct2 bimanual-YAM"
        f" MJCF ({MJCF_REL} of {HF_ASSET_REPO}). Do not edit by hand; re-run the converter. "
    ))

    # A `world` root with a fixed mount joint, the usual ROS pattern. It is load-bearing here: it
    # makes the cuRobo base_link coincide with the SIM WORLD origin instead of the robot's own base,
    # so every world-frame convention the Franka pipeline already relies on carries over untouched --
    # scene_specs object coordinates, the workspace cuboids, M2T2's table-plane bounds, and the
    # `world_from_cam` extrinsic the client puts on the wire. Isaac then spawns the robot at the
    # origin and the mount offset rides along inside the articulation, so the planner and the
    # simulator cannot disagree about where the arms are.
    ET.SubElement(robot, "link", name="world")
    mount = ET.SubElement(robot, "joint", name="bimanual_base_mount", type="fixed")
    ET.SubElement(mount, "parent", link="world")
    ET.SubElement(mount, "child", link="bimanual_base")
    ET.SubElement(mount, "origin", xyz=fmt(mount_xyz), rpy="0 0 0")

    for name in order:
        body = bodies[name]
        link = ET.SubElement(robot, "link", name=name)

        mass, com, inertia = inertials.get(name, (0.0, np.zeros(3), np.zeros((3, 3))))
        if mass > 0.0:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz=fmt(com), rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
            ET.SubElement(
                inertial, "inertia",
                ixx=f"{inertia[0, 0]:.9g}", ixy=f"{inertia[0, 1]:.9g}", ixz=f"{inertia[0, 2]:.9g}",
                iyy=f"{inertia[1, 1]:.9g}", iyz=f"{inertia[1, 2]:.9g}", izz=f"{inertia[2, 2]:.9g}",
            )

        for g in body.geoms:
            if g.is_visual or g.gtype == "mesh":
                visual = ET.SubElement(link, "visual")
                _origin(visual, g.pos, g.quat)
                if not _geometry(visual, g, mesh_files, mesh_dir):
                    link.remove(visual)
                    continue
                if g.rgba is not None:
                    mat = ET.SubElement(visual, "material", name=f"{name}_mat")
                    ET.SubElement(mat, "color", rgba=fmt(g.rgba))
            else:
                collision = ET.SubElement(link, "collision")
                _origin(collision, g.pos, g.quat)
                if not _geometry(collision, g, mesh_files, mesh_dir):
                    link.remove(collision)

        if body.parent is None:
            continue
        if body.joint_name is None:
            joint = ET.SubElement(robot, "joint", name=f"{name}_fixed_joint", type="fixed")
        else:
            jtype = "revolute" if body.joint_type == "hinge" else "prismatic"
            joint = ET.SubElement(robot, "joint", name=body.joint_name, type=jtype)
        ET.SubElement(joint, "parent", link=body.parent)
        ET.SubElement(joint, "child", link=name)
        _origin(joint, body.pos, body.quat)
        if body.joint_name is not None:
            assert body.joint_axis is not None and body.joint_range is not None
            ET.SubElement(joint, "axis", xyz=fmt(body.joint_axis))
            lo, hi = float(body.joint_range[0]), float(body.joint_range[1])
            # Effort comes from the MJCF actuator's forcerange, so the dm4340 shoulder joints (28 Nm)
            # and the dm4310 wrist joints (10 Nm) are not flattened to one number. Isaac reads these
            # for its joint drives; cuRobo reads only the position limits.
            effort = efforts.get(body.joint_name, 28.0 if body.joint_type == "hinge" else 40.0)
            velocity = 3.14 if body.joint_type == "hinge" else 0.2
            ET.SubElement(joint, "limit", lower=f"{lo:.9g}", upper=f"{hi:.9g}",
                          effort=f"{effort:.9g}", velocity=f"{velocity:.9g}")
            # MJCF models joint friction as `frictionloss`, with no viscous damping term.
            ET.SubElement(joint, "dynamics", damping="0.0", friction="0.1")

    if grasp_frame_z:
        # cuTAMP plans to a named ee_link, so each hand gets a massless `<arm>_grasp_frame`, the
        # analogue of `grasp_frame` in ur5e_robotiq_2f_85.urdf / fr3_robotiq_2f_85.urdf.
        #
        # Placement matters and is NOT the MJCF's own `*_grasp_site`: cuTAMP composes
        # `world_from_ee = world_from_grasp @ tool_from_ee`, and both existing robots put the ee
        # origin exactly on the DISTAL FINGERTIP PLANE with +z along the approach and x along the
        # jaw axis (verified: the Panda's tool_from_ee z-offset 0.105 lands on its 0.1034 fingertip
        # plane, and the Robotiq's pad meshes end at z = -0.0007 in its grasp_frame). link_6's own
        # axes already satisfy that (jaws separate along +/-x, pads extend along +z), so the frame
        # is a pure translation up the approach axis and the Robotiq's tool_from_ee rotation carries
        # over unchanged. The MJCF grasp_site would instead add a spurious -90deg roll about z.
        for arm in ARMS:
            ET.SubElement(robot, "link", name=f"{arm}_grasp_frame")
            joint = ET.SubElement(robot, "joint", name=f"{arm}_grasp_frame_joint", type="fixed")
            ET.SubElement(joint, "parent", link=f"{arm}_link_6")
            ET.SubElement(joint, "child", link=f"{arm}_grasp_frame")
            _origin(joint, (0.0, 0.0, grasp_frame_z[arm]), (1.0, 0.0, 0.0, 0.0))

    ET.indent(robot, space="  ")
    return ET.ElementTree(robot)


# --------------------------------------------------------------------------- #
# cuRobo collision-sphere yml fragment                                          #
# --------------------------------------------------------------------------- #
def collision_sphere_yaml(
    bodies: Dict[str, Body], order: List[str], max_samples: int = 6, indent: str = "      "
) -> str:
    """Render the `collision_spheres:` block of a cuRobo robot yml for every collidable link."""
    lines: List[str] = []
    for name in order:
        if name in FIXED_BASE_LINKS:
            continue
        spheres = body_collision_spheres(bodies[name], max_samples)
        if not spheres:
            continue
        lines.append(f"{indent}{name}:")
        for center, radius in spheres:
            lines.append(f"{indent}  - center: [{center[0]:.5f}, {center[1]:.5f}, {center[2]:.5f}]")
            lines.append(f"{indent}    radius: {radius:.5f}")
    return "\n".join(lines)


def collidable_links(bodies: Dict[str, Body], order: List[str], max_samples: int = 6) -> List[str]:
    return [n for n in order if n not in FIXED_BASE_LINKS and body_collision_spheres(bodies[n], max_samples)]


# --------------------------------------------------------------------------- #
# cuRobo robot config (one per arm)                                             #
# --------------------------------------------------------------------------- #
# The gripper is locked OPEN for planning: cuRobo's collision model should reflect the widest the
# hand ever gets while it approaches, and the fingers are commanded outside the plan anyway (cuTAMP
# emits discrete {"type": "gripper"} steps). -0.0475 is the MJCF actuator's ctrlrange limit, just
# inside the -0.0485 joint limit.
FINGER_OPEN = -0.0475

# Elbow-up posture over the table, used as the cuRobo retract/null-space config and as the pose the
# idle arm is locked at so it stays folded back out of the working arm's way.
#
# This is OURS, not MolmoAct2's -- an earlier comment here claimed it mirrored their "rest" keyframe,
# which is false in two ways. The MJCF this script converts
# (bimanual_yam_linear_flattened.xml) has NO keyframe at all (mj nkey == 0); the only bimanual home
# pose in the asset set lives in the sibling bimanual_yam.xml and reads
#     qpos = 0 1.047 1.047 0.1 -0.1 0 | 0 0 | 0 1.047 1.047 0.1 -0.1 0 | 0 0
# i.e. joint2 = joint3 = 1.047 = pi/3 with small wrist offsets, NOT (pi/4, pi/2, 0, 0, 0).
# Keep the two straight before "syncing" them: this value is what the scene-6 IK reachability sweep
# was run at, and every cuRobo yml, the Isaac ArticulationCfg and the demos all start from it.
ARM_RETRACT = (0.0, np.pi / 4, np.pi / 2, 0.0, 0.0, 0.0)



def _chain(bodies: Dict[str, Body], leaf: str) -> List[str]:
    """Link names from the root down to ``leaf``, inclusive."""
    out: List[str] = []
    cur: Optional[str] = leaf
    while cur is not None:
        out.append(cur)
        cur = bodies[cur].parent
    return list(reversed(out))


def _self_collision_ignore(bodies: Dict[str, Body], links: List[str]) -> Dict[str, List[str]]:
    """Pairs to skip in self-collision: links within 2 hops on the same arm, plus the finger pairs.

    Cross-arm pairs are deliberately NOT ignored -- with one arm locked and the other planning, the
    idle arm is exactly the obstacle we want cuRobo to respect.
    """
    depth = {name: len(_chain(bodies, name)) for name in links}
    ignore: Dict[str, List[str]] = {}
    for i, a in enumerate(links):
        arm_a = a.split("_")[0]
        for b in links[i + 1:]:
            if b.split("_")[0] != arm_a:
                continue
            chain_b = _chain(bodies, b)
            related = a in chain_b or b in _chain(bodies, a)
            near = related and abs(depth[a] - depth[b]) <= 2
            # The two fingers of one hand open and close past each other; never a real collision.
            fingers = a.endswith("_finger") and b.endswith("_finger")
            # Both fingers hang off link_6 and are always in contact with it.
            hand = a.endswith("_link_6") and b.endswith("_finger")
            if near or fingers or hand:
                ignore.setdefault(a, []).append(b)
    return ignore


def attached_link_name(arm: str, dual: bool) -> str:
    """cuRobo link a grasped object is attached to. Per-arm in dual mode; the single-arm name is kept
    verbatim so every existing robot config and `motion_solver.solve_curobo` are unaffected."""
    return f"{arm}_attached_object" if dual else "attached_object"


def _attached_links(arms: Sequence[str], dual: bool) -> List[str]:
    """The `extra_links` + `extra_collision_spheres` yml lines for the grasped-object slots."""
    entries = []
    for a in arms:
        name = attached_link_name(a, dual)
        entries.append(
            f'      "{name}": {{"parent_link_name": "{a}_grasp_frame", "link_name": "{name}",'
            f' "fixed_transform": [0,0,0,1,0,0,0], "joint_type":"FIXED", "joint_name": "{a}_attach_joint"}}'
        )
    return [
        "    extra_links: {",
        ",\n".join(entries),
        "    }",
        "    extra_collision_spheres: {"
        + ", ".join(f'"{attached_link_name(a, dual)}": 4' for a in arms) + "}",
    ]


def curobo_yaml(
    bodies: Dict[str, Body],
    order: List[str],
    arm: str,          # "left" / "right" plan that arm; "dual" plans BOTH arms as one 12-DOF chain
    *,
    max_samples: int = 6,
    urdf_rel: str = "yam_description/bimanual_yam.urdf",
    asset_root: str = "yam_description",
) -> str:
    """Render a complete cuRobo robot yml that plans for ``arm`` and locks everything else."""
    dual = arm == "dual"
    other = None if dual else ("right" if arm == "left" else "left")
    links = collidable_links(bodies, order, max_samples)
    joint_names = [f"{a}_joint{i}" for a in ARMS for i in range(1, 7)]
    joint_names += [f"{a}_{side}_finger" for a in ARMS for side in ("left", "right")]

    # Everything except the planning arm's 6 revolute joints is locked, so cuRobo's active DOF is 6:
    # both grippers (held open) and the whole idle arm (parked at ARM_RETRACT, still collision-checked).
    lock: Dict[str, float] = {}
    if not dual:
        for i, val in enumerate(ARM_RETRACT, start=1):
            lock[f"{other}_joint{i}"] = float(val)
    for a in ARMS:
        for side in ("left", "right"):
            lock[f"{a}_{side}_finger"] = FINGER_OPEN

    retract = []
    for a in ARMS:
        retract += [float(v) for v in ARM_RETRACT]
    retract += [FINGER_OPEN] * 4

    hand_arms = ARMS if dual else (arm,)
    attached = [attached_link_name(a, dual) for a in hand_arms]
    ignore = _self_collision_ignore(bodies, links)
    hand_links = [n for n in links
                  if any(n.startswith(a) for a in hand_arms) and (n.endswith("_link_6") or n.endswith("_finger"))]

    def yaml_list(items: Sequence[str], indent: str) -> str:
        return "[\n" + "".join(f"{indent}  '{it}',\n" for it in items) + f"{indent}]"

    lines = [
        "# Generated by droid-sim-evals/tools/yam_mjcf_to_urdf.py; do not edit by hand.",
        (f"# Bimanual YAM planning with BOTH arms as one 12-DOF chain: only the four finger joints are"
         if dual else
         f"# Bimanual YAM planning with the {arm.upper()} arm. The {other} arm and both grippers are"),
        ("# locked. Both grasp frames are tracked, so one configuration can be constrained at both hands."
         if dual else
         "# locked, so cuRobo exposes 6 active DOF while still collision-checking the whole robot."),
        "robot_cfg:",
        "  kinematics:",
        f'    urdf_path: "{urdf_rel}"',
        f'    asset_root_path: "{asset_root}"',
        '    usd_path: ""',
        '    isaac_usd_path: ""',
        "    usd_flip_joints: {}",
        "    usd_flip_joint_limits: []",
        "",
        '    base_link: "world"',
        # In dual mode cuRobo still has ONE ee_link, so the left grasp frame is the "ee" and the
        # right one is requested as an extra tracked link; both poses come back from one
        # get_state() call, which is what lets a single 12-DOF configuration be constrained at
        # both hands simultaneously.
        f'    ee_link: "{"left" if dual else arm}_grasp_frame"',
        ('    link_names: ["left_grasp_frame", "right_grasp_frame"]' if dual else "    link_names: null"),
        "    lock_joints: {" + ", ".join(f"'{k}': {v:.6g}" for k, v in lock.items()) + "}",
        "",
        # cuTAMP attaches a grasped object here (see extra_collision_spheres); the pose is identity
        # relative to the grasp frame, matching ur5e_robotiq_2f_85.yml.
        *_attached_links(hand_arms if dual else (arm,), dual),
        "",
        "    collision_link_names: " + yaml_list(links + attached, "    "),
        "    collision_spheres:",
        collision_sphere_yaml(bodies, order, max_samples),
        # Negative buffer: _sweep() inflates every radius to guarantee gap-free coverage with few
        # spheres (arm capsules 0.032 -> 0.039, fingertip pads 0.0071 -> 0.0093), and cuRobo subtracts
        # this from all of them, which lands the model back near the TRUE geometry rather than
        # thinning it below. It matters at the fingertips: with the inflated radii the jaws cannot be
        # driven to within 12 mm of the table without the pad spheres reporting a collision, so a toy
        # lying on the table can only be gripped across its top ~2 cm. See YAM_GRASP_DEPTH.
        "    collision_sphere_buffer: -0.006",
        "",
        "    self_collision_ignore:",
    ]
    for a, bs in ignore.items():
        lines.append(f"      {a}: " + yaml_list(bs, "      "))
    # A grasped object sits between its own hand's fingers, so it must not self-collide with them.
    # In dual mode the two attached objects are deliberately NOT made to ignore each other: two held
    # objects colliding is a real failure the planner should see.
    for a in hand_arms:
        own = [n for n in links if n.startswith(a) and (n.endswith("_link_6") or n.endswith("_finger"))]
        lines.append(f"      {attached_link_name(a, dual)}: " + yaml_list(own, "      "))
    lines += [
        "",
        "    self_collision_buffer: {" + ", ".join(f"'{n}': 0.0" for n in links + attached) + "}",
        "",
        "    use_global_cumul: True",
        "    mesh_link_names: " + yaml_list(links, "    "),
        "",
        "    cspace:",
        "      joint_names: " + yaml_list(joint_names, "      "),
        "      retract_config: [" + ", ".join(f"{v:.6g}" for v in retract) + "]",
        "      null_space_weight: [" + ", ".join("1.0" for _ in joint_names) + "]",
        "      cspace_distance_weight: [" + ", ".join("1.0" for _ in joint_names) + "]",
        "      max_jerk: 500.0",
        "      max_acceleration: 12.0",
        "      position_limit_clip: 0.0",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #
def fetch_mjcf() -> Path:
    """Download the MolmoAct2 YAM MJCF + meshes from HuggingFace and return the MJCF path."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(HF_ASSET_REPO, repo_type="dataset", allow_patterns=["assets/yam/**"])
    mjcf = Path(local) / MJCF_REL
    if not mjcf.exists():
        raise FileNotFoundError(f"{mjcf} not found in the {HF_ASSET_REPO} snapshot")
    return mjcf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mjcf", type=Path, default=None,
                    help="path to bimanual_yam_linear_flattened.xml (default: download from HuggingFace)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory for bimanual_yam.urdf + meshes/ "
                         "(conventionally ../cuTAMP/cutamp/robots/assets/yam_description)")
    ap.add_argument("--name", default="bimanual_yam", help="URDF robot name")
    ap.add_argument("--max-spheres-per-geom", type=int, default=6,
                    help="cap on collision spheres swept from one collider (default 6)")
    ap.add_argument("--mount-xyz", type=float, nargs=3, default=list(MOUNT_XYZ), metavar=("X", "Y", "Z"),
                    help=f"where the shared base sits in the sim world frame (default {MOUNT_XYZ})")
    ap.add_argument("--curobo-cfg-dir", type=Path, default=None,
                    help="where to write bimanual_yam_{left,right}.yml (default: the parent of --out)")
    args = ap.parse_args()

    mjcf = args.mjcf or fetch_mjcf()
    print(f"MJCF: {mjcf}")

    bodies, order, mesh_files = parse_mjcf(mjcf)
    inertials = compiled_inertials(mjcf)
    print(f"parsed {len(order)} bodies, {sum(b.joint_name is not None for b in bodies.values())} joints")

    jaws = {arm: measure_jaws(mjcf, arm, FINGER_OPEN) for arm in ARMS}
    grasp_frame_z = {arm: jaws[arm][0] for arm in ARMS}
    for arm, (tip_z, base_z, half_open) in jaws.items():
        print(f"  {arm} hand: pads z=[{base_z:.4f}, {tip_z:.4f}] in link_6, "
              f"opening {2 * half_open:.4f} m -> {arm}_grasp_frame at z={tip_z:.4f}")

    out = args.out.resolve()
    mesh_out = out / "meshes"
    mesh_out.mkdir(parents=True, exist_ok=True)
    mesh_src = mjcf.parent / "assets"
    copied = 0
    for fname in sorted(set(mesh_files.values())):
        src = mesh_src / fname
        if not src.exists():
            raise FileNotFoundError(f"mesh {src} referenced by the MJCF is missing")
        shutil.copy2(src, mesh_out / fname)
        copied += 1
        # OBJ files may reference a sibling .mtl; carry it along so the visuals keep their material.
        mtl = src.with_suffix(".mtl")
        if mtl.exists():
            shutil.copy2(mtl, mesh_out / mtl.name)
    print(f"copied {copied} meshes -> {mesh_out}")

    tree = build_urdf(bodies, order, mesh_files, inertials, robot_name=args.name, grasp_frame_z=grasp_frame_z,
                      mount_xyz=tuple(args.mount_xyz), efforts=actuator_efforts(mjcf))
    urdf_path = out / f"{args.name}.urdf"
    tree.write(urdf_path, encoding="unicode", xml_declaration=True)
    print(f"wrote {urdf_path}")

    # The cuRobo robot configs live one level up (next to ur5e_robotiq_2f_85.yml) so their paths
    # resolve the same way the existing robots' do, via `external_asset_path` = the assets dir.
    k = args.max_spheres_per_geom
    links = collidable_links(bodies, order, k)
    n_spheres = sum(len(body_collision_spheres(bodies[n], k)) for n in links)
    cfg_dir = args.curobo_cfg_dir.resolve() if args.curobo_cfg_dir else out.parent
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for arm in (*ARMS, "dual"):
        path = cfg_dir / f"{args.name}_{arm}.yml"
        path.write_text(curobo_yaml(
            bodies, order, arm, max_samples=k,
            urdf_rel=f"{out.name}/{args.name}.urdf", asset_root=out.name,
        ))
        print(f"wrote {path}")
    print(f"  ({len(links)} collision links, {n_spheres} spheres)")


if __name__ == "__main__":
    main()
