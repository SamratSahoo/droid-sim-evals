"""Spawn a rigid or deformable body from a preprocessed mesh file (.obj/.ply).

The custom plate + toys are scanned assets that don't ship with PhysX physics, so we
bring them in by reusing Isaac Lab's native mesh-spawn path
(:func:`isaaclab.sim.spawners.meshes.meshes._spawn_mesh_geom_from_mesh`) -- the same
code that backs ``MeshCuboidCfg``/``MeshSphereCfg`` -- but feeding it geometry loaded
from a file instead of a trimesh primitive. This gives full control over physics:

  * the **plate** spawns as a rigid body with a convex collider, and
  * the **toys** spawn as PhysX FEM **soft bodies** (tetrahedralized at spawn from a
    watertight mesh), via ``deformable_props`` + a ``DeformableBodyMaterialCfg``.

Used by :class:`SceneCfg.dynamic_scene` when a ``scene<N>_<V>.json`` sidecar is present.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

import trimesh

import isaacsim.core.utils.prims as prim_utils
from pxr import PhysxSchema, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files import from_files_cfg
from isaaclab.sim.spawners.meshes import meshes_cfg
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.utils import clone, get_all_matching_child_prims
from isaaclab.utils import configclass


def _load_trimesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


@clone
def spawn_mesh_from_file(
    prim_path: str,
    cfg: "FileMeshCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh prim from a mesh file with the configured physics applied.

    Decorated with :func:`clone` so a regex ``prim_path`` is spawned across all matching
    (per-env) paths, matching the built-in mesh spawners.
    """
    mesh = _load_trimesh(cfg.mesh_path)
    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, None)

    # Optionally override the rigid collider approximation (e.g. convexDecomposition so a
    # concave plate keeps its rim instead of being hulled into a solid dome).
    if cfg.collision_approximation is not None and cfg.collision_props is not None:
        mesh_prim = prim_utils.get_prim_at_path(f"{prim_path}/geometry/mesh")
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision_api.GetApproximationAttr().Set(cfg.collision_approximation)
        if cfg.collision_approximation == "convexDecomposition":
            PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh_prim)

    return prim_utils.get_prim_at_path(prim_path)


@configclass
class FileMeshCfg(meshes_cfg.MeshCfg):
    """A :class:`MeshCfg` whose geometry is loaded from a mesh file (``mesh_path``).

    Set ``rigid_props`` + ``collision_props`` (+ ``mass_props``) for a rigid body, or
    ``deformable_props`` + a :class:`DeformableBodyMaterialCfg` for a soft body -- exactly
    as with the built-in ``MeshCuboidCfg``.
    """

    func: Callable = spawn_mesh_from_file

    mesh_path: str = MISSING
    """Absolute path to the mesh file to spawn (.obj/.ply/.stl; trimesh-loadable)."""

    collision_approximation: str | None = None
    """If set (rigid only), override the collider approximation on the mesh prim,
    e.g. ``"convexHull"`` or ``"convexDecomposition"``. ``None`` keeps the default
    (``convexHull`` for a non-primitive mesh)."""


@clone
def spawn_usd_rigid(
    prim_path: str,
    cfg: "UsdRigidCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Reference a USD/USDZ asset and make it a rigid body with a convex collider.

    Unlike ``UsdFileCfg`` (whose ``rigid_props``/``collision_props`` only *modify*
    already-authored physics), this *applies* a rigid body + collision to a plain
    visual asset (e.g. the scanned, textured toy usdz) — keeping its material/textures.
    The transform/scale come from ``translation``/``orientation`` (init_state) + ``cfg.scale``.
    """
    prim_utils.create_prim(
        prim_path, usd_path=cfg.usd_path, translation=translation, orientation=orientation, scale=cfg.scale
    )
    schemas.define_rigid_body_properties(prim_path, cfg.rigid_props or sim_utils.RigidBodyPropertiesCfg())
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)
    approx = cfg.collision_approximation or "convexHull"
    for mesh_prim in get_all_matching_child_prims(prim_path, predicate=lambda p: p.GetTypeName() == "Mesh"):
        mesh_path = str(mesh_prim.GetPath())
        mca = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mca.CreateApproximationAttr().Set(approx)
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh_prim)
        schemas.define_collision_properties(mesh_path, cfg.collision_props or sim_utils.CollisionPropertiesCfg())
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class UsdRigidCfg(from_files_cfg.UsdFileCfg):
    """Reference a textured USD/USDZ and spawn it as a rigid body with a convex collider.

    Set ``rigid_props`` + ``collision_props`` + ``mass_props`` like a normal rigid spawn;
    ``scale`` scales the referenced asset; the pose comes from the owning cfg's init_state.
    """

    func: Callable = spawn_usd_rigid

    collision_approximation: str | None = "convexHull"
    """Collider approximation applied to the asset's mesh(es). Must be convex for a
    dynamic rigid body (``convexHull`` or ``convexDecomposition``)."""
