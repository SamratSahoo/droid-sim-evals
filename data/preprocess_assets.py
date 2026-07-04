#!/usr/bin/env python3
"""Preprocess the scene 8-12 assets into Isaac-ready form (run with the eval .venv python).

Two jobs, both needing the bundled Isaac build (pxr + the MJCF importer), so this launches a
headless Isaac app once:

  1. CABINET (scene 11): ``assets/custom/cabinet/white_cabinet.xml`` is a MuJoCo MJCF 3-drawer
     cabinet (three prismatic ``*_level`` slide joints) whose *visual* meshes are legacy MuJoCo
     binary ``.msh`` files the Isaac MJCF importer can't read. We parse each ``.msh`` -> ``.obj``
     (trimesh), rewrite the MJCF to reference the OBJs, then run Isaac Lab's ``MjcfConverter``
     (fix_base=True -> world-anchored) to emit an articulated USD under ``custom/cabinet/processed/``.

  2. USDZ/USDA size measurement (scenes 8 + 10): the scanned Sketchfab assets (whiteboard, eraser,
     e-stop buttons, and the articulated e-stop USDA) are authored ~2-4.7 "units" with
     metersPerUnit=0.01 + upAxis=Y, but Isaac treats 1 unit = 1 m, so each spawns ~20-40x too big.
     We measure each asset's true axis-aligned bbox (pxr BBoxCache) and print the ``scale`` that maps
     its longest axis to a sensible tabletop target -- the values that go into the scene sidecars.

Outputs a summary + ``assets/custom/processed_assets.json`` (per-asset extents + recommended scale).

Usage (from droid-sim-evals):
    .venv/bin/python data/preprocess_assets.py
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh

_DSE = Path(__file__).resolve().parents[1]          # droid-sim-evals/
ASSETS = _DSE / "assets"
CABINET_DIR = ASSETS / "custom" / "cabinet"
PROCESSED = ASSETS / "custom" / "processed_assets.json"

# usdz/usda assets to measure -> a sensible tabletop "longest horizontal axis" target (metres).
# (rotation to stand upright / lie flat is chosen per-scene in the sidecar, verified visually.)
MEASURE = {
    "estop_button_articulated": (CABINET_DIR.parent / "button" / "estop_articulated" / "estop_button.usda", 0.12),
    "Emergency_Stop_Button":    (CABINET_DIR.parent / "button" / "Emergency_Stop_Button.usdz", 0.12),
    "Button_Key_factory":       (CABINET_DIR.parent / "button" / "Button_Key__factory_8MB.usdz", 0.10),
    "Whiteboard":               (CABINET_DIR.parent / "whiteboard" / "Whiteboard.usdz", 0.30),
    "Whiteboard_Eraser":        (CABINET_DIR.parent / "whiteboard" / "Whiteboard_Eraser.usdz", 0.11),
}


# --------------------------------------------------------------------------- #
# Legacy MuJoCo .msh (binary) -> .obj                                          #
# --------------------------------------------------------------------------- #
def load_legacy_msh(path: Path):
    """Parse a legacy MuJoCo binary ``.msh`` -> (verts (V,3), faces (F,3), uv (V,2)|None).

    Layout: 4 int32 counts (nvert, nnormal, ntexcoord, nface), then float32 vertices (V*3),
    float32 normals (N*3), float32 texcoords (T*2), int32 faces (F*3).
    """
    data = path.read_bytes()
    nv, nn, nt, nf = struct.unpack("<4i", data[:16])
    off = 16
    verts = np.frombuffer(data, "<f4", nv * 3, off).reshape(-1, 3).astype(np.float64); off += nv * 12
    off += nn * 12  # skip normals (trimesh recomputes)
    uv = None
    if nt:
        uv = np.frombuffer(data, "<f4", nt * 2, off).reshape(-1, 2).astype(np.float64); off += nt * 8
    faces = np.frombuffer(data, "<i4", nf * 3, off).reshape(-1, 3).astype(np.int64)
    return verts, faces, (uv if uv is not None and len(uv) == len(verts) else None)


def convert_cabinet_meshes() -> list[tuple[str, str]]:
    """Convert every ``*_vis.msh`` under the cabinet dir to a sibling ``.obj``. Returns [(msh_rel, obj_rel)]."""
    pairs = []
    for msh in sorted(CABINET_DIR.rglob("*_vis.msh")):
        verts, faces, uv = load_legacy_msh(msh)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        if uv is not None:
            mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
        obj = msh.with_suffix(".obj")
        mesh.export(obj)
        pairs.append((msh.relative_to(CABINET_DIR).as_posix(), obj.relative_to(CABINET_DIR).as_posix()))
        print(f"  .msh -> .obj: {obj.relative_to(_DSE)}  (V={len(verts)}, F={len(faces)})")
    return pairs


def rewrite_mjcf_for_isaac() -> Path:
    """Copy white_cabinet.xml -> white_cabinet_isaac.xml, OBJ meshes + a single articulation root.

    Two edits: (1) point every ``<mesh file>`` at the converted ``.obj``; (2) FLATTEN the redundant
    empty wrapper bodies (worldbody -> <body> -> object -> base -> drawers) so ``base`` sits directly
    under ``worldbody``. Those pass-through wrappers (no joints, no offsets) otherwise make the MJCF
    importer emit *two* PhysicsArticulationRoots (worldBody + a nested body), which Isaac Lab rejects
    ("Failed to find a single articulation"). With ``base`` as the lone root, fix_base anchors it and
    the three slide-joint drawers form one clean articulation.
    """
    import xml.etree.ElementTree as ET

    src = CABINET_DIR / "white_cabinet.xml"
    txt = src.read_text().replace("_vis.msh", "_vis.obj")
    root = ET.fromstring(txt)
    wb = root.find("worldbody")
    base = wb.find('.//body[@name="base"]')  # the real fixed body (geoms + site + 3 drawer children)
    for child in list(wb):
        wb.remove(child)
    wb.append(base)
    dst = CABINET_DIR / "white_cabinet_isaac.xml"
    ET.ElementTree(root).write(dst, encoding="unicode")
    return dst


# --------------------------------------------------------------------------- #
# Isaac-side: bbox measurement + MJCF->USD conversion                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app

    # The bundled headless experience doesn't enable the MJCF importer, so MjcfConverter's
    # MJCFCreateImportConfig command isn't registered -> enable the extension explicitly.
    import omni.kit.app
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.asset.importer.mjcf", True
    )

    from pxr import Usd, UsdGeom
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    results: dict = {}

    # --- 1. Cabinet: .msh -> .obj, rewrite MJCF, convert to USD ---
    print("\n=== Cabinet: converting legacy .msh visuals -> .obj ===")
    convert_cabinet_meshes()
    isaac_mjcf = rewrite_mjcf_for_isaac()
    out_dir = CABINET_DIR / "processed"
    out_dir.mkdir(exist_ok=True)
    print(f"=== Cabinet: MJCF -> USD via MjcfConverter ({isaac_mjcf.name}) ===")
    conv = MjcfConverter(MjcfConverterCfg(
        asset_path=str(isaac_mjcf),
        usd_dir=str(out_dir),
        usd_file_name="white_cabinet.usd",
        # fix_base=False -> a SINGLE articulation root (the real `base` rigid body). fix_base=True
        # additionally synthesizes a `worldBody` root, so Isaac Lab sees TWO PhysicsArticulationRoots
        # and rejects the prim. We instead world-anchor at spawn via the sidecar's "fix_base": true
        # -> ArticulationRootPropertiesCfg.fix_root_link=True (base has a RigidBodyAPI, so it works).
        fix_base=False,
        import_sites=True,
        self_collision=False,
        make_instanceable=False,
        force_usd_conversion=True,
    ))
    cab_usd = Path(conv.usd_path)
    print(f"  cabinet USD -> {cab_usd}")

    # The MjcfConverter always emits TWO PhysicsArticulationRoots -- a synthetic massless ``worldBody``
    # AND the real rigid ``base`` -- but Isaac Lab's Articulation loader requires exactly one under the
    # spawn path ("Failed to find a single articulation"). Strip the root API from the synthetic
    # worldBody (via a root-layer listOp deletion, which composes even if the API was applied in a
    # sublayer), keeping it on the real rigid ``base`` so the sidecar's fix_root_link can anchor it.
    from pxr import PhysxSchema, UsdPhysics

    cstage = Usd.Stage.Open(str(cab_usd))
    root_prims = [p for p in cstage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(root_prims) > 1:
        keep = next((p for p in root_prims if p.HasAPI(UsdPhysics.RigidBodyAPI)), root_prims[0])
        cstage.SetEditTarget(cstage.GetRootLayer())
        for p in root_prims:
            if p != keep:
                p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                if p.HasAPI(PhysxSchema.PhysxArticulationAPI):
                    p.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        cstage.GetRootLayer().Save()
        cstage = Usd.Stage.Open(str(cab_usd))  # reopen to confirm the composed result

    roots = [str(p.GetPath()) for p in cstage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    joints = [p.GetName() for p in cstage.Traverse() if "Joint" in str(p.GetTypeName()) and "Fixed" not in str(p.GetTypeName())]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    cb = cache.ComputeWorldBound(cstage.GetPseudoRoot()).ComputeAlignedRange().GetSize()
    print(f"  cabinet USD: articulation_roots={roots} (want 1), moving joints={joints}, "
          f"bbox(m)~=[{cb[0]:.3f},{cb[1]:.3f},{cb[2]:.3f}]")
    results["cabinet"] = {
        "usd": cab_usd.relative_to(ASSETS).as_posix(),
        "articulation_roots": roots,
        "joints": joints,
        "bbox_m": [round(float(x), 4) for x in cb],
    }

    # --- 2. Measure usdz/usda extents ---
    print("\n=== USDZ/USDA extents (raw units; Isaac treats 1 unit = 1 m) ===")
    for name, (path, target_m) in MEASURE.items():
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        stage = Usd.Stage.Open(str(path))
        prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        size = rng.GetSize()          # raw units on each axis
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        longest = max(sx, sy, sz)
        scale = round(target_m / longest, 6) if longest > 0 else 1.0
        thin_axis = int(np.argmin([sx, sy, sz]))  # 0=X 1=Y 2=Z (smallest extent = the "flat" axis)
        print(f"  {name:26s} raw(units)=[{sx:7.2f},{sy:7.2f},{sz:7.2f}]  scale={scale:.6f} -> "
              f"[{sx*scale:.3f},{sy*scale:.3f},{sz*scale:.3f}] m  thin_axis={'XYZ'[thin_axis]}")
        results[name] = {
            "path": path.relative_to(ASSETS).as_posix(),
            "raw_units": [round(sx, 3), round(sy, 3), round(sz, 3)],
            "target_longest_m": target_m,
            "scale": scale,
            "scaled_m": [round(sx * scale, 4), round(sy * scale, 4), round(sz * scale, 4)],
            "thin_axis": "XYZ"[thin_axis],
        }

    PROCESSED.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {PROCESSED.relative_to(_DSE)}")
    simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
