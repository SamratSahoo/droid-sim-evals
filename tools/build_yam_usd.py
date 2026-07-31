#!/usr/bin/env python3
"""Convert the generated bimanual-YAM URDF into a USD that Isaac Sim can spawn.

Second half of the YAM asset pipeline; ``tools/yam_mjcf_to_urdf.py`` writes the URDF (and the cuRobo
configs) first. Both consumers read the SAME URDF, which is the point: the joint names Isaac exposes
on the articulation are exactly the ones cuRobo plans over, so the eval driver can map plan waypoints
onto sim joints by name and be sure they agree.

Needs the Isaac app running before ``isaaclab.sim`` is importable at all, so this is a standalone
script rather than a function someone can call from the eval::

    .venv/bin/python tools/build_yam_usd.py

Writes ``assets/yam/bimanual_yam.usd`` plus the converter's ``configuration/`` sub-layers and
baked meshes beside it (override the location with ``--out``). Re-run after regenerating the URDF;
pass ``--force`` to reconvert when the output already exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO.parent / "cuTAMP" / "cutamp" / "robots" / "assets" / "yam_description" / "bimanual_yam.urdf"
DEFAULT_OUT = REPO / "assets" / "yam" / "bimanual_yam.usd"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="reconvert even if the USD already exists")
    args = ap.parse_args()

    if not args.urdf.exists():
        raise SystemExit(f"{args.urdf} not found; run tools/yam_mjcf_to_urdf.py first")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = UrdfConverterCfg(
        asset_path=str(args.urdf.resolve()),
        usd_dir=str(out.parent),
        usd_file_name=out.name,
        force_usd_conversion=args.force or not out.exists(),
        # One robot, spawned once per env: instancing buys nothing and makes the resulting stage
        # harder to inspect when a joint or collider looks wrong.
        make_instanceable=False,
        fix_base=True,
        # The URDF's massless `world` link carries the mount offset to the sim world origin; naming
        # it here keeps PhysX from picking some other link as the articulation root.
        root_link_name="world",
        # False (the default) would collapse the `world` -> `bimanual_base` mount and the two
        # `*_grasp_frame` joints away. Keeping them costs two massless links and preserves the frames
        # cuRobo names, which is worth far more when debugging a pose mismatch.
        merge_fixed_joints=False,
        # The URDF's collision geometry is cylinders standing in for the MJCF's capsules (URDF has no
        # capsule primitive); this restores the rounded caps instead of leaving sharp-edged cylinders
        # to catch on the table.
        replace_cylinders_with_capsules=True,
        # Gains are re-specified per actuator group by the ArticulationCfg in
        # src/sim_evals/environments/bimanual_yam.py, which is what actually drives the sim; these
        # only set the USD defaults.
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=10.0),
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"wrote {converter.usd_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
