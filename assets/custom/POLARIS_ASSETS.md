# PolaRiS-ported assets (scenes 8 & 9)

Scenes 8 (`DROID-PanClean`) and 9 (`DROID-BlockStackKitchen`) are ported from PolaRiS
([owhan/PolaRiS-Hub](https://huggingface.co/datasets/owhan/PolaRiS-Hub) /
[github.com/arhanjain/PolaRiS](https://github.com/arhanjain/PolaRiS)). We reuse PolaRiS's scanned
USDZ object meshes on **our** table+dome+robot base (`scene8_0.usd`/`scene9_0.usd`) — i.e. we drop
PolaRiS's scanned background scenes (`g60_stovetop_zed`, `g60_kitchen_table_zed`) and its Gaussian-
splat renderer, and spawn the objects via our sidecar `kind:"usd_rigid"` path (see
`eval/scene_specs.py` for the ported rubrics and `assets/scene{8,9}_0.json` for the layout).

Each mesh is XY-centered with its base at local z=0, so a sidecar `pos.z` = the table-top rest
height (~0.045-0.05). Scales are PolaRiS's per-object `xformOp:scale` (the raw USDZ meshes are in
arbitrary units); non-uniform scale is supported (the sponge). The frying pan / green tray are the
fixed `kinematic` targets; the sponge / cubes are the pick objects.

## `pan_clean/` — DROID-PanClean  (instruction: "Use the yellow sponge to scrub the blue handle frying pan")
Rubric: reach(sponge) → lift(sponge) → sponge within pan (gripper released).

| file | role | scale | physical size (m) |
|------|------|-------|-------------------|
| `frying_pan.usdz` | target (kinematic) | 0.35 | 0.350 × 0.209 × 0.037 |
| `sponge.usdz` | pick | (0.09, 0.07, 0.09) | 0.090 × 0.067 × 0.025 |
| `coke.usdz` | distractor | 0.09 | 0.050 × 0.050 × 0.090 |
| `bell_pepper.usdz` | distractor | 0.08 | 0.080 × 0.080 × 0.074 |
| `mustard.usdz`, `ketchup.usdz`, `sushi.usdz`, `blue_mug.usdz` | (library; unused) | 0.13 / 0.13 / 0.08 / 0.09 | — |

## `block_stack_kitchen/` — DROID-BlockStackKitchen  (instruction: "Place and stack the blocks on top of the green tray")
Rubric: reach(each cube) → lift(each cube) → each cube within tray (gripper released) → green stacked on wood.

| file | role | scale | physical size (m) |
|------|------|-------|-------------------|
| `green_tray.usdz` | target (kinematic) | 0.22 (PolaRiS 0.28) | 0.160 × 0.220 × 0.023 (@0.22) |
| `green_cube.usdz` | pick + stack (top) | 0.06 | 0.060 × 0.060 × 0.060 |
| `wood_cube.usdz` | pick + stack (base) | 0.06 | 0.060 × 0.060 × 0.060 |
| `tomato.usdz` | distractor | 0.06 | 0.060 × 0.060 × 0.048 |
| `uw_corn.usdz` | distractor | 0.08 | 0.080 × 0.033 × 0.033 |
| `pink_bowl.usdz`, `purple_plate.usdz` | (library; unused) | 0.13 / 0.13 | — |

The tray is scaled down from PolaRiS's 0.28 to 0.22 to fit our smaller reachable workspace
(x~[0.30,0.60], y~[-0.12,0.24]); distractors are a curated subset of PolaRiS's clutter (the full
per-scene mesh library is kept in these folders). Object poses are a first cut adapted to our
workspace — verify placement/graspability in a smoke render and nudge as needed.
