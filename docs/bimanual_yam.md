# Bimanual YAM

A second embodiment for the Isaac sim evals: the bimanual YAM from
[MolmoAct2's `sim_eval`](https://github.com/allenai/molmoact2/tree/main/sim_eval) — two 6-DOF arms
with linear parallel-jaw grippers on a shared base — running TiPToP on the same scenes as the DROID
Franka.

Two things differ from the Franka setup and drive most of the design:

* **Perception is a fixed third-person camera**, not a wrist camera. It is the MolmoAct2 `top_cam`
  verbatim: mounted on `bimanual_base` at `(0.15, 0, 0.80)`, 69.4° HFOV (RealSense D435i), looking
  80° down. Its RGB + depth + intrinsics + extrinsics are what go over the TiPToP wire, and it is
  the right pane of every recorded video. (FoundationStereo is not in the loop — the simulator
  renders true depth, so the server runs with `depth_estimator=None` and reads it directly.)
* **Both arms can be used three ways.** The sim-eval rollout runs them in SEQUENCE, one TiPToP plan
  per arm (see [Using both arms](#using-both-arms)). cuTAMP additionally supports planning both arms
  SIMULTANEOUSLY -- one configuration constrained at both hands -- either on two separate objects
  ([parallel](#simultaneous-dual-arm-planning)) or on the same one
  ([handover](#handover)). The sequential path is what the Isaac eval drives; both simultaneous
  modes plan, execute in Isaac and are recorded by `tools/replay_dual_plan.py`.

## Building the assets

Two steps, because the second one needs the Isaac app:

```bash
# 1. MJCF -> URDF + the two per-arm cuRobo configs (any interpreter with `mujoco` + `numpy`)
python tools/yam_mjcf_to_urdf.py --out ../cuTAMP/cutamp/robots/assets/yam_description

# 2. that same URDF -> USD for Isaac
.venv/bin/python tools/build_yam_usd.py --force
```

Both consumers read the **same URDF**, which is the point: the joint names Isaac exposes are exactly
the ones cuRobo plans over, so plan waypoints map onto sim joints by name and cannot silently
disagree. The URDF also carries a `world -> bimanual_base` fixed joint at `MOUNT_XYZ`, which makes
cuRobo's base frame coincide with this repo's world frame — so scene coordinates, camera extrinsics
and plan waypoints all live in one frame with no conversions anywhere.

The robot rests **on the table** at `(0.24, 0, 0.0551)`. That pose was picked by sweeping cuRobo IK
for top-down grasps over the scene-6 region: it splits the workspace so each arm owns one side
(9 sites left-only, 9 right-only, 18 shared, none unreachable).

## Running

```bash
# M2T2 grasp server (TiPToP needs it)
(cd ../M2T2 && pixi run python server.py --port 8123 &)

# TiPToP planning server, left arm
(cd ../tiptop && TIPTOP_CONFIG=tiptop_yam.yml \
   pixi run python -m tiptop.tiptop_websocket_server --port 8765 --num-particles 512 &)

.venv/bin/python eval/yam_tiptop_eval.py
```

`GEMINI_API_KEY` must be set in the **server's** environment. Each rollout writes to
`runs/<date>/<time>/`: the video, plus the third-person RGB and a turbo-colormapped depth image for
every planner query, so a run is self-contained.

## Using both arms

cuTAMP has no operator for choosing an arm — it optimizes over one chain, with one `ee_link`. Each
arm therefore gets its own cuRobo config over the shared 16-joint URDF, differing only in `ee_link`
and `lock_joints`; the idle arm is locked at its rest posture and stays in the collision model, so
the planner respects it as an obstacle.

`--bimanual` runs the two in sequence inside one rollout: split the objects between the arms, ask
each arm's server for a plan covering its own objects, execute those plans back to back. Scene 6
**variant 1** exists for this — the toys are spread to `y = ±0.30`, too wide for either arm alone.

```bash
(cd ../tiptop && TIPTOP_CONFIG=tiptop_yam.yml       pixi run python -m tiptop.tiptop_websocket_server --port 8765 --num-particles 512 &)
(cd ../tiptop && TIPTOP_CONFIG=tiptop_yam_right.yml pixi run python -m tiptop.tiptop_websocket_server --port 8766 --num-particles 512 &)

.venv/bin/python eval/yam_tiptop_eval.py --bimanual --variant 1
```

This is **sequential** bimanual. For both arms acting at the same instant, see
[Simultaneous dual-arm planning](#simultaneous-dual-arm-planning) and [Handover](#handover).

The arm assignment (`_split_by_arm`) is a harness-level rule — objects go to the arm on their side of
the robot's midline, which is the geometry the reachability sweep measured. TiPToP still does all the
perception, grasp selection and motion planning itself.

## Notes on the port

Things that were not obvious and are worth knowing before changing any of it:

* **Grasp depth.** `<arm>_grasp_frame` sits on the distal fingertip plane, matching both Franka
  configs. But the YAM's jaws are 70 mm deep, so putting the M2T2 contact point there leaves only
  the top ~20 mm of a 35 mm toy between the pads, and the grasp slips. `YAM_GRASP_DEPTH = -0.012`
  drops the fingertip plane to about the table surface. It is bounded by the *table*, not by reach.
* **Collision-sphere buffer.** The sphere generator inflates radii to guarantee gap-free coverage
  with few spheres; `collision_sphere_buffer: -0.006` takes that back off. Without it the fingertip
  spheres report a table collision before the jaws get deep enough to grip.
* **Flipped grasps.** For a parallel jaw, rotating a grasp 180° about its approach axis is the same
  physical grasp, and M2T2 emits only one representative. A 7-DOF arm can usually roll its wrist to
  reach whichever one it gets; a 6-DOF arm cannot. `perception.augment_flipped_grasps` (on in the
  YAM configs, off for the Franka) adds the twin — on one scene-6 run the far toy had 3 grasps, of
  which the arm could reach 0 as emitted and all 3 flipped.
* **Contact friction.** The MJCF specifies no per-geom friction, so the pads fall back to PhysX's
  0.5 and drop everything. MolmoAct2 binds a high-friction material to the finger links after load;
  we cannot, because the URDF importer emits the collider prims as instance proxies and USD forbids
  authoring on those — so `set_contact_friction` raises the simulation's *default* material instead.
* **Fixed base links.** The two arm-base housings are dropped from the cuRobo collision set. They are
  rigidly world-fixed, so checking them can only ever produce a constant false positive, and with the
  robot resting on the table their capsules intersect the table slab. `ur5e_robotiq_2f_85.yml` omits
  its `base_link` for the same reason.

## Joint ordering — three stacks, three different orders

The same 16 joints with the same names come out in a **different index order in each stack**. All
three were read off live objects (`mujoco.MjModel`, cuRobo `CudaRobotModel`, IsaacLab `Articulation`):

| Stack | Order | Traversal |
| --- | --- | --- |
| **MuJoCo** (MJCF) | `left_joint1..6`, `left_left_finger`, `left_right_finger`, `right_joint1..6`, `right_left_finger`, `right_right_finger` | depth-first / XML document order |
| **cuRobo** (URDF) | `left_joint1..6`, `right_joint1..6` — the 12 active DOF; the 4 fingers are locked, and land at the **tail** (12–15) if unlocked | chain order, per tracked EE |
| **Isaac** (USD → PhysX) | `left_joint1`, `right_joint1`, `left_joint2`, `right_joint2`, … `left_joint6`, `right_joint6`, then the 4 fingers at 12–15 | **breadth-first — the two arms are interleaved** |

Two consequences, neither of which is optional:

* **Never copy a raw joint vector between stacks — always remap by name.** MuJoCo and cuRobo disagree
  at 8 of 16 indices *even though neither interleaves the arms*: MuJoCo keeps each hand's fingers with
  its own arm (slots 6, 7 and 14, 15) while cuRobo hoists all four to the tail. Index 6 is a
  **right-arm revolute** in cuRobo and a **left-hand prismatic** in MuJoCo. Feeding a MuJoCo `qpos`
  straight into Isaac drives `left_joint4` with a finger command and commands a finger to −0.5 m,
  10× outside its `[-0.0485, 0]` limit.
* **What makes our pipeline correct is name resolution, not layout.** IsaacLab's `JointAction`
  resolves its configured `joint_names` to indices, so the `left_arm` term comes out as ids
  `[0, 2, 4, 6, 8, 10]` and writing a contiguous `[left_joint1..6]` block through it is right. The
  `init_state.joint_pos` is a name-keyed dict for the same reason. `verify_action_mapping(env)` (in
  `yam_environment.py`) asserts all of it at startup and is called by the replay driver.

**The rest posture cannot catch a mistake here**, which is why the assertion exists. `(0, π/4, π/2,
0, 0, 0)` repeated for both arms has value multiset `{0.0 ×8, π/4 ×2, π/2 ×2}`, so **161,279 distinct
permutations leave it bit-identical**. Measured at runtime: a whole left↔right block swap moves
every one of the 22 bodies by `0.000000000 m`; so does reversing the wrist joints, and so does
swapping joint1 with joint6. Controls that *should* be caught (swapping joint2 with joint3, or
treating the interleaved order as contiguous) move bodies 0.68 m and 0.67 m, so the probe does
discriminate — the symmetric pose is simply blind. Any ordering check must use **distinct values per
joint**, e.g. `left = (0.1 … 0.6)`, `right = (−0.1 … −0.6)`.

### The rest pose is ours, not MolmoAct2's

`ARM_RETRACT` / `ARM_REST` = `(0, π/4, π/2, 0, 0, 0)` is a value this port chose, and an earlier
comment in `yam_mjcf_to_urdf.py` wrongly claimed it mirrored a MolmoAct2 keyframe. The MJCF this repo
actually converts (`bimanual_yam_linear_flattened.xml`) has **no keyframe at all**; the only bimanual
home pose in the asset set is in the sibling `bimanual_yam.xml`:

```xml
<key name="home" qpos="0 1.047 1.047 0.1 -0.1 0  0 0   0 1.047 1.047 0.1 -0.1 0  0 0"/>
<!-- qpos order: left arm (6) + left gripper (2) + right arm (6) + right gripper (2) = 16 DOF -->
```

— joint2 = joint3 = `1.047` = **π/3**, with small wrist offsets, not `(π/4, π/2, 0, 0, 0)`. Our value
is what the scene-6 IK reachability sweep was run at and what every cuRobo yml, the Isaac
`ArticulationCfg` and all the demos start from, so it should not be "synced" to theirs casually. (That
keyframe's own comment is independent confirmation of MuJoCo's per-arm-contiguous layout above.)

## Simultaneous dual-arm planning

`--bimanual` above runs the arms one after the other. cuTAMP now also supports both arms acting **at
the same time**: one configuration at which the left hand is on one object and the right hand is on
another. Demo:

```bash
cd ../tiptop
pixi run python -m cutamp.scripts.dual_arm_demo                  # 4 operators, both arms at once
pixi run python -m cutamp.scripts.dual_arm_demo --arm-mode single # 8 operators, one at a time
```

Same scene and goal; dual mode plans

```
MoveFree(q0, traj1, q1)
PickBoth(cube_a, grasp1, cube_b, grasp2, q1)
MoveHoldingBoth(cube_a, grasp1, cube_b, grasp2, q1, traj2, q2)
PlaceBoth(cube_a, grasp1, pose1, tray_a, cube_b, grasp2, pose2, tray_b, q2)
```

`q1` is a single 12-DOF configuration constrained at **both** grasp frames simultaneously.

### How it works

* **Lockstep operators.** `PickBoth` / `PlaceBoth` / `MoveHoldingBoth` act on both hands at one
  timestep, so hand state stays symmetric and `HandEmpty()` still means "both hands empty". That is
  what let this land without changing a single fluent, existing operator, or goal file. They live in
  their own `dual_tamp_operators` list, selected by `TAMPConfiguration.arm_mode`, so a skeleton is
  never a mix of one-hand and two-hand operators.
* **One 12-DOF chain.** `bimanual_yam_dual.yml` locks only the four finger joints and tracks both
  grasp frames in `link_names`, so a single `get_state(q)` returns both hands' poses and all 154
  collision spheres. Cross-arm collision therefore comes for free through the existing
  `Motion/self_collision` term.
* **`DualKinematicConstraint(q, action_a, action_b)`** with its own cost method, rather than two
  `KinematicConstraint`s: the cost function zips its kinematic constraints 1:1 against rollout
  timesteps, and two constraints sharing a configuration would break that invariant for single-arm
  skeletons too. Its values are emitted as `(b, T*A)` — folding the arm axis into the time axis —
  which is exactly what `CostReducer` (sums `dim=1`) and `ConstraintChecker` (`.all(dim=1)`) already
  mean, so **neither reducer needed changing** and the registered weights and tolerances apply
  unchanged.
* **Collision-aware seeding.** cuRobo has no simultaneous multi-pose IK, and none is needed: the arms
  share no joints, so each is solved against its own single-arm config and the results are written
  into that arm's column slice. The catch is that each solve runs with the *other* arm parked, so
  independently-chosen elbow branches routinely intersect — several branches are requested per arm
  and the combination with the largest cross-arm clearance is kept.

Every dual-arm branch is gated on `RobotContainer.arms` being non-empty, which only the dual
container sets, so the single-arm pipeline is untouched — `arms == ()` for all seven single-arm
robots (panda, fr3_robotiq, ur5, panda_robotiq, fr3_franka, bimanual_yam_left, bimanual_yam_right),
each at its original DOF.

### Limits

* **Reachability is the binding constraint, not the planner.** The arms are 0.48 m apart: put both
  targets between the shoulders and the upper arms genuinely intersect (cuTAMP reports
  `Motion/self_collision > 0` and correctly rejects it); put them too far outboard and the hands
  cannot reach. The demo scene sits in the band between.
* **Arm assignment is a separate skeleton per permutation.** Only one keeps each arm on its own
  side, so `num_initial_plans` must be large enough to reach it — the crossed one is rejected
  geometrically, not symbolically.
* **Hardware execution is still gated.** The plan's gripper steps carry an `"arm"` key that
  `tiptop/planning.py::serialize_plan` and `tiptop/execute_plan.py` do not yet thread, so a dual plan
  must not be run on a real robot until they do.

### Executing a dual-arm plan

cuTAMP and Isaac live in different Python environments, so the plan is handed over as JSON:

```bash
cd ../tiptop
pixi run python -m cutamp.scripts.dual_arm_demo --arm-mode dual --save-plan /tmp/dual_plan.json

cd ../droid-sim-evals
.venv/bin/python tools/replay_dual_plan.py --plan /tmp/dual_plan.json
```

`solve_curobo_dual` (in `cutamp/motion_solver.py`) refines the optimized configurations into
trajectories. It differs from `solve_curobo` in three ways, each forced by the dual chain:

* **Joint-space segments.** `plan_single(js, Pose)` constrains one `ee_link` and leaves the other
  arm's six joints in the null space, and cuRobo has no two-pose IK — so segments are planned with
  `plan_single_js` between cuTAMP's optimized 12-DOF configurations, the only motion call that is
  DOF-agnostic. It needs a far more generous plan config than the single-arm cartesian hops (graph
  planner on, finetune on, 15 s timeout); trajopt alone fails on it.
* **The grasp targets are disabled as obstacles for the approach.** cuTAMP checks the gripper against
  an object's *sampled surface spheres*, but cuRobo's world model treats it as a solid, so closing on
  a cube reads as a collision and the grasp configuration is unreachable. Only the two objects being
  grasped are disabled, and only for that segment.
* **A joint-space approach and retract around every grasp and release,** built by solving each arm's
  IK for its own end-effector displaced along the tool axis (`back_off_configuration`) or raised 10 cm
  (`lift_configuration`). Both are necessary, for opposite reasons:
  * *Approach.* A joint-space goal cannot say "come in along the tool's −z", so without a pre-grasp
    waypoint the hand arrives from an arbitrary direction — and since the target is necessarily
    disabled as an obstacle for that segment, it gets swept aside rather than gripped. This was
    visible: the first handover run pushed the bar 2 cm across the table instead of picking it up.
    Splitting the segment confines that blindness to a short straight-in descent. It also fixed the
    parallel demo's slipping cube, so both cubes now land in their trays.
  * *Retract.* Immediately after a grasp the held object is an attached body still resting on the
    table, so every later plan would start in collision. When even that start state is rejected, the
    attached spheres are temporarily detached for the solve (`temporarily_detached`), mirroring what
    `solve_curobo` does for its cartesian retract.

  A mid-air handover pose sits near the edge of both workspaces, where a full 12 cm stand-off is often
  unreachable, so shorter ones are tried before giving up.

## Handover

Both hands on the **same** object: the left arm picks a bar up and passes it to the right arm, which
puts it in a tray the left arm cannot reach.

```bash
cd ../tiptop
pixi run python -m cutamp.scripts.dual_arm_demo --task handover --save-plan /tmp/handover_plan.json

cd ../droid-sim-evals
.venv/bin/python tools/replay_dual_plan.py --plan /tmp/handover_plan.json --scene 15
```

```
MoveFree(q0, traj1, q1)
PickGiver(bar, grasp1, q1)
MoveHoldingGiver(bar, grasp1, q1, traj2, q2)
Handover(bar, grasp1, grasp2, pose1, q2)
MoveHoldingTaker(bar, grasp2, q2, traj3, q3)
PlaceTaker(bar, grasp2, pose2, tray, q3)
```

`q2` is one 12-DOF configuration at which **both** hands hold the bar, and `pose1` is a free mid-air
object pose that both are tied to.

### How it works

* **Fixed roles, so no `Arm` type is needed.** The giver is always `arms[0]` and the taker `arms[1]`,
  which is what keeps the domain as small as the parallel one. `HeldByGiver` / `HeldByTaker` fluents
  replace the symmetric `Held`, since after `PickGiver` the hands are genuinely asymmetric.
* **`arm_active`, a per-timestep mask.** Parallel dual-arm is *lockstep* — both hands act at every
  timestep — which is exactly why it needed no new machinery. A handover breaks that: `PickGiver`
  constrains one hand, `Handover` two, `PlaceTaker` one again. The rollout therefore emits a
  `(T, A)` boolean mask and the cost gathers only the active `(t, arm)` pairs before reshaping to
  `(b, n_active)`, so the existing reducers still see one flat constraint axis.
  `DualKinematicConstraint` is variadic for the same reason: its arity is how many hands that timestep
  constrains, and *which* hands comes from the rollout.
* **The handover pose is sampled in a mid-air box** derived from the arm mounts
  (`TAMPWorld.handover_region`), not hard-coded, so it follows the robot if the base pose changes.

### What makes a handover feasible

Measured before any of it was built, and it decided the demo scene:

| object | IK-feasible poses | of those, arm-clear | best clearance |
|---|---|---|---|
| 5 cm cube | 156 / 3072 | 47 | **1.4 mm** |
| 22 cm bar, hands at ±8 cm | 83 / 3072 | 83 | **127 mm** |
| 30 cm bar, hands at ±12 cm | 49 / 3072 | 49 | **169 mm** |

Two YAM grippers are ~10 cm across, so on a small object they simply cannot both fit — a handover
needs a **long** object, and the two grasps have to be at opposite ends of it. That is not something
the optimizer can fix later, because grasps are sampled, not optimized. Two consequences:

* **The grasps are slid apart at sample time.** The heuristic 4-DOF sampler pins every grasp to the
  object's vertical axis (`translation[:, :2] = 0`), so every candidate it emits is at the same point
  and "pick the farthest candidate" separates nothing. `_slide_grasps_to_end` offsets the giver's
  grasp to the object's `+` end and the taker's to the `−` end along its longest horizontal axis,
  backing off 2 cm so the pads stay on solid material. Separation went 0 cm → 15–19 cm.
* **The mid-air yaw is half-constrained.** The giver holds the `+` end, so that end has to point to
  the giver's own side of the robot or the arms must cross, which is never collision-free. A uniform
  yaw gets this wrong half the time; adding π to exactly those samples raised the satisfying-particle
  yield from 0–32 / 1024 to 39–83 / 1024.

### Motion-planning specifics beyond the parallel case

* **The taker's approach is planned against an attached body.** The object belongs to the giver at
  that moment, so cuRobo reads the taker driving its jaws onto it as a *self*-collision and reports
  `TRAJOPT_FAIL` — no trajectory exists. cuTAMP has already checked that configuration's
  gripper-vs-object clearance under its own sphere model, so the attachment is hidden for that one
  solve. The pre-grasp hop still sees it, so the taker cannot swipe the bar out of the giver's hand on
  the way in.
* **The handoff is a detach/attach pair.** The taker closes first and the giver opens second, so the
  object is never unsupported; cuRobo can only model it as attached to one link, so the exchange is
  `detach(giver) → attach(taker)` in between.
* **The giver then withdraws along its own approach axis.** Its open jaws still surround an object
  that now belongs to the taker, and lifting both arms cannot separate them — they would rise
  together.
* **The placed object stays disabled until the arm has withdrawn.** Re-enabling it first makes the
  retract's own start state read as a world collision (`INVALID_START_STATE_WORLD_COLLISION`).

### Scene

`assets/scene15_0.{usd,json}` — a byte copy of `scene6_0.usd` plus a sidecar with the bar and the
tray, mirroring `build_handover_env()` exactly. The bar starts at `y = +0.26` (left side) and the tray
sits at `y = −0.26` (right side), out of the left arm's reach, so the object can only get there by
being passed between the hands.

A successful run ends with the bar at roughly `(0.53, −0.25, 0.080)` — tray top `0.0551` plus the
bar's 25 mm half-height, centred within a couple of centimetres.

## Task-plan additions

Everything below lives in `cuTAMP/cutamp/tamp_domain.py` unless noted. The headline is how *little*
had to be added: **no existing fluent, operator or goal file changed**, and every existing robot still
resolves to the same operator set. Types are cuTAMP's existing ones — `movable`, `grasp`, `pose`,
`surface`, `conf`, `traj`.

### Parameters

| Parameter | Type | Purpose |
| --- | --- | --- |
| `obj_a`, `obj_b` | `movable` | the two objects of a lockstep operator; `_a` binds to the robot's first arm, `_b` to the second |
| `grasp_a`, `grasp_b` | `grasp` | one grasp per hand |
| `placement_a`, `placement_b` | `pose` | one placement per hand |
| `surface_a`, `surface_b` | `surface` | one target surface per hand |
| `grasp_g` | `grasp` | the **giver's** grasp on the handed-over object |
| `grasp_t` | `grasp` | the **taker's** grasp on the same object |
| `hand_pose` | `pose` | the object's world pose at the instant both hands hold it — a free continuous variable the optimizer solves for, exactly like a placement, except in mid-air rather than on a surface |

Distinct *names* are what make this work: `Fluent.__call__` renames its parameter to the one it is
called with, so `Holding(obj_a)` and `Holding(obj_b)` are two groundings of the same unchanged fluent.

### Predicates (fluents)

Only the handover needed new ones. Parallel dual-arm needed **zero**, because it is lockstep — both
hands act at every timestep, so `HandEmpty()` still means "both hands empty".

| Fluent | Inputs | Why |
| --- | --- | --- |
| `HeldByGiver` | `(obj: movable)` | after `PickGiver` the hands are genuinely asymmetric, so the symmetric `Holding` can no longer say which hand holds the object |
| `HeldByGiverGrasp` | `(obj: movable, grasp: grasp)` | which grasp that hand is using |
| `HeldByTaker` | `(obj: movable)` | same, after the exchange |
| `HeldByTakerGrasp` | `(obj: movable, grasp: grasp)` | |

Roles are **fixed** — giver is always `arms[0]`, taker `arms[1]` — which is what keeps the arm out of
the type system entirely: no `Arm` type, no per-arm variant of every fluent. The cost is that the
planner cannot choose which arm picks.

### Constraint

| Constraint | Inputs | Notes |
| --- | --- | --- |
| `DualKinematicConstraint` | `(conf, *actions)` — **variadic** | in `cutamp/task_planning/constraints.py`. The *number* of actions is how many hands that timestep constrains: two for `PickBoth`/`Handover`, one for `PickGiver`/`PlaceTaker`. *Which* hands comes from the rollout's `arm_active`, not from here. |

It is a separate class from `KinematicConstraint` on purpose: the cost function zips its kinematic
constraints 1:1 against rollout timesteps, and two `KinematicConstraint`s sharing one configuration
would break that invariant for single-arm skeletons too.

### Operators — parallel (`dual_tamp_operators`)

`[MoveFree, PickBoth, MoveHoldingBoth, PlaceBoth]`. `MoveFree` is reused verbatim: it constrains no
end-effector, and its `Motion` / `TrajectoryLength` terms work at whatever width the configuration is.

| Operator | Parameters | Constraints | Costs |
| --- | --- | --- | --- |
| `PickBoth` | `(obj_a, grasp_a, obj_b, grasp_b, q)` | `DualKinematicConstraint(q, grasp_a, grasp_b)`, `CollisionFreeGrasp` ×2 | `GraspCost` ×2 |
| `MoveHoldingBoth` | `(obj_a, grasp_a, obj_b, grasp_b, q_start, traj, q_end)` | `CollisionFreeHolding` ×2, `Motion` | `TrajectoryLength` |
| `PlaceBoth` | `(obj_a, grasp_a, placement_a, surface_a, obj_b, grasp_b, placement_b, surface_b, q)` | `DualKinematicConstraint(q, placement_a, placement_b)`, `StablePlacement` ×2, `CollisionFreePlacement` ×2 | — |

All three declare `distinct_params=(("obj_a", "obj_b"),)`. `MoveHoldingBoth` exists only so the plan
does not ground plain `MoveHolding` once per held object with identical effects; like `MoveFree` it
constrains no end-effector and the rollout skips it.

Effects stay symmetric throughout — `PickBoth` deletes `HandEmpty()` and adds `Holding` for both
objects; `PlaceBoth` restores `HandEmpty()` — which is precisely why no fluent needed changing.

### Operators — handover (`handover_tamp_operators`)

`[MoveFree, PickGiver, MoveHoldingGiver, Handover, MoveHoldingTaker, PlaceTaker]`.

| Operator | Parameters | Key effects | Constraints |
| --- | --- | --- | --- |
| `PickGiver` | `(obj, grasp_g, q)` | `HandEmpty` → `HeldByGiver(obj)`, `HeldByGiverGrasp(obj, grasp_g)` | `DualKinematicConstraint(q, grasp_g)` (one hand), `CollisionFreeGrasp(obj, grasp_g)` |
| `MoveHoldingGiver` | `(obj, grasp_g, q_start, traj, q_end)` | `At(q_start)` → `At(q_end)` | `CollisionFreeHolding`, `Motion` |
| `Handover` | `(obj, grasp_g, grasp_t, hand_pose, q)` | `HeldByGiver*` → `HeldByTaker*` | `DualKinematicConstraint(q, grasp_g, grasp_t)` (both hands), `CollisionFreeGrasp(obj, grasp_t)` |
| `MoveHoldingTaker` | `(obj, grasp_t, q_start, traj, q_end)` | `At(q_start)` → `At(q_end)` | `CollisionFreeHolding`, `Motion` |
| `PlaceTaker` | `(obj, grasp_t, placement, surface, q)` | `HeldByTaker*` → `HandEmpty()`, `On(obj, surface)` | `DualKinematicConstraint(q, placement)` (one hand), `StablePlacement`, `CollisionFreePlacement` |

`Handover` is the only operator where two *different* grasps are tied to one *shared* object pose —
that is what makes it an exchange rather than two independent reaches. Note it takes `hand_pose` as a
parameter but does not name it in a constraint: the cost function registers it through
`obj_to_first_place`, so the object is collision-checked in mid-air.

### Supporting machinery

| Addition | Where | Inputs / shape |
| --- | --- | --- |
| `TAMPOperator.distinct_params` | `task_planning/tamp_structs.py` | `Sequence[tuple]` of parameter-name pairs that must bind to **different** objects. Preconditions cannot express inequality, so without it a two-object operator grounds with both hands on one cube. Defaults to `()`, so single-arm search is unchanged. |
| distinct-params rejection | `task_planning/search.py` | applied before `operator.ground(...)`; `any()` over the empty default is `False` |
| `_GROUND_OP_CACHE` key fix | `task_planning/tamp_structs.py` | now `(self.name, sorted(substitutions))` — the substitution map alone aliases the moment two operators share a parameter-name set |
| `ArmSpec` / `RobotContainer.arms` | `cutamp/robots/__init__.py` | per-arm `name`, `ee_link`, `joint_slice`, `tool_from_ee`, `gripper_spheres`, `attached_link`, `link_index`. `arms == ()` for all seven single-arm robots, and non-empty is what gates every dual-arm branch |
| rollout keys | `cutamp/rollout.py` | `arm_names: tuple`; `arm_active: List[List[bool]]` (T×A); `ee_position_arms` `(b,T,A,3)`, `ee_quaternion_arms` `(b,T,A,4)`, `world_from_{tool,ee}_desired_arms` `(b,T,A,4,4)`; `gripper_close_arms`, `action_params_arms` |
| `TAMPConfiguration.arm_mode` | `cutamp/config.py` | `"single"` \| `"dual"`; cross-validated against `robot == "bimanual_yam_dual"` in both directions |
| `TAMPConfiguration.dual_task` | `cutamp/config.py` | `"parallel"` \| `"handover"` — selects the operator list via `_operators_for(config)` in `algorithm.py` |

A skeleton is built from exactly one operator list, never a mix, so a dual skeleton can never contain
a single-arm `Pick` that would break the hand-state symmetry.

### Skeleton lengths for the same goal

| Operator set | Goal | Ops |
| --- | --- | --- |
| `all_tamp_operators` (single-arm) | two objects onto two trays | 8 |
| `dual_tamp_operators` | same | 4 |
| `handover_tamp_operators` | one bar onto one tray | 6 |
