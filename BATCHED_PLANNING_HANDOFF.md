# Batched TAMP Planning for scene-6 datagen — Status & Handoff

_Last updated: 2026-07-09 early AM (sessions `9c16a950`/`c287b6fb`, follows `3170bc09`)._

> **STATUS: ALL BLOCKERS ROOT-CAUSED AND FIXED. Batched planning is stable in production and the
> pipeline now collapses each batch into a SINGLE group.** The cuRobo `illegal memory access` that
> ended the previous session was a **warp-mesh use-after-free introduced by this fork's name-keyed
> mesh cache** — reproduced offline in seconds, fixed with an `(env_idx, name)` cache key, and proven
> by a 1-hour crash-free production soak (38 batches, 8 episodes, 0 CUDA errors). Five further
> blockers were found and fixed behind it (retime guard + two retime batch bugs, rep-world surface
> AND sphere leakage, fragmentation), then the shared-attach yield cap was solved (grasp-aligned
> selection + anchor-exact blob + detached lift-off). **Final: 10-14/16 plans per batch at ~6.4 s per
> plan (~2x serial CG); the full 300-episode run is live on the batched config: job `3508869`.**

---

## 0. TL;DR

- **Phase 1 (CUDA graphs, serial):** proven, stable, ~8.7 s/plan. Still the fallback:
  `sbatch --export=ALL slurm/datagen_8gpu_neuronic.slurm cfg/datagen_vae35k.yml`
- **Phase 2 (batched):** correct AND stable now. Five distinct bugs stood between the original design
  and a working batched pipeline; all five are fixed (§2). A batch of 8 scenes now canonicalizes into
  **one group** solved by **one cached batched solver** (§3).
- **Batched config (the winning one, ~2x serial CG):**
  `USE_BATCH_PLAN=1 BATCH_PLAN_SIZE=16 CUTAMP_BATCH_PAD=16 CUTAMP_REFINE_RANKS=3 CUTAMP_MG_CACHE=2`
  plus `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` (SLASWP cure on shared nodes). All gated (§5).

## 1. Pipeline recap

`tamp_data_gen.py` (Isaac worker) requests plans from the tiptop websocket server; with
`TIPTOP_BATCH_PLAN=1` the server's `_batch_loop` collects up to `TIPTOP_BATCH_SIZE` post-perception
requests and solves them together via `cutamp.run_cutamp_batched`: scenes are canonicalized, grouped
by structural signature, each group gets one batched particle optimization + one batched cuRobo
refinement (`solve_curobo_batched` → `plan_batch_env`, full-B with masks).

## 2. The five bugs (in discovery order) — all fixed

| # | Bug | Fix | Proof |
|---|---|---|---|
| 1 | **B=1 batch-axis squeeze**: cuRobo squeezes `interpolated_plan` to `[T,dof]` when a batch is size 1; `ip[b]` indexed the time axis → every singleton segment silently unusable ("0-live") | discriminate on `.position.ndim` in `_masked_plan` / `_append_batched_traj` | job 3505600: 54/54 segs usable, `ep_000` |
| 2 | **Warp-mesh use-after-free (THE crash)**: this curobo fork keys `_wp_mesh_cache` by mesh NAME and `del`s on collision; batched builds load the same names (plate/toy0/…) into every env, so env *i*'s load freed env *j<i*'s warp meshes while their ids stayed in `_mesh_tensor_list` → non-deterministic `illegal memory access` in whichever kernel ran next (MPPI/newton/`coll_check_fn` were all just messengers). Demo scenes never crashed (primitives, no meshes); singletons never crashed (no cross-env name reuse) | key the cache by `(env_idx, name)` (`curobo/src/curobo/geom/sdf/world_mesh.py`) | offline repro `repro_batched_mesh.py`: unfixed = 7 dangling ids + the exact Warp 700 crash in seconds; fixed = clean through builds/churn/queries/world swaps. Production soak job 3507060: 1h05m, 38 batches, 0 CUDA errors |
| 3 | **Retime guard killed every B>1 group**: `tiptop.yml` sets `time_dilation_factor: 0.2`, so `_plan_batch_attempts` always calls `result.retime_trajectory(0.2)`, which raised `only single result is supported` for batched results → multi-env groups produced ZERO plans (masked by bug 2 until it was fixed) | remove the single-result guard (`curobo motion_gen.py:1336`) — the body already handles `[B,T,dof]` per-env (same `scale_by_dt` + `get_batch_interpolated_trajectory` machinery as plan-time) | job 3507060: 71 hits = every multi-env group; gone after fix |
| 4 | **Rep-world surface leakage**: the batched `CostFunction` cached `surface_to_aabb/obb/target_z` from the group REP's world; scene-6 randomizes the plate per scene → non-rep scenes' placements were optimized toward the REP's plate and rejected by their own ranking | `cf.batched_surfaces` + `scene_env_idx` row-gather in `stable_placement_costs` (per-scene aabb/obb/target_z) | measured live: `#satisfying per scene [87, 0, 0]`, `[61, 0]`, `[97, 26, 0, 0]` (non-rep ≈ 0 unless plates overlapped by luck) |
| 5 | **Fragmentation** (speed, not correctness): toy names / toy count / surface label / grasp rep shattered batches into singleton groups; each group paid a fresh ~8-9 s solver build (measured: 33 builds / 0 reuses initially) | single-group canonicalization v2 + fixed-size batch padding (§3) | offline job 3507176 ALL PASS; live job 3507442 |

Historical red herrings, now explained: "graph search is a wall" (bug 1), "MPPI kernels are unsafe at
n_envs≥4" (bug 2 surfacing in MPPI), "seeds=4 misaligned address" (bug 2 again), "build-count memory
corruption accumulation" (bug 2; the solver-cache experiment could never have fixed it).

## 3. Single-group batching (canonicalization v2)

Every scene is normalized to ONE canonical form (gate: `CUTAMP_CANONICALIZE`, default on):

1. **Toy roles** (v1): movables renamed `toy0..k-1` by pick order.
2. **Surface** (`CUTAMP_CANON_SURFACE`, default `plate`): the single Place surface is renamed to the
   canonical name (static obstacle + TAMPWorld indexes + re-grounded skeleton). The tiptop
   `_batch_loop` aliases its loosened `{surface}_in_xy/_support` tolerances under the canonical name.
3. **Grasp representation** (`_normalize_scene_grasps`): 4/6-DOF fallback grasp blocks are converted
   to fixed `[pps,4,4]` matrices + fabricated unit confidences. Lossless: `ParticleOptimizer` only
   steps `{Pose, Conf}` types, so grasp blocks are frozen after initialization either way.
4. **Toy count** (`CUTAMP_CANON_COUNT`, default 3): scenes with k<3 real toys get phantom toys —
   small template cuboids registered as real movables and parked at free spots — plus appended
   `MoveFree/Pick/MoveHolding/Place` cycles using the task planner's positional naming
   (`q{2j+1}/q{2j+2}`, `grasp{j+1}`, `pose{j+1}`), making the padded skeleton structurally identical
   to a natural 3-toy skeleton. Fabricated particles: grasp = clone of `grasp1`, placements
   initialized on the scene's own plate, q's from `q0`.

Two masking layers keep phantoms out of results:
- **Ranking** uses the scene's pre-padding `skeleton_rank`, so phantom constraints never gate a
  scene's satisfying particles.
- **Refinement** masks phantom ops per env (`solve_curobo_batched(phantom_objs_per_env=…)`): a
  Pick/Place on an env's phantom toy skips that env (it coasts; phantom cycles are always a suffix of
  an env's real ops). Emitted plans contain no phantom motions (offline-verified).

Plus the throughput layer:
- **Fixed-size batch padding** (`CUTAMP_BATCH_PAD`, default 8): every group's refine batch is padded
  to 8 envs with dead clones (`initial_live`), so the batched-solver cache key is effectively the
  signature alone → the `(8, sig)` solver is built once per server lifetime and reused every batch.
  `plan_batch_env` wall-time is flat in B (7.74×/scene at B=8, job 3505678), so padding is ~free.

## 4. Files changed (all in-tree, no reverts needed)

- **curobo/src/curobo/geom/sdf/world_mesh.py** — warp-mesh cache keyed `(env_idx, name)` (bug 2).
- **curobo/src/curobo/wrap/reacher/motion_gen.py** — `retime_trajectory` batched support (bug 3).
- **cuTAMP/cutamp/cost_function.py** — `batched_surfaces` per-scene surface targets (bug 4).
- **cuTAMP/cutamp/algorithm.py** — surface canonicalization, grasp normalization, count padding
  (`_pad_scene_to_canonical`), `_batched_surface_targets`, mg-cache reuse via full per-env world
  reloads (`_update_batched_world`), fixed-size batch padding, `skeleton_rank` ranking.
- **cuTAMP/cutamp/motion_solver.py** — `initial_live` (dead padding envs), `phantom_objs_per_env`
  op masking, B=1 squeeze fix (bug 1, prior session).
- **tiptop/tiptop/tiptop_websocket_server.py** — canonical-surface tolerance alias in `_batch_loop`.
- **droid-sim-evals/slurm/datagen_8gpu_neuronic.slurm** — forwards `CUTAMP_BATCH_PAD`,
  `CUTAMP_CANON_SURFACE`, `CUTAMP_CANON_COUNT`.

## 5. How to run

Full datagen, batched (recommended once §7 confirms):
```bash
cd /n/fs/tamp-vla/tamp-vla/droid-sim-evals
sbatch --export=ALL,USE_BATCH_PLAN=1,BATCH_PLAN_SIZE=8 slurm/datagen_8gpu_neuronic.slurm cfg/datagen_vae35k.yml
```
Phase-1 fallback (drop `USE_BATCH_PLAN`). A/B knobs: `CUTAMP_CANONICALIZE=0`, `CUTAMP_CANON_COUNT=0`,
`CUTAMP_CANON_SURFACE=''`, `CUTAMP_BATCH_PAD=1` progressively restore older behaviors.

Key log greps (`runs/tamp_data/job_<JID>/logs/tiptop_*.log`):
`"scenes ->"` (want `-> 1 batchable group`), `"satisfying per scene"` (non-rep > 0 proves bug-4 fix),
`"mg-cache]"` (REUSE-dominated), `"plans in"` (s/scene), `"illegal memory access|misaligned"` (must
stay 0), `find … -name 'ep_[0-9]*'` (episodes).

## 6. Evidence (job log)

| Job | What | Result |
|---|---|---|
| 3507051/3507056 | offline warp-mesh UAF repro (unfixed vs fixed) | unfixed: 7 dangling ids + Warp 700 crash; fixed: clean incl. reuse-path world swaps + 3 MotionGen build/warmup cycles |
| 3507060 | live soak, UAF fix only | COMPLETED 1h05m at its 8-episode target: 38 batches, 80 builds/112 reuses, 0 CUDA errors; all multi-env groups lost to retime (71 hits) + surface leakage ([87,0,0]) — both since fixed |
| 3507176 | offline single-group validation | ALL PASS: pad-by-phantom keeps 1 group + identical plan structure + zero phantom motions; surface rename clean |
| 3507611 | first live single-group run | **`8 scenes -> 1 batchable group (sizes [8])` every batch**; 1 solver build per server + all REUSE; 0 CUDA errors; exposed the retime broadcast bug (0 plans) |
| 3507706 | + retime broadcast fix | plans flow end-to-end (1-5/8 per batch); exposed residual yield skew = REP-geometry sphere leakage |
| 3507724 | + per-scene object spheres | **8/8 scenes satisfying in nearly every batch (64-217 each)** — optimizer yield at singleton level; refine attrition (shared attach blob) now the cap |
| 3507756 | + B=16, pad=16, 3 refine ranks | **16-scene single groups, 4.2-4.6 s/scene**, 1-7 plans/batch (median ~5) ≈ 12-17 s/plan; 0 errors |

## 7. Final verdict: BATCHED WINS — full run launched on it

**Measured endpoint (job 3508697, ab config, 2xL40): 10-14 of 16 plans per batch at 4.3-4.4 s/scene
= ~6.4 s per produced plan — ~2x serial Phase-1 CG (~11-13 s/plan). Zero CUDA faults, zero group
failures, episodes flowing.** The full `datagen_vae35k` (300 episodes, 8 GPUs) is launched on this
config: **job `3508869`**, with
`USE_BATCH_PLAN=1 BATCH_PLAN_SIZE=16 CUTAMP_BATCH_PAD=16 CUTAMP_REFINE_RANKS=3 CUTAMP_MG_CACHE=2
OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`.

### The held-segment (attach) saga — what actually worked
The shared attach blob was the final yield cap (refine ~35% while the optimizer satisfied 16/16).
The iteration ladder, each step measured live:
- **Rep-world blob** (original): wrong shape for 15/16 envs -> attrition 6->4->2 per pick-place cycle.
- **Union blob, global 50-sphere cover**: covers bulge 2-3 cm below the true toys -> blob swallows
  fingers/table -> 0/16.
- **Union, per-env covers (3 spheres/env)**: still bulges below toy bottoms -> table collision at the
  grasp start state -> 0/16.
- **Exact union via 16x attach budget (800 spheres)**: OOMs self-collision buffers on a shared L40.
- **x4 budget (200) + grasp alignment**: works at <=5 live envs (56% yield), re-bulges at 16 -> 0/16.
- **WINNER — three composed pieces, each individually cheap:**
  1. **Grasp-aligned rank selection** (`_aligned_rows`, algorithm.py): scene 0 anchors; every other
     scene picks its best-confidence unused ranked particle whose grasp ROTATIONS best match the
     anchor -> all envs hold their toys similarly -> one blob can approximate everyone.
  2. **Anchor-exact blob** (`_attach_shared_blob`): the blob is the anchor env's true 50-sphere set
     (its own geometry via `batched_obj_spheres`) — no cover bulge, no budget/memory pressure.
  3. **Detached lift-off**: the Place retract (5 cm lift) plans with the blob DETACHED and reattaches
     from grasp-time state — at the grasp start the held toy touches the table, so ANY attached blob
     makes waypoint 0 invalid (the serial pipeline survives only via its INVALID_START detach-replan
     fallback; this is its batched analog, applied preemptively).
  Also included: **batched go-home** (one full-B masked pose plan to FK(q0); the old per-env
  `plan_single_js` collision-checked env 0's world for every env).

### Evidence (yield-lever runs)
| Job | Config | Result |
|---|---|---|
| 3507993 / 3508010 | union blob, global / per-env covers | 0/16 (blob bulge kills held segs) |
| 3508161 | 16x attach budget | OOM (self-collision buffers) |
| 3508207 | x4 budget + alignment + detach-liftoff | 3/4, 2/5 at small groups; OOM at 2nd solver build |
| 3508296 | + CUTAMP_MG_CACHE=2 | server-1 OOM persisted; x4 cover re-bulged at 16 envs |
| **3508697** | **anchor-exact blob + alignment + detach-liftoff, budget x1** | **10-14/16 plans, 4.3 s/scene = ~6.4 s/plan, 0 errors, episodes** |

Remaining nice-to-haves: MPPI/seed re-enable A/B (their historical crashes were bug 2); pure-fallback
scenes lose confidence ranking (rare); perception oddballs (0/4+ toys, multi-surface) fall out of the
single group by design.
