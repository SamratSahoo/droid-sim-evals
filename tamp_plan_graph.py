#!/usr/bin/env python3
"""Run a single cuTAMP solve on the toys scene and write the TAMP plan graph.

This drives cuTAMP (the sibling ``../cuTAMP`` repo) on a standalone toys environment
(``assets/toys_tamp.yml``, mirroring DROID sim "scene 6": place three toys on a plate),
then writes a factor-graph representation of the resulting plan:

    * ``<out>/plan_graph.json`` -- node-link JSON (operators, variables, constraints, costs)
      for easy programmatic analysis.
    * ``<out>/plan_graph.dot``  -- Graphviz DOT for visualization.
    * ``<out>/plan_graph.png``  -- rendered image (only if the ``dot`` binary is installed).

Two modes:

    Full solve (default)
        Runs the real cuTAMP optimizer on GPU and annotates the graph's constraint/cost
        factors with the concrete values of a satisfying particle. Requires the cuTAMP
        environment (cuRobo, a CUDA GPU, PyTorch). Example::

            python tamp_plan_graph.py --out runs/toys_graph -n 512

    Skeleton only (``--skeleton-only``)
        Builds the graph from the symbolic task plan alone -- no GPU, no cuRobo, no torch.
        Useful to inspect the plan structure anywhere. Example::

            python tamp_plan_graph.py --skeleton-only --out runs/toys_graph

The cuTAMP repo is located automatically at ``../cuTAMP`` (override with ``--cutamp-root``).
"""

import argparse
import logging
import sys
from pathlib import Path

_log = logging.getLogger("tamp_plan_graph")

_HERE = Path(__file__).resolve().parent
_DEFAULT_ENV = _HERE / "assets" / "toys_tamp.yml"
_DEFAULT_CUTAMP_ROOT = _HERE.parent / "cuTAMP"


def _add_cutamp_to_path(cutamp_root: Path) -> None:
    if not cutamp_root.exists():
        raise FileNotFoundError(
            f"cuTAMP repo not found at {cutamp_root}. Pass --cutamp-root to point at it."
        )
    sys.path.insert(0, str(cutamp_root))


def build_skeleton_graph(env_path: Path, out_dir: Path, max_skeletons: int = 1) -> dict:
    """CPU-only path: build the plan graph from the symbolic task plan (no torch/cuRobo/GPU)."""
    import yaml

    from cutamp.tamp_domain import get_initial_state, all_tamp_operators, all_tamp_fluents
    from cutamp.task_planning import task_plan_generator
    from cutamp.plan_graph import build_plan_graph, save_plan_graph

    env_dict = yaml.safe_load(env_path.read_text())
    types = env_dict.get("types", {})
    movables = types.get("Movable", [])
    surfaces = types.get("Surface", [])
    sticks = types.get("Stick", [])
    buttons = types.get("Button", [])

    initial_state = get_initial_state(movables=movables, surfaces=surfaces, sticks=sticks, buttons=buttons)

    name_to_fluent = {f.name: f for f in all_tamp_fluents}
    goal_atoms = set()
    for atom_dict in env_dict.get("goal", []):
        (fluent_name, values), = atom_dict.items()
        goal_atoms.add(name_to_fluent[fluent_name].ground(*values))
    goal_state = frozenset(goal_atoms)

    _log.info("Task planning for goal: %s", sorted(str(a) for a in goal_state))
    plan_gen = task_plan_generator(
        initial_state, goal_state, operators=all_tamp_operators, max_plan_skeletons=max_skeletons
    )
    try:
        plan_skeleton = next(iter(plan_gen))
    except StopIteration:
        raise RuntimeError("Task planner found no plan skeleton for the toys goal")

    _log.info("Found skeleton with %d operators", len(plan_skeleton))
    graph = build_plan_graph(
        plan_skeleton,
        name=env_dict.get("name", env_path.stem),
        initial_state=initial_state,
        goal_state=goal_state,
        solved=None,
        extra_metadata={"mode": "skeleton_only", "source_env": str(env_path)},
    )
    paths = save_plan_graph(graph, out_dir)
    _log.info("Wrote: %s", ", ".join(f"{k}={v}" for k, v in paths.items()))
    return graph


def run_full_solve(env_path: Path, out_dir: Path, args: argparse.Namespace) -> None:
    """GPU path: run the real cuTAMP solve; run_cutamp writes the plan graph into out_dir."""
    from cutamp.algorithm import run_cutamp
    from cutamp.config import TAMPConfiguration, validate_tamp_config
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.envs.utils import load_env
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol, setup_logging

    setup_logging()
    env = load_env(str(env_path))
    _log.info("Loaded env:\n%s", env)

    config = TAMPConfiguration(
        num_particles=args.num_particles,
        robot=args.robot,
        grasp_dof=4,
        approach="optimization",
        num_opt_steps=args.num_opt_steps,
        max_loop_dur=args.max_duration,
        num_initial_plans=args.num_initial_plans,
        curobo_plan=args.motion_plan,
        enable_visualizer=False,
        # New features in this cuTAMP fork.
        placement_check="obb",
        placement_shrink_dist=0.0,
        prop_satisfying_break=0.1,
        # The whole point of this script:
        save_plan_graph=True,
        experiment_root=str(out_dir.parent),
    )
    validate_tamp_config(config)

    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

    # Pass experiment_dir=out_dir so the plan graph lands directly in the requested directory.
    _, num_satisfying, failure_reason = run_cutamp(
        env,
        config,
        cost_reducer,
        constraint_checker,
        experiment_id=out_dir.name,
        experiment_dir=out_dir,
    )
    if failure_reason:
        _log.warning("cuTAMP did not fully solve: %s", failure_reason)
    _log.info("Done. num_satisfying=%s. Plan graph written under %s", num_satisfying, out_dir)

    graph_json = out_dir / "plan_graph.json"
    if graph_json.exists():
        _log.info("Plan graph: %s", graph_json)
    else:
        _log.warning("Expected plan graph at %s but it was not written", graph_json)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", type=Path, default=_DEFAULT_ENV, help="Path to the toys cuTAMP env YAML")
    parser.add_argument("--out", type=Path, default=_HERE / "runs" / "toys_plan_graph", help="Output directory")
    parser.add_argument("--cutamp-root", type=Path, default=_DEFAULT_CUTAMP_ROOT, help="Path to the cuTAMP repo")
    parser.add_argument(
        "--skeleton-only",
        action="store_true",
        help="Build the graph from the symbolic task plan only (no GPU / torch / cuRobo).",
    )
    parser.add_argument("-n", "--num-particles", type=int, default=512, help="Particles (full solve only)")
    parser.add_argument("--num-opt-steps", type=int, default=1000, help="Optimization steps (full solve only)")
    parser.add_argument("--num-initial-plans", type=int, default=30, help="Initial skeletons to sample (full solve)")
    parser.add_argument("--max-duration", type=float, default=None, help="Max optimize seconds (full solve only)")
    parser.add_argument("--motion-plan", action="store_true", help="Also run cuRobo motion refinement (full solve)")
    parser.add_argument("--robot", default="fr3_robotiq", help="Robot embodiment (full solve only)")
    parser.add_argument(
        "--self-contained",
        action="store_true",
        help="Inline vis-network into the HTML so it opens offline (downloads/caches the lib once).",
    )
    parser.add_argument("--vis-js", type=Path, default=None, help="Path to vis-network.min.js to inline (implies --self-contained)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    _add_cutamp_to_path(args.cutamp_root)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env_path = args.env.resolve()

    if args.skeleton_only:
        _log.info("Skeleton-only mode (no GPU). env=%s out=%s", env_path, out_dir)
        build_skeleton_graph(env_path, out_dir)
    else:
        _log.info("Full cuTAMP solve. env=%s out=%s", env_path, out_dir)
        run_full_solve(env_path, out_dir, args)

    # Optionally rewrite the HTML with vis-network inlined so it works offline. We regenerate from
    # the just-written plan_graph.json, so this covers both the skeleton and full-solve paths.
    if args.self_contained or args.vis_js is not None:
        _maybe_make_self_contained(out_dir, args)


def _resolve_vis_js(args) -> "str | None":
    """Return vis-network JS source to inline, or None if it can't be obtained."""
    if args.vis_js is not None:
        return Path(args.vis_js).read_text(encoding="utf-8")
    # Cache under the cuTAMP repo's assets so repeated runs don't re-download.
    cache = Path(args.cutamp_root) / "cutamp" / "assets" / "vis-network.min.js"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    url = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"
    try:
        import urllib.request

        _log.info("Downloading vis-network for offline HTML (one-time) ...")
        with urllib.request.urlopen(url, timeout=30) as resp:
            js = resp.read().decode("utf-8")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(js, encoding="utf-8")
        return js
    except Exception as e:
        _log.warning("Could not fetch vis-network (%s); HTML will use the CDN instead.", e)
        return None


def _maybe_make_self_contained(out_dir: Path, args) -> None:
    import json as _json

    from cutamp.plan_graph import save_plan_graph_html

    vis_js = _resolve_vis_js(args)
    if vis_js is None:
        return
    graph_json = out_dir / "plan_graph.json"
    if not graph_json.exists():
        _log.warning("No plan_graph.json found in %s; cannot inline vis-network.", out_dir)
        return
    graph = _json.loads(graph_json.read_text())
    html_path = save_plan_graph_html(graph, out_dir / "plan_graph.html", inline_vis=vis_js)
    _log.info("Wrote self-contained (offline) HTML: %s", html_path)


if __name__ == "__main__":
    main()
