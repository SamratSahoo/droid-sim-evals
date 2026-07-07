#!/usr/bin/env python3
"""Recompute whole-number scores + summary from existing results.json -- NO sim re-run.

Every ``results.json`` stores the per-env boolean criteria (``per_env``), so after a threshold-only change
in ``scene_specs.py`` (that affects only aggregation, not which criteria the evaluator emits) you can
re-score all collected rollouts instantly:

    .venv/bin/python eval/rescore.py

Under the count model each criterion is an independent whole point: per-env score = number of criteria met;
success = all met. Rewrites each ``runs/pi05-eval-v2/<policy>/<scene>/results.json`` (dense = mean_points/
max_points, sparse = success_rate, + mean_points/max_points/per_env_points/criterion_rates) and regenerates
``summary.{json,txt}``. Results collected under the OLD gated/fractional model (no ``max_points`` key) are
skipped -- they must be re-run, not rescored.
"""

import json
import sys
from pathlib import Path

import numpy as np

_EVAL = Path(__file__).resolve().parent
_DSE = _EVAL.parent
sys.path[:0] = [str(_DSE), str(_EVAL)]
from scene_specs import get_scene, score_env

_OUT = _DSE / "runs" / "pi05-eval-v2"
SCENE_TASKS = {6: "toys", 8: "pan_clean", 9: "block_stack", 10: "button", 11: "cabinet", 12: "blocks_plate"}


def _fmt_summary(summary: dict, scenes: list) -> str:
    cols = [SCENE_TASKS.get(s, str(s)) for s in scenes]
    w = max(14, max((len(c) for c in cols), default=0) + 2)
    head = f"{'policy':32s} | " + " | ".join(f"{c:>{w}s}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, per in summary.items():
        cells = []
        for s in scenes:
            r = per.get(str(s))
            cells.append(f"{r['mean_points']:.1f}/{r['max_points']} {r['sparse']:.2f}" if r else "  --/-- ")
        lines.append(f"{name:32s} | " + " | ".join(f"{c:>{w}s}" for c in cells))
    lines.append("\n(cells are MEAN_POINTS/MAX_POINTS SUCCESS_RATE: mean_points = avg # of whole-point boolean "
                 "sub-goals met per rollout; success_rate = frac envs meeting ALL criteria)")
    return "\n".join(lines)


def main() -> None:
    summary: dict = {}
    for rp in sorted(_OUT.glob("*/*/results.json")):
        r = json.loads(rp.read_text())
        if "per_env" not in r or not r["per_env"]:
            print(f"skip {rp} (no per_env)")
            continue
        if "max_points" not in r:
            print(f"skip {rp} (old gated/fractional model -- re-run the eval, cannot rescore)")
            continue
        try:
            spec = get_scene(r["scene"])                 # KeyError for a removed scene
        except KeyError:
            print(f"skip {rp} (unknown/removed scene {r['scene']})")
            continue
        per_env = r["per_env"]
        criteria = list(per_env[0].keys())
        scored = [score_env(e, spec) for e in per_env]
        pts = [int(s["points"]) for s in scored]
        maxp = int(scored[0]["max_points"])
        r["max_points"] = maxp
        r["per_env_points"] = pts
        r["mean_points"] = float(np.mean(pts))
        r["success_rate"] = float(np.mean([s["success"] for s in scored]))
        r["dense_score"] = r["mean_points"] / maxp if maxp else 0.0
        r["sparse_score"] = r["success_rate"]
        r["criterion_rates"] = {c: float(np.mean([e[c] for e in per_env])) for c in criteria}
        rp.write_text(json.dumps(r, indent=2))
        policy = rp.parent.parent.name
        summary.setdefault(policy, {})[str(r["scene"])] = {
            "dense": r["dense_score"], "sparse": r["sparse_score"],
            "mean_points": r["mean_points"], "max_points": maxp,
        }
        print(f"{policy}/{SCENE_TASKS.get(r['scene'], r['scene']):12s} "
              f"mean_points={r['mean_points']:.2f}/{maxp} success_rate={r['sparse_score']:.3f}")
    if summary:
        scenes = sorted({int(s) for per in summary.values() for s in per})
        (_OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        (_OUT / "summary.txt").write_text(_fmt_summary(summary, scenes))
        print(f"\nrewrote {_OUT / 'summary.json'} + summary.txt")
    else:
        print("no rescorable results.json found under", _OUT)


if __name__ == "__main__":
    main()
