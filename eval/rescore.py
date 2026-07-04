#!/usr/bin/env python3
"""Recompute dense/sparse scores + summary from existing results.json -- NO sim re-run.

Every ``results.json`` stores the per-env raw criteria (``per_env``), so after changing the scoring
structure in ``scene_specs.py`` (multiplicative on-table gates, sequential-order credit, or a criterion
threshold that only affects aggregation) you can re-score all collected rollouts instantly:

    .venv/bin/python eval/rescore.py

Rewrites each ``runs/pi05-eval-v2/<policy>/<scene>/results.json`` (dense/sparse + raw & credited rates)
and regenerates ``runs/pi05-eval-v2/summary.{json,txt}``.
"""

import json
import sys
from pathlib import Path

import numpy as np

_EVAL = Path(__file__).resolve().parent
_DSE = _EVAL.parent
sys.path[:0] = [str(_DSE), str(_EVAL)]
from scene_specs import get_scene, resolve_gates, score_env

_OUT = _DSE / "runs" / "pi05-eval-v2"
SCENE_TASKS = {6: "toys", 7: "cubes", 8: "whiteboard", 9: "stack", 10: "button", 11: "cabinet", 12: "blocks_plate"}


def _fmt_summary(summary: dict, scenes: list) -> str:
    cols = [SCENE_TASKS.get(s, str(s)) for s in scenes]
    w = max(14, max((len(c) for c in cols), default=0) + 2)
    head = f"{'policy':32s} | " + " | ".join(f"{c:>{w}s}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, per in summary.items():
        cells = [(f"{per[str(s)]['dense']:.2f}/{per[str(s)]['sparse']:.2f}" if str(s) in per else "  --/-- ")
                 for s in scenes]
        lines.append(f"{name:32s} | " + " | ".join(f"{c:>{w}s}" for c in cells))
    lines.append("\n(cells are DENSE/SPARSE: dense=on-table-gated, sequence-credited mean; sparse=frac envs all-pass)")
    return "\n".join(lines)


def main() -> None:
    summary: dict = {}
    for rp in sorted(_OUT.glob("*/*/results.json")):
        r = json.loads(rp.read_text())
        if "per_env" not in r or not r["per_env"]:
            print(f"skip {rp} (no per_env)")
            continue
        spec = get_scene(r["scene"])
        per_env = r["per_env"]
        criteria = list(per_env[0].keys())
        scored = [score_env(e, spec) for e in per_env]
        r["dense_score"] = float(np.mean([s["dense"] for s in scored]))
        r["sparse_score"] = float(np.mean([s["sparse"] for s in scored]))
        r["gates"] = resolve_gates(spec, criteria)
        r["sequence"] = spec.sequence
        r["criterion_rates"] = {c: float(np.mean([e[c] for e in per_env])) for c in criteria}
        r["credited_rates"] = {c: float(np.mean([s["credited"][c] for s in scored])) for c in criteria}
        rp.write_text(json.dumps(r, indent=2))
        policy = rp.parent.parent.name
        summary.setdefault(policy, {})[str(r["scene"])] = {"dense": r["dense_score"], "sparse": r["sparse_score"]}
        print(f"{policy}/{SCENE_TASKS.get(r['scene'], r['scene']):12s} dense={r['dense_score']:.3f} "
              f"sparse={r['sparse_score']:.3f}  gates={r['gates']} seq={bool(r['sequence'])}")
    if summary:
        scenes = sorted({int(s) for per in summary.values() for s in per})
        (_OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        (_OUT / "summary.txt").write_text(_fmt_summary(summary, scenes))
        print(f"\nrewrote {_OUT / 'summary.json'} + summary.txt")
    else:
        print("no results.json found under", _OUT)


if __name__ == "__main__":
    main()
