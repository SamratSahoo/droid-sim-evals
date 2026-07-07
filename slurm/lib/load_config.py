#!/usr/bin/env python3
"""Render a slurm YAML config into shell ``export`` lines (+ a cuRobo-overrides JSON sidecar).

Each ``*.slurm`` under ``droid-sim-evals/slurm/`` sources this to turn its ``cfg/<name>.yml``
default (optionally overlaid with a user-supplied YAML passed as the script's first arg) into
environment variables, so runtime knobs live in version-controlled config files instead of being
baked into the shell scripts.

Usage (inside a slurm script)::

    eval "$(<dse>/.venv/bin/python slurm/lib/load_config.py \
              --default cfg/<name>.yml [--config "$USER_CFG"] \
              [--overrides-out /path/tamp_overrides.json])"

Rules:
  * ``--config`` (if given) is deep-merged OVER ``--default``, so a user file need only set the
    keys it wants to change.
  * Every top-level scalar (str/int/float/bool) key ``K`` becomes ``export UPPER_K=<shell-quoted>``
    (bools render as ``true`` / ``false``; ``null`` keys are skipped).
  * The reserved top-level mapping ``tamp_overrides`` is written verbatim as JSON to
    ``--overrides-out`` and exposed as ``export TAMP_OVERRIDES_JSON=<path>`` -- but only when it is
    non-empty AND an output path was provided. This is the cuRobo cost-override dict the tiptop
    planning server consumes (vae_manifold_weight, rnd_novelty_weight, smooth_weight,
    time_dilation_factor[_literal], ...); see tiptop.motion_planning.apply_cost_overrides.
  * Any other top-level mapping/list is skipped with a note to stderr (comments only; stdout stays
    a clean set of ``export`` lines safe to ``eval``).
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

import yaml

RESERVED_OVERRIDES = "tamp_overrides"


def deep_merge(base: dict, over: dict | None) -> dict:
    """Recursively merge ``over`` into ``base`` (over wins); returns a new dict."""
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def emit_scalar(key: str, val) -> None:
    sval = ("true" if val else "false") if isinstance(val, bool) else str(val)
    print(f"export {key.upper()}={shlex.quote(sval)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--default", required=True, help="path to the default cfg/<name>.yml")
    ap.add_argument("--config", default=None, help="optional user YAML deep-merged over the default")
    ap.add_argument("--overrides-out", default=None, help="where to write the tamp_overrides JSON sidecar")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.default).read_text()) or {}
    if not isinstance(cfg, dict):
        sys.exit(f"load_config: default {args.default} must be a YAML mapping")
    if args.config:
        user = yaml.safe_load(Path(args.config).read_text()) or {}
        if not isinstance(user, dict):
            sys.exit(f"load_config: user config {args.config} must be a YAML mapping")
        cfg = deep_merge(cfg, user)
        print(f"# load_config: merged user config {args.config}", file=sys.stderr)

    overrides = cfg.get(RESERVED_OVERRIDES) or {}
    for k, v in cfg.items():
        if k == RESERVED_OVERRIDES:
            continue
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            print(f"# load_config: skipping non-scalar top-level key {k!r}", file=sys.stderr)
            continue
        emit_scalar(k, v)

    if overrides and args.overrides_out:
        outp = Path(args.overrides_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(overrides, indent=2))
        print(f"export TAMP_OVERRIDES_JSON={shlex.quote(str(outp))}")
        print(f"# load_config: wrote {len(overrides)} tamp overrides -> {outp}", file=sys.stderr)
    elif overrides:
        print("# load_config: WARNING tamp_overrides present but no --overrides-out; ignored", file=sys.stderr)


if __name__ == "__main__":
    main()
