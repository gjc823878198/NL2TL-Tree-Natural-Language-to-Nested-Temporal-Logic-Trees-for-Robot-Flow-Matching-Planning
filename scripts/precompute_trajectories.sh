#!/usr/bin/env bash
# Offline utility: run the planner for every case WITHOUT ROS/Gazebo and dump
# each trajectory to outputs/trajectories/<case>__<backend>.json, for quick
# inspection / sanity-checking of plan_waypoints output.
#
# Note: the live ROS closed loop (tb3_follower.py) re-plans with TeLoGraF every
# 5 s from the robot's current pose -- it does NOT read these caches. This script
# is purely for offline trajectory inspection.
#
# Usage (no need to source ROS):
#   bash scripts/precompute_trajectories.sh                 # role backend
#   BACKEND=telograf bash scripts/precompute_trajectories.sh
#   BACKEND=fallback bash scripts/precompute_trajectories.sh

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$HERE/.." && pwd)"
cd "$CODE_ROOT"

# role: telograf-primary cases -> telograf; fallback cases -> auto
ROLE_BACKEND="${BACKEND:-role}"

python3 - "$ROLE_BACKEND" <<'PY'
import sys, json
from pathlib import Path
sys.path.insert(0, ".")
from planner import CASES, plan_waypoints

backend_arg = sys.argv[1]
out = Path("outputs/trajectories"); out.mkdir(parents=True, exist_ok=True)
for c in CASES:
    role = c.get("role", "fallback")
    backend = (("telograf" if role == "telograf" else "auto")
               if backend_arg == "role" else backend_arg)
    print(f"computing {c['id']:18s} backend={backend} ...", flush=True)
    wp = plan_waypoints(c, n_steps=96, backend=backend)
    # write one file per backend alias for easy lookup:  <case>__<backend>.json
    for key in {backend, "auto", "role"}:
        (out / f"{c['id']}__{key}.json").write_text(json.dumps(
            {"case": c["id"], "backend": backend,
             "waypoints": [[float(x), float(y)] for x, y in wp]}))
    print(f"  -> {len(wp)} pts, start={tuple(round(v,2) for v in wp[0])}, "
          f"end={tuple(round(v,2) for v in wp[-1])}", flush=True)
print("done. caches in outputs/trajectories/")
PY