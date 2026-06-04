"""
Pre-flight completability gauge for the demo.

BEFORE committing to the full flow-matching run, we cheaply estimate whether the
task can finish within its deadline:

  1. A* a collision-free path through the goals on the dynamics model
     (the planner's classical fallback backend -- no flow sampling, fast);
  2. inflate its length by a *detour factor* for the clutter the reactive flow
     planner + tracker will actually weave through (the user picks a coarse
     environment level: open / normal / complex -- denser clutter => longer
     realized path than the A* shortest path);
  3. convert to time at the nominal speed and ADD the predicted planner latency
     (the world keeps turning while we plan -- the same planning-latency-aware
     accounting used elsewhere);
  4. compare to the deadline and return a graded *completability*, fed back to
     the user before anything moves.

This is a lightweight, planning-free feasibility check that complements the
in-the-loop best-of-N self-check: that one asks "can the flow planner satisfy
this spec here?"; this one asks "is the deadline even reachable, planning time
included?" -- answered up front.

The numeric detour factors and the predicted-latency split below are a coarse
prior tuned once for the demo map; they are deliberately NOT presented as magic
constants in the write-up, only the concept is.
"""
from __future__ import annotations
import contextlib
import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nl_grounding as _G                                          # noqa: E402  (sets CODE on sys.path)

# coarse environment level -> detour (tortuosity) factor: denser clutter makes
# the realized weaving path longer relative to the A* estimate.  (Coarse prior
# for the demo map; the WRITE-UP states only the concept, never these numbers.)
ENV_LEVELS = ["open", "normal", "complex"]
_DETOUR = {"open": 1.0, "normal": 1.12, "complex": 1.35}
# predicted planner wall-clock to reserve: a cold-start, plus a warm re-plan per
# extra leg (latency-aware -- we do not assume planning is free).
_COLD_START_S = 8.0
_WARM_REPLAN_S = 3.0


def _leg_len(start, goal, obstacles) -> float:
    from planner import plan_waypoints
    gx, gy, gr = goal
    case = {"id": "feas_leg", "role": "fallback",
            "tree": {"op": "finally", "interval": None,
                     "children": [{"op": "atom", "name": "g"}]},
            "grounding": {"g": {"kind": "reach", "x": gx, "y": gy,
                                "z": 0.0, "r": gr}},
            "map_hint": {"world_bounds": list(_G.WORLD_BOUNDS),
                         "start": [float(start[0]), float(start[1])],
                         "style": "telograf_native"}}
    with contextlib.redirect_stdout(io.StringIO()):
        path = plan_waypoints(case, n_steps=60, backend="fallback",
                              obstacles=obstacles)
    return sum(math.hypot(path[i][0] - path[i - 1][0],
                          path[i][1] - path[i - 1][1])
               for i in range(1, len(path)))


def estimate(start, goals, deadline_s, env_level, obstacles, *, v_nom=0.18):
    """Pre-flight estimate. `goals` is the ordered list of (x, y, r). Returns a
    dict with the execution-time and planning-time estimates, the deadline
    margin, and a graded `completability` in [0, 1]."""
    detour = _DETOUR.get(env_level, _DETOUR["normal"])
    pt = (float(start[0]), float(start[1]))
    raw = 0.0
    for g in goals:
        raw += _leg_len(pt, g, obstacles)
        pt = (g[0], g[1])
    path_len = raw * detour
    t_exec = path_len / v_nom
    t_plan = _COLD_START_S + max(0, len(goals) - 1) * _WARM_REPLAN_S
    t_finish = t_exec + t_plan
    margin = (deadline_s - t_finish) if deadline_s else float("nan")
    comp = max(0.0, min(1.0, margin / deadline_s)) if deadline_s else 0.0
    return {"t_exec": t_exec, "t_plan": t_plan, "t_finish": t_finish,
            "deadline": deadline_s, "margin": margin, "completability": comp,
            "level": env_level, "detour": detour, "path_len": path_len}


def verdict(est) -> str:
    d, m = est["deadline"], est["margin"]
    if not d:
        return "no deadline"
    if m <= 0:
        return "unlikely"
    return "tight" if m < 0.10 * d else "likely"


def summary(est) -> str:
    """One-line, human-facing gauge (no raw factors)."""
    return (f"pre-flight completability: {verdict(est).upper()} "
            f"({est['completability']*100:.0f}% time-budget margin) | "
            f"est. finish {est['t_finish']:.0f}s "
            f"(move {est['t_exec']:.0f}s + plan {est['t_plan']:.0f}s) "
            f"vs deadline {est['deadline']:.0f}s | env: {est['level']}")


if __name__ == "__main__":
    from sim_ros2.scenario import CYLINDERS
    obs = [{"kind": "circle", "x": x, "y": y, "r": r} for (x, y, r) in CYLINDERS]
    goals = [_G.LANDMARKS[n] for n in ("B", "A", "C")]
    for lvl in ENV_LEVELS:
        e = estimate(_G.START, goals, 180.0, lvl, obs)
        print(summary(e))
