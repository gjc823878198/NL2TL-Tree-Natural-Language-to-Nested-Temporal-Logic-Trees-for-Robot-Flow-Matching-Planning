"""
Sensor-grounded "always keep safe" STL predicate.

We express obstacle avoidance as ONE generic temporal-logic clause

        keep_safe  :=  G ( not unsafe )

and ground the single predicate `unsafe` with the live obstacle set the robot
actually perceives (LiDAR returns and/or known walls), rather than enumerating
a fixed avoid atom per obstacle:

        unsafe(p)        :=  p is inside ANY sensed/known obstacle disk
        rho(unsafe, p)   :=  max_k ( r_k - dist(p, c_k) )      (>0 inside)
        rho(not unsafe)  :=  min_k ( dist(p, c_k) - r_k )      (signed clearance)
        rho(G not unsafe):=  min_t  rho(not unsafe, p_t)       (worst clearance)

Why a union predicate instead of N avoid atoms: TeLoGraF's GNN encodes one disk
per avoid atom, so dozens of sensed disks would push it out of distribution.
A single `unsafe` predicate keeps the symbolic spec compact and lets the SAME
sensor set (a) ground the STL "keep safe" semantics + robustness used for
verification, and (b) drive the planner's test-time guidance.  The GNN keeps
seeing the in-distribution abstract spec; the sensed geometry enters through
this predicate's grounding + guidance.
"""
from __future__ import annotations

import math
from typing import List, Tuple

KEEP_SAFE_ATOM = "keep_safe"


def obstacles_to_disks(obstacles) -> List[dict]:
    """Normalise mixed circle/rect obstacle dicts to a flat list of disks."""
    if not obstacles:
        return []
    from planner.telograf_infer import _obstacles_to_disks
    return [{"x": float(x), "y": float(y), "r": float(r)}
            for (x, y, r) in _obstacles_to_disks(obstacles)]


def keep_safe_grounding(obstacles) -> dict:
    """Grounding entry that grounds the `unsafe` predicate with `obstacles`."""
    return {KEEP_SAFE_ATOM: {"kind": "keep_safe",
                             "disks": obstacles_to_disks(obstacles)}}


def keep_safe_clause() -> dict:
    """The STL subtree  G(not unsafe)."""
    return {"op": "globally", "interval": None, "children": [
        {"op": "not", "interval": None, "children": [
            {"op": "atom", "name": KEEP_SAFE_ATOM}]}]}


def with_keep_safe(tree: dict, grounding: dict, obstacles):
    """Return (full_tree, full_grounding) = original AND G(not unsafe),
    with `unsafe` grounded by `obstacles`.  Used for the verifiable spec and
    its robustness; the GNN-facing planning tree stays the original `tree`."""
    full = {"op": "and", "interval": None, "children": [tree, keep_safe_clause()]}
    g = dict(grounding)
    g.update(keep_safe_grounding(obstacles))
    return full, g


def keep_safe_robustness(obstacles, traj: List[Tuple[float, float]]) -> float:
    """rho( G not unsafe ) of `traj` against `obstacles`.

    Now a thin wrapper over the duration-aware monitor (keep_safe_robustness_da,
    duration-severity).  On a satisfied path this equals the worst clearance, so
    every number we report is unchanged; under violation it additionally
    penalises how long the path is unsafe."""
    if len(traj) == 0:
        return float("-inf")
    return keep_safe_robustness_da(obstacles, traj, mode="duration_severity")


def clearance_series(obstacles, traj: List[Tuple[float, float]]) -> List[float]:
    """Signed clearance  rho(not unsafe, p_t) = min_k(dist(p_t,c_k) - r_k)  at
    each step (>0 = clear of every obstacle by that margin; <0 = inside one)."""
    disks = obstacles_to_disks(obstacles)
    if not disks:
        return [1e3] * len(traj)          # no obstacles -> trivially clear
    return [min(math.hypot(x - d["x"], y - d["y"]) - d["r"] for d in disks)
            for (x, y) in traj]


def keep_safe_robustness_da(obstacles, traj: List[Tuple[float, float]],
                            mode: str = "duration_severity") -> float:
    """Duration-aware rho( G not unsafe ) (Finkeldei et al.).

    The standard STL value is the WORST clearance (min over time), which only
    captures the single most-severe instant and ignores how LONG the path is
    unsafe.  We instead aggregate the signed-clearance series soundly:

      mode="worst"             : min_t clearance        (standard STL)
      mode="duration"          : if ever unsafe, -(fraction of time unsafe);
                                 else min_t clearance        (H^D)
      mode="duration_severity" : if ever unsafe, the MEAN negative clearance
                                 over the horizon; else min_t clearance  (H^DS)

    On a SATISFIED path (no violation) every mode returns the worst clearance,
    so the certificate and the numbers we report are unchanged; the modes differ
    only under violation, where duration-(severity) additionally penalises the
    DURATION of the incursion.  The sign always matches Boolean satisfaction
    (rho>0 iff the path never enters an obstacle), so rho>0 stays a valid
    keep-safe certificate (sound, bounded, monotonic)."""
    if len(traj) == 0:
        return float("-inf")
    series = clearance_series(obstacles, traj)
    neg = [s for s in series if s < 0]
    if not neg or mode == "worst":        # satisfied, or standard worst-case
        return min(series)
    T = len(series)
    if mode == "duration":
        return -len(neg) / T
    return sum(neg) / T                    # duration_severity (default)
