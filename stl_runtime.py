"""
Runtime STL certificate over a *time-stamped* trajectory (real seconds).

Companion to ``stl_robustness.py``.  That module evaluates the exact (sound,
non-smoothed) quantitative robustness, but on a trajectory whose index axis is
abstract ("1 step = 1 time unit").  The closed loop needs the timed operators to
refer to *wall-clock seconds*, so a deadline ``F_{[0,D]} reach`` means "enter the
goal within D seconds of real driving".  This module anchors the index axis to
seconds and exposes a single ``reach-by-deadline`` certificate that is used as

  (a) the planner's *temporal gate*  (does the plan reach in time?), and
  (b) the closed-loop *runtime shield* (did the executed path reach in time?).

Anchoring rule.  A geometric path is resampled to a uniform ``dt``-second grid
under a constant nominal tracking speed ``v`` (arclength / v = time), so step
``i`` is exactly ``t = i*dt`` seconds.  A trajectory that is *already* uniform in
time (e.g. the executed ``travelled`` list, one entry per control tick) is used
directly with ``dt`` = the control period -- no resampling needed.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from stl_robustness import robustness

Pt = Tuple[float, float]


def resample_by_time(traj: Sequence[Pt], dt: float, speed: float) -> List[Pt]:
    """Resample a geometric path to a uniform ``dt``-second grid, assuming the
    robot traverses it at constant ``speed`` m/s (so step ``i`` is ``i*dt`` s).
    """
    if len(traj) < 2:
        return [tuple(p) for p in traj]
    s = [0.0]
    for i in range(1, len(traj)):
        s.append(s[-1] + math.hypot(traj[i][0] - traj[i - 1][0],
                                    traj[i][1] - traj[i - 1][1]))
    total_t = s[-1] / max(speed, 1e-6)
    n = max(2, int(round(total_t / dt)) + 1)
    out: List[Pt] = []
    j = 0
    for k in range(n):
        target = (k * dt) * speed
        while j < len(s) - 1 and s[j + 1] < target:
            j += 1
        if j >= len(traj) - 1:
            out.append((float(traj[-1][0]), float(traj[-1][1])))
            continue
        seg = s[j + 1] - s[j]
        a = 0.0 if seg < 1e-9 else (target - s[j]) / seg
        out.append((float(traj[j][0] + a * (traj[j + 1][0] - traj[j][0])),
                    float(traj[j][1] + a * (traj[j + 1][1] - traj[j][1]))))
    return out


def reach_by_deadline_rho(goal: Tuple[float, float, float],
                          traj_uniform: Sequence[Pt],
                          dt: float, deadline_s: float) -> float:
    """rho( F_{[0,deadline]} reach goal ) over a trajectory uniformly sampled at
    ``dt`` seconds/step.  >0 iff the path enters the goal disk by the deadline,
    with the margin = goal_radius - closest approach inside the window.
    """
    if not traj_uniform:
        return float("-inf")
    gx, gy, gr = goal
    K = max(0, min(len(traj_uniform) - 1, int(round(deadline_s / dt))))
    tree = {"op": "finally", "interval": [0, K],
            "children": [{"op": "atom", "name": "g"}]}
    grounding = {"g": {"x": gx, "y": gy, "r": gr, "kind": "reach"}}
    return robustness(tree, grounding, list(traj_uniform))


def plan_reach_by_deadline(plan: Sequence[Pt],
                           goal: Tuple[float, float, float],
                           deadline_s: float, speed: float,
                           dt: float = 0.15) -> Tuple[float, float]:
    """Temporal certificate for a *planned* (geometric) path: time-stamp it at
    ``speed`` and test reach-by-deadline.  Returns (rho_seconds, est_time_s).
    """
    uni = resample_by_time(plan, dt, speed)
    est = (len(uni) - 1) * dt
    return reach_by_deadline_rho(goal, uni, dt, deadline_s), est
