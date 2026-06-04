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


# --------------------------------------------------------------------------
# Planning-latency-aware timed robustness  (our contribution)
#
# The reach-by-deadline robustness above compares the path's EXECUTION time to
# the deadline and implicitly assumes planning is free -- i.e. that wall-clock
# time *freezes* while the planner thinks.  It does not: the flow-matching
# planner pays a model cold-start on the first call and a few seconds per
# re-plan, all of which also count against a "reach within D seconds" deadline.
# We PREDICT that planning latency and DEBIT it from the temporal budget, so the
# timed guarantee stays honest, and we hand the *tightened* deadline back to the
# next re-plan so subsequent trajectories are planned against the time that will
# actually remain.
# --------------------------------------------------------------------------

class PlanLatencyModel:
    """Online predictor of the planner's wall-clock latency.

    The first call is slow (model cold-start); warm re-plans are faster, so an
    EMA over observed plan durations (cold-start-seeded) predicts the next plan's
    cost.  ``consumed()`` is the wall-clock already spent planning this episode;
    ``reserve(n)`` is what to debit from the budget now -- everything already
    spent plus a prediction for ``n`` re-plans still expected on this leg."""

    def __init__(self, cold_start_s: float = 12.0, ema: float = 0.4):
        self._est = float(cold_start_s)
        self._ema = float(ema)
        self._n = 0
        self._total = 0.0

    def observe(self, dt_s: float) -> None:
        dt_s = max(0.0, float(dt_s))
        self._est = dt_s if self._n == 0 else \
            (1.0 - self._ema) * self._est + self._ema * dt_s
        self._n += 1
        self._total += dt_s

    def predict_next(self) -> float:
        return self._est

    def consumed(self) -> float:
        return self._total

    def reserve(self, n_future_replans: int = 0) -> float:
        return self._total + max(0, int(n_future_replans)) * self._est


def latency_aware_reach_rho(plan: Sequence[Pt],
                            goal: Tuple[float, float, float],
                            deadline_s: float, speed: float,
                            latency: "PlanLatencyModel",
                            *, n_future_replans: int = 0,
                            dt: float = 0.15) -> Tuple[float, float, float, float]:
    """Planning-latency-aware reach-by-deadline robustness (our contribution).

        rho_latency = (deadline - reserved_planning_time) - execution_time
                    = rho_naive_time - reserved_planning_time.

    Returns ``(rho_latency_s, rho_naive_time_s, est_exec_s, reserved_s)``.  The
    naive temporal margin ignores planning cost; ours debits the predicted plan
    wall-clock, so ``rho_latency <= rho_naive`` always and a plan that "arrives
    in time" on paper but blows the deadline once its own planning latency is
    counted is correctly flagged (``rho_latency <= 0``)."""
    _, est = plan_reach_by_deadline(plan, goal, deadline_s, speed, dt)
    reserved = latency.reserve(n_future_replans)
    rho_naive_time = deadline_s - est
    return rho_naive_time - reserved, rho_naive_time, est, reserved
