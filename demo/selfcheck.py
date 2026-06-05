"""Best-of-N feasibility / OOD self-check surfaced to the demo user (2D + ROS).

Thin wrapper over ``planner.telograf_infer.telograf_selfcheck``: it batch-samples
the frozen TeLoGraF flow prior for the WHOLE task, measures per-goal
reach-shortfall, and decides flow vs A* (OOD) -- the paper's Tier-3 self-check
(Fig.~1, ``best-of-N self-check'').  It returns the candidate trajectories and
which goals even the best candidate cannot reach, so the demo can VISUALISE
*why* a spec is out-of-distribution (and why we then decompose / defer to A*).

The self-check runs on a REACH-ONLY joint case (the keep-safe G-not-unsafe clause
has no fixed geometry -- it is grounded by sensed obstacles at run time -- so it
cannot enter TeLoGraF's graph encoder; the obstacles are passed separately, the
same way the demo plans each leg).
"""
from __future__ import annotations


def _joint_reach_case(goals, names, deadline_s, world_bounds, start, n_samples):
    """A conjunction of timed reaches over all goals -- the joint spec whose
    in/out-of-distribution status the prior is asked about."""
    grounding, children = {}, []
    for i, g in enumerate(goals):
        gx, gy, gr = g[0], g[1], g[2]
        nm = f"reach_{names[i] if i < len(names) else i}"
        grounding[nm] = {"kind": "reach", "x": gx, "y": gy, "z": 0.0, "r": gr}
        children.append({"op": "finally", "interval": [0, int(deadline_s)],
                         "children": [{"op": "atom", "name": nm}]})
    tree = ({"op": "and", "interval": None, "children": children}
            if len(children) > 1 else children[0])
    return {"id": "selfcheck_joint", "role": "telograf", "tree": tree,
            "grounding": grounding,
            "map_hint": {"world_bounds": list(world_bounds), "start": list(start),
                         "style": "telograf_native", "telograf_samples": n_samples}}


def run(info, obstacles=None, *, world_bounds, start, n_samples=16) -> dict:
    """Return the self-check report (see ``telograf_selfcheck``), or
    ``{"available": False, ...}`` when the flow backend is missing -- the demo
    reports that honestly rather than faking a result."""
    try:
        from planner.telograf_infer import telograf_selfcheck
        case = _joint_reach_case(info["goals"], info["goal_names"],
                                 info["deadline_s"], world_bounds, start, n_samples)
        return telograf_selfcheck(case, extra_obstacles=obstacles or [],
                                  n_samples=n_samples)
    except Exception as e:                       # pragma: no cover - venue safety
        return {"available": False, "error": str(e)}


def summary(rep: dict) -> str:
    """One-line, attendee-facing summary of the self-check verdict."""
    if not rep.get("available"):
        msg = "best-of-N self-check: skipped (flow backend unavailable)"
        return msg + (f" [{rep['error']}]" if rep.get("error") else "")
    miss = [g for g in rep.get("per_goal", []) if not g["reached"]]
    verdict = ("FEASIBLE -> flow" if rep["feasible"]
               else f"OOD -> defer to A* ({len(miss)} goal(s) unreachable)")
    return (f"best-of-{rep['n_samples']} self-check: "
            f"satisfy-rate {rep['satisfy_rate']:.0%}, best shortfall "
            f"{rep['best_shortfall']:.2f} m (tol {rep['tol']:.1f} m) -> {verdict}")
