"""
Quantitative STL robustness monitor (rho) over a discrete 2-D trajectory.

This is the *evaluation-time* companion to the differentiable robustness used
for guidance: given an STL parse tree, a grounding (atom -> disk), and a
trajectory (list of (x, y) at integer timesteps), it returns the standard
quantitative robustness  rho  (positive = satisfied, with margin).  We use it
for the Tier-4 self-correction loop: a parse whose robustness cannot be made
positive by the planner is flagged back to the LLM with the offending value.

Quantitative semantics (Donze & Maler / Fainekos & Pappas), min/max form:
  atom            rho_t = r - dist(pos_t, centre)        (>0 inside the disk)
  not phi         -rho(phi)
  and             min of children
  or              max of children
  imply(A,B)      max(-rho_A, rho_B)
  iff(A,B)        min(imply(A,B), imply(B,A))
  globally[a,b]   min over t in [a,b] of rho(sub)
  finally [a,b]   max over t in [a,b] of rho(sub)
  until[a,b](A,B) max_{t in [a,b]} min( rho_B(t), min_{t'<=t} rho_A(t') )

Intervals index into the trajectory (1 step = 1 time unit); interval=null
means the whole horizon.  Atom robustness is "inside-positive" so an avoid
constraint is written G(not(atom)) and reads as "always outside", which
matches planner/case_examples.py.
"""
from __future__ import annotations

import math
from typing import List, Tuple


def _atom_rho_series(name: str, grounding: dict,
                     traj: List[Tuple[float, float]]) -> List[float]:
    """Per-timestep robustness of an atom (inside-positive).

    A normal disk atom: r - distance(pos, centre).
    A `keep_safe` predicate (kind=="keep_safe") is grounded by a UNION of
    sensed/known obstacle disks; its inside-positive robustness is
    max_k (r_k - dist), i.e. >0 iff the point is inside ANY obstacle.  With no
    disks it is never unsafe, so we return a large negative constant (so
    `not unsafe` is safely positive)."""
    g = grounding[name]
    if g.get("kind") == "keep_safe":
        disks = g.get("disks", [])
        if not disks:
            return [-1e3] * len(traj)
        return [max(d["r"] - math.hypot(x - d["x"], y - d["y"]) for d in disks)
                for (x, y) in traj]
    cx, cy, r = g["x"], g["y"], g["r"]
    return [r - math.hypot(x - cx, y - cy) for (x, y) in traj]


def _interval_idx(interval, T: int) -> Tuple[int, int]:
    """Map an STL [a,b] interval to inclusive trajectory index bounds."""
    if not interval:
        return 0, T - 1
    a, b = interval
    return max(0, int(a)), min(T - 1, int(b))


def robustness_series(node: dict, grounding: dict,
                      traj: List[Tuple[float, float]]) -> List[float]:
    """Per-timestep robustness of `node` over the trajectory."""
    T = len(traj)
    op = node.get("op")
    if op == "atom":
        return _atom_rho_series(node["name"], grounding, traj)

    ch = node.get("children") or []
    if op == "not":
        return [-v for v in robustness_series(ch[0], grounding, traj)]
    if op == "and":
        series = [robustness_series(c, grounding, traj) for c in ch]
        return [min(vals) for vals in zip(*series)]
    if op == "or":
        series = [robustness_series(c, grounding, traj) for c in ch]
        return [max(vals) for vals in zip(*series)]
    if op == "imply":
        a = robustness_series(ch[0], grounding, traj)
        b = robustness_series(ch[1], grounding, traj)
        return [max(-av, bv) for av, bv in zip(a, b)]
    if op == "iff":
        a = robustness_series(ch[0], grounding, traj)
        b = robustness_series(ch[1], grounding, traj)
        imp1 = [max(-av, bv) for av, bv in zip(a, b)]
        imp2 = [max(-bv, av) for av, bv in zip(a, b)]
        return [min(p, q) for p, q in zip(imp1, imp2)]
    if op in ("globally", "finally"):
        sub = robustness_series(ch[0], grounding, traj)
        lo, hi = _interval_idx(node.get("interval"), T)
        out = []
        for t in range(T):
            # window [t+lo, t+hi] within the trajectory (future-relative)
            w = sub[min(T - 1, t + lo): min(T, t + hi + 1)] or [sub[min(T - 1, t)]]
            out.append(min(w) if op == "globally" else max(w))
        return out
    if op == "until":
        a = robustness_series(ch[0], grounding, traj)
        b = robustness_series(ch[1], grounding, traj)
        lo, hi = _interval_idx(node.get("interval"), T)
        out = []
        for t in range(T):
            best = -math.inf
            for tau in range(t + lo, min(T, t + hi + 1)):
                pre = min(a[t:tau + 1]) if tau >= t else math.inf
                best = max(best, min(b[tau], pre))
            out.append(best if best != -math.inf else -1e3)
        return out
    raise ValueError(f"unknown op for robustness: {op!r}")


def robustness(tree: dict, grounding: dict,
               traj: List[Tuple[float, float]]) -> float:
    """Scalar STL robustness at t=0 (the spec's satisfaction margin)."""
    if len(traj) == 0:
        return float("-inf")
    return robustness_series(tree, grounding, traj)[0]
