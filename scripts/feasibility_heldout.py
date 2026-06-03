"""
#4 Held-out validation of the feasibility self-check signal.

The 6 demo tasks are in-sample for the threshold.  Here we generate a HELD-OUT
set the threshold never saw -- random single timed-reaches (in-distribution for
the frozen flow prior) and random k-leg nested sequential specs (out of
distribution) -- and check that the best-of-N reach-shortfall still separates
them.

    python3 scripts/feasibility_heldout.py
"""
from __future__ import annotations
import io, contextlib, math, random, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner.telograf_infer import telograf_feasibility          # noqa: E402

R = 0.5
BOUNDS = 3.0


def _reach(name, x, y):
    return {"op": "finally", "interval": [20, 55],
            "children": [{"op": "atom", "name": name}]}


def simple_case(i, rng):
    gx, gy = rng.uniform(-BOUNDS, BOUNDS), rng.uniform(1.0, BOUNDS)
    sx, sy = rng.uniform(-BOUNDS, BOUNDS), rng.uniform(-BOUNDS, -1.0)
    return {"id": f"heldout_simple_{i}", "role": "telograf",
            "nl": "reach the goal in time", "tree": _reach("prop_g", gx, gy),
            "grounding": {"prop_g": {"kind": "reach", "x": gx, "y": gy,
                                     "z": 0.0, "r": R}},
            "map_hint": {"world_bounds": [-4, 4, -4, 4], "start": [sx, sy],
                         "style": "telograf_native", "telograf_samples": 32,
                         "gz_world": "maze"}}


def nested_case(i, k, rng):
    pts = [(rng.uniform(-BOUNDS, BOUNDS), rng.uniform(-BOUNDS, BOUNDS))
           for _ in range(k)]
    grounding = {f"prop_{j+1}": {"kind": "reach", "x": px, "y": py, "z": 0.0,
                                 "r": R} for j, (px, py) in enumerate(pts)}

    def build(j):
        leaf = {"op": "atom", "name": f"prop_{j+1}"}
        if j == k - 1:
            return {"op": "finally", "interval": None, "children": [leaf]}
        return {"op": "finally", "interval": None, "children": [
            {"op": "and", "interval": None, "children": [leaf, build(j + 1)]}]}

    return {"id": f"heldout_nested{k}_{i}", "role": "telograf",
            "nl": f"visit {k} zones in order", "tree": build(0),
            "grounding": grounding,
            "map_hint": {"world_bounds": [-4, 4, -4, 4],
                         "start": [-3.3, -3.3], "style": "telograf_native",
                         "telograf_samples": 32, "gz_world": "maze"}}


def feas(case):
    with contextlib.redirect_stdout(io.StringIO()):
        f = telograf_feasibility(case, n_steps=64)
    return f.get("best_cost"), f.get("satisfy_rate"), f.get("feasible")


def main():
    rng = random.Random(7)
    print(f"{'group':16s} {'shortfall':>10s} {'satisfy':>8s} {'feasible':>9s}")
    print("-" * 48)
    simple, nested = [], []
    for i in range(5):
        bc, sr, fe = feas(simple_case(i, rng))
        simple.append(bc)
        print(f"{'simple_reach':16s} {bc:10.2f} {sr:8.2f} {fe!s:>9s}")
    for i in range(5):
        k = 3 + (i % 2)                          # 3- and 4-leg nests
        bc, sr, fe = feas(nested_case(i, k, rng))
        nested.append(bc)
        print(f"{'nested'+str(k)+'leg':16s} {bc:10.2f} {sr:8.2f} {fe!s:>9s}")
    print("-" * 48)
    print(f"in-distribution simple : shortfall mean={statistics.mean(simple):.2f} "
          f"max={max(simple):.2f}")
    print(f"OOD nested             : shortfall mean={statistics.mean(nested):.2f} "
          f"min={min(nested):.2f}")
    print(f"SEPARATION: simple-max={max(simple):.2f} < nested-min={min(nested):.2f}"
          f"  -> {'CLEAN' if max(simple) < min(nested) else 'OVERLAP'}")


if __name__ == "__main__":
    main()
