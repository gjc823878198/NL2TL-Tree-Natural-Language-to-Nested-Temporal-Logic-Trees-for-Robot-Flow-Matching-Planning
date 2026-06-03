"""
Operator-normalisation contribution: TeLoGraF's graph vocabulary is the seven
primitives {and, or, not, finally, globally, until, reach}; it has no native
code for imply or iff.  planner.telograf_adapter.normalise() rewrites those
into the primitive basis via semantics-preserving STL identities:

    imply(A,B)  ->  or(not A, B)
    iff(A,B)    ->  and(or(not A, B), or(not B, A))

This test PROVES the rewriting preserves STL semantics exactly, by checking
that quantitative robustness rho is identical for the original (imply/iff) tree
and its rewrite, on many random trajectories.

Run with:  python3 tests/test_operator_rewriting.py
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.telograf_adapter import normalise, OP_CODE  # noqa: E402
from stl_robustness import robustness                      # noqa: E402

# Operators TeLoGraF's encoder has NO code for (must be rewritten away):
UNSUPPORTED = {"imply", "iff"}

GROUNDING = {"p1": {"x": 1.0, "y": 1.0, "r": 0.6},
             "p2": {"x": -1.5, "y": 0.5, "r": 0.6}}

TREES = {
    "imply": {"op": "globally", "interval": None, "children": [
        {"op": "imply", "interval": None, "children": [
            {"op": "atom", "name": "p1"},
            {"op": "finally", "interval": [0, 5],
             "children": [{"op": "atom", "name": "p2"}]}]}]},
    "iff": {"op": "iff", "interval": None, "children": [
        {"op": "finally", "interval": [0, 30],
         "children": [{"op": "atom", "name": "p1"}]},
        {"op": "globally", "interval": None, "children": [
            {"op": "not", "interval": None,
             "children": [{"op": "atom", "name": "p2"}]}]}]},
}


def _ops(node, acc=None):
    acc = acc if acc is not None else set()
    if isinstance(node, dict):
        if node.get("op"):
            acc.add(node["op"])
        for c in node.get("children") or []:
            _ops(c, acc)
    return acc


def main():
    rng = random.Random(0)
    failures = 0
    for name, tree in TREES.items():
        rew = normalise(tree)
        ops = _ops(rew)
        # (1) rewrite contains ONLY operators TeLoGraF can encode
        leftover = (ops - {"atom"}) - set(OP_CODE)
        unsupported_left = ops & UNSUPPORTED
        ok_basis = not leftover and not unsupported_left
        # (2) robustness is preserved exactly on random trajectories
        max_diff = 0.0
        for _ in range(3000):
            traj = [(rng.uniform(-4, 4), rng.uniform(-4, 4)) for _ in range(64)]
            max_diff = max(max_diff,
                           abs(robustness(tree, GROUNDING, traj) -
                               robustness(rew, GROUNDING, traj)))
        ok_rho = max_diff < 1e-9
        status = "PASS" if (ok_basis and ok_rho) else "FAIL"
        failures += status == "FAIL"
        print(f"[{status}] {name:6s}: rewritten_ops={sorted(ops)} "
              f"in-basis={ok_basis} max|drho|={max_diff:.1e}")
    if failures:
        print(f"\n{failures} FAILED")
        return 1
    print("\nall operator-rewriting equivalences hold (rho preserved exactly)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
