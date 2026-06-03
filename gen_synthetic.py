"""
Generate a small synthetic (natural language, STL formula) dataset from
templates. Useful when the real NL2TL dataset is gated behind a Google Drive
rate-limit, or when you just want to smoke-test the pipeline before training.

This mirrors the approach DeepSTL / NL2TL themselves use for data
bootstrapping: fix a few formula skeletons, sample predicates and intervals,
and produce a parallel English template description.

Output: JSONL, one {"natural": ..., "formula": ...} per line.
"""
import argparse
import json
import random
from pathlib import Path

PREDICATES = [f"prop_{i}" for i in range(1, 6)]


def _ab(rng):
    a = rng.randint(0, 50)
    b = a + rng.randint(1, 100)
    return a, b


def _cd(rng):
    c = rng.randint(0, 10)
    d = c + rng.randint(1, 20)
    return c, d


def t_globally(rng):
    p = rng.choice(PREDICATES)
    a, b = _ab(rng)
    nl = f"From time {a} to {b}, {p} must hold at every moment."
    fm = f"G[{a},{b}] {p}"
    return nl, fm


def t_finally(rng):
    p = rng.choice(PREDICATES)
    a, b = _ab(rng)
    nl = f"At some point between time {a} and {b}, {p} should occur."
    fm = f"F[{a},{b}] {p}"
    return nl, fm


def t_until(rng):
    p1, p2 = rng.sample(PREDICATES, 2)
    a, b = _ab(rng)
    nl = f"{p1} must hold continuously until {p2} occurs between time {a} and {b}."
    fm = f"{p1} U[{a},{b}] {p2}"
    return nl, fm


def t_globally_implies_finally(rng):
    p1, p2 = rng.sample(PREDICATES, 2)
    a, b = _ab(rng)
    c, d = _cd(rng)
    nl = (f"Throughout time {a} to {b}, whenever {p1} happens, {p2} must follow "
          f"within {c} to {d} time units.")
    fm = f"G[{a},{b}] ({p1} -> F[{c},{d}] {p2})"
    return nl, fm


def t_conj(rng):
    p1, p2 = rng.sample(PREDICATES, 2)
    a, b = _ab(rng)
    nl = f"During the interval [{a},{b}], both {p1} and {p2} must hold."
    fm = f"G[{a},{b}] ({p1} & {p2})"
    return nl, fm


def t_negation(rng):
    p = rng.choice(PREDICATES)
    a, b = _ab(rng)
    nl = f"{p} must never occur during the interval [{a},{b}]."
    fm = f"G[{a},{b}] !{p}"
    return nl, fm


def t_disj_finally(rng):
    p1, p2 = rng.sample(PREDICATES, 2)
    a, b = _ab(rng)
    nl = f"Within time {a} to {b}, at least one of {p1} or {p2} must occur."
    fm = f"F[{a},{b}] ({p1} | {p2})"
    return nl, fm


def t_iff(rng):
    p1, p2 = rng.sample(PREDICATES, 2)
    a, b = _ab(rng)
    nl = f"In the time window [{a},{b}], {p1} holds if and only if {p2} holds."
    fm = f"G[{a},{b}] ({p1} <-> {p2})"
    return nl, fm


def t_nested(rng):
    p1, p2, p3 = rng.sample(PREDICATES, 3)
    a, b = _ab(rng)
    c, d = _cd(rng)
    nl = (f"If {p1} holds at some time in [{a},{b}], then {p2} must continue "
          f"until {p3} occurs within {c} to {d} time units.")
    fm = f"F[{a},{b}] {p1} -> ({p2} U[{c},{d}] {p3})"
    return nl, fm


TEMPLATES = [
    t_globally, t_finally, t_until, t_globally_implies_finally,
    t_conj, t_negation, t_disj_finally, t_iff, t_nested,
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--n", type=int, default=50,
                    help="How many examples to generate (default 50).")
    ap.add_argument("-o", "--out", default="data/synthetic_nl_stl.jsonl",
                    help="Output JSONL path.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with out_p.open("w", encoding="utf-8") as f:
        for _ in range(args.n):
            tpl = rng.choice(TEMPLATES)
            nl, fm = tpl(rng)
            f.write(json.dumps({"natural": nl, "formula": fm}, ensure_ascii=False) + "\n")

    print(f"Wrote {args.n} synthetic NL-STL pairs to {out_p}")
    print("Next: python3 convert_dataset.py "
          f"{out_p} data/synthetic_converted.jsonl")


if __name__ == "__main__":
    main()
