"""
Smoke tests for stl_parser. Run with:
    python3 tests/test_parser.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stl_parser import parse_stl, ast_to_stl  # noqa: E402

CASES = [
    # (formula, predicate on resulting AST)
    ("prop_1",            lambda t: t == {"op": "atom", "name": "prop_1"}),
    ("G[0,10] prop_1",    lambda t: t["op"] == "globally" and t["interval"] == [0, 10]
                                    and t["children"][0]["name"] == "prop_1"),
    ("F[2,5] (prop_1 & prop_2)", lambda t: t["op"] == "finally"
                                            and t["children"][0]["op"] == "and"),
    ("prop_1 U[0,5] prop_2", lambda t: t["op"] == "until" and t["interval"] == [0, 5]),
    ("!prop_1",            lambda t: t["op"] == "not"),
    ("prop_1 & prop_2 & prop_3", lambda t: t["op"] == "and" and len(t["children"]) == 3),
    ("prop_1 | prop_2",    lambda t: t["op"] == "or"),
    ("prop_1 -> prop_2",   lambda t: t["op"] == "imply"),
    ("prop_1 <-> prop_2",  lambda t: t["op"] == "iff"),
    # NL2TL "word form" example from the paper
    ("( ( ( prop_2 until [176,415] prop_1 ) and prop_3 ) equal prop_4 )",
        lambda t: t["op"] == "iff"
                  and t["children"][0]["op"] == "and"
                  and t["children"][0]["children"][0]["op"] == "until"
                  and t["children"][0]["children"][0]["interval"] == [176, 415]),
    # word-form temporal ops mixed with symbolic logical ops
    ("globally [10,50] ( prop_1 imply finally [0,5] prop_2 )",
        lambda t: t["op"] == "globally"
                  and t["interval"] == [10, 50]
                  and t["children"][0]["op"] == "imply"),
]


def main():
    ok = 0
    fail = 0
    for s, pred in CASES:
        try:
            got = parse_stl(s)
            if pred(got):
                print(f"OK    {s!r}")
                ok += 1
            else:
                print(f"FAIL  {s!r}\n      got: {got}")
                fail += 1
        except Exception as e:
            print(f"ERROR {s!r}\n      {type(e).__name__}: {e}")
            fail += 1
    print(f"\n{ok} / {ok + fail} passed")

    # Round-trip sanity check (parse → render → parse again should give same AST)
    print("\nRound-trip check:")
    rt_ok = rt_fail = 0
    for s, _ in CASES:
        try:
            a = parse_stl(s)
            s2 = ast_to_stl(a)
            a2 = parse_stl(s2)
            if a == a2:
                rt_ok += 1
            else:
                rt_fail += 1
                print(f"  diverged: {s!r}")
                print(f"           rendered: {s2!r}")
        except Exception as e:
            rt_fail += 1
            print(f"  ERROR    {s!r}: {e}")
    print(f"{rt_ok} / {rt_ok + rt_fail} round-trip OK")

    sys.exit(0 if (fail == 0 and rt_fail == 0) else 1)


if __name__ == "__main__":
    main()
