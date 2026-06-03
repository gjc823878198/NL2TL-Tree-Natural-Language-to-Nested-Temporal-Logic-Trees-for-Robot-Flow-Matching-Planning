"""
Re-score an existing predictions JSONL with the looser metrics from
`tree_metrics`. Lets you compare exact-match vs. op-F1 / path-F1 / TED on
data you've already paid quota for, without re-querying the model.

Input lines must have `predicted_tree` and `gold_tree` fields (this is what
`nl_to_tree_groq.py --eval` produces). Lines with an `error` field are
counted as failures and skipped from metric averaging.

Usage:
  python3 rescore.py data/predictions.jsonl
  python3 rescore.py data/predictions.jsonl --out data/predictions.rescored.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

from tree_metrics import all_metrics, aggregate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Predictions JSONL from nl_to_tree_groq.py")
    ap.add_argument("--out", help="Optional path to write annotated JSONL")
    ap.add_argument("--show", type=int, default=3,
                    help="Print this many worst-and-best cases (default 3 each)")
    args = ap.parse_args()

    in_p = Path(args.input)
    if not in_p.exists():
        print(f"ERROR: not found: {in_p}", file=sys.stderr); sys.exit(2)

    annotated = []
    failures = 0
    skipped_no_gold = 0
    for line in in_p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "error" in obj:
            failures += 1
            annotated.append(obj)
            continue
        pred = obj.get("predicted_tree")
        gold = obj.get("gold_tree")
        if not pred or not gold:
            skipped_no_gold += 1
            annotated.append(obj)
            continue
        m = all_metrics(pred, gold)
        obj["metrics"] = m
        annotated.append(obj)

    # rows with metrics only
    scored = [o["metrics"] for o in annotated if "metrics" in o]
    agg = aggregate(scored)

    n_total = len(annotated)
    print(f"Total rows in file:   {n_total}")
    print(f"  Errors (no pred):   {failures}")
    print(f"  Missing gold:       {skipped_no_gold}")
    print(f"  Scored:             {len(scored)}")
    if not scored:
        sys.exit(0)
    print()
    print("=== Aggregate metrics (over scored rows) ===")
    print(f"  exact_match (strict):   {agg['exact_match']:.1%}")
    print(f"  op_f1   mean:           {agg.get('op_f1', 0):.3f}   (right operators used?)")
    print(f"  path_f1 mean:           {agg.get('path_f1', 0):.3f}   (right nesting?)")
    print(f"  ted     mean:           {agg.get('mean_ted', 0):.2f} edits per tree")
    print(f"  ted_norm mean:          {agg.get('ted_norm', 0):.3f}   (1.0 = identical)")
    print()

    # Print best and worst (excluding the exact matches at 1.0)
    scored_with_idx = [
        (i, o) for i, o in enumerate(annotated) if "metrics" in o
    ]
    near_misses = [
        (i, o) for i, o in scored_with_idx
        if not o["metrics"]["exact_match"]
    ]
    near_misses.sort(key=lambda x: -x[1]["metrics"].get("ted_norm", 0))

    if near_misses:
        print(f"=== Top {min(args.show, len(near_misses))} NEAR-MISSES "
              f"(exact_match=False but close) ===")
        for i, o in near_misses[:args.show]:
            m = o["metrics"]
            print(f"\n[#{i}] op_f1={m['op_f1']:.2f}  path_f1={m['path_f1']:.2f}  "
                  f"ted={m.get('ted','?')}  ted_norm={m.get('ted_norm',0):.3f}")
            print(f"  NL:   {(o.get('natural') or '')[:120]}")
            print(f"  PRED: {json.dumps(o['predicted_tree'], ensure_ascii=False)[:200]}")
            print(f"  GOLD: {json.dumps(o['gold_tree'], ensure_ascii=False)[:200]}")

        print()
        print(f"=== Top {min(args.show, len(near_misses))} WORST cases ===")
        worst = sorted(near_misses, key=lambda x: x[1]["metrics"].get("ted_norm", 0))
        for i, o in worst[:args.show]:
            m = o["metrics"]
            print(f"\n[#{i}] op_f1={m['op_f1']:.2f}  path_f1={m['path_f1']:.2f}  "
                  f"ted={m.get('ted','?')}  ted_norm={m.get('ted_norm',0):.3f}")
            print(f"  NL:   {(o.get('natural') or '')[:120]}")
            print(f"  PRED: {json.dumps(o['predicted_tree'], ensure_ascii=False)[:200]}")
            print(f"  GOLD: {json.dumps(o['gold_tree'], ensure_ascii=False)[:200]}")

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w") as f:
            for o in annotated:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"\nWrote annotated JSONL to: {out_p}")


if __name__ == "__main__":
    main()
