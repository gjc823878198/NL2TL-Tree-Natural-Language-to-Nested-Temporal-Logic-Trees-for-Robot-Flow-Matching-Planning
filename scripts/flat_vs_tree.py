"""
Ablation F: flat STL formula vs. nested STL tree as the LLM's output target.

Both use the SAME frozen LLM, SAME sentences, SAME gold trees, SAME metrics.
The only difference is what the model is asked to emit:
  - TREE  (our method): a nested JSON tree   -> scored directly  (reused from the
                        existing eval predictions, no new Groq calls)
  - FLAT  (baseline):   a single STL string   -> parsed by the Lark grammar into
                        a tree -> scored

    python3 scripts/flat_vs_tree.py --n 30            # run the flat baseline
    python3 scripts/flat_vs_tree.py --n 30 --dry      # sanity-check parsing only
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stl_parser import parse_stl                          # noqa: E402
from tree_metrics import exact_match, op_f1, path_f1      # noqa: E402

PRED = ROOT / "outputs" / "nl2tl_tree_eval" / "predictions_2026-05-29.jsonl"
SYS = ("You translate an English instruction into ONE Signal Temporal Logic "
       "formula string. Operators: G (globally), F (finally), U (until), "
       "! (not), & (and), | (or), -> (implies), <-> (iff); bounded variants "
       "take an interval like F[0,5]. Atoms are names like prop_1. Reply with "
       "JSON {\"formula\": \"<STL string>\"} and nothing else.")


def _records():
    return [json.loads(l) for l in PRED.read_text().splitlines() if l.strip()]


def _score(pred_tree, gold_tree):
    return (1.0 if exact_match(pred_tree, gold_tree) else 0.0,
            op_f1(pred_tree, gold_tree), path_f1(pred_tree, gold_tree))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--dry", action="store_true",
                    help="parse the GOLD formula instead of calling Groq")
    args = ap.parse_args()

    recs = _records()
    shots, test = recs[:args.shots], recs[args.shots:args.shots + args.n]

    flat, tree = [], []
    n_parse_fail = 0
    client = fewshot = None
    if not args.dry:
        import groq
        client = groq.Groq()
        fewshot = []
        for s in shots:
            fewshot.append({"role": "user", "content": s["natural"]})
            fewshot.append({"role": "assistant",
                            "content": json.dumps({"formula": s["formula"]})})

    for r in test:
        gold = r["gold_tree"]
        # TREE method: reuse the stored prediction's metrics
        m = r["metrics"]
        tree.append((float(m.get("exact_match", m.get("exact", 0.0))),
                     float(m.get("op_f1", 0.0)), float(m.get("path_f1", 0.0))))
        # FLAT method: get a formula string, parse, score
        if args.dry:
            formula = r["formula"]
        else:
            msg = [{"role": "system", "content": SYS}, *fewshot,
                   {"role": "user", "content": r["natural"]}]
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=msg,
                response_format={"type": "json_object"},
                temperature=0.0, max_tokens=512)
            formula = json.loads(resp.choices[0].message.content)["formula"]
        try:
            pred = parse_stl(formula)
            flat.append(_score(pred, gold))
        except Exception:
            n_parse_fail += 1
            flat.append((0.0, 0.0, 0.0))     # unparseable flat string = miss

    def agg(rows):
        n = len(rows)
        return tuple(round(sum(x[i] for x in rows) / n, 3) for i in range(3))

    fe, fo, fp = agg(flat)
    te, to, tp = agg(tree)
    print(f"n={len(test)}  flat_parse_failures={n_parse_fail}")
    print(f"{'method':6s} {'exact':>7s} {'op_f1':>7s} {'path_f1':>8s}")
    print(f"{'FLAT':6s} {fe:7.3f} {fo:7.3f} {fp:8.3f}")
    print(f"{'TREE':6s} {te:7.3f} {to:7.3f} {tp:8.3f}")
    print(f"{'Δ':6s} {te-fe:+7.3f} {to-fo:+7.3f} {tp-fp:+8.3f}")


if __name__ == "__main__":
    main()
