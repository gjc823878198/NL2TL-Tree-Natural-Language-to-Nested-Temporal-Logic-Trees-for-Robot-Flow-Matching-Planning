"""Reviewer experiments, run across MULTIPLE Groq models (spreads the free-tier
quota) with a BALANCED comparison (same N sentences scored for tree AND flat in
every model):

  (1) flat STL string  vs.  nested STL tree  -- same frozen LLM, same sentences,
      same gold trees, same metrics; only the output target differs.
  (2) self-correction ablation on the tree parser:
      greedy  vs.  +self-consistency (K samples, majority vote)  vs.  +round-trip
      scope refinement.

    python3 scripts/reviewer_exp.py --n 30 --k 5
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stl_parser import parse_stl, ast_to_stl                       # noqa: E402
from tree_metrics import exact_match, op_f1, path_f1              # noqa: E402
from nl_to_tree_groq import (nl_to_tree, nl_to_tree_self_consistent,  # noqa: E402
                             RateLimiter, with_retry)
from nl_to_tree_selfcorrect import roundtrip_scope_refine          # noqa: E402
import groq                                                        # noqa: E402

ARCHIVE = ROOT / "outputs" / "nl2tl_tree_eval" / "predictions_2026-05-29.jsonl"
OUT = ROOT / "outputs" / "nl2tl_tree_eval" / "reviewer_exp.json"
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
          "meta-llama/llama-4-scout-17b-16e-instruct"]
FLAT_SYS = ("You translate an English instruction into ONE Signal Temporal Logic "
            "formula string. Operators: G (globally), F (finally), U (until), "
            "! (not), & (and), | (or), -> (implies), <-> (iff); bounded variants "
            "take an interval like F[0,5]. Atoms are names like prop_1. Reply with "
            'JSON {"formula": "<STL string>"} and nothing else.')


def score(pred, gold):
    return [1.0 if exact_match(pred, gold) else 0.0,
            op_f1(pred, gold), path_f1(pred, gold)]


def canon(tree):
    """Route a predicted tree through the SAME grammar pipeline the flat
    baseline gets (flatten nested and/or, normalize ->,<-> to the operator
    basis), so exact-match is a fair head-to-head and not penalized for
    meaning-preserving structural differences.  Falls back to the raw tree if
    the tree is malformed and won't serialize/re-parse."""
    try:
        return parse_stl(ast_to_stl(tree))
    except Exception:
        return tree


def agg(rows):
    n = len(rows) or 1
    return [round(sum(r[i] for r in rows) / n, 3) for i in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--abl-models", default="llama-3.3-70b-versatile")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    models = args.models.split(",")

    recs = [json.loads(l) for l in ARCHIVE.read_text().splitlines() if l.strip()]
    shots, test = recs[:args.shots], recs[args.shots:args.shots + args.n]
    tree_fs = [{"natural": s["natural"], "tree": s["gold_tree"]} for s in shots]
    flat_fs = []
    for s in shots:
        flat_fs.append({"role": "user", "content": s["natural"]})
        flat_fs.append({"role": "assistant",
                        "content": json.dumps({"formula": s["formula"]})})
    client = groq.Groq()
    abl = set(args.abl_models.split(","))
    results = {"n": len(test), "k": args.k, "models": {}}

    def flat_parse(nl, model, lim):
        msg = [{"role": "system", "content": FLAT_SYS}, *flat_fs,
               {"role": "user", "content": nl}]
        r = with_retry(lambda: client.chat.completions.create(
            model=model, messages=msg, response_format={"type": "json_object"},
            temperature=0.0, max_tokens=512), limiter=lim)
        return json.loads(r.choices[0].message.content).get("formula", "")

    for model in models:
        lim = RateLimiter(rpm=28)
        m = {"flat": [], "tree": [], "tree_canon": [], "selfcons": [],
             "roundtrip": [], "flat_fail": 0, "n_flat": 0, "n_tree": 0}
        preds = []
        print(f"=== {model} ===", flush=True)
        for i, r in enumerate(test):
            gold, nl = r["gold_tree"], r["natural"]
            t = None
            try:
                t = nl_to_tree(nl, model=model, few_shot=tree_fs,
                               client=client, limiter=lim)
                m["tree"].append(score(t, gold))
                m["tree_canon"].append(score(canon(t), gold))
            except Exception:
                m["tree"].append([0.0, 0.0, 0.0])
                m["tree_canon"].append([0.0, 0.0, 0.0])
            m["n_tree"] += 1
            preds.append({"natural": nl, "gold": gold, "tree": t})
            try:
                pf = parse_stl(flat_parse(nl, model, lim))
                m["flat"].append(score(pf, gold))
            except Exception:
                m["flat_fail"] += 1
                m["flat"].append([0.0, 0.0, 0.0])
            m["n_flat"] += 1
            if model in abl:
                try:
                    sc = nl_to_tree_self_consistent(
                        nl, model=model, few_shot=tree_fs, client=client,
                        k=args.k, limiter=lim)
                    m["selfcons"].append(score(sc, gold))
                except Exception:
                    m["selfcons"].append(m["tree"][-1])
                try:
                    rt = roundtrip_scope_refine(nl, t, client=client,
                                                model=model) if t else None
                    m["roundtrip"].append(score(rt, gold) if rt else m["tree"][-1])
                except Exception:
                    m["roundtrip"].append(m["tree"][-1])
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(test)}", flush=True)
        out = {"n_tree": m["n_tree"], "n_flat": m["n_flat"],
               "flat_fail": m["flat_fail"],
               "flat": agg(m["flat"]), "tree": agg(m["tree"]),
               "tree_canon": agg(m["tree_canon"])}
        slug = model.replace("/", "_")
        (Path(args.out).parent / f"revexp_preds_{slug}.jsonl").write_text(
            "\n".join(json.dumps(p) for p in preds))
        if model in abl:
            out["greedy"] = out["tree"]
            out["selfcons"] = agg(m["selfcons"])
            out["roundtrip"] = agg(m["roundtrip"])
        results["models"][model] = out
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(json.dumps(out), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("REVIEWER_EXP_DONE", flush=True)


if __name__ == "__main__":
    main()
