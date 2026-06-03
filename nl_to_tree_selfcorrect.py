"""
Tier-4: closed-loop NL -> STL parse with robustness/validation self-correction.

The frozen LLM parser (nl_to_tree_groq.nl_to_tree) is strong at picking the
right operators but mis-scopes about half the time (op-F1 0.93 vs path-F1 0.58
in our study).  Instead of accepting the first parse, we VERIFY it and, when a
check fails, feed a TARGETED diagnostic back to the (still-frozen) LLM and ask
it to revise -- a neuro-symbolic loop, no fine-tuning.

Three verifiers, cheapest first:
  1. structural / arity   -- every operator has a legal child count (deterministic)
  2. grounding consistency -- atom names exist; no atom is both reach and avoid
  3. STL robustness        -- if a grounding + start are given, plan a trajectory
                              and evaluate quantitative robustness rho
                              (stl_robustness.robustness).  rho <= 0 means the
                              parsed spec is unsatisfiable/contradictory, which
                              is fed back as "your parse scores rho=X; re-examine
                              the temporal scoping."

Each failed round appends the model's previous (rejected) JSON + the diagnostic
to the conversation and re-queries, up to `max_rounds`.

CLI:
  export GROQ_API_KEY=gsk_...
  python3 nl_to_tree_selfcorrect.py \
      --nl "Always avoid the hazard and eventually reach the goal within 10 steps." \
      --max-rounds 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nl_to_tree_groq import (
    SYSTEM_PROMPT, RateLimiter, build_messages, load_few_shot,
    nl_to_tree, with_retry, _validate_tree,
)

_ARITY = {
    "not": (1, 1), "and": (2, 99), "or": (2, 99),
    "imply": (2, 2), "iff": (2, 2),
    "globally": (1, 1), "finally": (1, 1), "until": (2, 2),
}

# A planned trajectory scoring below this STL robustness is treated as
# "the parse is unsatisfiable / contradictory" and fed back to the LLM.
ROB_MARGIN = 0.0


def _structural_issues(node, path="root") -> list:
    issues = []
    if not isinstance(node, dict):
        return [f"node at {path} is not an object"]
    op = node.get("op")
    if op == "atom":
        if not node.get("name"):
            issues.append(f"atom at {path} is missing its 'name'")
        return issues
    if op not in _ARITY:
        return [f"unknown operator {op!r} at {path}"]
    ch = node.get("children") or []
    lo, hi = _ARITY[op]
    if not (lo <= len(ch) <= hi):
        want = f"{lo}" if lo == hi else f"{lo}-{hi}"
        issues.append(f"operator '{op}' at {path} needs {want} child(ren), "
                      f"got {len(ch)}")
    if op in ("globally", "finally", "until") and node.get("interval"):
        a, b = node["interval"]
        if a > b:
            issues.append(f"interval [{a},{b}] at {path}.{op} is reversed (a>b)")
    for i, c in enumerate(ch):
        issues += _structural_issues(c, f"{path}.{op}[{i}]")
    return issues


def _grounding_issues(node, grounding: dict) -> list:
    issues, names = [], set()

    def walk(n):
        if isinstance(n, dict):
            if n.get("op") == "atom":
                names.add(n.get("name"))
            for c in n.get("children") or []:
                walk(c)
    walk(node)
    for nm in names:
        if nm not in grounding:
            issues.append(f"atom '{nm}' is not in the grounding "
                          f"(known: {sorted(grounding)})")
    return issues


def _robustness_issue(tree, grounding, start, bounds) -> str | None:
    """Plan a trajectory for the parsed spec and report its STL robustness."""
    try:
        from planner import plan_waypoints
        from stl_robustness import robustness
    except Exception:
        return None
    case = {
        "id": "selfcorrect", "tree": tree, "grounding": grounding,
        "map_hint": {"start": list(start or [-3.0, -3.0]),
                     "world_bounds": list(bounds or [-4, 4, -4, 4]),
                     "telograf_samples": 8},
    }
    try:
        traj = plan_waypoints(case, n_steps=64, backend="fallback")
    except Exception:
        return None                      # can't plan -> skip this verifier
    rho = robustness(tree, grounding, traj)
    if rho <= ROB_MARGIN:
        return (f"when planned, the parsed specification scores STL robustness "
                f"rho={rho:.2f} (<=0 means it cannot be satisfied -- e.g. a "
                f"reach goal lies inside an avoid region, or a time bound is "
                f"impossible). Re-examine the temporal scoping and operator "
                f"nesting; the operators are likely right but mis-nested.")
    return None


def roundtrip_scope_refine(nl: str, tree: dict, *, client, model: str,
                           max_rounds: int = 1) -> dict:
    """Round-trip / paraphrase scope self-refine (brainstorm method #4).

    Ask the SAME frozen LLM to read the candidate tree back into plain English
    with the operator SCOPE made explicit, compare that paraphrase to the
    original instruction, and emit a corrected tree if (and only if) the
    nesting/scope disagrees.  A wrong nesting (globally(imply) vs
    imply(globally)) produces a meaningfully different paraphrase, so the
    otherwise-invisible scope inversion becomes a checkable discrepancy.
    References: Madaan et al. "Self-Refine" (NeurIPS 2023, arXiv:2303.17651);
    Lyu et al. "Faithful Chain-of-Thought Reasoning" (IJCNLP-AACL 2023)."""
    import json as _json
    cur = tree
    for _ in range(max(1, max_rounds)):
        prompt = (
            f"Original instruction:\n{nl}\n\n"
            f"Candidate STL tree (JSON):\n{_json.dumps(cur, ensure_ascii=False)}\n\n"
            "Step 1: paraphrase this tree back into plain English, making the "
            "operator SCOPE explicit -- e.g. distinguish 'always, (if A then "
            "eventually B)'  from  '(always if A), then eventually B'.\n"
            "Step 2: compare your paraphrase with the original instruction. If "
            "the nesting/scope matches, reply EXACTLY {\"ok\": true}. If it does "
            "NOT match, reply with the CORRECTED tree as {\"tree\": {...}} using "
            "the same JSON schema -- keep the operators, fix only the nesting.")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0, max_tokens=2048)
            data = _json.loads(resp.choices[0].message.content or "{}")
        except Exception:
            break
        if data.get("ok") is True or "tree" not in data:
            break                                       # confirmed, or no change
        try:
            cur = _validate_tree(data["tree"])
        except Exception:
            break
    return cur


def diagnose(tree: dict, grounding: dict | None = None,
             start=None, bounds=None) -> list:
    """Return a list of human/LLM-readable issues, cheapest verifier first."""
    issues = _structural_issues(tree)
    if issues:
        return issues
    if grounding:
        issues += _grounding_issues(tree, grounding)
        if issues:
            return issues
        rob = _robustness_issue(tree, grounding, start, bounds)
        if rob:
            issues.append(rob)
    return issues


def selfcorrect(nl: str, *, model: str, few_shot: list, client,
                grounding: dict | None = None, start=None, bounds=None,
                max_rounds: int = 2, limiter: RateLimiter | None = None) -> dict:
    """Parse NL -> tree with up to `max_rounds` robustness/validation
    self-corrections.  Returns {tree, rounds, accepted, transcript}."""
    messages = build_messages(nl, few_shot)
    transcript = []
    tree = None
    for rnd in range(max_rounds + 1):
        def _call():
            return client.chat.completions.create(
                model=model, messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0, max_tokens=2048)
        resp = with_retry(_call, max_retries=5, limiter=limiter)
        text = resp.choices[0].message.content or ""
        try:
            data = json.loads(text)
            tree = data.get("tree", data if data.get("op") else None)
            tree = _validate_tree(tree)
        except Exception as e:
            issues = [f"output was not a valid tree JSON: {e}"]
        else:
            issues = diagnose(tree, grounding, start, bounds)
        transcript.append({"round": rnd, "tree": tree, "issues": issues})
        if not issues:
            return {"tree": tree, "rounds": rnd, "accepted": True,
                    "transcript": transcript}
        if rnd == max_rounds:
            break
        # feed the rejected answer + targeted diagnostic back to the LLM
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content":
            "That parse has problems:\n- " + "\n- ".join(issues) +
            "\nReturn a corrected JSON tree (same one-key 'tree' format). "
            "Keep the operators you are confident about; fix the flagged nodes."})
    return {"tree": tree, "rounds": max_rounds, "accepted": False,
            "transcript": transcript}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl", required=True)
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--few-shot", default="data/nl2tl_converted.jsonl")
    ap.add_argument("--n-shot", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=2)
    args = ap.parse_args(argv)

    import groq
    client = groq.Groq()
    fs_path = Path(args.few_shot)
    few_shot = load_few_shot(fs_path, n=args.n_shot) if fs_path.exists() else []

    out = selfcorrect(args.nl, model=args.model, few_shot=few_shot,
                      client=client, max_rounds=args.max_rounds,
                      limiter=RateLimiter(rpm=30))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
