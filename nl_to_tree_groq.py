"""
Route A (Groq edition): call Groq's free Llama-3.3 endpoint to convert a
natural-language task into our STL parse-tree JSON.

Why Groq:
  Generous free tier (30 RPM / 14,400 RPD on llama-3.3-70b-versatile), no
  card required, ~500 tokens/sec inference. Compatible with OpenAI-style
  chat-completions API.

Requires:  export GROQ_API_KEY=gsk_...
Get a free key (no card):  https://console.groq.com/keys

CLI:
  python3 nl_to_tree_groq.py --nl "Always within time 5 to 10, prop_1 must hold."

  python3 nl_to_tree_groq.py --input data/nl2tl_converted.jsonl \
      --output data/predictions.jsonl --limit 100 --eval
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import groq

from tree_metrics import all_metrics, aggregate


# ---------------------- rate limiting + retry ----------------------

class RateLimiter:
    """Enforce a minimum interval between calls (target: free-tier RPM)."""

    def __init__(self, rpm: int = 30):
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last_call = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    def mark_failure(self, retry_after: float):
        self._last_call = time.monotonic() + retry_after - self.min_interval


_RETRY_MARKERS = ("429", "rate_limit", "503", "500", "overloaded",
                  "service_unavailable", "internal_error")
_RETRY_DELAY_PAT = re.compile(r"retry.after['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)")


def _classify(exc: Exception) -> tuple[bool, float | None]:
    """Is this exception retryable? Optional suggested wait in seconds.

    Fails fast on "tokens per day" exhaustion — that resets only at the day
    rollover, so the next 100 retries will all hit the same wall.
    """
    s = str(exc).lower()
    # Per-day TPD/RPD bucket: retrying within the same minute is hopeless.
    if "tokens per day" in s or "requests per day" in s or "tpd" in s or "rpd" in s:
        return False, None
    if isinstance(exc, groq.RateLimitError):
        m = _RETRY_DELAY_PAT.search(s)
        return True, float(m.group(1)) if m else None
    if isinstance(exc, groq.APIStatusError):
        if exc.status_code in (500, 502, 503, 504):
            return True, None
        return False, None
    if any(m in s for m in _RETRY_MARKERS):
        m = _RETRY_DELAY_PAT.search(s)
        return True, float(m.group(1)) if m else None
    return False, None


def with_retry(fn, *, max_retries: int = 5, base_delay: float = 2.0,
               limiter: RateLimiter | None = None):
    """Call fn() with exponential backoff on retryable Groq errors."""
    last = None
    for attempt in range(max_retries + 1):
        try:
            if limiter is not None:
                limiter.wait()
            return fn()
        except Exception as e:
            retry, suggested = _classify(e)
            if not retry or attempt == max_retries:
                raise
            if suggested:
                delay = max(suggested, base_delay)
            else:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            delay = min(delay, 60.0)
            print(f"  retry {attempt+1}/{max_retries} after {delay:.1f}s "
                  f"({type(e).__name__})", file=sys.stderr)
            if limiter is not None:
                limiter.mark_failure(delay)
            time.sleep(delay)
            last = e
    raise last  # pragma: no cover


# ---------------------- prompts + few-shot ----------------------

SYSTEM_PROMPT = """You translate natural-language task descriptions into Signal Temporal Logic (STL) parse trees, serialized as JSON.

The STL grammar uses these operators (children counts in parentheses):
  atom (0)        : a predicate;             fields: op="atom", name="prop_1", interval=null, children=null
  not (1)         : logical negation;        fields: op="not", interval=null, children=[child], name=null
  and / or  (>=2) : conjunction / disjunction; same fields, multiple children
  imply / iff (2) : "A -> B" / "A <-> B"; children=[A, B]
  globally (1)    : G[a,b] sub  -> "sub holds at every moment in [a,b]"
  finally  (1)    : F[a,b] sub  -> "sub holds at some moment in [a,b]"
  until    (2)    : A U[a,b] B  -> "A holds continuously until B happens within [a,b]"

For globally / finally / until, set interval=[a,b]; if the description gives no time bounds, set interval=null (unbounded).
Atomic predicates use the form prop_1, prop_2, ... (rename freely if the NL gives concrete names).
Always include all four fields (op, name, interval, children); set unused ones to null.

Output a JSON object with exactly one top-level key, "tree", whose value is the root node. Do not output anything else."""


def load_few_shot(path: Path, n: int = 8, seed: int = 0) -> list:
    """Sample diverse few-shot examples from a converted JSONL dataset."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rng = random.Random(seed)
    rng.shuffle(lines)

    by_op = {}
    for ln in lines:
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not obj.get("ok", True) or not obj.get("tree"):
            continue
        op = obj["tree"]["op"]
        by_op.setdefault(op, []).append(obj)

    picked = []
    for _op, group in by_op.items():
        picked.append(group[0])
        if len(picked) >= n:
            break
    flat = [o for v in by_op.values() for o in v]
    rng.shuffle(flat)
    for o in flat:
        if len(picked) >= n:
            break
        if o not in picked:
            picked.append(o)
    return picked[:n]


def build_messages(nl: str, few_shot: list) -> list:
    """OpenAI-style chat messages: system + few-shot user/assistant pairs + query."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot:
        messages.append({"role": "user", "content": ex["natural"]})
        messages.append({
            "role": "assistant",
            "content": json.dumps({"tree": ex["tree"]}, ensure_ascii=False),
        })
    messages.append({"role": "user", "content": nl})
    return messages


def _validate_tree(node) -> dict:
    """Light validation: every node has a valid op; atoms have a name."""
    if not isinstance(node, dict):
        raise ValueError(f"node is not a dict: {type(node).__name__}")
    op = node.get("op")
    if op not in ("atom", "not", "and", "or", "imply", "iff",
                  "globally", "finally", "until"):
        raise ValueError(f"invalid op: {op!r}")
    if op == "atom":
        if not node.get("name"):
            raise ValueError("atom missing name")
        return node
    for child in (node.get("children") or []):
        _validate_tree(child)
    return node


def nl_to_tree(nl: str, *, model: str, few_shot: list,
               client: groq.Groq,
               limiter: RateLimiter | None = None,
               max_retries: int = 5) -> dict:
    """Call Groq and return the predicted STL tree."""
    messages = build_messages(nl, few_shot)

    def _call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=2048,
        )

    resp = with_retry(_call, max_retries=max_retries, limiter=limiter)
    text = resp.choices[0].message.content
    if not text:
        raise RuntimeError("empty response from Groq")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"non-JSON response: {e}\nraw: {text[:500]}")
    tree = data.get("tree") if isinstance(data, dict) else None
    if tree is None and isinstance(data, dict) and data.get("op"):
        tree = data  # some answers omit the wrapper
    if tree is None:
        raise RuntimeError(f"missing `tree` key in response: {data}")
    return _validate_tree(tree)


def nl_to_tree_self_consistent(nl: str, *, model: str, few_shot: list,
                               client: groq.Groq, k: int = 5,
                               temperature: float = 0.7,
                               limiter: RateLimiter | None = None,
                               max_retries: int = 5) -> dict:
    """Self-consistency parsing (brainstorm method #1).

    Sample `k` parses at temperature>0, canonicalize each tree
    (and/or-commutative), and return the MAJORITY tree.  Rationale: operators
    are stable across samples (op-F1~0.93), so the only thing that fluctuates
    is the NESTING/scope; voting on the canonical structure rescues the
    majority-correct scope that greedy decoding loses to a single near-miss.
    Reference: Wang et al., "Self-Consistency Improves Chain of Thought
    Reasoning in Language Models" (ICLR 2023, arXiv:2203.11171)."""
    from collections import Counter
    from tree_metrics import canonical_key

    messages = build_messages(nl, few_shot)
    cands = []
    for _ in range(max(1, k)):
        def _call():
            return client.chat.completions.create(
                model=model, messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature, max_tokens=2048)
        try:
            resp = with_retry(_call, max_retries=max_retries, limiter=limiter)
            text = resp.choices[0].message.content or ""
            data = json.loads(text)
            tree = data.get("tree") if isinstance(data, dict) else None
            if tree is None and isinstance(data, dict) and data.get("op"):
                tree = data
            cands.append(_validate_tree(tree))
        except Exception:
            continue                                   # drop bad samples
    if not cands:
        return nl_to_tree(nl, model=model, few_shot=few_shot, client=client,
                          limiter=limiter, max_retries=max_retries)
    keys = [canonical_key(t) for t in cands]
    winner, _ = Counter(keys).most_common(1)[0]
    for t, key in zip(cands, keys):
        if key == winner:
            return t                                   # representative of majority
    return cands[0]


# --------------------------- exact-match (back-compat) ---------------------------
from tree_metrics import exact_match as tree_equal  # noqa: E402, F401


# ---------------------------------- CLI ----------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--nl", help="Single natural-language sentence to convert.")
    g.add_argument("--input", help="JSONL file with `natural` per line.")
    ap.add_argument("--output", help="Output JSONL (required with --input).")
    ap.add_argument("--limit", type=int, default=0, help="Cap on --input rows.")
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip the first N input rows before evaluating. Use a "
                    "different offset each day (0, 50, 100, ...) so daily runs "
                    "test DISJOINT rows -- the 3-day average is then a real "
                    "~150-row estimate, not 3x the same deterministic 50.")
    ap.add_argument("--eval", action="store_true",
                    help="If input rows also have `tree`, compute all metrics.")
    ap.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Groq model id. Defaults to llama-3.3-70b-versatile (free, 30 RPM, "
             "14400 RPD). Alternatives: llama-3.1-8b-instant (faster, weaker), "
             "qwen2.5-32b, mixtral-8x7b-32768, etc. See console.groq.com/playground.",
    )
    ap.add_argument("--few-shot-file", default="data/nl2tl_converted.jsonl",
                    help="JSONL to draw few-shot examples from.")
    ap.add_argument("--n-shots", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-consistency", type=int, default=1, metavar="K",
                    help="#1: sample K parses (temperature>0) and majority-vote "
                    "the canonical tree structure (default 1 = off). Raises "
                    "path-F1 by rescuing the majority-correct nesting; costs "
                    "K x tokens.")
    ap.add_argument("--sc-temperature", type=float, default=0.7,
                    help="sampling temperature for --self-consistency (default 0.7)")
    ap.add_argument("--roundtrip", action="store_true",
                    help="#4: after parsing, ask the LLM to paraphrase the tree "
                    "back to NL with explicit scope and self-refine if the "
                    "nesting disagrees (+1 token/call).")
    ap.add_argument("--rpm", type=int, default=25,
                    help="Target RPM (default 25, safe under free-tier 30 RPM).")
    ap.add_argument("--max-retries", type=int, default=5)
    args = ap.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: set environment variable GROQ_API_KEY before running.\n"
              "Get a free key at: https://console.groq.com/keys",
              file=sys.stderr)
        sys.exit(2)

    fs_path = Path(args.few_shot_file)
    if not fs_path.exists():
        print(f"ERROR: few-shot file not found: {fs_path}", file=sys.stderr)
        sys.exit(2)
    few_shot = load_few_shot(fs_path, n=args.n_shots, seed=args.seed)
    print(f"Loaded {len(few_shot)} few-shot examples (ops: "
          f"{[e['tree']['op'] for e in few_shot]}).", file=sys.stderr)

    client = groq.Groq()
    limiter = RateLimiter(rpm=args.rpm)
    print(f"Rate limit: {args.rpm} req/min "
          f"({limiter.min_interval:.1f}s between calls), "
          f"max {args.max_retries} retries on 429/5xx.", file=sys.stderr)
    if args.self_consistency > 1:
        print(f"#1 self-consistency: {args.self_consistency} samples @ T="
              f"{args.sc_temperature}, majority-vote canonical structure.",
              file=sys.stderr)
    if args.roundtrip:
        print("#4 round-trip scope self-refine: ON.", file=sys.stderr)

    def _parse(text):
        if args.self_consistency > 1:
            t = nl_to_tree_self_consistent(
                text, model=args.model, few_shot=few_shot, client=client,
                k=args.self_consistency, temperature=args.sc_temperature,
                limiter=limiter, max_retries=args.max_retries)
        else:
            t = nl_to_tree(text, model=args.model, few_shot=few_shot,
                           client=client, limiter=limiter,
                           max_retries=args.max_retries)
        if args.roundtrip:
            from nl_to_tree_selfcorrect import roundtrip_scope_refine
            t = roundtrip_scope_refine(text, t, client=client, model=args.model)
        return t

    if args.nl is not None:
        tree = _parse(args.nl)
        print(json.dumps({"natural": args.nl, "tree": tree},
                         ensure_ascii=False, indent=2))
        return

    in_p, out_p = Path(args.input), Path(args.output or "")
    if not args.output:
        print("ERROR: --output is required with --input", file=sys.stderr)
        sys.exit(2)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_ok = n_err = 0
    scored = []
    t0 = time.time()
    with in_p.open() as fin, out_p.open("w") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            if i < args.offset:                       # skip first --offset rows
                continue
            if args.limit and i >= args.offset + args.limit:
                break
            row = json.loads(line)
            nl = row.get("natural") or row.get("nl")
            if not nl:
                continue
            n_total += 1
            try:
                pred = _parse(nl)
                rec = {"natural": nl, "predicted_tree": pred,
                       "formula": row.get("formula")}
                if args.eval and row.get("tree"):
                    rec["gold_tree"] = row["tree"]
                    m = all_metrics(pred, row["tree"])
                    rec["metrics"] = m
                    scored.append(m)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
                print(f"  [{n_total}] ok", end="\r", file=sys.stderr)
            except Exception as e:
                n_err += 1
                fout.write(json.dumps({"natural": nl, "error": str(e)},
                                      ensure_ascii=False) + "\n")
                print(f"  [{n_total}] ERR {type(e).__name__}: {e}",
                      file=sys.stderr)
                if "tokens per day" in str(e).lower() or \
                   "requests per day" in str(e).lower():
                    print(f"\nBailing out — daily token/request quota exhausted on "
                          f"{args.model}. Two fixes:\n"
                          f"  1. Try --model llama-3.1-8b-instant  "
                          f"(500k TPD vs 100k for 70b)\n"
                          f"  2. Try --n-shots 4  (cuts tokens per call ~half)\n"
                          f"  3. Wait for daily reset (UTC midnight).",
                          file=sys.stderr)
                    break
    elapsed = time.time() - t0

    print(f"\n\nDone.  Total: {n_total}  OK: {n_ok}  Errors: {n_err}",
          file=sys.stderr)
    if args.eval and scored:
        agg = aggregate(scored)
        print(f"\n=== Metrics over {agg['n']} scored rows ===", file=sys.stderr)
        print(f"  exact_match (strict):   {agg['exact_match']:.1%}",
              file=sys.stderr)
        print(f"  op_f1   mean:           {agg.get('op_f1', 0):.3f}   "
              f"(right operators used?)", file=sys.stderr)
        print(f"  path_f1 mean:           {agg.get('path_f1', 0):.3f}   "
              f"(right nesting/scoping?)", file=sys.stderr)
        if agg.get("mean_ted") is not None:
            print(f"  ted     mean:           {agg['mean_ted']:.2f} edits/tree",
                  file=sys.stderr)
            print(f"  ted_norm mean:          {agg.get('ted_norm', 0):.3f}   "
                  f"(1.0 = identical)", file=sys.stderr)
    print(f"Wall time: {elapsed:.1f}s  ({elapsed/max(n_total,1):.2f}s per call)",
          file=sys.stderr)
    print(f"Wrote: {out_p}", file=sys.stderr)


if __name__ == "__main__":
    main()
