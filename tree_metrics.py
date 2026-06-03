"""
Looser matching metrics for STL parse trees.

Why these matter:
  exact-match is brittle — a tree that's structurally close (right operators,
  right scoping but one nested swap) scores 0/1 the same as random garbage.
  These metrics give graded credit so trends are visible at small sample sizes
  and so the "near miss" failure mode is separable from "completely wrong".

Metrics:
  exact_match : 0/1 strict equality after sort-normalizing and/or children
  op_f1       : F1 over operator multiset.  "Did you use the right ingredients?"
  path_f1     : F1 over root-to-leaf path multiset.  "Did you put them in the
                right structural relations?"  Sensitive to scoping/nesting.
  ted         : tree edit distance (Zhang-Shasha via `zss`). Lower = closer.
  ted_norm    : 1 - TED / max(|pred|, |gold|).  1.0 = identical, 0 = far.

Reading the numbers:
  - op_f1 high but path_f1 low  →  right operators, wrong nesting
  - op_f1 low                   →  wrong operator vocabulary chosen
  - exact_match 0 + ted_norm > 0.8  →  semantically very close, a single
                                       nested swap or naming mismatch
"""
from __future__ import annotations

import json
from collections import Counter

try:
    import zss
    _HAVE_ZSS = True
except ImportError:
    _HAVE_ZSS = False


# --------------------------- structural helpers ---------------------------

def tree_size(node: dict) -> int:
    """Number of nodes in the tree."""
    if not isinstance(node, dict):
        return 0
    if node.get("op") == "atom":
        return 1
    children = node.get("children") or []
    return 1 + sum(tree_size(c) for c in children)


def _label(node: dict) -> str:
    """A label that bundles op + interval + atom name for matching."""
    op = node.get("op")
    if op == "atom":
        return f"atom:{node.get('name')}"
    iv = node.get("interval")
    return f"{op}:{tuple(iv)}" if iv else op


def _children(node: dict) -> list:
    if not isinstance(node, dict) or node.get("op") == "atom":
        return []
    return node.get("children") or []


# --------------------------- exact match (canonical) ---------------------------

def _norm(node: dict) -> dict:
    if not isinstance(node, dict):
        return node
    op = node.get("op")
    if op == "atom":
        return {"op": "atom", "name": node.get("name")}
    children = [_norm(c) for c in (node.get("children") or [])]
    if op in ("and", "or"):  # commutative
        children = sorted(children, key=lambda x: json.dumps(x, sort_keys=True))
    return {"op": op, "interval": node.get("interval"), "children": children}


def exact_match(pred: dict, gold: dict) -> bool:
    return _norm(pred) == _norm(gold)


def canonical_key(node: dict) -> str:
    """Stable string key for a tree under and/or-commutativity.

    Two trees that are equal up to reordering of commutative (and/or) children
    map to the SAME key.  Used for self-consistency majority voting over
    sampled parses (vote on canonical STRUCTURE, not raw string)."""
    return json.dumps(_norm(node), sort_keys=True, ensure_ascii=False)


# --------------------------- operator-multiset F1 ---------------------------

def op_multiset(node: dict) -> Counter:
    """Count every label in the tree."""
    if not isinstance(node, dict):
        return Counter()
    c = Counter()
    c[_label(node)] += 1
    for ch in _children(node):
        c.update(op_multiset(ch))
    return c


def _multiset_f1(a: Counter, b: Counter) -> float:
    inter = sum((a & b).values())
    sa, sb = sum(a.values()), sum(b.values())
    if sa == 0 and sb == 0:
        return 1.0
    p = inter / sa if sa else 0.0
    r = inter / sb if sb else 0.0
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def op_f1(pred: dict, gold: dict) -> float:
    return _multiset_f1(op_multiset(pred), op_multiset(gold))


# --------------------------- root-to-leaf path F1 ---------------------------

def root_to_leaf_paths(node: dict, prefix: tuple = ()) -> Counter:
    """Multiset of all root-to-leaf paths (each path is a tuple of labels).

    Captures scoping: globally(imply(...)) and imply(globally(...)) share the
    same op_multiset but produce different paths.
    """
    if not isinstance(node, dict):
        return Counter()
    cur = prefix + (_label(node),)
    if node.get("op") == "atom" or not _children(node):
        return Counter({cur: 1})
    out = Counter()
    for ch in _children(node):
        out.update(root_to_leaf_paths(ch, cur))
    return out


def path_f1(pred: dict, gold: dict) -> float:
    return _multiset_f1(root_to_leaf_paths(pred), root_to_leaf_paths(gold))


# --------------------------- tree edit distance ---------------------------

def tree_edit_distance(pred: dict, gold: dict) -> int:
    """Zhang-Shasha tree edit distance via the `zss` library."""
    if not _HAVE_ZSS:
        raise ImportError("`zss` not installed; `pip install zss`")
    return int(zss.simple_distance(
        pred, gold, get_children=_children, get_label=_label
    ))


def ted_normalized(pred: dict, gold: dict) -> float:
    """1 - TED / max(|pred|, |gold|). 1.0 = identical, 0 = totally different."""
    sa, sb = tree_size(pred), tree_size(gold)
    if sa == 0 and sb == 0:
        return 1.0
    if not _HAVE_ZSS:
        return float("nan")
    t = tree_edit_distance(pred, gold)
    return max(0.0, 1.0 - t / max(sa, sb))


# --------------------------- bundle ---------------------------

def all_metrics(pred: dict, gold: dict) -> dict:
    out = {
        "exact_match": exact_match(pred, gold),
        "op_f1":       op_f1(pred, gold),
        "path_f1":     path_f1(pred, gold),
        "pred_size":   tree_size(pred),
        "gold_size":   tree_size(gold),
    }
    if _HAVE_ZSS:
        out["ted"]      = tree_edit_distance(pred, gold)
        out["ted_norm"] = ted_normalized(pred, gold)
    return out


def aggregate(rows: list[dict]) -> dict:
    """Aggregate metric dicts from `all_metrics` into mean / hit-rate summary."""
    if not rows:
        return {}
    n = len(rows)
    em = sum(1 for r in rows if r.get("exact_match"))
    keys = ("op_f1", "path_f1", "ted_norm")
    means = {k: sum(r.get(k, 0.0) for r in rows) / n for k in keys
             if any(k in r for r in rows)}
    return {
        "n": n,
        "exact_match": em / n,
        **means,
        "mean_ted": (sum(r.get("ted", 0) for r in rows) / n) if any("ted" in r for r in rows) else None,
    }


# --------------------------- self-test ---------------------------

if __name__ == "__main__":
    # demonstrate on the two mismatches from the user's earlier run
    pred = {"op": "not", "interval": None, "children": [
        {"op": "iff", "interval": None, "children": [
            {"op": "until", "interval": None, "children": [
                {"op": "atom", "name": "prop_2"},
                {"op": "atom", "name": "prop_1"},
            ]},
            {"op": "atom", "name": "prop_3"},
        ]},
    ]}
    gold = {"op": "iff", "interval": None, "children": [
        {"op": "not", "interval": None, "children": [
            {"op": "until", "interval": None, "children": [
                {"op": "atom", "name": "prop_2"},
                {"op": "atom", "name": "prop_1"},
            ]},
        ]},
        {"op": "atom", "name": "prop_3"},
    ]}
    m = all_metrics(pred, gold)
    print("Example: scope swap of `not` and `iff`")
    print(json.dumps(m, indent=2))
