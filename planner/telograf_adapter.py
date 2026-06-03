"""
Adapter: our nested STL-tree JSON  ->  TeLoGraF input graph.

TeLoGraF (arXiv:2505.00562) feeds the GNN encoder a syntax-tree graph with
one node per operator / atom.  We must:

  (a) Normalise our tree to the operator set TeLoGraF supports.
      It only handles G / F / U / and / or / not over atomic
      reach / avoid primitives, so we rewrite:
          imply(A, B)   ->  or(not(A), B)
          iff(A, B)     ->  and(or(not(A),B), or(not(B),A))
          not(atom)     ->  atom with kind="avoid"   (folded)
          chained and / or   ->  binary form (TeLoGraF assumes binary)

  (b) Replace each abstract atom `prop_i` with a grounded predicate using
      the per-case `grounding` dict (x, y, z, r, kind).

  (c) Produce the TeLoGraF node-feature matrix and edge_index.
      Node feature layout (8 dims, per TeLoGraF paper sec. 3.2):
          [op_type, t_start, t_end, x, y, z, r, until_order]
      Missing fields use -1.0 as the sentinel value.

The exact integer codes for `op_type` are pinned in OP_CODE below.  If the
TeLoGraF source uses a different numbering, edit OP_CODE to match -- the
rest of the adapter is unaffected.  See README.md for verification steps.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Operator codes -- pinned to TeLoGraF's actual encoding in
#   external/TeLoGraF/code/stl_to_seq_utils.py::stl_to_seq
# which lays each node out as:
#   [node_type_i, ts, te, x, y, z, r, n_child]
# and uses the integer table below.  There is NO separate "avoid" atom;
# avoidance is represented as a NOT node parenting a reach atom.
# ---------------------------------------------------------------------------
OP_CODE = {
    "and":      0,
    "or":       1,
    "not":      2,
    "finally":  5,           # TeLoGraF: "eventually"
    "globally": 6,           # TeLoGraF: "always"
    "until":    7,
    "reach":    8,           # atomic predicate (the only atom kind)
}
SENTINEL = -1.0
FEATURE_DIM = 8


# ---------------------------------------------------------------------------
# Step (a): normalise tree to TeLoGraF operator set
# ---------------------------------------------------------------------------
def normalise(node: dict) -> dict:
    """Rewrite imply / iff into the TeLoGraF operator set.

    TeLoGraF accepts {and, or, not, finally, globally, until, reach} so:
      imply(A, B)  ->  or(not(A), B)
      iff(A, B)    ->  and(imply(A,B), imply(B,A))
    `not(atom)` stays as-is -- TeLoGraF represents avoidance with a NOT
    node parenting the reach atom (NOT a separate "avoid" atom).
    And/or chains are kept n-ary; TeLoGraF's GNN aggregates over children
    via scatter, so binarisation is unnecessary and would distort tree depth.
    """
    if not isinstance(node, dict):
        raise TypeError(f"node must be dict, got {type(node).__name__}")
    op = node["op"]

    if op == "atom":
        return {"op": "atom", "name": node["name"]}

    if op == "imply":
        a, b = node["children"]
        return normalise({
            "op": "or", "interval": None,
            "children": [{"op": "not", "interval": None,
                          "children": [a]}, b],
        })

    if op == "iff":
        a, b = node["children"]
        ab = {"op": "imply", "interval": None, "children": [a, b]}
        ba = {"op": "imply", "interval": None, "children": [b, a]}
        return normalise({
            "op": "and", "interval": None, "children": [ab, ba],
        })

    if op == "not":
        return {"op": "not", "interval": None,
                "children": [normalise(node["children"][0])]}

    if op in ("and", "or"):
        kids = [normalise(c) for c in node["children"]]
        if len(kids) == 1:
            return kids[0]
        return {"op": op, "interval": None, "children": kids}

    if op in ("globally", "finally"):
        return {"op": op, "interval": node.get("interval"),
                "children": [normalise(node["children"][0])]}

    if op == "until":
        l, r = node["children"]
        return {"op": "until", "interval": node.get("interval"),
                "children": [normalise(l), normalise(r)]}

    raise ValueError(f"unknown op: {op}")


# ---------------------------------------------------------------------------
# Step (b) + (c): ground atoms and emit the (node_features, edge_index)
# ---------------------------------------------------------------------------
@dataclass
class TeLoGraFGraph:
    """Plain-Python container; convert to torch_geometric.Data at the call
    site so this file has no hard torch dependency."""
    node_features: list   # list of [8] floats
    edge_index:    list   # list of [src, dst] pairs (child -> parent)
    until_left_right: list  # for each Until parent, [left_child_idx, right_child_idx]
    op_names:      list   # debug labels per node

    def as_tg_data(self):  # lazy torch import
        import torch
        from torch_geometric.data import Data
        x = torch.tensor(self.node_features, dtype=torch.float32)
        ei = torch.tensor(self.edge_index, dtype=torch.long).t().contiguous() \
             if self.edge_index else torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=ei,
                    until_left_right=self.until_left_right)


def _atom_feature(g: dict, left_child: int = -1) -> list:
    """Reach-atom node feature, matching TeLoGraF's training-time
    encoding in `train_gstl_v1.py::get_graph_stl_embed_from_tree`:
        [node_type=8, ta=-1, tb=-1, obj_x, obj_y, obj_z, obj_r, left_child]
    There is no separate `avoid` atom in TeLoGraF; `not(reach)` is the
    way to express avoidance.  `left_child` is +1 only when this atom
    is the first child of an `until` operator, -1 otherwise.
    """
    return [OP_CODE["reach"],
            SENTINEL, SENTINEL,                       # ta, tb -1 for atoms
            g["x"], g["y"], g.get("z", 0.0), g["r"],  # geometry
            left_child]                               # -1 (or +1 if first child of until)


def _internal_feature(op: str, ts: float, te: float,
                       left_child: int = -1) -> list:
    """Non-atom node feature.  TeLoGraF fills (obj_x, obj_y, obj_z, obj_r)
    with -1, -1, -1, -1 for non-reach nodes and uses `left_child` as the
    last column (= +1 only for until's first child)."""
    return [OP_CODE[op],
            ts, te,
            SENTINEL, SENTINEL, SENTINEL, SENTINEL,   # x, y, z, r = -1
            left_child]


def _interval(node) -> tuple[float, float]:
    iv = node.get("interval")
    if iv is None:
        return (SENTINEL, SENTINEL)
    lo, hi = iv
    return (float(lo) if lo != "inf" else SENTINEL,
            float(hi) if hi != "inf" else SENTINEL)


def to_telograf_graph(tree: dict, grounding: dict) -> TeLoGraFGraph:
    """Walk the normalised tree, emit node features + edges in child->parent
    convention required by TeLoGraF.

    `left_child` is propagated down (parent decides per-child) to handle
    the until-first-child = +1 case exactly like TeLoGraF's
    `get_graph_stl_embed_from_tree`.
    """
    tree = normalise(tree)
    feats: list = []
    edges: list = []
    until_pairs: list = []
    op_names: list = []

    def add(node, left_child: int = -1) -> int:
        idx = len(feats)
        op = node["op"]
        op_names.append(op)

        if op == "atom":
            g = grounding.get(node["name"])
            if g is None:
                raise KeyError(f"atom {node['name']} not in grounding")
            feats.append(_atom_feature(g, left_child=left_child))
            return idx

        ts, te = _interval(node)
        feats.append(_internal_feature(op, ts, te, left_child=left_child))

        # Children's left_child bit:
        #   - until's first child gets +1, second child gets -1
        #   - every other case: -1
        is_until = (op == "until")
        child_idxs = []
        for ci, c in enumerate(node["children"]):
            lc = 1 if (is_until and ci == 0) else -1
            child_idxs.append(add(c, left_child=lc))

        for cid in child_idxs:
            edges.append([cid, idx])     # child -> parent (TeLoGraF convention)

        if is_until:
            until_pairs.append({"parent": idx,
                                "left":   child_idxs[0],
                                "right":  child_idxs[1]})

        return idx

    add(tree, left_child=-1)
    return TeLoGraFGraph(feats, edges, until_pairs, op_names)


# ---------------------------------------------------------------------------
# Convenience: drive the whole pipeline for a case dict
# ---------------------------------------------------------------------------
def case_to_graph(case: dict) -> TeLoGraFGraph:
    return to_telograf_graph(case["tree"], case["grounding"])


if __name__ == "__main__":
    from case_examples import CASES
    for c in CASES:
        g = case_to_graph(c)
        print(f"{c['id']:20s}  nodes={len(g.node_features)}  "
              f"edges={len(g.edge_index)}  ops={g.op_names}")
