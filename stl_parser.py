"""
STL (Signal Temporal Logic) parser, AST, and renderers.

Targets NL2TL-style formulas with both symbolic and word forms:
  G[a,b]  /  globally [a,b]
  F[a,b]  /  finally  [a,b]
  U[a,b]  /  until    [a,b]
  &  /  and          |  /  or
  !  /  not / negation
  -> /  imply        <-> / equal / iff
  atomic predicates: prop_1, prop_2, ... (any identifier)

JSON AST schema:
  atom:        {"op": "atom", "name": "prop_1"}
  non-atom:    {"op": <op>, "interval": [lo, hi] | None, "children": [...]}
    where <op> in: not, and, or, imply, iff, globally, finally, until
"""

from lark import Lark, Transformer, v_args

_GRAMMAR = r"""
?start: iff

?iff:   imply
      | iff IFF imply           -> iff_op

?imply: orx
      | imply IMPLY orx         -> imply_op

?orx:   andx
      | orx OR andx             -> or_op

?andx:  untilx
      | andx AND untilx         -> and_op

?untilx: unary
       | unary UNTIL interval unary  -> until_op
       | unary UNTIL unary           -> until_unbounded

?unary: NOT unary                    -> not_op
      | GLOBALLY interval unary      -> glob_op
      | GLOBALLY unary               -> glob_unbounded
      | FINALLY  interval unary      -> fin_op
      | FINALLY  unary               -> fin_unbounded
      | atom

?atom:  NAME                    -> atom_name
      | "(" iff ")"

interval: "[" bound "," bound "]"
?bound: NUMBER                  -> num_b
      | INF                     -> inf_b

INF.2:      "infinite" | "inf"
NOT.2:      "!"   | "not"      | "negation"
GLOBALLY.2: "globally" | "G"
FINALLY.2:  "finally"  | "F"
UNTIL.2:    "until"    | "U"
AND.2:      "and"      | "&"
OR.2:       "or"       | "|"
IMPLY.2:    "imply"    | "->"
IFF.2:      "iff"      | "equal" | "<->"

NAME: /[A-Za-z_][A-Za-z0-9_]*/

%import common.NUMBER
%import common.WS
%ignore WS
"""


def _flatten(op, *children):
    """Flatten chained associative operators: and(and(a,b),c) -> and(a,b,c)."""
    out = []
    for c in children:
        if isinstance(c, dict) and c.get("op") == op:
            out.extend(c["children"])
        else:
            out.append(c)
    return {"op": op, "interval": None, "children": out}


@v_args(inline=True)
class _ToAST(Transformer):
    def atom_name(self, name):
        return {"op": "atom", "name": str(name)}

    def not_op(self, _kw, child):
        return {"op": "not", "interval": None, "children": [child]}

    def glob_op(self, _kw, interval, child):
        return {"op": "globally", "interval": interval, "children": [child]}

    def glob_unbounded(self, _kw, child):
        return {"op": "globally", "interval": None, "children": [child]}

    def fin_op(self, _kw, interval, child):
        return {"op": "finally", "interval": interval, "children": [child]}

    def fin_unbounded(self, _kw, child):
        return {"op": "finally", "interval": None, "children": [child]}

    def until_op(self, left, _kw, interval, right):
        return {"op": "until", "interval": interval, "children": [left, right]}

    def until_unbounded(self, left, _kw, right):
        return {"op": "until", "interval": None, "children": [left, right]}

    def and_op(self, left, _kw, right):
        return _flatten("and", left, right)

    def or_op(self, left, _kw, right):
        return _flatten("or", left, right)

    def imply_op(self, left, _kw, right):
        return {"op": "imply", "interval": None, "children": [left, right]}

    def iff_op(self, left, _kw, right):
        return {"op": "iff", "interval": None, "children": [left, right]}

    def num_b(self, n):
        s = str(n)
        return float(s) if "." in s else int(s)

    def inf_b(self, _):
        return "inf"

    def interval(self, lo, hi):
        return [lo, hi]


_parser = Lark(_GRAMMAR, parser="earley", start="start")


def parse_stl(text: str) -> dict:
    """Parse an STL formula string into the JSON AST schema."""
    tree = _parser.parse(text.strip())
    return _ToAST().transform(tree)


# ----------------------- AST -> STL string (reverse render) -----------------------

def ast_to_stl(node: dict) -> str:
    """Render the AST back to a canonical STL string (for round-trip check)."""
    op = node["op"]
    if op == "atom":
        return node["name"]
    iv = node.get("interval")
    iv_s = f"[{iv[0]},{iv[1]}]" if iv else ""
    kids = [ast_to_stl(c) for c in node["children"]]
    if op == "not":
        return f"!({kids[0]})"
    if op == "globally":
        return f"G{iv_s} ({kids[0]})"
    if op == "finally":
        return f"F{iv_s} ({kids[0]})"
    if op == "until":
        return f"({kids[0]}) U{iv_s} ({kids[1]})"
    if op == "and":
        return "(" + " & ".join(kids) + ")"
    if op == "or":
        return "(" + " | ".join(kids) + ")"
    if op == "imply":
        return f"({kids[0]}) -> ({kids[1]})"
    if op == "iff":
        return f"({kids[0]}) <-> ({kids[1]})"
    raise ValueError(f"Unknown op: {op}")


# ----------------------- AST -> Graphviz DOT (for viz) -----------------------

_OP_LABEL = {
    "and":      "∧  AND",
    "or":       "∨  OR",
    "not":      "¬  NOT",
    "imply":    "→  IMPLY",
    "iff":      "↔  IFF",
    "globally": "□  G  (globally)",
    "finally":  "◇  F  (finally)",
    "until":    "U  (until)",
}

_FILL = {
    "atom":     "#fff3c4",   # yellow
    "globally": "#cfe8ff",   # blue
    "finally":  "#cfe8ff",
    "until":    "#cfe8ff",
    "and":      "#e6e6e6",   # grey
    "or":       "#e6e6e6",
    "not":      "#ffd9d9",   # red
    "imply":    "#e6d9ff",   # purple
    "iff":      "#e6d9ff",
}


def ast_to_dot(node: dict, title: str = "") -> str:
    """Render AST as a Graphviz DOT string. Pass directly to st.graphviz_chart."""
    lines = [
        "digraph G {",
        '  rankdir=TB;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12];',
        '  edge [arrowsize=0.8];',
    ]
    if title:
        safe = title.replace('"', "'")
        lines.append(f'  labelloc="t"; label="{safe}"; fontname="Helvetica"; fontsize=11;')

    counter = [0]

    def add(n):
        nid = f"n{counter[0]}"
        counter[0] += 1
        op = n["op"]
        fill = _FILL.get(op, "#ffffff")
        if op == "atom":
            label = n["name"]
        else:
            label = _OP_LABEL.get(op, op)
            iv = n.get("interval")
            if iv:
                label += f"\\n[{iv[0]}, {iv[1]}]"
        # quote internal quotes if any
        label = label.replace('"', '\\"')
        lines.append(f'  {nid} [label="{label}", fillcolor="{fill}"];')
        if op != "atom":
            for child in n["children"]:
                cid = add(child)
                lines.append(f"  {nid} -> {cid};")
        return nid

    add(node)
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke test
    import json
    samples = [
        "prop_1",
        "G[0,10] prop_1",
        "F[2,5] (prop_1 & prop_2)",
        "prop_1 U[0,5] prop_2",
        "!prop_1",
        "( ( ( prop_2 until [176,415] prop_1 ) and prop_3 ) equal prop_4 )",
        "globally [10,50] ( prop_1 imply finally [0,5] prop_2 )",
    ]
    for s in samples:
        ast = parse_stl(s)
        print("IN :", s)
        print("AST:", json.dumps(ast, ensure_ascii=False))
        print("OUT:", ast_to_stl(ast))
        print()
