"""
NL -> nested STL tree -> grounded planner case  (the paper's Stage-1 pipeline,
wired to the closed-loop demo map).

This is the ONE shared module behind both demo front-ends (run_2d.py, run_ros.py).
It uses the SAME map and task family as the closed-loop simulation
(planner case `closed_loop_multi` / code/sim_ros2/scenario.py):

  * three fixed goal regions  A=(+3,-3)  B=(+3,+3)  C=(-3,+3)  (radius 0.45 m),
  * a start at (-3,-3) inside a uniform field of cylinder obstacles the robot
    senses online (the obstacles are NOT in the spec; the keep-safe predicate
    G(not unsafe) is grounded by the sensed disks).

The user types a natural-language task that refers to those regions, e.g.

    "Visit A, then B, while always staying clear of every obstacle,
     and finish within 180 seconds."

`nl_to_case()` runs it through the paper's frozen-LLM parser
(`nl_to_tree`, code/nl_to_tree_groq.py) to get a nested STL tree, then GROUNDS
the region-named atoms (reach_A/reach_B/reach_C, unsafe) to the fixed map
coordinates and returns a planner `case` dict the flow-matching planner /
TurtleBot3 follower consume directly.

Booth-safe: if no GROQ_API_KEY is set (or the API call fails), it falls back to a
deterministic keyword parser so the demo still runs offline -- the README says so
explicitly; the LLM path is the real one.
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

# the paper's code/ tree (planner, nl_to_tree_groq, scenario).  Search upward so
# the demo works whether it lives in "Demo supplement/demo/" (sibling of code/) or
# inside the repo at "code/demo/".
def _find_code_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "planner").is_dir() and (p / "nl_to_tree_groq.py").exists():
            return p
        if (p / "code" / "planner").is_dir():
            return p / "code"
    return start


CODE = _find_code_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(CODE))

# ---- the fixed demo map (identical to closed_loop_multi / scenario.py) ----
LANDMARKS = OrderedDict([                 # name -> (x, y, radius)
    ("A", (3.0, -3.0, 0.45)),
    ("B", (3.0,  3.0, 0.45)),
    ("C", (-3.0, 3.0, 0.45)),
])
START = (-3.0, -3.0)
WORLD_BOUNDS = [-3.8, 3.8, -3.8, 3.8]
DEFAULT_DEADLINE_S = 170.0                # scenario.REACH_DEADLINE_S

# default Groq model for the live parse (override with $NL2TL_MODEL). 8b-instant
# is the booth default: fast and with a high daily token budget, and the few-shot
# makes it reliable on these templated region tasks. The larger 70b is stronger on
# hard sentences but has a low free-tier daily token cap that a busy booth exhausts.
DEFAULT_MODEL = os.environ.get("NL2TL_MODEL", "llama-3.1-8b-instant")

# ---- few-shot exemplars: REGION-NAMED atoms, so grounding is a dict lookup ----
# (the LLM is steered by these few-shot pairs only -- parameter-update-free.)
DEMO_FEWSHOT = [
    {"natural": "Go to region A.",
     "tree": {"op": "finally", "interval": None,
              "children": [{"op": "atom", "name": "reach_A"}]}},
    {"natural": "Visit A, then B.",
     "tree": {"op": "and", "interval": None, "children": [
         {"op": "finally", "interval": None,
          "children": [{"op": "atom", "name": "reach_A"}]},
         {"op": "finally", "interval": None,
          "children": [{"op": "atom", "name": "reach_B"}]}]}},
    {"natural": "Reach region C within 60 seconds.",
     "tree": {"op": "finally", "interval": [0, 60],
              "children": [{"op": "atom", "name": "reach_C"}]}},
    {"natural": "Go to A while always staying clear of every obstacle.",
     "tree": {"op": "and", "interval": None, "children": [
         {"op": "finally", "interval": None,
          "children": [{"op": "atom", "name": "reach_A"}]},
         {"op": "globally", "interval": None, "children": [
             {"op": "not", "interval": None,
              "children": [{"op": "atom", "name": "unsafe"}]}]}]}},
    {"natural": "Visit B, then A, then C in order, while always avoiding "
                "obstacles, and finish within 180 seconds.",
     "tree": {"op": "and", "interval": None, "children": [
         {"op": "finally", "interval": [0, 180],
          "children": [{"op": "atom", "name": "reach_B"}]},
         {"op": "finally", "interval": [0, 180],
          "children": [{"op": "atom", "name": "reach_A"}]},
         {"op": "finally", "interval": [0, 180],
          "children": [{"op": "atom", "name": "reach_C"}]},
         {"op": "globally", "interval": None, "children": [
             {"op": "not", "interval": None,
              "children": [{"op": "atom", "name": "unsafe"}]}]}]}},
    {"natural": "Reach A and C, keeping safe the whole time.",
     "tree": {"op": "and", "interval": None, "children": [
         {"op": "finally", "interval": None,
          "children": [{"op": "atom", "name": "reach_A"}]},
         {"op": "finally", "interval": None,
          "children": [{"op": "atom", "name": "reach_C"}]},
         {"op": "globally", "interval": None, "children": [
             {"op": "not", "interval": None,
              "children": [{"op": "atom", "name": "unsafe"}]}]}]}},
]

_SAFE_WORDS = ("unsafe", "obstacle", "collision", "clear", "avoid", "safe")


# --------------------------- LLM parse (the real path) ---------------------------

def make_client():
    """A Groq client if GROQ_API_KEY is set, else None (booth offline mode)."""
    if not os.environ.get("GROQ_API_KEY"):
        return None
    try:
        import groq
        return groq.Groq()
    except Exception:
        return None


def parse_nl_llm(nl: str, *, model: str, client) -> dict | None:
    """Frozen-LLM parse -> nested STL tree (the paper's nl_to_tree). None on any
    failure so the caller can fall back to the deterministic parser."""
    try:
        from nl_to_tree_groq import nl_to_tree
        return nl_to_tree(nl, model=model, few_shot=DEMO_FEWSHOT, client=client)
    except Exception as e:
        print(f"[nl_grounding] LLM parse failed ({e}); using keyword fallback",
              file=sys.stderr)
        return None


# --------------------------- deterministic fallback ---------------------------

def parse_nl_keyword(nl: str) -> dict:
    """Offline fallback: region letters in order of mention + keep-safe + deadline,
    assembled into the same nested STL tree the LLM would emit. Deterministic, no
    network -- so the demo still runs with no API key."""
    order = []
    for m in re.finditer(r"\b([ABC])\b", nl, flags=re.I):
        L = m.group(1).upper()
        if L not in order:
            order.append(L)
    if not order:                       # nothing recognised -> default tour
        order = list(LANDMARKS)
    keep = any(w in nl.lower() for w in _SAFE_WORDS)
    dl = _deadline_from_text(nl)
    iv = [0, int(dl)] if dl else None
    children = [{"op": "finally", "interval": iv,
                 "children": [{"op": "atom", "name": f"reach_{L}"}]}
                for L in order]
    if keep:
        children.append({"op": "globally", "interval": None, "children": [
            {"op": "not", "interval": None,
             "children": [{"op": "atom", "name": "unsafe"}]}]})
    if len(children) == 1:
        return children[0]
    return {"op": "and", "interval": None, "children": children}


def _deadline_from_text(nl: str) -> float | None:
    m = re.search(r"within\s+(\d+(?:\.\d+)?)\s*(s|sec|second|seconds|"
                  r"min|minute|minutes)", nl, flags=re.I)
    if not m:
        return None
    val = float(m.group(1))
    return val * 60.0 if m.group(2).lower().startswith("min") else val


# --------------------------- grounding ---------------------------

def _reach_atoms_in_order(node, out):
    """DFS left-to-right; collect region letters of every reach_* atom in the
    order the tree lists them (= the visit order for a sequence tour)."""
    if not isinstance(node, dict):
        return
    if node.get("op") == "atom":
        m = re.search(r"reach[_\- ]?([abc])", node.get("name", ""), flags=re.I) \
            or re.fullmatch(r"\s*([abc])\s*", node.get("name", ""), flags=re.I)
        if m:
            L = m.group(1).upper()
            if L not in out and L in LANDMARKS:
                out.append(L)
        return
    for c in node.get("children") or []:
        _reach_atoms_in_order(c, out)


def _has_keepsafe(node) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("op") == "atom":
        return any(w in node.get("name", "").lower() for w in ("unsafe", "obstacle"))
    return any(_has_keepsafe(c) for c in node.get("children") or [])


def _max_finally_upper(node) -> float | None:
    best = None
    if isinstance(node, dict):
        if node.get("op") == "finally" and node.get("interval"):
            best = float(node["interval"][1])
        for c in node.get("children") or []:
            u = _max_finally_upper(c)
            if u is not None:
                best = u if best is None else max(best, u)
    return best


def ground(tree: dict, nl: str = "") -> dict:
    """Resolve a parsed STL tree against the fixed demo map.

    Returns {goals, goal_names, keep_safe, deadline_s} where `goals` is the
    ORDERED list of (x, y, r) the robot visits (sequence order = tree order)."""
    order = []
    _reach_atoms_in_order(tree, order)
    if not order:                       # parser produced no usable reach atom
        order = list(LANDMARKS)
    goals = [LANDMARKS[L] for L in order]
    keep_safe = _has_keepsafe(tree) or any(w in nl.lower() for w in _SAFE_WORDS)
    dl = _deadline_from_text(nl) or _max_finally_upper(tree) or DEFAULT_DEADLINE_S
    return {"goals": goals, "goal_names": order,
            "keep_safe": bool(keep_safe), "deadline_s": float(dl)}


def build_case(nl: str, tree: dict, g: dict) -> dict:
    """A planner `case` dict (tree + grounded atoms + map hint) the flow planner
    and the TurtleBot3 follower consume directly. Reach atoms are an ordered
    grounding dict so the follower visits them in spec order."""
    grounding = OrderedDict()
    for name, (x, y, r) in zip(g["goal_names"], g["goals"]):
        grounding[f"reach_{name}"] = {"kind": "reach", "x": x, "y": y,
                                      "z": 0.0, "r": r}
    if g["keep_safe"]:
        # grounded online by the sensed cylinder disks (handled by the runtime
        # keep-safe shield); kept here so the spec is self-describing.
        grounding["unsafe"] = {"kind": "avoid", "sensed": True}
    return {
        "id": "nl_demo", "role": "telograf", "nl": nl, "tree": tree,
        "grounding": grounding,
        "map_hint": {"world_bounds": list(WORLD_BOUNDS), "start": list(START),
                     "style": "telograf_native", "telograf_samples": 8,
                     "reach_deadline_s": g["deadline_s"], "nominal_speed": 0.18},
    }


# --------------------------- one-call orchestration ---------------------------

def nl_to_case(nl: str, *, model: str = DEFAULT_MODEL, use_llm: bool = True):
    """NL -> (case, tree, grounding-info, used_llm). The single entry point both
    front-ends call."""
    tree = None
    used_llm = False
    if use_llm:
        client = make_client()
        if client is not None:
            tree = parse_nl_llm(nl, model=model, client=client)
            used_llm = tree is not None
    if tree is None:
        tree = parse_nl_keyword(nl)
    g = ground(tree, nl)
    case = build_case(nl, tree, g)
    return case, tree, g, used_llm


def stl_string(tree: dict) -> str:
    """Pretty one-line STL string for display (uses the paper's serializer)."""
    try:
        from stl_parser import ast_to_stl
        return ast_to_stl(tree)
    except Exception:
        return json.dumps(tree)


if __name__ == "__main__":
    demo_nl = " ".join(sys.argv[1:]) or \
        "Visit A, then B, while always staying clear of every obstacle, " \
        "and finish within 180 seconds."
    case, tree, g, used_llm = nl_to_case(demo_nl)
    print("NL   :", demo_nl)
    print("parse:", "LLM (frozen)" if used_llm else "keyword fallback (offline)")
    print("STL  :", stl_string(tree))
    print("goals:", " -> ".join(g["goal_names"]),
          f"| keep_safe={g['keep_safe']} | deadline={g['deadline_s']:.0f}s")
    print("case :", json.dumps({k: case[k] for k in ("id", "grounding")},
                               indent=1))
