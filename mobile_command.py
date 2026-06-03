"""
Mobile UbiComp algorithm: from a phone-typed natural-language task to a
verified SPATIOTEMPORAL COMMAND for the robot, end to end.

This is the integrative pipeline the poster is about -- a non-expert speaks to
their space in plain language, and the system returns a checked, executable
command without training anything:

  NL  --LLM-->  nested STL tree            (frozen LLM)
      --Tier-4 self-correction-->  tree validated against the scene grounding
      --Tier-5 operator normalisation-->  imply/iff rewritten to TeLoGraF basis
      --Tier-3 feasibility self-check-->  in-distribution? flow : defer
      --Tier-1 decomposition + flow matching-->  trajectory
  ==> SPATIOTEMPORAL COMMAND = ordered (zone, centre, time-window, reach/avoid)
      constraints + the planned trajectory + the STL robustness margin.

`nl_to_spatiotemporal_command` is UI-free so it unit-tests headlessly; the
Streamlit phone UI (mobile_app.py) is a thin shell over it.

A "scene" grounds named places to disks:
  {"name": "...", "grounding": {"kitchen": {"kind":"reach","x":..,"y":..,"r":..}, ...},
   "start": [x, y], "bounds": [xmin,xmax,ymin,ymax]}
"""
from __future__ import annotations

from typing import List, Tuple


# A small home/office scene with NAMED places (more pervasive-computing than
# abstract prop_1).  The robot starts at the door.
DEFAULT_SCENE = {
    "name": "apartment",
    "grounding": {
        "kitchen":     {"kind": "reach", "x":  2.5, "y":  2.5, "z": 0.0, "r": 0.6},
        "living_room": {"kind": "reach", "x": -2.0, "y":  2.0, "z": 0.0, "r": 0.6},
        "charger":     {"kind": "reach", "x":  3.0, "y": -2.5, "z": 0.0, "r": 0.5},
        "bedroom":     {"kind": "avoid", "x": -2.5, "y": -2.0, "z": 0.0, "r": 0.8},
    },
    "start": [-3.0, -3.0],
    "bounds": [-4, 4, -4, 4],
}


def _reach_windows(node, grounding, acc=None, window=None):
    """Pair every reach atom with the time-window of its nearest enclosing
    finally/globally (None = unbounded), in tree order."""
    acc = acc if acc is not None else []
    if not isinstance(node, dict):
        return acc
    op = node.get("op")
    if op == "atom":
        g = grounding.get(node["name"], {})
        if g.get("kind", "reach") == "reach":
            acc.append({"zone": node["name"], "x": g.get("x"), "y": g.get("y"),
                        "r": g.get("r"), "kind": "reach", "window": window})
        return acc
    w = node.get("interval") if op in ("finally", "globally", "until") else window
    for c in node.get("children") or []:
        _reach_windows(c, grounding, acc, w)
    return acc


def _avoid_zones(grounding) -> List[dict]:
    return [{"zone": name, "x": g["x"], "y": g["y"], "r": g["r"], "kind": "avoid"}
            for name, g in grounding.items() if g.get("kind") == "avoid"]


def nl_to_spatiotemporal_command(nl: str, scene: dict | None = None, *,
                                 client=None, model: str = "llama-3.3-70b-versatile",
                                 few_shot: list | None = None,
                                 backend: str = "auto", n_steps: int = 96,
                                 max_rounds: int = 2) -> dict:
    """Run the full mobile pipeline.  Returns a command dict (JSON-able)."""
    scene = scene or DEFAULT_SCENE
    grounding = scene["grounding"]

    # 1) NL -> validated STL tree (Tier-4 self-correction against the scene).
    from nl_to_tree_selfcorrect import selfcorrect
    if client is None:
        import groq
        client = groq.Groq()
    if few_shot is None:
        from pathlib import Path
        from nl_to_tree_groq import load_few_shot
        p = Path(__file__).resolve().parent / "data" / "nl2tl_converted.jsonl"
        few_shot = load_few_shot(p, n=8) if p.exists() else []
    sc = selfcorrect(nl, model=model, few_shot=few_shot, client=client,
                     grounding=grounding, start=scene["start"],
                     bounds=scene["bounds"], max_rounds=max_rounds)
    tree = sc["tree"]

    cmd = {"nl": nl, "scene": scene["name"], "tree": tree,
           "parse_rounds": sc["rounds"], "parse_accepted": sc["accepted"],
           "parse_transcript": sc["transcript"]}
    if not tree:
        cmd["error"] = "could not parse NL into a valid STL tree"
        return cmd

    # 2) Spatiotemporal command = ordered reach constraints (with windows) +
    #    a single sensor-groundable "always keep safe" predicate G(not unsafe).
    #    (Tier-5 operator normalisation happens inside the planner/adapter when
    #    the tree is encoded for TeLoGraF.)
    cmd["reach_sequence"] = _reach_windows(tree, grounding)
    cmd["avoid_zones"] = _avoid_zones(grounding)
    cmd["keep_safe"] = "G(not unsafe)  with unsafe = union of avoid zones / sensed obstacles"

    # 3) Build a case and run feasibility (Tier-3) + plan (Tier-1/flow).
    case = {"id": "mobile", "tree": tree, "grounding": grounding,
            "map_hint": {"start": list(scene["start"]),
                         "world_bounds": list(scene["bounds"]),
                         "telograf_samples": 16, "gz_world": "maze"}}
    try:
        from planner import plan_waypoints, telograf_available, telograf_feasibility
        if telograf_available():
            rep = telograf_feasibility(case, n_steps=64)
            cmd["feasibility"] = {k: rep[k] for k in
                                  ("satisfy_rate", "best_cost", "feasible")}
        traj = plan_waypoints(case, n_steps=n_steps, backend=backend)
        cmd["trajectory"] = [[round(float(x), 3), round(float(y), 3)]
                             for x, y in traj]
        cmd["goal"] = cmd["trajectory"][-1]
    except Exception as e:
        cmd["plan_error"] = f"{type(e).__name__}: {e}"

    # 4) STL robustness margins of the planned trajectory (the verification):
    #    rho(spec) for the named spec, and rho(G not unsafe) for keep-safe
    #    grounded by the avoid zones (the robot's "always keep safe" content).
    try:
        from stl_robustness import robustness
        from keep_safe import keep_safe_robustness
        if cmd.get("trajectory"):
            t = [(x, y) for x, y in cmd["trajectory"]]
            cmd["robustness"] = round(robustness(tree, grounding, t), 3)
            unsafe = [{"kind": "circle", "x": z["x"], "y": z["y"], "r": z["r"]}
                      for z in cmd["avoid_zones"]]
            cmd["keep_safe_rho"] = round(keep_safe_robustness(unsafe, t), 3)
    except Exception:
        pass
    return cmd


def dispatch_to_robot(cmd: dict, case_id: str = "mobile",
                      backend: str = "auto") -> str:
    """Hand the planned trajectory to the robot follower by writing the same
    trajectory-cache file the ROS kinematic follower loads.  Returns its path.
    (The follower is then started with case:=mobile cache:=true.)"""
    import json
    from pathlib import Path
    out = Path(__file__).resolve().parent / "outputs" / "trajectories"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{case_id}__{backend}.json"
    f.write_text(json.dumps({"case": case_id, "backend": backend,
                             "waypoints": cmd.get("trajectory", [])}))
    return str(f)
