"""
TeLoGraF inference wrapper.

Goal: given a case dict (from `planner.case_examples`), produce a 2-D
waypoint list `[(x, y), ...]` that we then feed into either the matplotlib
animator (`sim2d/run_demo.py`) or the ROS 2 closed-loop follower
(`sim_ros2/tb3_follower.py`).

This module is the **single boundary** between our project and TeLoGraF.
If TeLoGraF (and its venv) is installed under `code/external/TeLoGraF` we
load its model and run the flow-matching planner; otherwise we fall back
to a robustness-shaped path planner so the demo still runs end-to-end
without a heavy dependency.

Public API (kept stable so callers do not change when TeLoGraF lands):

    waypoints = plan_waypoints(case, n_steps=64, ckpt=None)
        returns a list of (x, y) tuples starting at `case.map_hint.start`
        and ending inside the last reach-zone of the case.

    plan_waypoints(case, backend="auto")
        backend = "auto"     -> TeLoGraF if importable, else "fallback"
        backend = "telograf" -> TeLoGraF, raise if not importable
        backend = "fallback" -> always the robustness-shaped planner

The fallback planner is NOT a no-op: it does avoid `avoid`-atoms and
chains through `reach`-atoms in the order they appear in the tree.
See `_robustness_path()` below.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

from .case_examples import get_case
from .telograf_adapter import case_to_graph, OP_CODE, normalise


# ---------------------------------------------------------------------------
# 1) TeLoGraF importer
# ---------------------------------------------------------------------------
# We do not import torch / torch_geometric at module load -- only when the
# user actually selects the "telograf" backend.  Keeps the matplotlib demo
# fast and lets the project run on machines without a GPU/conda env.

_TELOGRAF_ROOT = Path(__file__).resolve().parents[1] / "external" / "TeLoGraF"


def telograf_available() -> bool:
    """True iff `code/external/TeLoGraF/code/` is on disk."""
    return (_TELOGRAF_ROOT / "code" / "train_gstl_v1.py").exists()


def _find_checkpoint() -> str | None:
    """Look for a downloaded TeLoGraF checkpoint."""
    ckpts = sorted(_TELOGRAF_ROOT.glob("exps/**/models/model_last.ckpt"))
    return str(ckpts[0]) if ckpts else None


def _obstacles_to_disks(obstacles: List[dict],
                        long_rect_split: float = 2.0) -> List[Tuple[float, float, float]]:
    """Convert the planner's obstacle dict-list into a sparse list of
    `(x, y, r)` disks for TeLoGraF.  TeLoGraF only knows circular atoms,
    so:
      - circles pass through unchanged,
      - small/square rectangles become ONE bounding disk centred on the AABB,
      - long thin rectangles (e.g. walls) get split along the long axis
        into a small chain (>= 2 disks) so the planner can see *where*
        the gap is rather than a single huge disk swallowing the whole row.

    Goal: keep the augmented graph well under ~30 nodes so we stay inside
    TeLoGraF's training graph-size distribution.
    """
    out: List[Tuple[float, float, float]] = []
    for o in obstacles:
        kind = o.get("kind", "circle")
        if kind == "circle":
            out.append((o["x"], o["y"], o["r"]))
            continue
        if kind != "rect":
            raise ValueError(f"unknown obstacle kind: {kind!r}")

        w, h = o["w"], o["h"]
        long_axis = "x" if w >= h else "y"
        long_len  = max(w, h)
        short_len = min(w, h)
        # half-diagonal of the short cross-section + a small pad
        r = math.hypot(short_len / 2, short_len / 2) + 0.05

        # how many disks along the long axis?  1 for small footprints,
        # 2-3 for walls so the planner can see the gap.
        n = max(1, int(math.ceil(long_len / long_rect_split)))
        for i in range(n):
            f = (i + 0.5) / n - 0.5     # offset fraction in [-0.5, 0.5]
            if long_axis == "x":
                out.append((o["x"] + f * w, o["y"], r))
            else:
                out.append((o["x"], o["y"] + f * h, r))
    return out


def _augment_case_with_obstacles(case: dict,
                                  extra_obstacles: List[dict] | None
                                  ) -> dict:
    """Bake extra (procedural) obstacles into a copy of `case` so that
    TeLoGraF sees them as part of the STL formula.

    The original case has:
        tree     = <user STL formula>
        grounding= {prop_1: reach, prop_2: avoid, ...}

    We synthesise additional avoid atoms `_obs_k` (k = 1...N), each one
    grounded to a disk position taken from `extra_obstacles`, then wrap
    them into `globally(not(_obs_k))` clauses conjoined onto the root:

        tree' = and(<original tree>, G(!_obs_1), G(!_obs_2), ...)

    That way the encoder sees the same tree the user wrote, **plus** an
    explicit "always avoid every procedural obstacle" preamble.
    """
    if not extra_obstacles:
        return case

    disks = _obstacles_to_disks(extra_obstacles)
    if not disks:
        return case

    new_grounding = dict(case["grounding"])
    avoid_clauses = []
    for i, (x, y, r) in enumerate(disks):
        name = f"_obs_{i+1}"
        new_grounding[name] = {"kind": "avoid", "x": x, "y": y,
                                "z": 0.0, "r": r}
        avoid_clauses.append({
            "op": "globally", "interval": None,
            "children": [{
                "op": "not", "interval": None,
                "children": [{"op": "atom", "name": name}],
            }],
        })

    augmented_tree = {
        "op": "and", "interval": None,
        "children": [case["tree"]] + avoid_clauses,
    }

    new_case = dict(case)
    new_case["tree"]      = augmented_tree
    new_case["grounding"] = new_grounding
    return new_case


def _telograf_subprocess(case: dict, n_steps: int, ckpt: str | None,
                          extra_obstacles: List[dict] | None = None,
                          n_samples: int = 1,
                          return_all: bool = False,
                          return_full: bool = False):
    """Run TeLoGraF's flow-matching planner via a subprocess shell-out.

    With `n_samples > 1` the subprocess batch-samples that many flow
    trajectories in ONE model forward (faithful to TeLoGraF's `test_muls`
    best-of-N selection).  `return_all=True` returns the list of all
    sampled trajectories so the caller can pick the best by STL
    robustness; otherwise just the first one (back-compat).

    TeLoGraF's inference path lives inside `train_gstl_v1.py --fix -T <run>`
    in the upstream repo (see `run_icml2025_test.sh`).  There is no clean
    Python-level "load this checkpoint, sample this graph" entry point --
    the script does it all through argparse + a private `args.Namespace`
    matching the training config.

    Rather than duplicate hundreds of lines of training-config wiring
    here, we shell out: `python train_gstl_v1.py --env simple --encoder
    gnn --flow --fix -T <run_dir> -b 1 --num_evals 1 --export <our_json>`
    and parse the JSON it writes (see `tools/telograf_export.py` for the
    small patch we apply on first run so that the eval loop dumps a
    single (x, y) trajectory).

    NOTE: this needs (a) the TeLoGraF venv at `.venv-telograf/`, and
    (b) a downloaded checkpoint at
    `external/TeLoGraF/exps/<run_id>/models/model_last.ckpt`.  Without
    either, the caller in `plan_waypoints` falls back to the robustness
    planner.
    """
    import json
    import subprocess

    if not telograf_available():
        raise ImportError(
            f"TeLoGraF not found at {_TELOGRAF_ROOT}.  "
            "Run `bash scripts/install_telograf.sh` from the code/ dir."
        )

    ckpt_path = ckpt or _find_checkpoint()
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "No TeLoGraF checkpoint found under "
            f"{_TELOGRAF_ROOT}/exps/**/models/model_last.ckpt.\n"
            "Download one (e.g. `g0128-075243_simple_gnn_F`) from the "
            "Google Drive link in the TeLoGraF README and unzip it into "
            f"{_TELOGRAF_ROOT}/exps/."
        )

    # Procedural obstacles are NOT injected into the STL graph for
    # TeLoGraF by default.  Reason: stuffing 30+ extra avoid atoms into
    # a graph that the simple_gnn_F checkpoint was trained on (typical
    # ~10 nodes) pushes the encoder out of distribution, and the
    # diffusion samples drift off the map.  Set
    # TELOGRAF_AUGMENT_OBSTACLES=1 to opt back in for experimentation;
    # otherwise the model sees only the user's original STL formula
    # (which it is in-distribution for), and the procedural avoidance
    # is left to the controller (or the A* fallback).
    if os.environ.get("TELOGRAF_AUGMENT_OBSTACLES") == "1":
        augmented = _augment_case_with_obstacles(case, extra_obstacles)
    else:
        augmented = case

    # Full obstacle set the diffusion guidance must honour = STL avoid
    # atoms (which the GNN already sees) + procedural walls/furniture
    # (which it does NOT see, fed here so the STLCG-style refinement in
    # the subprocess pushes the trajectory out of them).  Rectangles are
    # decomposed into a sparse set of bounding circles.
    refine_obstacles = [{"x": g["x"], "y": g["y"], "r": g["r"]}
                        for g in augmented["grounding"].values()
                        if g.get("kind") == "avoid"]
    for (ox, oy, orr) in _obstacles_to_disks(extra_obstacles or []):
        refine_obstacles.append({"x": ox, "y": oy, "r": orr})

    # Encode the (now-augmented) STL graph into a small JSON that
    # tools/telograf_export.py will load on the TeLoGraF side.
    graph = case_to_graph(augmented)
    case_payload = {
        "id":         augmented["id"],
        "tree":       augmented["tree"],
        "grounding":  augmented["grounding"],
        "start":      list(augmented["map_hint"]["start"]),
        "n_steps":    n_steps,
        "n_samples":  n_samples,
        "refine_obstacles": refine_obstacles,
        "node_feats": graph.node_features,
        "edge_index": graph.edge_index,
        "ckpt":       ckpt_path,
    }

    tools_dir = _TELOGRAF_ROOT.parent.parent / "tools"
    tools_dir.mkdir(exist_ok=True)
    exporter = tools_dir / "telograf_export.py"
    if not exporter.exists():
        raise FileNotFoundError(
            f"TeLoGraF export bridge missing at {exporter}.  "
            "Create it from the template in the README (planner/README.md "
            "section 'Wiring TeLoGraF inference'); the project ships a "
            "stub that you fill in once you've downloaded a checkpoint."
        )

    venv_python = _TELOGRAF_ROOT.parent.parent / ".venv-telograf" / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(
            f"TeLoGraF venv missing at {venv_python}.  "
            "Run `bash scripts/install_telograf.sh`."
        )

    payload_str = json.dumps(case_payload)
    # Cap CPU threads for the torch subprocess.  Left uncapped, PyTorch/BLAS grab
    # EVERY core (e.g. 11/16), which thrashes against a running Gazebo GUI + RViz
    # and can blow ONE plan up from ~5 s to minutes (oversubscription + thermal
    # throttling on hybrid P/E cores).  A small cap is actually FASTER under
    # contention; override with TELOGRAF_THREADS=N.  Must be set in the child's
    # env (read at torch/BLAS import), so we inject it here.
    n_threads = os.environ.get("TELOGRAF_THREADS") or str(
        max(2, min(6, (os.cpu_count() or 4) // 2)))
    thread_env = {k: n_threads for k in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    proc = subprocess.run(
        [str(venv_python), str(exporter)],
        input=payload_str, capture_output=True, text=True,
        cwd=str(_TELOGRAF_ROOT / "code"),
        env={**os.environ, **thread_env,
             "PYTHONPATH": f"{_TELOGRAF_ROOT/'code'}:" + os.environ.get("PYTHONPATH", "")},
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"telograf_export.py exit={proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr ---\n{proc.stderr[-2000:]}"
        )

    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        raise RuntimeError(
            f"telograf_export.py did not return JSON on its last line:\n"
            f"{proc.stdout[-1000:]}"
        ) from e

    if return_full:
        return result          # {waypoints, samples, best_raw, ...}
    if return_all:
        samples = result.get("samples", [result["waypoints"]])
        return [[(float(x), float(y)) for x, y in s] for s in samples]
    return [(float(x), float(y)) for x, y in result["waypoints"]]


def _group_shortfall(traj, alts, cur) -> Tuple[float, int]:
    """Shortfall for ONE reach group = reach ANY alternative in it.

    Returns (min-shortfall over the alternatives, advanced cursor).  A singleton
    group is the usual "must reach this atom"; a multi-alternative group models a
    disjunctive goal (e.g. an `imply`/`or` that normalises to reach-N-or-reach-S),
    where touching either alternative satisfies it."""
    best_short, best_cur = 9.9, cur
    for (rx, ry, rr) in alts:
        d = min((math.hypot(traj[k][0] - rx, traj[k][1] - ry)
                 for k in range(cur, len(traj))), default=9.9)
        short = max(0.0, d - rr)
        if short < best_short:
            best_short = short
            nc = cur
            for k in range(cur, len(traj)):
                if math.hypot(traj[k][0] - rx, traj[k][1] - ry) < rr + 0.4:
                    nc = k
                    break
            best_cur = nc
    return best_short, best_cur


def _traj_reach_cost(traj, case) -> float:
    """Reach-shortfall ONLY: sum over reach GROUPS (in order) of how far the
    closest waypoint stays OUTSIDE the goal disk; a disjunctive group counts the
    best (closest) alternative.  0 means the trajectory satisfies every group.

    This is the right feasibility signal for the Tier-3 self-check: test-time
    guidance can push samples OUT of obstacles (so collision cost is fixable),
    but it cannot make the prior visit a goal region it never approaches.  So
    reach-shortfall on the raw prior predicts whether flow + guidance can
    succeed, whereas raw collision cost does not."""
    cost, cur = 0.0, 0
    for alts in _reach_groups(case):
        short, cur = _group_shortfall(traj, alts, cur)
        cost += short
    return cost


def _traj_stl_cost(traj, case, obstacles) -> float:
    """Soft cost for best-of-N selection: lower = better STL satisfaction.

    cost = sum over reach atoms of min-distance-to-atom (want to touch each)
         + collision penalty (heavy) for entering any obstacle.
    Used to rank TeLoGraF samples; a cost ~0 means the sample reaches every
    goal and stays clear of every obstacle.
    """
    all_obs = list(_avoid_obstacles(case)) + list(obstacles or [])
    cost = 0.0
    cur = 0
    for alts in _reach_groups(case):      # disjunctive group -> best alternative
        short, cur = _group_shortfall(traj, alts, cur)
        cost += short                     # 0 if a waypoint enters the group
    for (x, y) in traj:
        if _obstacle_blocked(x, y, all_obs, 0.10):
            cost += 5.0                   # heavy per-waypoint collision penalty
    return cost


# ---------------------------------------------------------------------------
# Tier-1 contribution: nested STL tree decomposition (NO retraining)
# ---------------------------------------------------------------------------
# The frozen simple_gnn_F checkpoint is in-distribution for a SINGLE timed
# reach  F[t1,t2](goal)  (its strongest template, ~24/24 clean) but goes OOD
# on >=3 nested sequential reaches like  F(A & F(B & F(C))).  Rather than
# retrain, we exploit the *tree structure*: split a sequential-visit spec into
# per-leg timed-reach sub-specs, sample each leg with the frozen flow-matching
# model (each leg IS in-distribution), and stitch the legs.  This recovers
# genuine flow-matching trajectories for specs the model cannot satisfy
# jointly -- a test-time, training-free use of the nested STL tree.
#
# We ONLY decompose pure sequential-visit patterns (nested finally/and over
# reach atoms).  Connectives whose meaning is NOT "do these in order"
# (imply / until / iff / or) are left to the joint encoder.  Disable entirely
# with TELOGRAF_DECOMPOSE=0.

def _tree_ops(node, acc=None) -> set:
    """Set of every operator string appearing in the STL tree."""
    acc = acc if acc is not None else set()
    if isinstance(node, dict):
        if node.get("op"):
            acc.add(node["op"])
        for c in node.get("children", []) or []:
            _tree_ops(c, acc)
    return acc


def _should_decompose(case: dict) -> bool:
    """True iff the case is a pure sequential-visit spec with >=2 reaches."""
    if os.environ.get("TELOGRAF_DECOMPOSE") == "0":
        return False
    if len(_seq_reach_targets(case)) < 2:
        return False
    # Only "visit A, then B, ..." patterns; reject non-sequential semantics.
    return not (_tree_ops(case["tree"]) & {"imply", "until", "iff", "or"})


def _build_reach_subcase(case: dict, goal: Tuple[float, float, float],
                         start: Tuple[float, float], avoids: List[dict],
                         seg_idx: int,
                         window: Tuple[int, int] = (20, 55)) -> dict:
    """One in-distribution timed-reach leg:  F[window](reach goal) [& G !o ...].

    Built in the exact shape of the working `reach_avoid` template so the
    frozen GNN encoder stays in-distribution.  `start` becomes the leg's ego
    state; STL avoid atoms (if any) are carried so the per-leg guidance keeps
    honouring them."""
    gx, gy, gr = goal
    grounding = {"prop_g": {"kind": "reach", "x": gx, "y": gy, "z": 0.0, "r": gr}}
    children = [{"op": "finally", "interval": list(window),
                 "children": [{"op": "atom", "name": "prop_g"}]}]
    for j, o in enumerate(avoids):
        nm = f"prop_o{j + 1}"
        grounding[nm] = {"kind": "avoid", "x": o["x"], "y": o["y"],
                         "z": o.get("z", 0.0), "r": o["r"]}
        children.append({"op": "globally", "interval": None, "children": [
            {"op": "not", "interval": None, "children": [
                {"op": "atom", "name": nm}]}]})
    tree = children[0] if len(children) == 1 else \
        {"op": "and", "interval": None, "children": children}
    return {
        "id":        f"{case['id']}__seg{seg_idx}",
        "role":      "telograf",
        "tree":      tree,
        "grounding": grounding,
        "map_hint":  {**case["map_hint"], "start": [start[0], start[1]]},
    }


def _telograf_plan_decomposed(case: dict, n_steps: int, ckpt: str | None,
                              extra_obstacles: List[dict] | None
                              ) -> List[Tuple[float, float]]:
    """Plan a sequential-visit spec leg-by-leg with the frozen flow model."""
    reaches = _seq_reach_targets(case)
    avoids = [{"x": g["x"], "y": g["y"], "z": g.get("z", 0.0), "r": g["r"]}
              for g in case["grounding"].values() if g.get("kind") == "avoid"]
    n_samples = int(case.get("map_hint", {}).get("telograf_samples", 1))
    seg_steps = max(8, n_steps // len(reaches))

    print(f"[telograf] DECOMPOSE nested STL into {len(reaches)} timed-reach "
          f"legs; each sampled by flow matching, then stitched (no retrain)",
          file=sys.stderr)

    stitched: List[Tuple[float, float]] = []
    cur: Tuple[float, float] = tuple(case["map_hint"]["start"])
    for i, (rx, ry, rr) in enumerate(reaches):
        sub = _build_reach_subcase(case, (rx, ry, rr), cur, avoids, i)
        seg = _telograf_subprocess(sub, seg_steps, ckpt,
                                   extra_obstacles=extra_obstacles,
                                   n_samples=n_samples, return_all=False)
        seg = [(float(x), float(y)) for x, y in seg]
        stitched.extend(seg if i == 0 else seg[1:])
        cur = (rx, ry)        # next leg departs from this goal

    if os.environ.get("TELOGRAF_NO_REPAIR") == "1":
        return stitched

    ok, why = _trajectory_validates(stitched, case, extra_obstacles or [])
    if ok:
        print(f"[telograf] decomposed stitch ({len(reaches)} flow legs) PASSES "
              f"full STL+collision -- GENUINE flow matching, no A*",
              file=sys.stderr)
        return stitched

    print(f"[telograf] decomposed stitch fails ({why}) -> A* repair between "
          f"reach anchors (TeLoGraF shape kept where valid)", file=sys.stderr)
    repaired = _repair_trajectory(stitched, case, extra_obstacles or [], n_steps)
    ok2, why2 = _trajectory_validates(repaired, case, extra_obstacles or [])
    if not ok2:
        print(f"[telograf] WARN: decomposed repair still imperfect ({why2}); "
              f"falling back to pure A*", file=sys.stderr)
        return _robustness_path(case, n_steps, extra_obstacles=extra_obstacles)
    return repaired


# ---------------------------------------------------------------------------
# Tier-3 contribution: best-of-N feasibility / OOD self-check
# ---------------------------------------------------------------------------
# TeLoGraF samples a BATCH of N candidate trajectories (its learned prior) in
# one forward pass.  The SPREAD of how well those raw samples satisfy the STL
# is a free, calibrated signal of whether the spec is in- or out-of-
# distribution for the frozen model: if many samples already satisfy it, the
# model is "confident"; if even the best raw sample is far off, the model is
# OOD/uncertain and we should defer to the classical A* backend.  This turns
# the flow-vs-A* choice from a hard-coded per-case `role` into a measured,
# reportable decision (and a poster figure).

# PER-REACH-ATOM shortfall tolerance (metres).  If the best prior sample
# misses the goal disks by more than this PER ATOM, the test-time guidance
# (whose goal-pull rescues ~1-2 m/atom -- measured) cannot close the gap, so
# the spec is out-of-distribution and we defer to A*.  Empirically separates
# in-distribution cases (best <=1.8 m/atom) from a nested 3-reach spec fed
# jointly to the GNN (~4.4 m/atom).  See tools/feasibility_report.py.
SAT_COST_TOL = 2.5


def _samples_feasibility(samples, case: dict,
                         obstacles: List[dict]) -> dict:
    """Per-sample reach-shortfall statistics over the best-of-N prior.

    Feasibility is decided on PER-ATOM reach-shortfall (what guidance CANNOT
    fix), not on raw collision cost (which it can) -- see _traj_reach_cost."""
    pts = [[(float(x), float(y)) for x, y in s] for s in samples]
    n_reach = max(1, len(_seq_reach_targets(case)))
    costs = sorted(_traj_reach_cost(s, case) / n_reach for s in pts)  # per-atom
    n = len(costs)
    if n == 0:
        return {"n_samples": 0, "best_cost": float("inf"),
                "mean_cost": float("inf"), "satisfy_rate": 0.0,
                "feasible": False, "costs": [], "n_reach": n_reach}
    sat = sum(1 for c in costs if c <= SAT_COST_TOL)
    return {
        "n_samples":    n,
        "n_reach":      n_reach,
        "best_cost":    costs[0],          # per-atom reach-shortfall, best sample
        "mean_cost":    sum(costs) / n,
        "satisfy_rate": sat / n,           # fraction of prior within guidance range
        "feasible":     costs[0] <= SAT_COST_TOL,
        "costs":        costs,
    }


def telograf_feasibility(case: dict | str, n_steps: int = 64,
                         ckpt: str | None = None,
                         extra_obstacles: List[dict] | None = None,
                         n_samples: int | None = None) -> dict:
    """Public Tier-3 entry: batch-sample the frozen flow prior and report how
    feasible / in-distribution the STL spec is for it.  Used by
    tools/feasibility_report.py to produce the self-check figure."""
    if isinstance(case, str):
        case = get_case(case)
    ns = n_samples or int(case.get("map_hint", {}).get("telograf_samples", 16))
    samples = _telograf_subprocess(case, n_steps, ckpt,
                                   extra_obstacles=extra_obstacles,
                                   n_samples=ns, return_all=True)
    report = _samples_feasibility(samples, case, extra_obstacles or [])
    report["case"] = case["id"]
    return report


def _telograf_plan(case: dict, n_steps: int, ckpt: str | None,
                    extra_obstacles: List[dict] | None = None,
                    ) -> List[Tuple[float, float]]:
    """Run TeLoGraF's flow-matching planner + STLCG-style guidance.

    Pipeline (faithful to how TeLoGraF is used in the paper):
      0. If the spec is a nested sequential-visit pattern, DECOMPOSE it into
         per-leg timed reaches and plan each with flow matching (Tier-1).
      1. Batch-sample `n_samples` flow trajectories (TeLoGraF `test_muls`)
         conditioned on the STL graph -- the learned trajectory *prior*.
      2. Inside the subprocess: pick the best by STL cost, then run
         STLCG-style differentiable refinement against the FULL fed
         obstacle set (STL avoid atoms + procedural walls/furniture) so
         the diffusion output is pushed out of every obstacle and onto
         the goal (TeLoGraF's CTG/LTLDoG test-time guidance, applied to
         the 2-D waypoints).
      3. Validate.  If it passes -> GENUINE TeLoGraF (+guidance), no A*.
         Only if guidance still can't satisfy it do we A*-repair / fall
         back to pure A* (clearly logged).

    `n_samples` from `map_hint["telograf_samples"]` (default 1).
    `TELOGRAF_NO_REPAIR=1` skips the A* safety net (inspect raw guidance).
    """
    if _should_decompose(case):
        return _telograf_plan_decomposed(case, n_steps, ckpt, extra_obstacles)

    n_samples = int(case.get("map_hint", {}).get("telograf_samples", 1))

    # ONE subprocess call returns BOTH the refined best-of-N (waypoints) and
    # the raw prior samples -- so the Tier-3 self-check is free.
    result = _telograf_subprocess(case, n_steps, ckpt,
                                  extra_obstacles=extra_obstacles,
                                  n_samples=n_samples, return_full=True)
    guided = [(float(x), float(y)) for x, y in result["waypoints"]]

    # Tier-3: feasibility / OOD self-check over the best-of-N prior samples.
    report = _samples_feasibility(result.get("samples", [result["waypoints"]]),
                                  case, extra_obstacles or [])
    print(f"[telograf] self-check: best-of-{report['n_samples']} prior "
          f"satisfy-rate={report['satisfy_rate']:.0%}, "
          f"best_cost={report['best_cost']:.2f} -> "
          f"{'FEASIBLE (flow)' if report['feasible'] else 'OOD/uncertain'}",
          file=sys.stderr)
    if os.environ.get("TELOGRAF_SELFCHECK") == "1" and not report["feasible"]:
        print("[telograf] self-check: prior is OOD -> routing to A* by "
              "measured decision (not try-then-fail)", file=sys.stderr)
        return _robustness_path(case, n_steps, extra_obstacles=extra_obstacles)

    if os.environ.get("TELOGRAF_NO_REPAIR") == "1":
        return guided

    ok, why = _trajectory_validates(guided, case, extra_obstacles or [])
    if ok:
        print(f"[telograf] best-of-{n_samples} diffusion + STLCG guidance "
              f"PASSES STL+collision check (pure TeLoGraF, no A*)",
              file=sys.stderr)
        return guided

    print(f"[telograf] diffusion+guidance still fails: {why} -> "
          f"A* repair keeping TeLoGraF shape between anchors", file=sys.stderr)
    repaired = _repair_trajectory(guided, case, extra_obstacles or [], n_steps)

    ok2, why2 = _trajectory_validates(repaired, case, extra_obstacles or [])
    if not ok2:
        print(f"[telograf] WARN: repaired still imperfect: {why2}; "
              f"falling back to pure A*", file=sys.stderr)
        return _robustness_path(case, n_steps, extra_obstacles=extra_obstacles)
    return repaired


# ---------------------------------------------------------------------------
# 2) Fallback robustness-shaped planner
# ---------------------------------------------------------------------------
# Without TeLoGraF we still need *some* trajectory to drive the simulator.
# The goal is to (a) hit every reach-atom in tree order, (b) detour around
# every avoid-atom, (c) stay inside the world_bounds.  This is enough to
# verify the rest of the pipeline (ROS, controller, RViz).

def _seq_reach_targets(case: dict) -> List[Tuple[float, float, float]]:
    """Walk the tree left-to-right and return all `reach` ground-zones."""
    reaches: List[Tuple[float, float, float]] = []

    def walk(node):
        if not isinstance(node, dict):
            return
        if node.get("op") == "atom":
            g = case["grounding"][node["name"]]
            if g.get("kind", "reach") == "reach":
                reaches.append((g["x"], g["y"], g["r"]))
            return
        for child in node.get("children", []) or []:
            walk(child)

    walk(case["tree"])
    return reaches


def _reach_groups(case: dict) -> List[List[Tuple[float, float, float]]]:
    """Reach targets as ordered GROUPS, after normalising imply/iff to the basis.

    Each group is a list of (x,y,r) ALTERNATIVES: a singleton group must be
    reached (conjunctive/sequential reach), while a multi-alternative group is a
    DISJUNCTIVE goal (reach any one) -- the shape an `or`, or an `imply`/`iff`
    that normalises to `or`, induces.  This is what lets operator normalization
    feed a `reach-N-or-reach-S` spec to the frozen planner and have the
    feasibility self-check score it as flow-feasible (reach either) rather than
    wrongly demanding both."""
    tree = normalise(case["tree"])
    groups: List[List[Tuple[float, float, float]]] = []

    def reaches_under(node, acc):
        if not isinstance(node, dict):
            return
        if node.get("op") == "atom":
            g = case["grounding"][node["name"]]
            if g.get("kind", "reach") == "reach":
                acc.append((g["x"], g["y"], g["r"]))
            return
        for c in node.get("children", []) or []:
            reaches_under(c, acc)

    def walk(node):
        if not isinstance(node, dict):
            return
        op = node.get("op")
        if op == "or":                         # disjunctive goal: reach any
            alts: List[Tuple[float, float, float]] = []
            reaches_under(node, alts)
            if alts:
                groups.append(alts)
            return                             # do not split the OR further
        if op == "atom":
            g = case["grounding"][node["name"]]
            if g.get("kind", "reach") == "reach":
                groups.append([(g["x"], g["y"], g["r"])])
            return
        for c in node.get("children", []) or []:
            walk(c)

    walk(tree)
    return groups


def _avoid_obstacles(case: dict) -> List[dict]:
    """Obstacles that come from the case's STL `grounding` (avoid-atoms).

    Returned in the same dict format as the external `obstacles=...`
    parameter to plan_waypoints, so the two streams merge cleanly:

        {"kind": "circle", "x": <cx>, "y": <cy>, "r": <r>}
    """
    return [{"kind": "circle", "x": g["x"], "y": g["y"], "r": g["r"]}
            for g in case["grounding"].values()
            if g.get("kind") == "avoid"]


def _obstacle_blocked(x: float, y: float, obstacles: List[dict],
                       clearance: float) -> bool:
    """Is the point (x, y) inside any obstacle, inflated by `clearance`?

    obstacles: list of dicts, each one of
        {"kind": "circle", "x": cx, "y": cy, "r": r}
        {"kind": "rect",   "x": cx, "y": cy, "w": w,  "h": h}     # centred AABB
    """
    for o in obstacles:
        kind = o.get("kind", "circle")
        if kind == "circle":
            if math.hypot(x - o["x"], y - o["y"]) < o["r"] + clearance:
                return True
        elif kind == "rect":
            hx, hy = o["w"] / 2, o["h"] / 2
            # nearest point on AABB to (x, y)
            cx = max(o["x"] - hx, min(x, o["x"] + hx))
            cy = max(o["y"] - hy, min(y, o["y"] + hy))
            if math.hypot(x - cx, y - cy) < clearance:
                return True
        else:
            raise ValueError(f"unknown obstacle kind: {kind!r}")
    return False


def _astar(start: Tuple[float, float], goal: Tuple[float, float],
           obstacles: List[dict], bounds: Tuple[float, float, float, float],
           *, resolution: float = 0.15, clearance: float = 0.30
           ) -> List[Tuple[float, float]]:
    """8-connected A* on a uniform grid.

    Falls back to a straight line if no path is found (e.g. goal landed
    inside an inflated obstacle); caller can decide what to do.
    """
    import heapq

    xmin, xmax, ymin, ymax = bounds
    nx = int(math.ceil((xmax - xmin) / resolution)) + 1
    ny = int(math.ceil((ymax - ymin) / resolution)) + 1

    def in_grid(i, j):  return 0 <= i < nx and 0 <= j < ny
    def to_world(i, j): return (xmin + i * resolution, ymin + j * resolution)
    def to_grid(p):
        i = max(0, min(nx - 1, round((p[0] - xmin) / resolution)))
        j = max(0, min(ny - 1, round((p[1] - ymin) / resolution)))
        return (i, j)

    blocked = [[False] * ny for _ in range(nx)]
    for i in range(nx):
        for j in range(ny):
            wx, wy = to_world(i, j)
            blocked[i][j] = _obstacle_blocked(wx, wy, obstacles, clearance)

    # If start or goal landed in an inflated obstacle, sweep outwards a
    # few cells looking for a free one.  Real robot would not start
    # inside a wall, but planner failures shouldn't crash the demo.
    def _nudge_to_free(g):
        i0, j0 = g
        if not blocked[i0][j0]:
            return (i0, j0)
        for radius in range(1, 8):
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if max(abs(di), abs(dj)) != radius:
                        continue
                    i, j = i0 + di, j0 + dj
                    if in_grid(i, j) and not blocked[i][j]:
                        return (i, j)
        return (i0, j0)  # give up

    s = _nudge_to_free(to_grid(start))
    g = _nudge_to_free(to_grid(goal))

    NEIGH = [(1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (-1, 1), (1, -1), (-1, -1)]

    came_from: dict = {}
    g_score = {s: 0.0}
    heap: list = [(0.0, s)]
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == g:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            world_path = [start] + [to_world(*p) for p in path[1:-1]] + [goal]
            return world_path
        ci, cj = cur
        for di, dj in NEIGH:
            ni, nj = ci + di, cj + dj
            if not in_grid(ni, nj) or blocked[ni][nj]:
                continue
            step = math.hypot(di, dj) * resolution
            tg = g_score[cur] + step
            if tg < g_score.get((ni, nj), float("inf")):
                came_from[(ni, nj)] = cur
                g_score[(ni, nj)] = tg
                # straight-line heuristic
                h = math.hypot(ni - g[0], nj - g[1]) * resolution
                heapq.heappush(heap, (tg + h, (ni, nj)))

    return [start, goal]   # no path -> straight line as last resort


def _pin_reaches(traj: List[Tuple[float, float]],
                 case: dict) -> List[Tuple[float, float]]:
    """Snap the trajectory so it genuinely passes through each reach-atom
    centre (in tree order).  STL `reach` means "get inside the goal disk";
    TeLoGraF gets us the obstacle-aware *shape* and lands near each goal,
    so we nudge the single closest waypoint onto the goal centre.  This is
    the trajectory analogue of pinning the start, and makes the followed
    path actually satisfy `reach` rather than grazing the disk edge.
    """
    if not traj:
        return traj
    reaches = _seq_reach_targets(case)
    if not reaches:
        return traj
    traj = [tuple(p) for p in traj]
    cur = 0
    last_reach_idx = None
    for (rx, ry, rr) in reaches:
        # closest waypoint to this goal, searching forward (keep order)
        best_i, best_d = cur, float("inf")
        for k in range(cur, len(traj)):
            d = math.hypot(traj[k][0] - rx, traj[k][1] - ry)
            if d < best_d:
                best_d, best_i = d, k
        # Only snap if the model already brought us reasonably close
        # (otherwise this would teleport through obstacles).
        if best_d <= rr + 0.6:
            traj[best_i] = (rx, ry)
            cur = best_i
            last_reach_idx = best_i
    # The mission ends once the FINAL reach atom is reached -- a timed
    # `F[t1,t2]` lets the diffusion wander after the goal, which looks odd
    # (robot drives past the goal).  Truncate there so the followed path
    # ends ON the goal.
    if last_reach_idx is not None and last_reach_idx >= 1:
        traj = traj[: last_reach_idx + 1]
    return traj


def _interp(pts: List[Tuple[float, float]], n_steps: int) -> List[Tuple[float, float]]:
    if len(pts) < 2:
        return pts
    seglens = [math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(pts, pts[1:])]
    total = sum(seglens) or 1e-6
    out: List[Tuple[float, float]] = [pts[0]]
    for (a, b), L in zip(zip(pts, pts[1:]), seglens):
        k = max(1, round(n_steps * L / total))
        for i in range(1, k + 1):
            t = i / k
            out.append((a[0] + t * (b[0] - a[0]),
                        a[1] + t * (b[1] - a[1])))
    return out


# ---------------------------------------------------------------------------
# 2.5) Trajectory validator + repair
# ---------------------------------------------------------------------------
# TeLoGraF's diffusion samples can sometimes (a) not actually pass through
# every reach-atom in the STL, (b) cut through a procedural wall the model
# never saw, or (c) overshoot the time bound of an `F[a,b]` reach.  The
# repair pass guarantees the returned trajectory satisfies all three.
#
# Strategy:
#   1. Pull the STL's "must-visit" reach atoms in tree order ->  anchors
#   2. For each consecutive (anchor_i, anchor_{i+1}) pair, take TeLoGraF's
#      sub-trajectory between the indices closest to those anchors.
#   3. Project every waypoint of that sub-trajectory out of any obstacle
#      (avoid-atom or procedural).  Where the projected line still
#      collides, replace that sub-trajectory with an A* path.
#   4. Concatenate, re-interpolate to `n_steps`, return.

def _trajectory_validates(traj: List[Tuple[float, float]],
                          case: dict,
                          obstacles: List[dict],
                          reach_tol: float = 0.4,
                          collision_clearance: float = 0.15) -> Tuple[bool, str]:
    """Quick STL-style check.  Returns (ok, reason_if_not)."""
    if not traj:
        return False, "empty trajectory"

    # 1) start
    start = case["map_hint"]["start"]
    d0 = math.hypot(traj[0][0] - start[0], traj[0][1] - start[1])
    if d0 > 0.5:
        return False, f"start mismatch: {d0:.2f} m away"

    # 2) every reach atom visited in order
    reaches = _seq_reach_targets(case)
    cur = 0
    for (rx, ry, rr) in reaches:
        hit_idx = None
        for k in range(cur, len(traj)):
            if math.hypot(traj[k][0] - rx, traj[k][1] - ry) < rr + reach_tol:
                hit_idx = k
                break
        if hit_idx is None:
            return False, f"never visited reach atom ({rx:.2f}, {ry:.2f})"
        cur = hit_idx

    # 3) no waypoint inside any obstacle (avoid + procedural)
    all_obstacles = list(_avoid_obstacles(case)) + list(obstacles or [])
    for (x, y) in traj:
        if _obstacle_blocked(x, y, all_obstacles, collision_clearance):
            return False, f"collides at ({x:.2f}, {y:.2f})"

    return True, "ok"


def _nearest_traj_index(traj: List[Tuple[float, float]],
                         pt: Tuple[float, float],
                         start_from: int = 0) -> int:
    """Index of the closest waypoint to `pt`, searching from start_from."""
    best_i, best_d = start_from, float("inf")
    for k in range(start_from, len(traj)):
        d = math.hypot(traj[k][0] - pt[0], traj[k][1] - pt[1])
        if d < best_d:
            best_d, best_i = d, k
    return best_i


def _repair_trajectory(traj: List[Tuple[float, float]],
                       case: dict,
                       obstacles: List[dict],
                       n_steps: int) -> List[Tuple[float, float]]:
    """Force-satisfy the case's STL constraints by stitching TeLoGraF's
    shape with A*-routed segments around obstacles.  Keeps the TeLoGraF
    "look" wherever it already was valid."""
    start = tuple(case["map_hint"]["start"])
    reaches = _seq_reach_targets(case)
    all_obstacles = list(_avoid_obstacles(case)) + list(obstacles or [])
    bounds = tuple(case["map_hint"].get("world_bounds", (-4, 4, -4, 4)))

    # Build anchor list: start -> r_1 -> r_2 -> ... -> r_K
    anchors: List[Tuple[float, float]] = [start]
    for (rx, ry, _r) in reaches:
        anchors.append((rx, ry))

    if len(anchors) < 2:
        return traj                                     # nothing to repair

    # Map each anchor to the closest index in TeLoGraF's trajectory,
    # monotonically increasing.
    anchor_idx: List[int] = [0]
    last_idx = 0
    for a in anchors[1:]:
        j = _nearest_traj_index(traj, a, start_from=last_idx)
        anchor_idx.append(j)
        last_idx = j
    anchor_idx[-1] = len(traj) - 1                     # last anchor = last waypoint

    repaired: List[Tuple[float, float]] = []
    for k in range(len(anchors) - 1):
        a0, a1 = anchors[k], anchors[k + 1]
        i0, i1 = anchor_idx[k], anchor_idx[k + 1]
        if i1 <= i0:
            i1 = min(len(traj) - 1, i0 + 1)
        segment = list(traj[i0 : i1 + 1])

        # Force segment endpoints to land exactly on the anchors.
        segment[0]  = a0
        segment[-1] = a1

        # Check whether the segment is collision-free under the *full*
        # obstacle set.  If not, replace it with an A* path.
        seg_ok = all(
            not _obstacle_blocked(x, y, all_obstacles, 0.15)
            for (x, y) in segment
        )
        if not seg_ok or len(segment) < 2:
            segment = _astar(a0, a1, all_obstacles, bounds,
                             clearance=0.25)

        # Push the TeLoGraF-shaped waypoints out of any near-miss obstacle.
        segment = [_push_out(p, all_obstacles, clearance=0.15) for p in segment]

        if k == 0:
            repaired.extend(segment)
        else:
            repaired.extend(segment[1:])               # avoid duplicate anchor

    return _interp(repaired, n_steps=n_steps)


def _push_out(p: Tuple[float, float],
              obstacles: List[dict],
              clearance: float) -> Tuple[float, float]:
    """If p is inside an inflated obstacle, push it out radially to the
    nearest free space.  Single-step; enough for the small overlaps the
    diffusion sample tends to produce."""
    px, py = p
    for o in obstacles:
        kind = o.get("kind", "circle")
        if kind == "circle":
            dx, dy = px - o["x"], py - o["y"]
            d = math.hypot(dx, dy) or 1e-6
            need = o["r"] + clearance
            if d < need:
                k = need / d
                return (o["x"] + dx * k, o["y"] + dy * k)
        elif kind == "rect":
            hx, hy = o["w"] / 2, o["h"] / 2
            cx = max(o["x"] - hx, min(px, o["x"] + hx))
            cy = max(o["y"] - hy, min(py, o["y"] + hy))
            dx, dy = px - cx, py - cy
            d = math.hypot(dx, dy)
            if d < clearance:
                # Inside the rect or within clearance; push to nearest face.
                if d < 1e-6:
                    # truly inside -- pick whichever face is closest
                    rights = [
                        (o["x"] - hx - clearance, py, abs(px - (o["x"] - hx))),
                        (o["x"] + hx + clearance, py, abs(px - (o["x"] + hx))),
                        (px, o["y"] - hy - clearance, abs(py - (o["y"] - hy))),
                        (px, o["y"] + hy + clearance, abs(py - (o["y"] + hy))),
                    ]
                    rights.sort(key=lambda r: r[2])
                    return (rights[0][0], rights[0][1])
                k = clearance / d
                return (cx + dx * k, cy + dy * k)
    return p


def _robustness_path(case: dict, n_steps: int,
                     extra_obstacles: List[dict] | None = None
                     ) -> List[Tuple[float, float]]:
    start = tuple(case["map_hint"]["start"])
    reaches = _seq_reach_targets(case)
    if not reaches:
        return [start]

    # All obstacles the planner is allowed to see, in dict form.
    obstacles = list(_avoid_obstacles(case))
    if extra_obstacles:
        obstacles.extend(extra_obstacles)

    bounds = tuple(case["map_hint"].get("world_bounds", (-4, 4, -4, 4)))

    waypts: List[Tuple[float, float]] = [start]
    cur = start
    for rx, ry, _ in reaches:
        seg = _astar(cur, (rx, ry), obstacles, bounds)
        waypts.extend(seg[1:])    # don't duplicate cur
        cur = (rx, ry)
    return _interp(waypts, n_steps=n_steps)


# ---------------------------------------------------------------------------
# 3) Public entry point
# ---------------------------------------------------------------------------
def plan_waypoints(case: dict | str,
                   n_steps: int = 64,
                   ckpt: str | None = None,
                   backend: str = "auto",
                   obstacles: List[dict] | None = None,
                   ) -> List[Tuple[float, float]]:
    """Plan a 2D trajectory for `case` using either TeLoGraF or the
    robustness-shaped fallback.

    Args:
        case:        case dict or case_id string
        n_steps:     number of waypoints to return
        ckpt:        path to a TeLoGraF checkpoint, or None for auto-pick
        backend:     "auto" | "telograf" | "fallback"
        obstacles:   extra obstacles the planner did NOT see through the
                     case's `grounding` -- typically the procedurally
                     placed walls / furniture / random disks the 2D
                     simulator generates.  Each item is a dict:
                         {"kind":"circle", "x":..., "y":..., "r":...}
                         {"kind":"rect",   "x":..., "y":..., "w":..., "h":...}
                     (centred AABB for "rect").  Only the robustness
                     fallback honours this argument; TeLoGraF reads
                     obstacles off the encoded STL graph.
    """
    if isinstance(case, str):
        case = get_case(case)

    if backend not in ("auto", "telograf", "fallback"):
        raise ValueError(f"unknown backend: {backend!r}")

    if backend == "fallback":
        traj = _robustness_path(case, n_steps, extra_obstacles=obstacles)
    elif backend == "telograf":
        traj = _telograf_plan(case, n_steps, ckpt, extra_obstacles=obstacles)
    else:  # auto
        traj = None
        if telograf_available():
            try:
                traj = _telograf_plan(case, n_steps, ckpt, extra_obstacles=obstacles)
            except (ImportError, FileNotFoundError) as e:
                print(f"[plan_waypoints] TeLoGraF unavailable ({e.__class__.__name__}): "
                      f"{e}\n               -> falling back to robustness path planner.",
                      file=sys.stderr)
        if traj is None:
            traj = _robustness_path(case, n_steps, extra_obstacles=obstacles)

    # STL `reach` means "enter the goal disk".  Snap the trajectory so it
    # genuinely passes through each reach-atom centre (no-op for the A*
    # path, which already ends on the goal).  Keeps TeLoGraF's avoidance
    # shape while guaranteeing the reach predicate is met.
    result = _pin_reaches(traj, case)

    # Temporal gate (Route B).  When the case carries a real-seconds reach
    # deadline, anchor the plan to wall-clock time (arclength / nominal speed)
    # and certify reach-by-deadline with the EXACT, sound monitor.  If the flow
    # plan is too slow to meet it, prefer the shorter A* path that does -- so the
    # planner is gated on the timed spec, not only the spatial one.
    mh = case.get("map_hint", {})
    if mh.get("reach_deadline_s") and backend != "fallback":
        from stl_runtime import plan_reach_by_deadline
        targets = _seq_reach_targets(case)
        if targets:
            goal = targets[-1]
            v_nom = float(mh.get("nominal_speed", 0.18))
            deadline = float(mh["reach_deadline_s"])
            rho_t, est = plan_reach_by_deadline(result, goal, deadline, v_nom)
            if rho_t <= 0.0:
                alt = _pin_reaches(
                    _robustness_path(case, n_steps, extra_obstacles=obstacles), case)
                rho_a, est_a = plan_reach_by_deadline(alt, goal, deadline, v_nom)
                print(f"[plan] temporal gate: flow ~{est:.0f}s (rho={rho_t:+.2f}) "
                      f"misses {deadline:.0f}s deadline; A* ~{est_a:.0f}s "
                      f"(rho={rho_a:+.2f})", file=sys.stderr)
                if rho_a > rho_t:
                    result = alt
    return result


__all__ = ["plan_waypoints", "telograf_available", "telograf_feasibility"]


if __name__ == "__main__":
    import argparse
    import json
    from .case_examples import CASES

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=CASES[0]["id"])
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "telograf", "fallback"])
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    wp = plan_waypoints(args.case, args.n_steps, args.ckpt, backend=args.backend)
    print(f"# case={args.case}  backend={args.backend}  "
          f"telograf_available={telograf_available()}  n_waypoints={len(wp)}")
    print(json.dumps(wp[:6] + ["..."] + wp[-3:], indent=2, default=str))
