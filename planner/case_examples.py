"""
Fixed case-study examples for end-to-end NL -> Tree -> TeLoGraF -> trajectory.

Each case bundles:
  - `nl`     : the natural-language utterance the user would type
  - `tree`   : the STL tree JSON our parser is expected to produce
  - `grounding` : maps every abstract atom `prop_i` to a TeLoGraF-style
                  geometric primitive  {kind, x, y, z, r}
                  where `kind` is "reach" (default) or "avoid".
  - `map_hint` : suggested map layout for the 2D / Gazebo simulator
                  (positions + obstacles + start pose).

The grounding is fixed per case (not learned) so the experiments are
reproducible regardless of LLM stochasticity. To run a different scene
you only edit the (x, y, r) fields here.

TeLoGraF supports two atomic semantics only: REACH and AVOID
(arXiv:2505.00562 sec.3.2), so each grounded predicate must be one of these.
"""

# Five cases that span the four STL templates TeLoGraF identifies.
# ---------------------------------------------------------------------------
# Case design after empirically profiling the TeLoGraF simple_gnn_F checkpoint
# (see code/README.md §7C "what TeLoGraF can/can't do"):
#
#   TeLoGraF (bare flow, no test-time guidance) is STRONG at:
#     * timed reach  F[t1,t2](goal)              -> 24/24 best-of-N clean
#     * reach + avoid where the obstacle is OFF the direct start->goal line
#   TeLoGraF is WEAK at:
#     * an obstacle dead-centre on the straight line (clearance ~0.9 < r)
#     * 3+ sequential reaches  (out of the 4 training STL templates)
#     * dense procedural clutter the STL never mentions (walls/furniture)
#
# So each case carries a `role`:
#   "telograf"  -> in-distribution; best-of-N raw TeLoGraF is expected to
#                  satisfy the STL with NO A* repair.  Scene = only the STL
#                  atoms (style "telograf_native").
#   "fallback"  -> deliberately hard (dense clutter / non-sequential logic).
#                  TeLoGraF is expected to FAIL; the A* fallback produces the
#                  valid trajectory.  The honest "diffusion-failure" demo.
#
# Tier-1 note: a NESTED SEQUENTIAL reach (visit A, then B, then C) used to be a
# fallback case (3 reaches > the 4 training templates -> OOD when fed jointly).
# planner.telograf_infer now DECOMPOSES such trees into per-leg timed reaches,
# samples each leg with the frozen flow model, and stitches -- recovering
# genuine flow matching with no retraining (set TELOGRAF_DECOMPOSE=0 to see the
# joint-encoding failure).  So seq_reach_ABC is now role "telograf".
#
# `telograf_samples` = best-of-N batch size for the telograf backend.
# ---------------------------------------------------------------------------
CASES = [
    # ========================== TeLoGraF-primary ==========================

    # Helper note: every telograf case feeds ALL its avoid atoms to the
    # diffusion planner.  The GNN gives the trajectory shape; an STLCG-style
    # differentiable guidance (test-time, in tools/telograf_export.py) pushes
    # the trajectory out of every obstacle and onto the goal.  The NL stays
    # generic ("keep safe" / "avoid all obstacles"); the grounding lists the
    # actual obstacle disks.

    # ---------- case 1: timed reach + keep safe (2 obstacles) ----------
    {
        "id":   "reach_within_T",
        "role": "telograf",
        "nl":   "Reach the goal within 70 seconds while always keeping safe from every obstacle.",
        "tree": {
            "op": "and", "interval": None, "children": [
                {"op": "finally", "interval": [0, 12], "children": [
                    {"op": "atom", "name": "prop_g"},
                ]},
                {"op": "globally", "interval": None, "children": [
                    {"op": "not", "interval": None, "children": [
                        {"op": "atom", "name": "prop_o1"}]}]},
                {"op": "globally", "interval": None, "children": [
                    {"op": "not", "interval": None, "children": [
                        {"op": "atom", "name": "prop_o2"}]}]},
            ],
        },
        "grounding": {
            "prop_g":  {"kind": "reach", "x":  2.5, "y":  2.5, "z": 0.0, "r": 0.6},
            "prop_o1": {"kind": "avoid", "x":  0.0, "y":  0.0, "z": 0.0, "r": 0.6},
            "prop_o2": {"kind": "avoid", "x": -1.2, "y":  1.5, "z": 0.0, "r": 0.55},
        },
        "map_hint": {
            "world_bounds": [-4, 4, -4, 4],
            "start":        [-3.0, -3.0],
            "style": "telograf_native",
            "telograf_samples": 48,
            "gz_world": "maze",
            # real-seconds reach deadline (the F[0,*] above is the abstract
            # horizon TeLoGraF is conditioned on; the *certified* timed task is
            # this wall-clock deadline -- see stl_runtime / Route B).
            "reach_deadline_s": 70.0,
            "nominal_speed": 0.18,
        },
    },

    # ---------- case 2: reach + always avoid all obstacles (3) ----------
    {
        "id":   "reach_avoid",
        "role": "telograf",
        "nl":   "Always avoid all obstacles and eventually, within the time bound, reach the goal.",
        "tree": {
            "op": "and", "interval": None, "children": [
                {"op": "finally", "interval": [20, 55], "children": [
                    {"op": "atom", "name": "prop_g"},
                ]},
                {"op": "globally", "interval": None, "children": [
                    {"op": "not", "interval": None, "children": [
                        {"op": "atom", "name": "prop_o1"}]}]},
                {"op": "globally", "interval": None, "children": [
                    {"op": "not", "interval": None, "children": [
                        {"op": "atom", "name": "prop_o2"}]}]},
                {"op": "globally", "interval": None, "children": [
                    {"op": "not", "interval": None, "children": [
                        {"op": "atom", "name": "prop_o3"}]}]},
            ],
        },
        "grounding": {
            "prop_g":  {"kind": "reach", "x":  3.0, "y":  3.0, "z": 0.0, "r": 0.6},
            "prop_o1": {"kind": "avoid", "x":  1.0, "y": -1.5, "z": 0.0, "r": 0.6},
            "prop_o2": {"kind": "avoid", "x": -1.5, "y":  1.0, "z": 0.0, "r": 0.6},
            "prop_o3": {"kind": "avoid", "x":  0.3, "y":  0.3, "z": 0.0, "r": 0.5},
        },
        "map_hint": {
            "world_bounds": [-4, 4, -4, 4],
            "start":        [-3.0, -3.0],
            "style": "telograf_native",
            "telograf_samples": 48,
            "gz_world": "maze",
        },
    },

    # ---------- case 3: reach a northern goal, avoid all obstacles (3) -----
    {
        "id":   "reach_goal_north",
        "role": "telograf",
        "nl":   "Keep safe from every obstacle and reach the north goal within the time bound.",
        "tree": {
            "op": "and", "interval": None, "children": [
                {"op": "finally", "interval": [20, 55], "children": [
                    {"op": "atom", "name": "prop_g"},
                ]},
                {"op": "globally", "interval": None, "children": [
                    {"op": "not", "interval": None, "children": [
                        {"op": "atom", "name": "prop_o1"}]}]},
                {"op": "globally", "interval": None, "children": [
                    {"op": "not", "interval": None, "children": [
                        {"op": "atom", "name": "prop_o2"}]}]},
                {"op": "globally", "interval": None, "children": [
                    {"op": "not", "interval": None, "children": [
                        {"op": "atom", "name": "prop_o3"}]}]},
            ],
        },
        "grounding": {
            "prop_g":  {"kind": "reach", "x":  0.0, "y":  3.0, "z": 0.0, "r": 0.6},
            "prop_o1": {"kind": "avoid", "x":  1.2, "y":  0.5, "z": 0.0, "r": 0.6},
            "prop_o2": {"kind": "avoid", "x": -1.5, "y":  1.0, "z": 0.0, "r": 0.55},
            "prop_o3": {"kind": "avoid", "x": -0.5, "y": -1.0, "z": 0.0, "r": 0.5},
        },
        "map_hint": {
            "world_bounds": [-4, 4, -4, 4],
            "start":        [-3.0, -3.0],
            "style": "telograf_native",
            "telograf_samples": 64,
            "gz_world": "maze",
        },
    },

    # ===================== decomposition + A*-fallback ====================

    # ---------- case 4: sequential reach A->B->C ----------
    # A nested sequential spec  F(A & F(B & F(C)))  is OUT of distribution for
    # the simple_gnn_F checkpoint when fed jointly (3 nested reaches > the 4
    # training templates).  Tier-1 *tree decomposition* (planner.telograf_infer
    # ._telograf_plan_decomposed) splits it into 3 in-distribution timed-reach
    # legs, samples each by flow matching, and stitches them -- so this case
    # now runs on GENUINE flow matching (no A*), verified end-to-end.  Disable
    # the decomposition with TELOGRAF_DECOMPOSE=0 to see the joint-encoding
    # failure it recovers from.
    {
        "id":   "seq_reach_ABC",
        "role": "telograf",
        "nl":   "Visit zone A, then zone B, then zone C, while always avoiding every obstacle in the room.",
        "tree": {
            "op": "finally", "interval": None, "children": [
                {"op": "and", "interval": None, "children": [
                    {"op": "atom", "name": "prop_1"},
                    {"op": "finally", "interval": None, "children": [
                        {"op": "and", "interval": None, "children": [
                            {"op": "atom", "name": "prop_2"},
                            {"op": "finally", "interval": None, "children": [
                                {"op": "atom", "name": "prop_3"},
                            ]},
                        ]},
                    ]},
                ]},
            ],
        },
        "grounding": {
            "prop_1": {"kind": "reach", "x": -2.5, "y": -2.0, "z": 0.0, "r": 0.4},
            "prop_2": {"kind": "reach", "x":  2.5, "y": -1.0, "z": 0.0, "r": 0.4},
            "prop_3": {"kind": "reach", "x":  0.0, "y":  2.5, "z": 0.0, "r": 0.4},
        },
        "map_hint": {
            "world_bounds": [-4, 4, -4, 4],
            "start":        [-3.5, -3.5],
            "style": "complex",            # walls + furniture TeLoGraF can't see
            "n_furniture": 4,
            "n_random_obstacles": 0,
            "telograf_samples": 16,
            "gz_world": "depot",
        },
    },

    # ---------- case 5: conditional response (imply) + clutter ----------
    {
        "id":   "trigger_response",
        "role": "fallback",
        "nl":   "If the trigger zone fires, reach the safe zone within 5 steps, while always avoiding every obstacle in the room.",
        "tree": {
            "op": "globally", "interval": None, "children": [
                {"op": "imply", "interval": None, "children": [
                    {"op": "atom", "name": "prop_1"},
                    {"op": "finally", "interval": [0, 5], "children": [
                        {"op": "atom", "name": "prop_2"},
                    ]},
                ]},
            ],
        },
        "grounding": {
            "prop_1": {"kind": "reach", "x": -1.5, "y": -1.5, "z": 0.0, "r": 0.4},
            "prop_2": {"kind": "reach", "x":  2.5, "y":  2.5, "z": 0.0, "r": 0.4},
        },
        "map_hint": {
            "world_bounds": [-4, 4, -4, 4],
            "start":        [-3.0, -3.0],
            "style": "complex",
            "n_furniture": 3,
            "n_random_obstacles": 2,
            "telograf_samples": 16,
            "gz_world": "warehouse",
        },
    },

    # ---------- case 6: either-goal via implication, solved by FLOW ----------
    # Demonstrates OPERATOR NORMALIZATION enabling the FROZEN flow planner:
    #   imply(not(F reach_N), F reach_S)  --normalise-->  or(F reach_N, F reach_S)
    # i.e. "reach the north dock OR the south dock" -- a disjunctive goal the
    # frozen GNN handles in-distribution (it reaches whichever is easier), so the
    # spec is solved by pure flow + guidance, NOT the A* fallback.  This is the
    # only one of our tasks where a rewritten ->/<-> operator is discharged by
    # flow rather than the classical backend.
    {
        "id":   "cond_reach_either",
        "role": "telograf",
        "nl":   "Reach the north dock; if you cannot, reach the south dock "
                "instead, while always keeping safe from the obstacles.",
        "tree": {
            "op": "and", "interval": None, "children": [
                {"op": "imply", "interval": None, "children": [
                    {"op": "not", "interval": None, "children": [
                        {"op": "finally", "interval": [20, 55], "children": [
                            {"op": "atom", "name": "prop_N"}]}]},
                    {"op": "finally", "interval": [20, 55], "children": [
                        {"op": "atom", "name": "prop_S"}]},
                ]},
                {"op": "globally", "interval": None, "children": [
                    {"op": "not", "interval": None, "children": [
                        {"op": "atom", "name": "prop_o1"}]}]},
                {"op": "globally", "interval": None, "children": [
                    {"op": "not", "interval": None, "children": [
                        {"op": "atom", "name": "prop_o2"}]}]},
            ],
        },
        "grounding": {
            "prop_N":  {"kind": "reach", "x": -2.6, "y":  2.8, "z": 0.0, "r": 0.6},
            "prop_S":  {"kind": "reach", "x":  2.6, "y": -2.4, "z": 0.0, "r": 0.6},
            "prop_o1": {"kind": "avoid", "x":  0.2, "y":  0.2, "z": 0.0, "r": 0.6},
            "prop_o2": {"kind": "avoid", "x": -1.6, "y": -0.8, "z": 0.0, "r": 0.5},
        },
        "map_hint": {
            "world_bounds": [-4, 4, -4, 4],
            "start":        [-3.2, -3.2],
            "style": "telograf_native",
            "telograf_samples": 48,
            "gz_world": "maze",
        },
    },

    # ===== closed-loop MULTI-GOAL demo (uniform cylinder field, sensed online) ==
    # The robot must VISIT three goal regions in order (A -> B -> C) while always
    # keeping safe.  The obstacles are a UNIFORM grid of cylinders the spec never
    # names -- the robot senses them online (LiDAR) and the keep-safe predicate
    # G(not unsafe) is grounded by the sensed disks.  Shared by the 2D demo
    # (sim_ros2/closed_loop_demo.py) and the Gazebo closed loop
    # (sim_ros2/tb3_follower.py); the cylinder field lives in sim_ros2/scenario.py.
    {
        "id": "closed_loop_multi", "role": "telograf",
        "nl": "Visit region B, then A, then C in order, while always staying "
              "clear of every obstacle.",
        # spec/visit order is B -> A -> C (so the robot crosses the cylinder
        # field diagonally); the A/B/C labels stay pinned to their positions.
        "tree": {"op": "and", "interval": None, "children": [
            {"op": "finally", "interval": [10, 45], "children": [
                {"op": "atom", "name": "reach_B"}]},
            {"op": "finally", "interval": [10, 45], "children": [
                {"op": "atom", "name": "reach_A"}]},
            {"op": "finally", "interval": [10, 45], "children": [
                {"op": "atom", "name": "reach_C"}]},
        ]},
        "grounding": {
            "reach_B": {"kind": "reach", "x":  3.0, "y":  3.0, "z": 0.0, "r": 0.45},
            "reach_A": {"kind": "reach", "x":  3.0, "y": -3.0, "z": 0.0, "r": 0.45},
            "reach_C": {"kind": "reach", "x": -3.0, "y":  3.0, "z": 0.0, "r": 0.45},
        },
        "map_hint": {
            "world_bounds": [-3.8, 3.8, -3.8, 3.8],
            "start":        [-3.0, -3.0],
            "style": "telograf_native",
            "telograf_samples": 8,
        },
    },
]


# The ROS closed loop runs in Gazebo Classic with an OPEN cylinder world
# generated per-case by sim_ros2/tb3_world_gen.py (red obstacle cylinders the
# TurtleBot3 senses online via LiDAR; goal regions are RViz markers only, not
# physical bodies).  Any leftover "gz_world" key in a case's map_hint is unused.


def get_case(case_id: str) -> dict:
    for c in CASES:
        if c["id"] == case_id:
            return c
    raise KeyError(f"unknown case: {case_id}; choose from "
                   f"{[c['id'] for c in CASES]}")


if __name__ == "__main__":
    import json
    for c in CASES:
        print(f"=== {c['id']} ===")
        print(f"  NL    : {c['nl']}")
        print(f"  atoms : {list(c['grounding'].keys())}")
        print(f"  tree  : {json.dumps(c['tree'], ensure_ascii=False)[:120]}...")
        print()
