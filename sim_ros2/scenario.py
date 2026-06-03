"""
Shared closed-loop scenario: a UNIFORM grid of cylinder obstacles + a MULTI-GOAL
sequence the robot must visit in order while keeping safe.

Used by BOTH the 2D matplotlib demo (closed_loop_demo.py) and the Gazebo closed
loop (tb3_world_gen.py + tb3_sim.launch.py + tb3_follower.py), so the two render
the SAME task.  The goal regions live in the STL spec (case `closed_loop_multi`
in planner/case_examples.py -- the single source of truth, shown as RViz markers,
NOT physical bodies).  The cylinders are NOT in the spec: the robot SENSES them
online with its LiDAR and the keep-safe predicate G(not unsafe) is grounded by
the sensed disks.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from planner import get_case                                       # noqa: E402

CASE_ID = "closed_loop_multi"
_CASE = get_case(CASE_ID)

START = tuple(_CASE["map_hint"]["start"])                          # (x, y)
# goals, in the order the spec asks for them (grounding dict is insertion-ordered);
# the name comes from the atom key (reach_B -> "B") so the visit order is whatever
# order the case lists them in.
_reaches = [(k, g) for k, g in _CASE["grounding"].items()
            if g.get("kind", "reach") == "reach"]
GOALS = [(g["x"], g["y"], g["r"]) for (_, g) in _reaches]
GOAL_NAMES = [k.replace("reach_", "").upper() for (k, _) in _reaches]
REGION = tuple(_CASE["map_hint"]["world_bounds"])                  # xmin,xmax,ymin,ymax
REACH_DEADLINE_S = 170.0          # finish all goals within this (real seconds);
#  B->A->C crosses the field diagonally twice (~21 m at the 0.18 m/s nominal
#  speed ~120 s), so a 170 s budget keeps a comfortable margin.
NOMINAL_SPEED = 0.18              # plan time-stamping speed (m/s, <= v_max)


def uniform_cylinders(n: int = 3, lo: float = -2.2, hi: float = 2.2,
                      r: float = 0.22, clear: float = 0.95):
    """A regular n x n grid of cylinder obstacles, evenly spaced over the region
    interior; drop any that sit too close (< clear) to the start or a goal so the
    field never blocks a goal disk or buries the spawn.  Spacing is kept wide
    enough (gap >~ 1 m) that a diagonal B->A->C crossing always has a clear lane
    with margin -- a denser grid leaves the flow/A* no collision-free path in the
    tight diagonal gaps."""
    keep = []
    step = (hi - lo) / (n - 1)
    for i in range(n):
        for j in range(n):
            gx, gy = lo + i * step, lo + j * step
            if math.hypot(gx - START[0], gy - START[1]) < clear + r:
                continue
            if any(math.hypot(gx - x, gy - y) < clear + r + gr
                   for (x, y, gr) in GOALS):
                continue
            keep.append((round(gx, 3), round(gy, 3), r))
    return keep


# the physical/sensed obstacle field (uniform grid), as (x, y, r) tuples
CYLINDERS = uniform_cylinders()


def task_nl() -> str:
    seq = ", then ".join(GOAL_NAMES)
    return (f"Visit region {seq} in order, while ALWAYS staying clear of every "
            f"obstacle, and finish within {REACH_DEADLINE_S:.0f} s.")


def task_stl() -> str:
    seq = " ; ".join(f"F(reach_{n})" for n in GOAL_NAMES)
    return (f"{seq}   subject to   G(not unsafe)"
            f"   [unsafe = inside any LiDAR-sensed cylinder]")


def print_task(logger=None) -> None:
    """Print the task this run must complete (NL + STL + scene)."""
    bar = "=" * 70
    lines = [
        bar,
        "TASK  (closed-loop, multi-goal):",
        "  NL  : " + task_nl(),
        "  STL : " + task_stl(),
        "  goals     : " + "  ".join(
            f"{n}=({x:+.1f},{y:+.1f})" for n, (x, y, _) in zip(GOAL_NAMES, GOALS)),
        f"  obstacles : {len(CYLINDERS)} cylinders, UNIFORM grid (sensed online, "
        f"not in spec)",
        f"  start     : ({START[0]:+.1f},{START[1]:+.1f})",
        bar,
    ]
    msg = "\n".join(lines)
    if logger is not None:
        logger.info("\n" + msg)
    else:
        print(msg, flush=True)


if __name__ == "__main__":
    print_task()
    print("CYLINDERS:", CYLINDERS)
