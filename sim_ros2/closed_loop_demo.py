"""
Closed-loop, sensor-driven re-planning demo (the loop §4 describes), rendered
RViz-style top-down.  Self-contained (no ROS / Gazebo needed): it runs the SAME
sense->plan->replan logic as `tb3_follower.py`, driven by a Python loop instead
of /odom + /scan callbacks.

Scenario (shared with the Gazebo loop, see sim_ros2/scenario.py): a UNIFORM grid
of cylinder obstacles and a MULTI-GOAL sequence -- the robot must visit goal
regions A -> B -> C in order while ALWAYS keeping safe.  The cylinders are NOT in
the STL spec: the robot senses them online within its LiDAR range and the
keep-safe predicate G(not unsafe) is grounded by the sensed disks; when a newly
sensed cylinder blocks the remaining path, the robot REPLANS from its current
pose with the frozen flow planner.

Outputs (top-down, RViz marker colours):
    outputs/sim_ros2/closed_loop/closed_loop_demo.png   (overhead snapshot)
    outputs/sim_ros2/closed_loop/closed_loop_demo.gif   (animated)

    python3 sim_ros2/closed_loop_demo.py
"""
from __future__ import annotations
import io, contextlib, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner import plan_waypoints                              # noqa: E402
from planner.telograf_infer import _obstacle_blocked            # noqa: E402
from keep_safe import keep_safe_robustness                      # noqa: E402
from sim_ros2 import scenario as S                              # noqa: E402

OUT = ROOT / "outputs" / "sim_ros2" / "closed_loop"
SENSE_RANGE = 2.0                               # LiDAR range (m)
START = S.START
GOALS = S.GOALS                                 # [(x,y,r), ...] visit in order
CYLS = [{"kind": "circle", "x": x, "y": y, "r": r} for (x, y, r) in S.CYLINDERS]


def _single_case(goal):
    """A single in-distribution timed-reach toward `goal` (one STL leg)."""
    gx, gy, gr = goal
    return {"id": "cl_leg", "role": "telograf",
            "tree": {"op": "finally", "interval": [20, 55],
                     "children": [{"op": "atom", "name": "g"}]},
            "grounding": {"g": {"kind": "reach", "x": gx, "y": gy,
                                "z": 0.0, "r": gr}},
            "map_hint": {"world_bounds": list(S.REGION), "start": list(START),
                         "style": "telograf_native", "telograf_samples": 8}}


def _plan(start_xy, goal, obstacles):
    c = _single_case(goal); c["map_hint"]["start"] = list(start_xy)
    with contextlib.redirect_stdout(io.StringIO()):
        return [tuple(p) for p in plan_waypoints(c, n_steps=90, backend="telograf",
                                                 obstacles=obstacles)]


def _sense(x, y, sensed):
    """Add any cylinder whose nearest edge is within SENSE_RANGE to `sensed`."""
    changed = False
    for c in CYLS:
        if c in sensed:
            continue
        if math.hypot(x - c["x"], y - c["y"]) - c["r"] < SENSE_RANGE:
            sensed.append(c); changed = True
    return changed


def _remaining_blocked(plan, idx, obstacles):
    for (x, y) in plan[idx:idx + 18]:
        if _obstacle_blocked(x, y, obstacles, 0.10):
            return True
    return False


def run():
    S.print_task()                              # show the task this run completes
    x, y = START
    yaw = 0.0
    travelled = [(x, y)]
    sensed = []
    _sense(x, y, sensed)
    replan_pts = []
    goal_arrivals = []                          # (name, traj-index) per reached goal
    dt, v_max, w_max, look, goal_tol = 0.06, 0.35, 1.5, 0.28, 0.40

    for gi, goal in enumerate(GOALS):
        plan = _plan((x, y), goal, sensed)
        idx = 0
        reached = False
        for _step in range(2500):
            if _sense(x, y, sensed) and _remaining_blocked(plan, idx, sensed):
                plan = _plan((x, y), goal, sensed)      # CLOSED-LOOP REPLAN
                idx = 0
                replan_pts.append((x, y))
            if idx >= len(plan):
                plan = _plan((x, y), goal, sensed); idx = 0; continue
            # follow the plan polyline FAITHFULLY: the plan is collision-free
            # w.r.t. the sensed cylinders (planner clearance ~0.2 m), so stepping a
            # fixed arclength along it lets the travelled path inherit that
            # clearance (pure-pursuit corner-cutting was what grazed the field).
            tx, ty = plan[idx]
            d = math.hypot(tx - x, ty - y)
            if d < 1e-3:
                idx += 1; continue
            step = min(v_max * dt, d)
            x += step * (tx - x) / d
            y += step * (ty - y) / d
            travelled.append((x, y))
            if d <= v_max * dt:
                idx += 1
            if math.hypot(goal[0] - x, goal[1] - y) < goal[2] + 0.05:
                reached = True
                goal_arrivals.append((S.GOAL_NAMES[gi], len(travelled)))
                break
        if not reached:
            goal_arrivals.append((S.GOAL_NAMES[gi] + "(missed)", len(travelled)))
    rho = keep_safe_robustness(CYLS, travelled)
    return travelled, replan_pts, sensed, goal_arrivals, rho


# --------------------------- rendering ---------------------------
def _draw_world(ax, replan_pts, title):
    xmin, xmax, ymin, ymax = S.REGION
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
    ax.grid(alpha=0.25); ax.set_title(title, fontsize=11)
    # uniform cylinder field (sensed obstacles, red)
    for c in CYLS:
        ax.add_patch(mp.Circle((c["x"], c["y"]), c["r"], facecolor="#e74c3c",
                               edgecolor="#922b21", alpha=0.55, lw=1.0))
    # goal regions A/B/C (green) + order labels
    for name, (gx, gy, gr) in zip(S.GOAL_NAMES, GOALS):
        ax.add_patch(mp.Circle((gx, gy), gr, facecolor="#2ecc71",
                               edgecolor="#1e8449", alpha=0.45, lw=1.2))
        ax.plot(gx, gy, marker="*", ms=16, color="#1e8449",
                markeredgecolor="k", markeredgewidth=0.5, zorder=6)
        ax.text(gx, gy + gr + 0.18, name, ha="center", fontsize=11,
                fontweight="bold", color="#1e8449")
    # start
    ax.plot(*START, marker="s", ms=11, color="k", zorder=6)
    ax.text(START[0], START[1] - 0.4, "start", ha="center", fontsize=8)
    # replan points + sensor ring
    for rp in replan_pts:
        ax.add_patch(mp.Circle(rp, SENSE_RANGE, facecolor="#00bcd4",
                               edgecolor="#00838f", alpha=0.05, lw=0.8, ls="--"))
        ax.plot(*rp, marker="X", ms=10, color="#8e44ad", zorder=7)


def render(travelled, replan_pts, sensed, goal_arrivals, rho):
    OUT.mkdir(parents=True, exist_ok=True)
    from matplotlib.cm import ScalarMappable
    V_NOM = S.NOMINAL_SPEED
    deadline = S.REACH_DEADLINE_S
    tr = np.asarray(travelled)
    seg_len = np.hypot(np.diff(tr[:, 0]), np.diff(tr[:, 1]))
    tcum = np.concatenate([[0.0], np.cumsum(seg_len)]) / V_NOM
    t_arr = float(tcum[-1])
    n_reached = sum(1 for (n, _) in goal_arrivals if "missed" not in n)
    order = r"$\to$".join(S.GOAL_NAMES)
    title = (rf"Closed-loop multi-goal ({order}) through a uniform "
             rf"cylinder field" "\n"
             rf"visited {n_reached}/{len(GOALS)} goals,  "
             rf"$\rho_{{\mathrm{{safe}}}}={rho:+.2f}$ m,  "
             rf"finish {t_arr:.0f} s $<$ {deadline:.0f} s deadline")
    # --- static top-down ---
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    _draw_world(ax, replan_pts, title)
    pts = tr.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_t = 0.5 * (tcum[:-1] + tcum[1:])
    lc = LineCollection(segs, cmap="rainbow", norm=plt.Normalize(0, deadline),
                        linewidth=3.0, zorder=5)
    lc.set_array(seg_t)
    ax.add_collection(lc)
    cb = fig.colorbar(ScalarMappable(norm=plt.Normalize(0, deadline),
                                     cmap="rainbow"), ax=ax,
                      fraction=0.046, pad=0.04)
    cb.set_label("travelled: time (s)", fontsize=9)
    cb.set_ticks([0, round(t_arr), deadline])
    cb.set_ticklabels(["0", f"{t_arr:.0f} (finish)", f"{deadline:.0f} (deadline)"])
    cb.ax.tick_params(labelsize=7)
    fig.savefig(OUT / "closed_loop_demo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- animated ---
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    _draw_world(ax, replan_pts, title)
    (line,) = ax.plot([], [], "-", color="#e67e22", lw=2.5, zorder=5)
    (dot,) = ax.plot([], [], "o", color="#d35400", ms=10, zorder=7)
    ring = mp.Circle((0, 0), SENSE_RANGE, facecolor="#00bcd4", alpha=0.06,
                     edgecolor="#00838f", lw=0.8, ls="--", zorder=2)
    ax.add_patch(ring)
    stride = 4
    frames = list(range(1, len(travelled) + 1, stride))
    if frames[-1] != len(travelled):
        frames.append(len(travelled))

    def update(i):
        line.set_data([p[0] for p in travelled[:i]], [p[1] for p in travelled[:i]])
        dot.set_data([travelled[i - 1][0]], [travelled[i - 1][1]])
        ring.center = travelled[i - 1]
        return line, dot, ring

    anim = FuncAnimation(fig, update, frames=frames, interval=60, blit=False)
    anim.save(OUT / "closed_loop_demo.gif", writer=PillowWriter(fps=16))
    plt.close(fig)
    print(f"wrote {OUT/'closed_loop_demo.png'} and .gif")
    print(f"  visited={n_reached}/{len(GOALS)} {[n for n,_ in goal_arrivals]}; "
          f"rho_safe={rho:+.2f}m; replans={len(replan_pts)}; "
          f"finish={t_arr:.0f}s/{deadline:.0f}s; travelled_pts={len(travelled)}")


if __name__ == "__main__":
    travelled, replan_pts, sensed, goal_arrivals, rho = run()
    render(travelled, replan_pts, sensed, goal_arrivals, rho)
