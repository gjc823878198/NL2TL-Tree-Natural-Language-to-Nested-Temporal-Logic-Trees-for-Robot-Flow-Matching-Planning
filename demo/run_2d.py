"""
2D demo: natural language -> nested STL tree -> closed-loop flow-matching plan,
rendered top-down (no ROS / Gazebo needed).

Same map as the closed-loop simulation (regions A/B/C in a uniform cylinder
field, start at (-3,-3)); the ONLY difference from code/sim_ros2/closed_loop_demo.py
is that the task comes from a natural-language sentence you type, parsed by the
paper's frozen-LLM pipeline (nl_grounding.nl_to_case), instead of a hard-coded
scenario. The robot visits the goals the sentence asks for, in order, while the
keep-safe predicate G(not unsafe) is grounded by the sensed cylinder disks and a
newly sensed cylinder on the path triggers a re-plan.

    # type the task in a pop-up box, then watch the plan:
    python3 run_2d.py
    # or pass it directly / force offline keyword parsing:
    python3 run_2d.py --nl "Visit C, then A, keeping safe, within 120 s" --no-llm

Outputs: outputs/nl_demo_2d.png (overhead) + outputs/nl_demo_2d.gif (animated).
"""
from __future__ import annotations
import argparse
import contextlib
import io
import math
import os
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import matplotlib
# Live on-screen animation when a display is available (watch the robot move in
# real time); fall back to a headless Agg "save the PNG/GIF only" mode otherwise.
_HEADLESS = (not os.environ.get("DISPLAY")) or bool(os.environ.get("NL2TL_HEADLESS"))
if _HEADLESS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nl_grounding as G                                            # noqa: E402

CODE = G.CODE
sys.path.insert(0, str(CODE))
from planner import plan_waypoints                                 # noqa: E402
from planner.telograf_infer import _obstacle_blocked               # noqa: E402
from keep_safe import keep_safe_robustness                         # noqa: E402
from stl_runtime import PlanLatencyModel                           # noqa: E402
from sim_ros2.scenario import CYLINDERS                            # noqa: E402

OUT = HERE / "outputs"
SENSE_RANGE = 2.0
V_NOM = 0.18                # plan time-stamping speed (m/s) == scenario.NOMINAL_SPEED;
#                            sim time at a point = travelled arc-length / V_NOM, in
#                            SECONDS -- the same unit as the task deadline.
CYLS = [{"kind": "circle", "x": x, "y": y, "r": r} for (x, y, r) in CYLINDERS]


def _leg_case(goal):
    gx, gy, gr = goal
    return {"id": "nl_leg", "role": "telograf",
            "tree": {"op": "finally", "interval": [20, 55],
                     "children": [{"op": "atom", "name": "g"}]},
            "grounding": {"g": {"kind": "reach", "x": gx, "y": gy,
                                "z": 0.0, "r": gr}},
            "map_hint": {"world_bounds": list(G.WORLD_BOUNDS),
                         "start": list(G.START),
                         "style": "telograf_native", "telograf_samples": 8}}


def _plan(start_xy, goal, obstacles):
    c = _leg_case(goal)
    c["map_hint"]["start"] = list(start_xy)
    with contextlib.redirect_stdout(io.StringIO()):
        return [tuple(p) for p in plan_waypoints(c, n_steps=90, backend="telograf",
                                                 obstacles=obstacles)]


def _sense(x, y, sensed):
    changed = False
    for c in CYLS:
        if c in sensed:
            continue
        if math.hypot(x - c["x"], y - c["y"]) - c["r"] < SENSE_RANGE:
            sensed.append(c)
            changed = True
    return changed


def _remaining_blocked(plan, idx, obstacles):
    return any(_obstacle_blocked(x, y, obstacles, 0.10)
               for (x, y) in plan[idx:idx + 18])


def run(goals, goal_names, on_step=None, on_status=None, on_plan_tick=None):
    """Closed-loop sense -> plan -> re-plan over the NL-derived goal sequence.

    Callbacks (for the live window, see live_run):
      on_step(travelled, sensed, replan_pts, gi, plan, plan_total_s)
      on_status(msg)
      on_plan_tick(plan_elapsed_s, plan_total_before_s) -- called REPEATEDLY while
        the planner is thinking and the robot is stopped, so the live clock keeps
        advancing (the world does not freeze while we plan)."""
    x, y = G.START
    travelled = [(x, y)]
    sensed = []
    _sense(x, y, sensed)
    replan_pts = []
    arrivals = []
    dt, v_max = 0.06, 0.35
    # measure the planner's wall-clock latency: while it thinks, the world waits,
    # so this time is debited from the deadline budget (latency-aware robustness).
    lat = PlanLatencyModel()

    def timed_plan(s, g, o):
        t0 = time.time()
        if on_plan_tick is None:                 # headless: just time it
            p = _plan(s, g, o)
            lat.observe(time.time() - t0)
            return p
        # live: run the planner in a thread and TICK the clock while it runs, so
        # the robot's sim-time keeps advancing as it waits (demonstrates that the
        # planning latency really elapses against the deadline).
        import threading
        box = {}

        def _work():
            box["p"] = _plan(s, g, o)
        before = lat.consumed()
        th = threading.Thread(target=_work, daemon=True)
        th.start()
        while th.is_alive():
            on_plan_tick(time.time() - t0, before)
            th.join(timeout=0.05)
        lat.observe(time.time() - t0)
        return box["p"]

    for gi, goal in enumerate(goals):
        if on_status:
            on_status(f"planning leg {gi+1}/{len(goals)} → {goal_names[gi]} "
                      f"(TeLoGraF flow-matching)…")
        plan = timed_plan((x, y), goal, sensed)
        idx = 0
        reached = False
        for _ in range(2500):
            if _sense(x, y, sensed) and _remaining_blocked(plan, idx, sensed):
                if on_status:
                    on_status(f"leg {gi+1}/{len(goals)} → {goal_names[gi]}: "
                              f"sensed obstacle blocks path — re-planning…")
                plan = timed_plan((x, y), goal, sensed)
                idx = 0
                replan_pts.append((x, y))
            if idx >= len(plan):
                plan = timed_plan((x, y), goal, sensed)
                idx = 0
                continue
            tx, ty = plan[idx]
            d = math.hypot(tx - x, ty - y)
            if d < 1e-3:
                idx += 1
                continue
            step = min(v_max * dt, d)
            x += step * (tx - x) / d
            y += step * (ty - y) / d
            travelled.append((x, y))
            if d <= v_max * dt:
                idx += 1
            if on_step and len(travelled) % 3 == 0:
                on_step(travelled, sensed, replan_pts, gi, plan, lat.consumed())
            if math.hypot(goal[0] - x, goal[1] - y) < goal[2] + 0.05:
                reached = True
                arrivals.append((goal_names[gi], len(travelled)))
                if on_step:
                    on_step(travelled, sensed, replan_pts, gi, plan, lat.consumed())
                break
        if not reached:
            arrivals.append((goal_names[gi] + "(missed)", len(travelled)))
    rho = keep_safe_robustness(CYLS, travelled)
    return travelled, replan_pts, arrivals, rho, lat.consumed()


# --------------------------- rendering ---------------------------

def _draw_world(ax, goals, goal_names, replan_pts, title):
    xmin, xmax, ymin, ymax = G.WORLD_BOUNDS
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax + 0.7)          # top headroom so region labels clear the title
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    for c in CYLS:
        ax.add_patch(mp.Circle((c["x"], c["y"]), c["r"], facecolor="#e74c3c",
                               edgecolor="#922b21", alpha=0.55, lw=1.0))
    visit = {n: i + 1 for i, n in enumerate(goal_names)}
    for name, (gx, gy, gr) in G.LANDMARKS.items():
        on = name in visit
        ax.add_patch(mp.Circle((gx, gy), gr,
                               facecolor="#2ecc71" if on else "#cfd8dc",
                               edgecolor="#1e8449" if on else "#90a4ae",
                               alpha=0.5 if on else 0.35, lw=1.2))
        # label = letter + coordinate in the robot-start-origin frame (the robot
        # spawns at START, so that point is the (0, 0) of these coordinates).
        rx, ry = gx - G.START[0], gy - G.START[1]
        lab = f"{name} ({rx:g}, {ry:g})"
        ax.text(gx, gy + gr + 0.2, lab, ha="center", fontsize=13,
                fontweight="bold",
                color="#1e8449" if on else "#90a4ae")
    ax.plot(*G.START, marker="s", ms=12, color="k", zorder=6)
    ax.text(G.START[0], G.START[1] - 0.45, "start (0, 0)", ha="center", fontsize=12,
            fontweight="bold")
    for rp in replan_pts:
        ax.plot(*rp, marker="X", ms=10, color="#8e44ad", zorder=7)


def _draw_tb3(ax, cx, cy, r, zorder=8):
    """Top-down TurtleBot3 sticker (round base + two wheels + LiDAR dome) drawn
    on a reached goal, to make clear the robot arrives INSIDE the target region
    (the per-leg path stops at the disk edge)."""
    ww, wl = 0.30 * r, 0.95 * r                      # wheel width / length
    for sx in (-1, 1):                               # left & right wheels (black)
        ax.add_patch(mp.Rectangle((cx + sx * 0.92 * r - ww / 2, cy - wl / 2),
                                  ww, wl, facecolor="#15181d", edgecolor="#000",
                                  lw=0.6, zorder=zorder))
    ax.add_patch(mp.Circle((cx, cy), r, facecolor="#5a6475",              # base plate
                           edgecolor="#11151c", lw=1.3, zorder=zorder + 1))
    ax.add_patch(mp.Circle((cx, cy), 0.46 * r, facecolor="#17a589",        # LiDAR dome
                           edgecolor="#0b5345", lw=1.0, zorder=zorder + 2))
    ax.add_patch(mp.Circle((cx, cy), 0.17 * r, facecolor="#0b5345",        # hub
                           zorder=zorder + 3))
    ax.add_patch(mp.Circle((cx, cy + 0.7 * r), 0.12 * r, facecolor="#f4d03f",  # front
                           edgecolor="#7d6608", lw=0.5, zorder=zorder + 3))


def render(nl, tree, info, travelled, replan_pts, arrivals, rho, used_llm,
           plan_s=0.0):
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = info["deadline_s"]
    v_nom = V_NOM
    tr = np.asarray(travelled)
    seg = np.hypot(np.diff(tr[:, 0]), np.diff(tr[:, 1]))
    tcum = np.concatenate([[0.0], np.cumsum(seg)]) / v_nom
    t_arr = float(tcum[-1])
    goals, names = info["goals"], info["goal_names"]
    n_reached = sum(1 for (n, _) in arrivals if "missed" not in n)
    order = r"$\to$".join(names)
    # latency-aware deadline accounting (our contribution): the planner spends
    # plan_s wall-clock while the world waits, so the honest margin debits it.
    rho_lat = (deadline - plan_s) - t_arr
    line2 = (rf"$\rho_{{\mathrm{{safe}}}}={rho:+.2f}$ m,  "
             rf"exec {t_arr:.0f} s $+$ plan {plan_s:.0f} s vs {deadline:.0f} s  "
             rf"($\rho^{{\mathrm{{lat}}}}_{{\mathrm{{time}}}}{{=}}{rho_lat:+.0f}$ s)")
    title = (rf"NL-driven closed loop ({order}),  visited {n_reached}/{len(goals)}"
             "\n" + line2)
    parse_tag = "frozen LLM" if used_llm else "keyword fallback (offline)"

    # layout: a LEFT text panel (NL + parsed STL tree, large + bold) | the map.
    fig = plt.figure(figsize=(11.0, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.52, 1.0], wspace=0.18)
    tax = fig.add_subplot(gs[0, 0])
    tax.axis("off")
    ax = fig.add_subplot(gs[0, 1])
    _draw_world(ax, goals, names, replan_pts, title)
    pts = tr.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="rainbow", norm=plt.Normalize(0, deadline),
                        linewidth=3.0, zorder=5)
    lc.set_array(0.5 * (tcum[:-1] + tcum[1:]))
    ax.add_collection(lc)
    # TurtleBot3 sticker on each reached goal -- the robot arrives INSIDE the
    # region even though the per-leg path stops at the disk edge.
    for gi, (nm, _) in enumerate(arrivals):
        if gi < len(goals) and "missed" not in nm:
            gx, gy, gr = goals[gi][0], goals[gi][1], goals[gi][2]
            _draw_tb3(ax, gx, gy, min(0.9 * gr, 0.42))
    cb = fig.colorbar(ScalarMappable(norm=plt.Normalize(0, deadline),
                                     cmap="rainbow"), ax=ax,
                      fraction=0.046, pad=0.04)
    cb.set_label("travelled: time (s)", fontsize=13, fontweight="bold")
    cb.set_ticks([0, round(t_arr), deadline])
    cb.set_ticklabels(["0", f"{t_arr:.0f}", f"{deadline:.0f}"])
    cb.ax.tick_params(labelsize=11)
    # left panel: the natural-language task and the parsed nested STL tree
    panel = (f"Natural-language task\n({parse_tag}):\n\n"
             "“" + "\n".join(textwrap.wrap(nl, 20)) + "”\n\n\n"
             "Parsed nested STL tree:\n\n"
             + "\n".join(textwrap.wrap(G.stl_string(tree), 20)))
    tax.text(0.0, 0.5, panel, transform=tax.transAxes, ha="left", va="center",
             fontsize=13, fontweight="bold", color="#1a1a1a",
             bbox=dict(boxstyle="round,pad=0.6", fc="#f4f7fb", ec="#9bb8d6",
                       lw=1.3))
    fig.savefig(OUT / "nl_demo_2d.png", dpi=150, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    _draw_world(ax, goals, names, replan_pts, title)
    (line,) = ax.plot([], [], "-", color="#e67e22", lw=2.5, zorder=5)
    (dot,) = ax.plot([], [], "o", color="#d35400", ms=10, zorder=7)
    ring = mp.Circle((0, 0), SENSE_RANGE, facecolor="#00bcd4", alpha=0.06,
                     edgecolor="#00838f", lw=0.8, ls="--", zorder=2)
    ax.add_patch(ring)
    time_txt = ax.text(0, 0, "", fontsize=12, fontweight="bold", color="#111",
                       ha="left", va="bottom", zorder=9,
                       bbox=dict(boxstyle="round,pad=0.25", fc="#fff3b0",
                                 ec="#b8860b", lw=1.0))
    frames = list(range(1, len(travelled) + 1, 4))
    if frames[-1] != len(travelled):
        frames.append(len(travelled))

    def upd(i):
        line.set_data([p[0] for p in travelled[:i]], [p[1] for p in travelled[:i]])
        dot.set_data([travelled[i - 1][0]], [travelled[i - 1][1]])
        ring.center = travelled[i - 1]
        time_txt.set_position((travelled[i - 1][0] + 0.25, travelled[i - 1][1] + 0.22))
        time_txt.set_text(f"t = {tcum[i - 1]:.0f} / {deadline:.0f} s")
        return line, dot, ring, time_txt

    anim = FuncAnimation(fig, upd, frames=frames, interval=60, blit=False)
    anim.save(OUT / "nl_demo_2d.gif", writer=PillowWriter(fps=16))
    plt.close(fig)
    print(f"wrote {OUT/'nl_demo_2d.png'} and .gif")
    print(f"  parse={parse_tag}; visited={n_reached}/{len(goals)} "
          f"{[n for n, _ in arrivals]}; rho_safe={rho:+.2f}m; "
          f"replans={len(replan_pts)}; finish={t_arr:.0f}s/{deadline:.0f}s")


def render_selfcheck(info, rep):
    """Visualise the best-of-N OOD self-check (paper Fig.~1): the candidate
    prior trajectories + which goals even the best candidate cannot reach -- i.e.
    *why* a spec is out-of-distribution and we defer to A* / decompose.
    Saved to outputs/nl_demo_2d_selfcheck.png (only when the flow backend ran)."""
    if not rep.get("available"):
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 6.8))
    xmin, xmax, ymin, ymax = G.WORLD_BOUNDS
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax + 0.7)
    ax.set_aspect("equal"); ax.grid(alpha=0.25); ax.tick_params(labelsize=12)
    feas = rep["feasible"]
    ax.set_title("Best-of-%d self-check:  %s\nbest per-atom shortfall %.2f m "
                 "(OOD if $>$%.1f m),  satisfy-rate %.0f%%"
                 % (rep["n_samples"],
                    "FEASIBLE — flow" if feas else "OOD — defer to A*",
                    rep["best_shortfall"], rep["tol"], 100 * rep["satisfy_rate"]),
                 fontsize=12.5, fontweight="bold", pad=12,
                 color="#1e7a46" if feas else "#b03a2e")
    for c in CYLS:                                          # sensed obstacles
        ax.add_patch(mp.Circle((c["x"], c["y"]), c["r"], facecolor="#e74c3c",
                     alpha=0.30, edgecolor="#922b21", lw=0.8, zorder=1))
    for s in rep["candidates"]:                            # N prior candidates
        xs = [p[0] for p in s]; ys = [p[1] for p in s]
        ax.plot(xs, ys, "-", color="#9aa0a6", lw=0.8, alpha=0.5, zorder=3)
    if rep.get("best"):                                    # best candidate
        ax.plot([p[0] for p in rep["best"]], [p[1] for p in rep["best"]],
                "-", color="#2c6fbb", lw=2.4, zorder=5, label="best-of-N prior")
    for g in rep["per_goal"]:                              # goals: reached / missed
        ok = g["reached"]
        ax.add_patch(mp.Circle((g["x"], g["y"]), g["r"], facecolor="none",
                     edgecolor="#1e7a46" if ok else "#b03a2e", lw=2.4, zorder=6))
        ax.plot(g["x"], g["y"], marker="*", ms=15,
                color="#1e7a46" if ok else "#b03a2e",
                markeredgecolor="k", markeredgewidth=0.5, zorder=7)
        if not ok:
            ax.text(g["x"], g["y"] + g["r"] + 0.18,
                    f"miss {g['shortfall']:.1f} m", ha="center", fontsize=11,
                    fontweight="bold", color="#b03a2e", zorder=8)
    ax.plot(*G.START, marker="s", ms=11, color="k", zorder=7)
    ax.text(G.START[0], G.START[1] - 0.45, "start", ha="center", fontsize=11)
    ax.legend(loc="lower center", fontsize=10, framealpha=0.9)
    fig.savefig(OUT / "nl_demo_2d_selfcheck.png", dpi=150, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {OUT/'nl_demo_2d_selfcheck.png'}")
    return OUT / "nl_demo_2d_selfcheck.png"


def live_run(goals, goal_names, info, nl, tree, used_llm):
    """Open an interactive window and animate the closed-loop run in REAL TIME:
    the robot dot, the growing travelled path, the current TeLoGraF plan (dashed),
    the sensor ring, and re-plan markers all update live as the robot drives."""
    plt.ion()
    fig = plt.figure(figsize=(11.0, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.52, 1.0], wspace=0.18)
    tax = fig.add_subplot(gs[0, 0]); tax.axis("off")
    ax = fig.add_subplot(gs[0, 1])
    _draw_world(ax, goals, goal_names, [], "live demo — starting…")
    parse_tag = "frozen LLM" if used_llm else "keyword fallback (offline)"
    panel = (f"Natural-language task\n({parse_tag}):\n\n"
             "“" + "\n".join(textwrap.wrap(nl, 20)) + "”\n\n\n"
             "Parsed nested STL tree:\n\n"
             + "\n".join(textwrap.wrap(G.stl_string(tree), 20)))
    tax.text(0.0, 0.5, panel, transform=tax.transAxes, ha="left", va="center",
             fontsize=13, fontweight="bold", color="#1a1a1a",
             bbox=dict(boxstyle="round,pad=0.6", fc="#f4f7fb", ec="#9bb8d6", lw=1.3))
    (path_ln,) = ax.plot([], [], "-", color="#e67e22", lw=2.6, zorder=5)
    (plan_ln,) = ax.plot([], [], "--", color="#2e86de", lw=1.5, alpha=0.7, zorder=4)
    (dot,) = ax.plot([], [], "o", color="#d35400", ms=13, zorder=8)
    ring = mp.Circle((0, 0), SENSE_RANGE, facecolor="#00bcd4", alpha=0.06,
                     edgecolor="#00838f", lw=0.8, ls="--", zorder=2)
    ax.add_patch(ring)
    # live simulation-time clock that follows the robot (seconds == task units)
    deadline = info["deadline_s"]
    time_txt = ax.text(0, 0, "", fontsize=13, fontweight="bold", color="#111",
                       ha="left", va="bottom", zorder=9,
                       bbox=dict(boxstyle="round,pad=0.25", fc="#fff3b0",
                                 ec="#b8860b", lw=1.0))
    seen = {"replan": 0, "arclen": 0.0, "n": 1, "t_exec": 0.0,
            "pos": tuple(G.START)}

    def _title(msg):
        ax.set_title(msg, fontsize=12.5, fontweight="bold", color="#111")
        fig.canvas.draw_idle(); plt.pause(0.001)

    def _set_clock(t_total, planning):
        # t_total = execution time + planning wall-clock so far (== sim seconds).
        rem = deadline - t_total
        x, y = seen["pos"]
        time_txt.set_position((x + 0.25, y + 0.22))
        tag = "  (planning)" if planning else (""  if rem >= 0 else "  LATE")
        time_txt.set_text(f"t = {t_total:.0f} / {deadline:.0f} s{tag}")
        time_txt.set_color("#7a1500" if (planning or rem < 0) else "#111")
        time_txt.get_bbox_patch().set_facecolor(
            "#ffb3a0" if planning else ("#ffd0c0" if rem < 0 else "#fff3b0"))

    def on_step(travelled, sensed, replan_pts, gi, plan, plan_total):
        xs = [p[0] for p in travelled]; ys = [p[1] for p in travelled]
        path_ln.set_data(xs, ys)
        dot.set_data([xs[-1]], [ys[-1]]); ring.center = (xs[-1], ys[-1])
        if plan and len(plan) >= 2:
            plan_ln.set_data([p[0] for p in plan], [p[1] for p in plan])
        for rp in replan_pts[seen["replan"]:]:
            ax.plot(*rp, marker="X", ms=10, color="#8e44ad", zorder=7)
        seen["replan"] = len(replan_pts)
        for k in range(seen["n"], len(xs)):          # accumulate arc length
            seen["arclen"] += math.hypot(xs[k] - xs[k-1], ys[k] - ys[k-1])
        seen["n"] = len(xs)
        seen["t_exec"] = seen["arclen"] / V_NOM
        seen["pos"] = (xs[-1], ys[-1])
        _set_clock(seen["t_exec"] + plan_total, planning=False)  # exec + planning
        ax.set_title(f"live — driving to goal {gi+1}/{len(goals)} "
                     f"({goal_names[gi]})", fontsize=12.5, fontweight="bold",
                     color="#111")
        fig.canvas.draw_idle(); plt.pause(0.003)

    def on_plan_tick(plan_elapsed, plan_total_before):
        # The robot is STOPPED waiting for the planner, but the clock keeps
        # advancing by the real planning wall-clock -- the world does not freeze
        # while we plan. This is the latency-aware contribution, live: you watch
        # the deadline shrink while the robot sits still.
        _set_clock(seen["t_exec"] + plan_total_before + plan_elapsed, planning=True)
        ax.set_title("live — ⏸ PLANNING (robot waiting); the clock keeps "
                     "advancing, eating the deadline",
                     fontsize=12.5, fontweight="bold", color="#b03a2e")
        fig.canvas.draw_idle(); plt.pause(0.01)

    res = run(goals, goal_names, on_step=on_step, on_status=_title,
              on_plan_tick=on_plan_tick)
    travelled, _, arrivals, rho, plan_s = res
    n_reached = sum(1 for (n, _) in arrivals if "missed" not in n)
    seg = [math.hypot(travelled[k][0]-travelled[k-1][0],
                      travelled[k][1]-travelled[k-1][1])
           for k in range(1, len(travelled))]
    t_exec = sum(seg) / V_NOM
    rho_lat = (deadline - plan_s) - t_exec     # latency-aware deadline margin (s)
    _title(f"done — visited {n_reached}/{len(goals)}, ρ_safe={rho:+.2f} m; "
           f"exec {t_exec:.0f}s + plan {plan_s:.0f}s vs {deadline:.0f}s "
           f"(ρ_time^lat={rho_lat:+.0f}s)")
    plt.ioff()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl", default=None, help="task sentence (else a pop-up box)")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the Groq call, use the deterministic keyword parser")
    ap.add_argument("--model", default=G.DEFAULT_MODEL)
    ap.add_argument("--render-only", action="store_true",
                    help="re-render the figure from the last cached run "
                         "(no LLM, no planning) -- for fast layout tweaks")
    ap.add_argument("--save-only", action="store_true",
                    help="skip the live on-screen window; just write the PNG/GIF")
    ap.add_argument("--env", default="normal", choices=["open", "normal", "complex"],
                    help="environment-complexity hint for the pre-flight gauge")
    ap.add_argument("--no-selfcheck", action="store_true",
                    help="skip the best-of-N OOD self-check visualization")
    args = ap.parse_args()
    import pickle
    cache = OUT / "last_run.pkl"
    if args.render_only:
        render(**pickle.loads(cache.read_bytes()))
        return
    nl, env = args.nl, args.env
    if nl is None:
        from nl_input import ask_nl
        nl, env = ask_nl()
    if not nl:
        print("no task entered; nothing to do")
        return
    case, tree, info, used_llm = G.nl_to_case(nl, model=args.model,
                                              use_llm=not args.no_llm)
    print("NL   :", nl)
    print("STL  :", G.stl_string(tree))
    print("plan :", " -> ".join(info["goal_names"]),
          f"| keep_safe={info['keep_safe']} | deadline={info['deadline_s']:.0f}s")
    # pre-flight completability gauge (BEFORE committing to the full flow run)
    import feasibility as F
    fe = F.estimate(G.START, info["goals"], info["deadline_s"], env, CYLS)
    print("check:", F.summary(fe))
    # best-of-N OOD self-check (paper Fig.~1): sample the flow prior for the whole
    # task and report/visualise whether it is in-distribution or must defer to A*.
    if not args.no_selfcheck:
        import selfcheck as SC
        sc = SC.run(info, obstacles=CYLS, world_bounds=G.WORLD_BOUNDS,
                    start=G.START)
        print("self :", SC.summary(sc))
        render_selfcheck(info, sc)
    live = (not _HEADLESS) and (not args.save_only)
    if live:
        print("opening live window — watch the robot drive (close it to exit)…")
        travelled, replan_pts, arrivals, rho, plan_s = live_run(
            info["goals"], info["goal_names"], info, nl, tree, used_llm)
    else:
        if _HEADLESS and not args.save_only:
            print("no display detected — running headless (PNG/GIF only).")
        travelled, replan_pts, arrivals, rho, plan_s = run(info["goals"],
                                                           info["goal_names"])
    d = dict(nl=nl, tree=tree, info=info, travelled=travelled,
             replan_pts=replan_pts, arrivals=arrivals, rho=rho, used_llm=used_llm,
             plan_s=plan_s)
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(d))
    render(**d)                                  # always save the PNG + GIF too
    if live:
        plt.show()                               # keep the live window open


if __name__ == "__main__":
    main()
