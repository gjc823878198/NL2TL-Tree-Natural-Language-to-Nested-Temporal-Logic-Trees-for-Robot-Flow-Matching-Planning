"""
Lightweight 2D simulator for visualising STL-conditioned trajectories.

A scene contains:
  - axis-aligned world bounds
  - reach circles (green): atoms with kind="reach"
  - avoid circles (red):   atoms with kind="avoid"  (treated as obstacles)
  - random circular obstacles (grey) sprinkled into the free space
  - optional rectangular obstacles + walls (grey boxes), forming
    procedural rooms / corridors that force visible detours
  - a start pose

Two map styles, selected per-case via `map_hint.style`:
  "simple"  : just random disks (default; same as the original simulator)
  "complex" : walls dividing the world + furniture-shaped rectangles
              so an obstacle-avoiding planner has to do real navigation,
              not a straight line.

This file is intentionally backend-agnostic: it does not depend on TeLoGraF.
You can feed it any trajectory `[(x,y), ...]` and it will render + animate.
When TeLoGraF is installed, plug its output here.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Disk:
    x: float
    y: float
    r: float
    kind: str = "obstacle"     # "reach" | "avoid" | "obstacle"
    name: str = ""


@dataclass
class Rect:
    """Axis-aligned box: (x, y) is the centre, (w, h) is the full size."""
    x: float
    y: float
    w: float
    h: float
    kind: str = "obstacle"     # "wall" | "obstacle"
    name: str = ""

    def aabb(self):
        return (self.x - self.w / 2, self.x + self.w / 2,
                self.y - self.h / 2, self.y + self.h / 2)


@dataclass
class Scene:
    bounds: tuple                       # (xmin, xmax, ymin, ymax)
    start:  tuple                       # (x, y)
    disks:  list[Disk] = field(default_factory=list)
    rects:  list[Rect] = field(default_factory=list)

    def reach_disks(self):  return [d for d in self.disks if d.kind == "reach"]
    def avoid_disks(self):  return [d for d in self.disks if d.kind != "reach"]

    def obstacles_for_planner(self) -> list:
        """Export every non-reach geometry in the dict format that
        `planner.plan_waypoints(obstacles=...)` expects.  This is what
        threads procedurally-placed walls, furniture, and random
        circles through to the robustness path planner so the
        trajectory actually goes *around* them."""
        out: list = []
        for d in self.disks:
            if d.kind == "reach":
                continue                                # never an obstacle
            out.append({"kind": "circle",
                        "x": d.x, "y": d.y, "r": d.r})
        for r in self.rects:                            # walls + furniture
            out.append({"kind": "rect",
                        "x": r.x, "y": r.y,
                        "w": r.w, "h": r.h})
        return out


# --------------------------- collision helpers ---------------------------

def _disk_disk_overlap(x, y, r, others, pad=0.2) -> bool:
    for o in others:
        if math.hypot(x - o.x, y - o.y) < r + o.r + pad:
            return True
    return False


def _disk_rect_overlap(x, y, r, rects, pad=0.2) -> bool:
    for q in rects:
        x0, x1, y0, y1 = q.aabb()
        cx = max(x0 - pad, min(x, x1 + pad))
        cy = max(y0 - pad, min(y, y1 + pad))
        if math.hypot(x - cx, y - cy) < r:
            return True
    return False


def _rect_rect_overlap(rx, ry, rw, rh, rects, pad=0.2) -> bool:
    for q in rects:
        if abs(rx - q.x) * 2 < rw + q.w + pad and \
           abs(ry - q.y) * 2 < rh + q.h + pad:
            return True
    return False


# --------------------------- complex map generation ---------------------------

def _gap_aligned_with_atoms(running_min: float, running_max: float,
                              perp_axis: str, wall_perp: float,
                              reach_atoms: list,
                              avoid_atoms: list,
                              pad: float, default_gap: float,
                              rng: random.Random) -> tuple:
    """Compute (gap_centre, gap_width) for a wall.

    A wall is described by:
      perp_axis = "x"  -> vertical wall at x=wall_perp, running along Y
                          axis from running_min..running_max
      perp_axis = "y"  -> horizontal wall at y=wall_perp, running along X
                          axis from running_min..running_max

    Any reach atom whose perpendicular distance to the wall is smaller
    than `r + pad` forces the gap to enclose its running-axis coordinate.
    Avoid atoms create the opposite constraint (the gap, and thus the
    only crossing, should NOT pass right through an avoid disk).  We
    grow the gap to cover all such reach atoms.
    """
    def _coords(a):
        return (a[0], a[1], a[2])

    affected = []        # list of (running_coord, atom_radius) for reach atoms on the wall
    for (ax, ay, ar) in reach_atoms:
        perp = (ax if perp_axis == "x" else ay)
        running = (ay if perp_axis == "x" else ax)
        if abs(perp - wall_perp) < ar + pad:
            affected.append((running, ar))

    if not affected:
        gap = default_gap
        half = gap / 2
        return (rng.uniform(running_min + 0.5 + half,
                             running_max - 0.5 - half),
                gap)

    # Make gap span every affected reach atom + pad.
    lo = min(r - rr - pad for (r, rr) in affected)
    hi = max(r + rr + pad for (r, rr) in affected)
    gap = max(default_gap, hi - lo)
    centre = (lo + hi) / 2
    half = gap / 2
    centre = max(running_min + 0.5 + half,
                 min(running_max - 0.5 - half, centre))
    return centre, gap


def _add_walls_with_gaps(scene: Scene, rng: random.Random):
    """Drop two interior walls forming an L-shape with a passable gap each.

    Wall gaps are auto-aligned to the case's reach atoms so we never
    fence a goal off behind a wall.
    """
    xmin, xmax, ymin, ymax = scene.bounds
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    thick = 0.2
    default_gap = 1.2
    pad = 0.4

    reach_atoms = [(d.x, d.y, d.r) for d in scene.reach_disks()]
    avoid_atoms = [(d.x, d.y, d.r) for d in scene.disks if d.kind == "avoid"]

    # ---- vertical wall at x = cx (runs along Y) ----------------------------
    gap_y, gap_v = _gap_aligned_with_atoms(
        ymin, ymax, perp_axis="x", wall_perp=cx,
        reach_atoms=reach_atoms, avoid_atoms=avoid_atoms,
        pad=pad, default_gap=default_gap, rng=rng,
    )
    seg_below_h = max(0.0, (gap_y - gap_v / 2) - ymin)
    seg_above_h = max(0.0, ymax - (gap_y + gap_v / 2))
    if seg_below_h > 0.5:
        scene.rects.append(Rect(cx, ymin + seg_below_h / 2,
                                thick, seg_below_h,
                                kind="wall", name="wall_v_below"))
    if seg_above_h > 0.5:
        scene.rects.append(Rect(cx, ymax - seg_above_h / 2,
                                thick, seg_above_h,
                                kind="wall", name="wall_v_above"))

    # ---- horizontal wall at y = cy on right half-plane (runs along X) ------
    gap_x, gap_h = _gap_aligned_with_atoms(
        cx, xmax, perp_axis="y", wall_perp=cy,
        reach_atoms=reach_atoms, avoid_atoms=avoid_atoms,
        pad=pad, default_gap=default_gap, rng=rng,
    )
    seg_left_w  = max(0.0, (gap_x - gap_h / 2) - cx)
    seg_right_w = max(0.0, xmax - (gap_x + gap_h / 2))
    if seg_left_w > 0.5:
        scene.rects.append(Rect(cx + seg_left_w / 2, cy,
                                seg_left_w, thick,
                                kind="wall", name="wall_h_left"))
    if seg_right_w > 0.5:
        scene.rects.append(Rect(xmax - seg_right_w / 2, cy,
                                seg_right_w, thick,
                                kind="wall", name="wall_h_right"))


def _add_furniture(scene: Scene, rng: random.Random,
                   n: int = 4, max_tries: int = 200):
    xmin, xmax, ymin, ymax = scene.bounds
    placed, tries = 0, 0
    while placed < n and tries < max_tries:
        tries += 1
        rw = rng.uniform(0.5, 1.2)
        rh = rng.uniform(0.5, 1.2)
        rx = rng.uniform(xmin + rw / 2 + 0.2, xmax - rw / 2 - 0.2)
        ry = rng.uniform(ymin + rh / 2 + 0.2, ymax - rh / 2 - 0.2)
        # keep clear of disks; give reach atoms an extra 0.4 m halo so
        # the trajectory can actually stop inside them.
        too_close = False
        for d in scene.disks:
            pad = 0.8 if d.kind == "reach" else 0.4
            if math.hypot(rx - d.x, ry - d.y) < d.r + max(rw, rh) / 2 + pad:
                too_close = True
                break
        if too_close:
            continue
        if _rect_rect_overlap(rx, ry, rw, rh, scene.rects, pad=0.4):
            continue
        if math.hypot(rx - scene.start[0], ry - scene.start[1]) < 1.0:
            continue
        scene.rects.append(Rect(rx, ry, rw, rh,
                                kind="obstacle", name=f"furn_{placed}"))
        placed += 1


# --------------------------- top-level scene builder ---------------------------

def build_scene(case: dict, seed: int = 0) -> Scene:
    """Build a Scene from a case dict (see planner/case_examples.py).

    map_hint fields used:
      world_bounds          : [xmin, xmax, ymin, ymax]
      start                 : [x, y]
      style                 : "telograf_native" | "simple" | "complex"
                              (default: "simple")
                              - telograf_native / simple: only the STL atoms
                                (+ optional random circles).  This is what
                                TeLoGraF's encoder actually sees, so its raw
                                output is collision-free here.
                              - complex: adds walls + furniture that are NOT
                                in the STL -> TeLoGraF can't avoid them, used
                                for the A*-fallback demo cases.
      n_random_obstacles    : N random circles
      n_furniture           : N rectangular furniture pieces (complex only)
    """
    h = case["map_hint"]
    xmin, xmax, ymin, ymax = h["world_bounds"]
    scene = Scene(bounds=(xmin, xmax, ymin, ymax),
                  start=tuple(h["start"]))
    # grounded atoms always become reach / avoid disks
    for name, g in case["grounding"].items():
        scene.disks.append(Disk(x=g["x"], y=g["y"], r=g["r"],
                                kind=g.get("kind", "reach"),
                                name=name))

    rng = random.Random(seed)
    style = h.get("style", "simple")

    if style == "complex":
        _add_walls_with_gaps(scene, rng)
        _add_furniture(scene, rng,
                       n=int(h.get("n_furniture", 4)))

    # always honour n_random_obstacles for backwards compatibility
    n_circ = int(h.get("n_random_obstacles", 0))
    tries, placed = 0, 0
    while placed < n_circ and tries < 300:
        tries += 1
        rr = rng.uniform(0.3, 0.6)
        rx = rng.uniform(xmin + rr, xmax - rr)
        ry = rng.uniform(ymin + rr, ymax - rr)
        if _disk_disk_overlap(rx, ry, rr, scene.disks):           continue
        if _disk_rect_overlap(rx, ry, rr, scene.rects):           continue
        if math.hypot(rx - scene.start[0], ry - scene.start[1]) < 1.0: continue
        scene.disks.append(Disk(rx, ry, rr, kind="obstacle",
                                name=f"obs_{placed}"))
        placed += 1

    return scene


# Planner lives in planner/telograf_infer.py::plan_waypoints -- both the
# 2D demo (sim2d/run_demo.py) and the ROS trajectory follower call it.

# --------------------------- rendering ---------------------------

def render(scene: Scene, traj: list, title: str = "",
           save: str | None = None, animate: bool = True, ms: int = 60):
    """Render a scene + a trajectory.  Saves an image (if save=...) or a
    GIF (if save ends in .gif), or opens a window if save is None."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6, 6))
    xmin, xmax, ymin, ymax = scene.bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.set_title(title or "")
    ax.grid(alpha=0.3)

    for r in scene.rects:
        color = "#3a3a3a" if r.kind == "wall" else "#a0a0a0"
        x0, _, y0, _ = r.aabb()
        ax.add_patch(mp.Rectangle((x0, y0), r.w, r.h,
                                  facecolor=color,
                                  edgecolor="#1a1a1a",
                                  alpha=0.85 if r.kind == "wall" else 0.7,
                                  linewidth=1.2))
    for d in scene.disks:
        color = {"reach": "#2ecc71",
                 "avoid": "#e74c3c",
                 "obstacle": "#7f8c8d"}.get(d.kind, "#7f8c8d")
        ax.add_patch(mp.Circle((d.x, d.y), d.r, color=color, alpha=0.45))
        ax.text(d.x, d.y, d.name, ha="center", va="center", fontsize=8)

    ax.plot(*scene.start, marker="s", color="black", markersize=10,
            label="start")

    if not animate or save and save.lower().endswith((".png", ".pdf", ".svg")):
        xs, ys = zip(*traj)
        ax.plot(xs, ys, "-", lw=2, color="#3498db", label="trajectory")
        ax.plot(xs[-1], ys[-1], marker="*", color="#f39c12", markersize=14,
                label="end")
        ax.legend(loc="upper left", fontsize=8)
        if save: plt.savefig(save, dpi=120, bbox_inches="tight")
        else:    plt.show()
        plt.close(fig); return

    line, = ax.plot([], [], "-", lw=2, color="#3498db", label="trajectory")
    head, = ax.plot([], [], marker="o", color="#e67e22", markersize=8)
    ax.legend(loc="upper left", fontsize=8)

    def update(i):
        if i == 0:
            line.set_data([], []); head.set_data([], []); return line, head
        xs, ys = zip(*traj[:i + 1])
        line.set_data(xs, ys); head.set_data([xs[-1]], [ys[-1]])
        return line, head

    anim = FuncAnimation(fig, update, frames=len(traj),
                         interval=ms, blit=True, repeat=False)
    if save:
        anim.save(save, writer=PillowWriter(fps=int(1000 / ms)))
    else:
        plt.show()
    plt.close(fig)
