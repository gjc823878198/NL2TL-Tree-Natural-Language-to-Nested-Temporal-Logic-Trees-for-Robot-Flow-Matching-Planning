"""
Publication figure: three TeLoGraF flow-matching plans on one cluttered map.

One wide figure, three panels (a)(b)(c) -- each a different NL task on the SAME
complex map -- with the planned trajectory coloured by TIME (rainbow), the NL
sentence and STL time-window annotated, and a single shared rainbow colorbar on
the right.  All three plans use TeLoGraF (flow matching + STLCG guidance); the
extra clutter is fed to the planner as obstacles (sensor-grounded keep-safe),
not as STL graph nodes, so the frozen GNN stays in-distribution.

    python3 outputs/paper/paper_figure_telograf.py
    -> telograf_cases.{png,pdf}   (this script lives NEXT TO its output, so it is
       self-evident the figure is produced by code, not hand-drawn)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable

# this file now lives at code/outputs/paper/, so the code root is 2 levels up
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from planner import get_case, plan_waypoints                       # noqa: E402
from keep_safe import keep_safe_robustness                          # noqa: E402

BOUNDS = (-4.2, 4.2, -4.2, 4.2)
CMAP = "rainbow"

# --- one shared, more-complex map: extra clutter on top of each case's own
#     avoid atoms (placed to leave navigable corridors to every goal). ---
EXTRA_OBSTACLES = [
    {"kind": "circle", "x": -1.8, "y": -0.4, "r": 0.40},
    {"kind": "circle", "x":  0.6, "y":  1.1, "r": 0.45},
    {"kind": "circle", "x":  1.9, "y": -2.1, "r": 0.40},
    {"kind": "circle", "x": -0.7, "y":  1.9, "r": 0.38},
    {"kind": "rect",   "x":  3.4, "y": -3.1, "w": 1.0, "h": 0.6},
]

# three TeLoGraF cases (panels a,b,c)
PANELS = [
    ("reach_within_T",  "(a)"),
    ("reach_avoid",     "(b)"),
    ("reach_goal_north","(c)"),
]


def _finally_window(node):
    """First finally/F interval found in the tree -> (t1, t2) or None."""
    if isinstance(node, dict):
        if node.get("op") == "finally" and node.get("interval"):
            return tuple(node["interval"])
        for c in node.get("children") or []:
            w = _finally_window(c)
            if w:
                return w
    return None


def _avoid_disks(case):
    return [{"kind": "circle", "x": g["x"], "y": g["y"], "r": g["r"]}
            for g in case["grounding"].values() if g.get("kind") == "avoid"]


def _reach_disks(case):
    return [(g["x"], g["y"], g["r"]) for g in case["grounding"].values()
            if g.get("kind", "reach") == "reach"]


def _wrap(s, width=46):
    import textwrap
    return "\n".join(textwrap.wrap(s, width))


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 10,
        "mathtext.fontset": "cm",
    })

    # ---- plan all three with TeLoGraF on the cluttered map ----
    data = []
    for cid, tag in PANELS:
        case = get_case(cid)
        traj = plan_waypoints(case, n_steps=120, backend="telograf",
                              obstacles=EXTRA_OBSTACLES)
        obstacles = _avoid_disks(case) + EXTRA_OBSTACLES
        rho = keep_safe_robustness(obstacles, traj)
        data.append({
            "tag": tag, "id": cid, "nl": case["nl"],
            "window": _finally_window(case["tree"]),
            "reaches": _reach_disks(case),
            "obstacles": obstacles, "start": tuple(case["map_hint"]["start"]),
            "traj": np.asarray(traj, float), "rho": rho,
        })

    # ---- figure: 3 panels + a slim shared colorbar on the right ----
    fig = plt.figure(figsize=(16.0, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.16)
    norm = plt.Normalize(0.0, 1.0)

    for k, d in enumerate(data):
        ax = fig.add_subplot(gs[0, k])
        ax.set_aspect("equal")
        ax.set_xlim(BOUNDS[0], BOUNDS[1])
        ax.set_ylim(BOUNDS[2], BOUNDS[3])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#999"); sp.set_linewidth(0.8)

        # obstacles (keep-safe / unsafe region) -- soft grey
        for o in d["obstacles"]:
            if o.get("kind") == "rect":
                ax.add_patch(mp.Rectangle(
                    (o["x"] - o["w"] / 2, o["y"] - o["h"] / 2), o["w"], o["h"],
                    facecolor="#5b6675", edgecolor="#3a4250", alpha=0.55, lw=0.8))
            else:
                ax.add_patch(mp.Circle((o["x"], o["y"]), o["r"],
                                       facecolor="#5b6675", edgecolor="#3a4250",
                                       alpha=0.55, lw=0.8))

        # goal disk(s) + star
        for (gx, gy, gr) in d["reaches"]:
            ax.add_patch(mp.Circle((gx, gy), gr, facecolor="#f4d35e",
                                   edgecolor="#caa83a", alpha=0.55, lw=1.0))
            ax.plot(gx, gy, marker="*", ms=16, color="#b8860b",
                    markeredgecolor="k", markeredgewidth=0.5, zorder=6)

        # start
        sx, sy = d["start"]
        ax.plot(sx, sy, marker="s", ms=10, color="k", zorder=6)
        ax.annotate("start", (sx, sy), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)

        # trajectory coloured by time (rainbow)
        tr = d["traj"]
        pts = tr.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        t = np.linspace(0.0, 1.0, len(segs))
        lc = LineCollection(segs, cmap=CMAP, norm=norm, linewidth=3.0,
                            capstyle="round", zorder=5)
        lc.set_array(t)
        ax.add_collection(lc)

        # title + NL/time-window annotation
        t1, t2 = d["window"] if d["window"] else ("?", "?")
        ax.set_title(f"{d['tag']} {d['id']}   "
                     r"$\mathbf{TeLoGraF}$ (flow $+$ STLCG)", pad=8)
        caption = (f"NL: “{_wrap(d['nl'])}”\n"
                   rf"STL: $F_{{[{t1},{t2}]}}(\mathrm{{reach}})\ \wedge\ "
                   rf"G\,\neg\,\mathrm{{unsafe}}$    "
                   rf"$\rho(G\neg\mathrm{{unsafe}})={d['rho']:+.2f}$ m")
        ax.text(0.5, -0.13, caption, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#cccccc"))

    # shared rainbow colorbar = time
    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label("normalized time   (start $\\rightarrow$ goal)", fontsize=10)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle("TeLoGraF flow-matching trajectories on a cluttered map "
                 "(trajectory colour = time)", fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.01, right=0.95, top=0.90, bottom=0.20)

    out_dir = ROOT / "outputs" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"telograf_cases.{ext}", dpi=300,
                    bbox_inches="tight")
    print("wrote", out_dir / "telograf_cases.png")
    for d in data:
        print(f"  {d['id']:18s} window={d['window']} "
              f"end={tuple(round(v,2) for v in d['traj'][-1])} "
              f"rho_keepsafe={d['rho']:+.2f}")


if __name__ == "__main__":
    main()
