"""
Case-study figure: a TeLoGraF flow-matching trajectory that THREADS THROUGH an
obstacle cluster (slalom), rather than detouring around the whole field.

The robot starts below a dense cluster and must reach a goal above it; three
central blockers force a left-right-left weave, while side blockers keep the
path inside the corridor (it cannot go around).  Trajectory colour = time
(rainbow); a colorbar on the right labels time.  Lives next to its output so it
is self-evident the figure is produced by code, not hand-drawn.

    python3 outputs/paper/figure_weave.py
    -> telograf_weave.{png,pdf}
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

ROOT = Path(__file__).resolve().parents[2]          # code/
sys.path.insert(0, str(ROOT))
from planner import plan_waypoints                                   # noqa: E402
from keep_safe import keep_safe_robustness                           # noqa: E402

BOUNDS = (-3.2, 3.2, -3.8, 3.8)
CMAP = "rainbow"
WINDOW = (20, 55)
NL = ("Eventually, within the time bound, reach the goal beyond the cluttered "
      "room, while always keeping safe from every obstacle.")

# three central blockers (STL avoid atoms, in-distribution for the GNN) ...
CENTER = [(0.0, -1.3, 0.45), (0.0, 0.4, 0.45), (0.0, 2.1, 0.45)]
# ... plus side blockers (fed as obstacles -> guidance) that confine the
# corridor so the only way through is to slalom between the central ones.
SIDE = [(-1.9, -0.5, 0.45), (1.9, -0.5, 0.45),
        (-1.9, 1.2, 0.45), (1.9, 1.2, 0.45)]
START = (0.0, -3.3)
GOAL = (0.0, 3.3, 0.5)


def _case():
    grounding = {"prop_g": {"kind": "reach", "x": GOAL[0], "y": GOAL[1],
                            "z": 0.0, "r": GOAL[2]}}
    children = [{"op": "finally", "interval": list(WINDOW),
                 "children": [{"op": "atom", "name": "prop_g"}]}]
    for i, (x, y, r) in enumerate(CENTER, 1):
        nm = f"prop_o{i}"
        grounding[nm] = {"kind": "avoid", "x": x, "y": y, "z": 0.0, "r": r}
        children.append({"op": "globally", "interval": None, "children": [
            {"op": "not", "interval": None,
             "children": [{"op": "atom", "name": nm}]}]})
    return {
        "id": "weave_through", "role": "telograf",
        "tree": {"op": "and", "interval": None, "children": children},
        "grounding": grounding,
        "map_hint": {"world_bounds": [-3, 3, -3.5, 3.5], "start": list(START),
                     "style": "telograf_native", "telograf_samples": 64,
                     "gz_world": "maze"},
    }


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "axes.titlesize": 13, "mathtext.fontset": "cm"})
    side_obs = [{"kind": "circle", "x": x, "y": y, "r": r} for (x, y, r) in SIDE]
    traj = np.asarray(plan_waypoints(_case(), n_steps=140, backend="telograf",
                                     obstacles=side_obs), float)
    all_obs = ([{"kind": "circle", "x": x, "y": y, "r": r} for (x, y, r) in CENTER]
               + side_obs)
    rho = keep_safe_robustness(all_obs, traj)

    fig = plt.figure(figsize=(6.4, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.045], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    norm = plt.Normalize(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlim(*BOUNDS[:2]); ax.set_ylim(*BOUNDS[2:])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#999"); sp.set_linewidth(0.8)

    for (x, y, r) in CENTER + SIDE:
        ax.add_patch(mp.Circle((x, y), r, facecolor="#5b6675",
                               edgecolor="#3a4250", alpha=0.55, lw=0.8))
    ax.add_patch(mp.Circle(GOAL[:2], GOAL[2], facecolor="#f4d35e",
                           edgecolor="#caa83a", alpha=0.55, lw=1.0))
    ax.plot(GOAL[0], GOAL[1], marker="*", ms=18, color="#b8860b",
            markeredgecolor="k", markeredgewidth=0.5, zorder=6)
    ax.plot(START[0], START[1], marker="s", ms=11, color="k", zorder=6)
    ax.annotate("start", START, textcoords="offset points", xytext=(8, -2),
                fontsize=9)
    ax.annotate("goal", GOAL[:2], textcoords="offset points", xytext=(8, 4),
                fontsize=9)

    pts = traj.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=CMAP, norm=norm, linewidth=3.2,
                        capstyle="round", zorder=5)
    lc.set_array(np.linspace(0.0, 1.0, len(segs)))
    ax.add_collection(lc)

    t1, t2 = WINDOW
    import textwrap
    cap = ("NL: “" + "\n".join(textwrap.wrap(NL, 52)) + "”\n"
           rf"STL: $F_{{[{t1},{t2}]}}(\mathrm{{reach}})\ \wedge\ "
           rf"G\,\neg\,\mathrm{{unsafe}}$    "
           rf"$\rho(G\neg\mathrm{{unsafe}})={rho:+.2f}$ m  "
           r"(TeLoGraF, flow$+$STLCG)")
    ax.set_title("Flow-matching plan threading an obstacle cluster")
    ax.text(0.5, -0.10, cap, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#cccccc"))

    cax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label("normalized time (start $\\rightarrow$ goal)", fontsize=9)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.subplots_adjust(left=0.02, right=0.9, top=0.93, bottom=0.22)
    out = ROOT / "outputs" / "paper"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"telograf_weave.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", out / "telograf_weave.png")
    print(f"  end={tuple(round(v,2) for v in traj[-1])}  "
          f"rho_keepsafe={rho:+.2f}  n={len(traj)}")


if __name__ == "__main__":
    main()
