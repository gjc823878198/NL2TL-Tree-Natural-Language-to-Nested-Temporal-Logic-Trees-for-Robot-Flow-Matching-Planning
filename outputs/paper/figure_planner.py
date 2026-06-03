"""
Figure for Sec.~2.2 (render to differentiable robustness) + Sec.~2.3 (frozen
flow-matching planner), TeLoGraF-informed.  Three linked panels:

  (a) nested STL tree, with reach (blue) / avoid (grey) atoms;
  (b) frozen GNN encoder -> conditioning, then flow-matching denoising
      x_T -> x_0 steered by the differentiable STL-robustness gradient
      (nabla rho); the three training-free mechanisms tagged underneath;
  (c) a REAL TeLoGraF flow-matching plan that slaloms through a dense obstacle
      cluster (not a hand-drawn curve), coloured by time, with the
      sensor-grounded keep-safe margin rho(G not unsafe) marked.

Code-generated (matplotlib), saved with the other paper figures.
    python3 outputs/paper/figure_planner.py  ->  planner.{png,pdf}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.collections import LineCollection

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from planner import plan_waypoints                                   # noqa: E402
from keep_safe import keep_safe_robustness                           # noqa: E402

REACH = "#3a6ea5"; AVOID = "#5b6675"; MECH = "#fff4d6"; MECH_E = "#b8860b"
GATE_E = "#1e7a46"; BOX = "#dfe9f5"; BOX_E = "#2c3e6b"

# ---- panel (c): real weave/slalom plan through an obstacle cluster ----
WIN = (20, 55)
CENTER = [(0.0, -1.3, 0.45), (0.0, 0.4, 0.45), (0.0, 2.1, 0.45)]
SIDE = [(-1.9, -0.5, 0.45), (1.9, -0.5, 0.45),
        (-1.9, 1.2, 0.45), (1.9, 1.2, 0.45)]
START = (0.0, -3.3); GOAL = (0.0, 3.3, 0.5)


def _weave_case():
    grounding = {"prop_g": {"kind": "reach", "x": GOAL[0], "y": GOAL[1],
                            "z": 0.0, "r": GOAL[2]}}
    children = [{"op": "finally", "interval": list(WIN),
                 "children": [{"op": "atom", "name": "prop_g"}]}]
    for i, (x, y, r) in enumerate(CENTER, 1):
        nm = f"prop_o{i}"
        grounding[nm] = {"kind": "avoid", "x": x, "y": y, "z": 0.0, "r": r}
        children.append({"op": "globally", "interval": None, "children": [
            {"op": "not", "interval": None,
             "children": [{"op": "atom", "name": nm}]}]})
    return {"id": "weave_through", "role": "telograf",
            "tree": {"op": "and", "interval": None, "children": children},
            "grounding": grounding,
            "map_hint": {"world_bounds": [-3, 3, -3.5, 3.5],
                         "start": list(START), "style": "telograf_native",
                         "telograf_samples": 64, "gz_world": "maze"}}


def main():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 13.5})
    fig = plt.figure(figsize=(16.8, 4.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 1.12], wspace=0.10)

    # ---------- (a) nested STL tree ----------
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    nodes = {"and": (5, 9, r"$\wedge$", "#d7e3f2"),
             "F":   (2.4, 6.4, r"$F_{[a,b]}$", "#d7e3f2"),
             "g":   (2.4, 3.6, "reach\n$g$", REACH),
             "G1":  (5, 6.4, r"$G$", "#d7e3f2"),
             "n1":  (5, 4.2, r"$\neg$", "#eee"),
             "o":   (5, 2.0, "avoid\n$o$", AVOID),
             "Gks": (7.8, 6.4, r"$G$", "#d7e3f2"),
             "nks": (7.8, 4.2, r"$\neg$", "#eee"),
             "u":   (7.8, 2.0, "unsafe\n(sensed)", AVOID)}
    edges = [("and", "F"), ("F", "g"), ("and", "G1"), ("G1", "n1"),
             ("n1", "o"), ("and", "Gks"), ("Gks", "nks"), ("nks", "u")]
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]],
                "-", color="#888", lw=1.3, zorder=1)
    for k, (x, y, lb, c) in nodes.items():
        ax.add_patch(Circle((x, y), 0.66, facecolor=c, edgecolor="#333",
                            lw=1.2, zorder=2))
        tc = "white" if c in (REACH, AVOID) else "#16213a"
        ax.text(x, y, lb, ha="center", va="center", fontsize=11,
                fontweight="bold", color=tc, zorder=3)
    ax.set_title("(a) Nested STL spec", fontsize=15, fontweight="bold")
    ax.text(5, 0.4, r"keep-safe $=G\,\neg\,$unsafe, grounded by sensed disks",
            ha="center", fontsize=10.5, fontweight="bold", color="#444")

    # ---------- (b) GNN + flow matching + robustness guidance ----------
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    def box(x, y, w, h, t, fc, ec, fs=11, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.3,rounding_size=2", lw=1.4,
                     facecolor=fc, edgecolor=ec, zorder=3))
        ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=4)

    def arr(x0, y0, x1, y1, ec="#2c3e6b", lw=1.8, style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                     mutation_scale=14, lw=lw, color=ec, zorder=2))

    # encoder -> conditioning
    box(1, 66, 27, 16, "frozen GNN\nencoder", "#d7e3f2", BOX_E, fs=12,
        bold=True)
    ax.text(14.5, 60.5, r"$\Rightarrow$ conditioning $z$", ha="center",
            fontsize=11, fontweight="bold")
    arr(28, 74, 39, 74)
    # flow / denoising row
    xs = [47, 62, 77, 92]; yr = 74
    ax.text(69.5, 92, "flow matching (frozen)", ha="center", fontsize=12.5,
            fontweight="bold", color="#16213a")
    for i, (xx, lb) in enumerate(zip(xs, [r"$x_T$", r"$x_t$", r"$\cdots$", r"$x_0$"])):
        ax.add_patch(Circle((xx, yr), 5.6, facecolor="#cfe0d2",
                            edgecolor=GATE_E, lw=1.4, zorder=3))
        ax.text(xx, yr, lb, ha="center", va="center", fontsize=11.5,
                fontweight="bold", zorder=4)
        if i < len(xs)-1:
            arr(xx+5.6, yr, xs[i+1]-5.6, yr, ec=GATE_E, lw=1.4)
    # robustness-gradient guidance: box below, arrows up into the row
    box(38, 47, 58, 9, r"differentiable STL robustness $\nabla\rho$ (STLCG)"
        "\nsteers sampling", "#fbe2df", "#b03a2e", fs=11, bold=True)
    for xx in xs[:-1]:
        arr(xx, 56, xx, yr-5.8, ec="#b03a2e", lw=1.3, style="-|>")
    # three mechanism tags, evenly spaced, NO overlap (gaps of 3 units)
    tags = ["operator normalization\n$\\rightarrow,\\leftrightarrow\\Rightarrow$ basis",
            "nested-tree\ndecomposition",
            "feasibility self-check\n$\\to$ A* if OOD"]
    txs = [1, 35, 69]                       # 1-31, 35-65, 69-99  (gap 3)
    for x, t in zip(txs, tags):
        box(x, 24, 30, 13, t, MECH, MECH_E, fs=10.5, bold=True)
    ax.text(50, 15, r"$\rho(G\varphi)=\min_t\rho(\varphi,t),\;\;"
            r"\rho(F\varphi)=\max_t\rho(\varphi,t),\;\;\rho>0 \Rightarrow$ satisfied",
            ha="center", fontsize=11, fontweight="bold", color="#333")
    ax.set_title("(b) Frozen flow-matching planner $+$ robustness guidance",
                 fontsize=15, fontweight="bold")

    # ---------- (c) REAL weave plan through an obstacle cluster ----------
    ax = fig.add_subplot(gs[0, 2])
    ax.set_aspect("equal", adjustable="datalim")     # fill box -> aligns (a)(b)
    side_obs = [{"kind": "circle", "x": x, "y": y, "r": r} for (x, y, r) in SIDE]
    traj = np.asarray(plan_waypoints(_weave_case(), n_steps=140,
                                     backend="telograf", obstacles=side_obs),
                      float)
    all_obs = ([{"kind": "circle", "x": x, "y": y, "r": r}
                for (x, y, r) in CENTER] + side_obs)
    rho = keep_safe_robustness(all_obs, traj)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.8, 3.8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#999")
    for (x, y, r) in CENTER + SIDE:
        ax.add_patch(Circle((x, y), r, facecolor=AVOID, edgecolor="#3a4250",
                            alpha=0.55, lw=0.8))
    ax.add_patch(Circle(GOAL[:2], GOAL[2], facecolor="#f4d35e",
                        edgecolor="#caa83a", alpha=0.6, lw=1.0))
    ax.plot(GOAL[0], GOAL[1], marker="*", ms=18, color="#b8860b",
            markeredgecolor="k", markeredgewidth=0.5, zorder=6)
    ax.plot(START[0], START[1], marker="s", ms=10, color="k", zorder=6)
    pts = traj.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="rainbow", norm=plt.Normalize(0, 1),
                        linewidth=3.4, capstyle="round", zorder=5)
    lc.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection(lc)
    ax.annotate(rf"$\rho(G\neg$unsafe$)={rho:+.2f}$", (0.0, -3.3),
                textcoords="offset points", xytext=(10, -2), fontsize=11,
                fontweight="bold", color="#1e7a46")
    ax.text(0.02, 0.98, "colour $=$ time", transform=ax.transAxes, ha="left",
            va="top", fontsize=10, fontweight="bold", color="#555")
    ax.set_title("(c) Real flow plan slaloming a cluster",
                 fontsize=15, fontweight="bold")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.04)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"planner.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", OUT / "planner.png", "rho_keepsafe=%+.2f" % rho,
          "end=", tuple(round(v, 2) for v in traj[-1]))


if __name__ == "__main__":
    main()
