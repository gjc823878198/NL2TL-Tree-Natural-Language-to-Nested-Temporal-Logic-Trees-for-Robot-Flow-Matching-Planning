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
    # two panels only: (a) tree + (b) planner.  The trajectory panel was dropped
    # -- the deployed timed-task trajectory is Fig.~\ref{fig:single} (telograf_
    # single), so a third trajectory panel here was redundant; two wider panels
    # leave more room for the labels (no internal squeeze).
    fig = plt.figure(figsize=(11.6, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.08)

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
    box(1, 64, 27, 20, "frozen GNN\nencoder", "#d7e3f2", BOX_E, fs=13,
        bold=True)
    ax.text(14.5, 58.5, r"$\Rightarrow$ conditioning $z$", ha="center",
            fontsize=12, fontweight="bold")
    arr(28, 74, 41.4, 74)
    # flow / denoising row
    xs = [47, 62, 77, 92]; yr = 74
    ax.text(69.5, 93, "flow matching (frozen)", ha="center", fontsize=13.5,
            fontweight="bold", color="#16213a")
    for i, (xx, lb) in enumerate(zip(xs, [r"$x_T$", r"$x_t$", r"$\cdots$", r"$x_0$"])):
        ax.add_patch(Circle((xx, yr), 5.6, facecolor="#cfe0d2",
                            edgecolor=GATE_E, lw=1.4, zorder=3))
        ax.text(xx, yr, lb, ha="center", va="center", fontsize=12.5,
                fontweight="bold", zorder=4)
        if i < len(xs)-1:
            arr(xx+5.6, yr, xs[i+1]-5.6, yr, ec=GATE_E, lw=1.4)
    # robustness-gradient guidance: box below, arrows up into the row
    box(33, 44, 65, 12, r"differentiable STL robustness $\nabla\rho$ (STLCG)"
        "\nsteers sampling", "#fbe2df", "#b03a2e", fs=12.5, bold=True)
    for xx in xs[:-1]:
        arr(xx, 56, xx, yr-5.6, ec="#b03a2e", lw=1.4, style="-|>")
    # three mechanism tags; the long-text boxes (1 and 3) get extra width so the
    # text does not touch the rounded border, with small even gaps.
    tags = ["operator normalization\n$\\rightarrow,\\leftrightarrow\\Rightarrow$ basis",
            "nested-tree\ndecomposition",
            "feasibility self-check\n$\\to$ A* if OOD"]
    tag_xw = [(0, 35), (37, 26), (65, 35)]   # (x, width): 0-35, 37-63, 65-100
    for (x, w), t in zip(tag_xw, tags):
        box(x, 22, w, 15, t, MECH, MECH_E, fs=11.5, bold=True)
    # render this formula with the STIX math fontset, whose \mathbf DOES bold
    # Greek (cm's does not) -- per-text, so the rest of the figure stays cm.
    ax.text(50, 10, r"$\mathbf{\rho(G\varphi)=\min_t\rho(\varphi,t),\;\;"
            r"\rho(F\varphi)=\max_t\rho(\varphi,t),\;\;\rho>0 \Rightarrow}$ "
            r"$\mathbf{satisfied}$",
            ha="center", fontsize=17, fontweight="bold", color="#111",
            math_fontfamily="stix")
    ax.set_title("(b) Frozen flow-matching planner $+$ robustness guidance",
                 fontsize=15, fontweight="bold")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.04)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"planner.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    print("wrote", OUT / "planner.png")


if __name__ == "__main__":
    main()
