"""
Figure 1 (architecture style, after Style_Reference/Figure1_Style_Reference.png):
inputs on the left -> a tinted central "frozen pipeline" block holding the three
stages (parse / render / flow-matching planner with a denoising row) -> output
on the right -> a deployment strip along the bottom. Every paper mechanism is a
colour-coded tag on the stage it acts on. Code-generated (matplotlib), saved
next to the other paper figures.

    python3 outputs/paper/figure_pipeline.py   ->  pipeline.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parent
# palette (muted, academic)
BG      = "#eaf2fb"; BG_E   = "#9bb8d6"      # central method block
ST      = "#dfe9f5"; ST_E   = "#2c3e6b"      # stage boxes
IN      = "#e7efe4"; IN_E   = "#4a7a52"      # inputs
OUTc    = "#f3e7df"; OUT_E  = "#b07a3a"      # output
MECH    = "#fff4d6"; MECH_E = "#b8860b"      # mechanism tags
ALT     = "#fbe2df"; ALT_E  = "#b03a2e"      # A* fallback
GATE    = "#e2f0e6"; GATE_E = "#1e7a46"      # self-check
DEP     = "#ededf3"; DEP_E  = "#5b5b78"      # deployment


def main():
    # Larger fonts throughout: the figure is placed at \textwidth (2-col span),
    # so it is downscaled ~2x; small labels must start big to stay legible.
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(15.5, 6.6))
    ax.set_xlim(0, 162); ax.set_ylim(13, 82); ax.axis("off")

    def box(x, y, w, h, t, fc, ec, fs=13.5, bold=True, z=3):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.3,rounding_size=1.3",
                     lw=1.4, facecolor=fc, edgecolor=ec, zorder=z))
        ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#16213a", zorder=z+1)

    def arr(x0, y0, x1, y1, ec="#2c3e6b", lw=1.9, style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                     mutation_scale=14, lw=lw, color=ec, zorder=2))

    def tag(xc, ytop, lines, w=29):
        h = 2.7*len(lines) + 1.8
        box(xc - w/2, ytop - h, w, h, "\n".join(lines), MECH, MECH_E, fs=11.0,
            bold=True)

    # ---- banner ----
    ax.text(81, 78.5, "NL2TL-Tree: training-free   "
            "(LLM and flow-matching planner are FROZEN — no fine-tuning)",
            ha="center", fontsize=17.5, fontweight="bold", color="#2c3e6b")

    # ---- inputs (left) ---- (centred on ymid so its arrow is horizontal)
    box(2, 37, 20, 13, "", IN, IN_E, z=3)
    ax.text(12, 46.5, "Natural-language\ntask  (phone)", ha="center",
            va="center", fontsize=12.5, fontweight="bold", color="#16213a", zorder=4)
    ax.text(12, 40.8, "“visit A, then B;\nalways keep safe”", ha="center",
            va="center", fontsize=10.5, style="italic", fontweight="bold",
            color="#33513a", zorder=4)

    # ---- central frozen pipeline block ----
    ax.add_patch(FancyBboxPatch((26, 19), 104, 50,
                 boxstyle="round,pad=0.4,rounding_size=2.0", lw=1.4,
                 facecolor=BG, edgecolor=BG_E, zorder=1))
    ax.text(78, 65.8, "Training-free pipeline  (nothing trained)", ha="center",
            fontsize=14.5, fontweight="bold", color="#2c3e6b")

    sy, sh = 37, 13
    ymid = sy + sh/2
    # stage 1: parse
    box(29, sy, 27, sh, "(1) Frozen LLM parse\n$\\rightarrow$ nested STL tree",
        ST, ST_E, fs=13.0, bold=True)
    tag(42.5, sy-4.2, ["self-consistency vote (#1)",
                       "round-trip scope refine (#4)",
                       "robustness-feedback self-correct"])
    # stage 2: render
    box(60, sy, 24, sh, "(2) Render robustness\nSTL $\\varphi$, diff.\\ $\\rho$",
        ST, ST_E, fs=13.0, bold=True)
    tag(72, sy-4.2, ["operator norm.: $\\rightarrow,\\leftrightarrow\\Rightarrow$ basis",
                     "sensor-grounded  $G\\,\\neg$unsafe"])
    # stage 3: flow-matching planner (with denoising row)
    box(88, sy, 39, sh, "", ST, ST_E, z=3)
    ax.text(107.5, sy+sh-2.4, "(3) Frozen flow-matching planner (TeLoGraF)",
            ha="center", fontsize=12.5, fontweight="bold", color="#16213a", zorder=5)
    # GNN encoder chip
    box(90, sy+2.7, 9.5, 5.3, "GNN\nencoder", "#d7e3f2", ST_E, fs=9.6, z=5)
    # denoising / flow row x_T -> x_0
    xs = [104, 110, 116, 122]; yrow = sy+5.3
    labels = ["$x_T$", "$x_t$", "$\\cdots$", "$x_0$"]
    for i, (xx, lb) in enumerate(zip(xs, labels)):
        ax.add_patch(Circle((xx, yrow), 2.0, facecolor="#cfe0d2",
                     edgecolor=GATE_E, lw=1.1, zorder=5))
        ax.text(xx, yrow, lb, ha="center", va="center", fontsize=10.5,
                fontweight="bold", zorder=6)
        if i < len(xs)-1:
            arr(xx+2.0, yrow, xs[i+1]-2.0, yrow, ec=GATE_E, lw=1.1)
    arr(99.5, yrow, 102.0, yrow, ec=ST_E, lw=1.1)
    ax.text(113, sy+1.5, "flow sampling $+$ STLCG $\\nabla\\rho$ guidance",
            ha="center", fontsize=10.2, fontweight="bold", color="#16213a", zorder=6)
    tag(107.5, sy-4.2, ["nested-tree decomposition",
                        "online $\\rho$ monitor $+$ replan"], w=35)

    # inter-stage arrows
    arr(56, ymid, 60, ymid); arr(84, ymid, 88, ymid)

    # feasibility self-check -> A* fallback (above stage 3)
    box(95, 55, 25, 5.8, "feasibility self-check (best-of-$N$)", GATE, GATE_E, fs=10.8, z=4)
    arr(107.5, sy+sh, 107.5, 55, ec=GATE_E, lw=1.1)
    box(122.5, 55, 6.8, 5.8, "A*", ALT, ALT_E, fs=11.5, bold=True, z=4)
    arr(120, 57.9, 122.5, 57.9, ec=ALT_E, lw=1.2)
    ax.text(121, 62.4, "OOD", ha="center", fontsize=9.6, fontweight="bold",
            color="#b03a2e")

    # input -> pipeline (horizontal, level with the inter-stage arrows)
    arr(22, ymid, 29, ymid)

    # ---- output (right) ---- (centred on ymid so its arrow is horizontal)
    box(133, 36, 26, 15, "", OUTc, OUT_E, z=3)
    ax.text(146, 47, "Output", ha="center", fontsize=13.0, fontweight="bold",
            color="#7a4d1c", zorder=4)
    ax.text(146, 41.6, "STL-satisfying\ntrajectory $\\tau$\n"
            "$\\rho(G\\neg$unsafe$)>0$  (certified)", ha="center", va="center",
            fontsize=11.0, fontweight="bold", zorder=4)
    arr(127, ymid, 133, ymid)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.99, bottom=0.01)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"pipeline.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", OUT / "pipeline.png")


if __name__ == "__main__":
    main()
