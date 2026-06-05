"""
Figure 1 (v2, NOT yet in the paper): the full NL2TL-Tree pipeline INCLUDING the
feasibility-feedback features --

  * a top forward band: phone NL (+ environment level) -> frozen LLM parse ->
    render to differentiable robustness -> frozen flow-matching planner -> output;
  * a middle "checks & feedback to the user" band:
      - PRE-FLIGHT feasibility gauge (A* + dynamics estimate, charged the
        predicted planning latency, vs the deadline -> likely / tight / unlikely),
      - best-of-N FEASIBILITY self-check (reach-shortfall > tau -> defer to A*,
        with a "why infeasible" visualization);
  * a bottom closed-loop band: MPPI tracking + online sound rho monitor ->
    stop / re-plan, with the planning-latency-aware deadline;
  * a feedback loop carrying the completability + the why-infeasible explanation
    back to the user.

Code-generated (matplotlib); saved next to the other paper figures but kept OUT
of the .tex for now.
    python3 outputs/paper/figure_pipeline_v2.py  ->  pipeline_v2.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parent
# palette
BG, BG_E = "#eaf2fb", "#9bb8d6"        # frozen pipeline block
ST, ST_E = "#dfe9f5", "#2c3e6b"        # stage boxes
IN, IN_E = "#e7efe4", "#4a7a52"        # user / input
OUTc, OUT_E = "#f3e7df", "#b07a3a"     # output
MECH, MECH_E = "#fff4d6", "#b8860b"    # mechanism tags
FEA, FEA_E = "#dff0ec", "#2a8a7a"      # feasibility / feedback (teal)
ALT, ALT_E = "#fbe2df", "#b03a2e"      # A* fallback
GATE, GATE_E = "#e2f0e6", "#1e7a46"    # flow denoising
DEP, DEP_E = "#ededf3", "#5b5b78"      # closed-loop deployment
FB = "#7a4dbf"                          # feedback arrows (purple)


def main():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(16, 8.8))
    ax.set_xlim(0, 162); ax.set_ylim(0, 96); ax.axis("off")

    def box(x, y, w, h, t, fc, ec, fs=12.5, bold=True, z=3):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.3,rounding_size=1.3", lw=1.4,
                     facecolor=fc, edgecolor=ec, zorder=z))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#16213a", zorder=z + 1)

    def arr(x0, y0, x1, y1, ec=ST_E, lw=1.9, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                     mutation_scale=15, lw=lw, color=ec, zorder=2,
                     linestyle=ls))

    def curved(x0, y0, x1, y1, rad, ec, lw=2.2, style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                     mutation_scale=16, lw=lw, color=ec, zorder=2,
                     connectionstyle=f"arc3,rad={rad}"))

    def tag(xc, ytop, lines, w=30):
        h = 2.6 * len(lines) + 1.8
        box(xc - w / 2, ytop - h, w, h, "\n".join(lines), MECH, MECH_E,
            fs=9.6, bold=True)

    # ---------- banner ----------
    ax.text(81, 92.5, "NL2TL-Tree: training-free pipeline   "
            "(frozen LLM $+$ frozen flow-matching planner; nothing fine-tuned)",
            ha="center", fontsize=17, fontweight="bold", color="#2c3e6b")

    ymid = 64                                       # main forward band centre
    # ---------- user / phone ----------
    box(2, 56, 22, 16, "", IN, IN_E, z=3)
    ax.text(13, 67.5, "Natural-language\ntask  (phone)", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="#16213a", zorder=4)
    ax.text(13, 61.0, "“visit A, then B;\nalways keep safe”", ha="center",
            va="center", fontsize=10, style="italic", fontweight="bold",
            color="#33513a", zorder=4)
    ax.text(13, 57.4, "+ env level: open/normal/complex", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color="#2a8a7a", zorder=4)

    # ---------- frozen pipeline block ----------
    ax.add_patch(FancyBboxPatch((27, 47), 105, 33,
                 boxstyle="round,pad=0.4,rounding_size=2.0", lw=1.4,
                 facecolor=BG, edgecolor=BG_E, zorder=1))
    ax.text(79, 77.3, "Training-free pipeline  (everything frozen)", ha="center",
            fontsize=13.5, fontweight="bold", color="#2c3e6b")

    sy, sh = 58, 13
    # stage 1
    box(30, sy, 26, sh, "(1) Frozen LLM parse\n$\\rightarrow$ nested STL tree",
        ST, ST_E, fs=12.5)
    tag(43, sy - 1.2, ["self-consistency vote",
                       "round-trip scope refine",
                       "robustness-feedback self-correct"], w=27)
    # stage 2
    box(60, sy, 23, sh, "(2) Render robustness\nSTL $\\varphi$, diff.\\ $\\rho$",
        ST, ST_E, fs=12.5)
    tag(71.5, sy - 1.2, ["operator norm. $\\rightarrow,\\leftrightarrow$ basis",
                         "sensor-grounded $G\\,\\neg$unsafe"], w=24)
    # stage 3 (flow planner with denoising row)
    box(87, sy, 42, sh, "", ST, ST_E, z=3)
    ax.text(108, sy + sh - 2.3, "(3) Flow-matching planner (TeLoGraF)",
            ha="center", fontsize=11.5, fontweight="bold", color="#16213a", zorder=5)
    box(89, sy + 2.6, 9.5, 5.3, "GNN\nencoder", "#d7e3f2", ST_E, fs=9.2, z=5)
    xs = [104, 110, 116, 122]; yrow = sy + 5.2
    for i, (xx, lb) in enumerate(zip(xs, ["$x_T$", "$x_t$", "$\\cdots$", "$x_0$"])):
        ax.add_patch(Circle((xx, yrow), 2.0, facecolor="#cfe0d2",
                     edgecolor=GATE_E, lw=1.1, zorder=5))
        ax.text(xx, yrow, lb, ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=6)
        if i < len(xs) - 1:
            arr(xx + 2.0, yrow, xs[i + 1] - 2.0, yrow, ec=GATE_E, lw=1.1)
    arr(99.0, yrow, 102.0, yrow, ec=ST_E, lw=1.1)
    ax.text(113, sy + 1.0, "flow sampling $+$ STLCG $\\nabla\\rho$ guidance",
            ha="center", fontsize=9.6, fontweight="bold", color="#16213a", zorder=6)
    tag(111, sy - 1.2, ["nested-tree decomposition",
                        "best-of-$N$ guided sampling"], w=32)

    # forward arrows
    arr(24, ymid, 30, ymid)
    arr(56, ymid, 60, ymid)
    arr(83, ymid, 87, ymid)

    # ---------- output ----------
    box(135, 56, 25, 16, "", OUTc, OUT_E, z=3)
    ax.text(147.5, 67, "Output", ha="center", fontsize=12.5, fontweight="bold",
            color="#7a4d1c", zorder=4)
    ax.text(147.5, 61.2, "STL-satisfying\ntrajectory $\\tau$\n"
            "$\\rho(G\\neg$unsafe$)>0$ (certified)", ha="center", va="center",
            fontsize=10, fontweight="bold", zorder=4)
    arr(129, ymid, 135, ymid)

    # ====================================================================
    # MIDDLE band: checks & feedback to the user
    # ====================================================================
    ax.text(60, 44.0, "Checks $+$ user feedback", ha="center",
            fontsize=12.5, fontweight="bold", color=FEA_E)

    # pre-flight feasibility gauge
    box(40, 28, 50, 12, "", FEA, FEA_E, z=3)
    ax.text(65, 37.3, "Pre-flight feasibility gauge  (A* $+$ dynamics)",
            ha="center", fontsize=11, fontweight="bold", color="#155", zorder=4)
    ax.text(65, 31.7, "A* path $\\times$ clutter $+$ predicted planning latency  "
            "vs. deadline\n$\\Rightarrow$ completability:  likely / tight / unlikely",
            ha="center", va="center", fontsize=9.6, fontweight="bold",
            color="#16213a", zorder=4)
    # from render down to the gauge
    arr(65, 49, 65, 40, ec=FEA_E, lw=1.6)

    # best-of-N self-check -> A* (OOD) with visualization
    box(95, 28, 38, 12, "", FEA, FEA_E, z=3)
    ax.text(114, 37.3, "Best-of-$N$ self-check", ha="center", fontsize=11,
            fontweight="bold", color="#155", zorder=4)
    ax.text(113, 31.7, "reach-shortfall $>\\tau$ ?\n"
            "$\\rightarrow$ visualize why $+$ defer", ha="center",
            va="center", fontsize=9.6, fontweight="bold", color="#16213a", zorder=4)
    box(126, 30.5, 6.5, 6.5, "A*", ALT, ALT_E, fs=11, z=4)
    ax.text(129.4, 38.4, "OOD", ha="center", fontsize=8.6, fontweight="bold",
            color=ALT_E)
    # from planner down to the self-check
    arr(110, 49, 110, 40, ec=FEA_E, lw=1.6)
    # A* fallback feeds the output
    curved(129.4, 37.0, 140, 56, 0.25, ALT_E, lw=1.8)

    # feedback loop: completability + why-infeasible -> back to the user
    curved(40, 33, 10, 56, -0.32, FB, lw=2.4)
    ax.text(17, 44.5, "feedback:\ncompletability\n$+$ why-infeasible",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color=FB,
            zorder=5)

    # ====================================================================
    # BOTTOM band: closed-loop execution
    # ====================================================================
    box(74, 6, 70, 12, "", DEP, DEP_E, z=3)
    ax.text(109, 15.3, "Closed-loop execution  (sim TurtleBot3)", ha="center",
            fontsize=11.5, fontweight="bold", color="#16213a", zorder=4)
    ax.text(109, 9.6, "MPPI tracking $\\sim$7 Hz  $+$  online sound $\\rho$ "
            "monitor $\\rightarrow$ stop / re-plan   |   "
            "planning-latency-aware deadline", ha="center", va="center",
            fontsize=9.6, fontweight="bold", color="#16213a", zorder=4)
    # output -> execution
    curved(147.5, 56, 138, 18, -0.2, DEP_E, lw=1.8)
    # execution -> re-plan back into the planner
    arr(92.5, 18.5, 92.5, 57, ec=DEP_E, lw=1.8)
    ax.text(90.5, 23, "re-plan", ha="right", va="center",
            fontsize=9.0, fontweight="bold", color=DEP_E, zorder=5)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.01)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"pipeline_v2.{ext}", dpi=200, bbox_inches="tight",
                    pad_inches=0.05)
    print("wrote", OUT / "pipeline_v2.png")


if __name__ == "__main__":
    main()
