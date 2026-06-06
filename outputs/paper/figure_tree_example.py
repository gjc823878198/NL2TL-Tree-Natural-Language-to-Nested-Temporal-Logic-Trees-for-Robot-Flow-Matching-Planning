"""Figure 2 (fig:example): the interpretable intermediate representation, as a
CRISP VECTOR figure that replaces the raster screenshot of the inspection tool.

Left: the parsed nested STL tree (vector nodes).  Right: the NL instruction, the
AST as a compact 4-line PSEUDO-CODE (instead of the 25-line JSON screenshot, which
turned to jaggies when scaled), and the round-trip STL string used to verify.

    python3 outputs/paper/figure_tree_example.py  ->  tree_example.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
IMPLY = "#d9c8f0"; IMPLY_E = "#7a5bbf"      # purple
ANDc = "#d6d6d6"; AND_E = "#7a7a7a"         # grey
ATOM = "#fce8a8"; ATOM_E = "#caa83a"        # yellow
INK = "#16213a"


def main():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig = plt.figure(figsize=(7.4, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.04)

    # ---------------- left: the parsed nested STL tree ----------------
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def node(x, y, w, h, t, fc, ec, fs=9.5):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.06,rounding_size=0.28", lw=1.3,
                     facecolor=fc, edgecolor=ec, zorder=3))
        ax.text(x, y, t, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=INK, zorder=4)

    def edge(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=12, lw=1.3, color="#444", zorder=2,
                     shrinkA=2, shrinkB=2))

    # positions
    P = {"imply": (5.0, 9.0), "case1": (2.1, 5.4), "and": (6.6, 5.4),
         "mv": (3.8, 1.5), "fin": (7.9, 1.5)}
    edge(*P["imply"], *P["case1"]); edge(*P["imply"], *P["and"])
    edge(*P["and"], *P["mv"]); edge(*P["and"], *P["fin"])
    node(*P["imply"], 3.4, 1.5, r"$\rightarrow$ imply", IMPLY, IMPLY_E, fs=10)
    node(*P["case1"], 2.9, 1.4, "case1", ATOM, ATOM_E)
    node(*P["and"], 2.7, 1.4, r"$\wedge$ and", ANDc, AND_E, fs=10)
    node(*P["mv"], 3.8, 1.4, "move_to_place2", ATOM, ATOM_E, fs=8.5)
    node(*P["fin"], 3.4, 1.4, "finish_task3", ATOM, ATOM_E, fs=8.5)

    # ---------------- right: NL + pseudo-code AST + round-trip STL ----------
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    ax.text(0.1, 9.6, "Natural language", fontsize=10.5, fontweight="bold",
            color="#2c3e6b")
    ax.text(0.1, 8.6, "“Always, whenever case1 happens, the\n"
            "robot must move to place2 and finish task3.”",
            fontsize=9, style="italic", va="top", color=INK)

    ax.text(0.1, 6.4, "AST (pseudo-code)", fontsize=10.5, fontweight="bold",
            color="#2c3e6b")
    ax.text(0.1, 5.5, "imply(\n"
            "    atom case1,\n"
            "    and(atom move_to_place2,\n"
            "        atom finish_task3))",
            fontsize=9.3, family="monospace", va="top", color="#1b3a2b")

    ax.text(0.1, 1.85, "Round-trip STL", fontsize=10.5, fontweight="bold",
            color="#2c3e6b")
    ax.text(0.1, 1.0, r"(case1) $\rightarrow$ (move_to_place2 $\wedge$ "
            "finish_task3)", fontsize=9.3, family="monospace", va="center",
            color="#111")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tree_example.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.04)
    print("wrote", OUT / "tree_example.png")


if __name__ == "__main__":
    main()
