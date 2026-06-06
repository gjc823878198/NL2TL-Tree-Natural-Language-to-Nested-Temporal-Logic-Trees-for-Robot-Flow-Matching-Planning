"""Clean flat black device silhouettes (transparent PNG) to flank the phone in
Figure 1's Natural-language task box: a smartwatch + a smart speaker."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Ellipse, Circle
from pathlib import Path

OUT = Path("/tmp/devgen"); OUT.mkdir(exist_ok=True)
BLACK = "#1a1a1a"


def fig():
    f = plt.figure(figsize=(2.56, 2.56), dpi=100)
    ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")
    return f, ax


def watch():
    f, ax = fig()
    # straps (tapered), black
    ax.add_patch(Polygon([(0.39, 0.70), (0.61, 0.70), (0.585, 0.97), (0.415, 0.97)],
                         closed=True, color=BLACK))
    ax.add_patch(Polygon([(0.415, 0.03), (0.585, 0.03), (0.61, 0.30), (0.39, 0.30)],
                         closed=True, color=BLACK))
    # crown
    ax.add_patch(Rectangle((0.705, 0.46), 0.055, 0.08, color=BLACK))
    # case (rounded square)
    ax.add_patch(FancyBboxPatch((0.30, 0.28), 0.40, 0.44,
                 boxstyle="round,pad=0,rounding_size=0.10", color=BLACK))
    # screen (white inset)
    ax.add_patch(FancyBboxPatch((0.355, 0.335), 0.29, 0.33,
                 boxstyle="round,pad=0,rounding_size=0.06", color="white"))
    f.savefig(OUT / "watch.png", transparent=True); plt.close(f)


def speaker():
    f, ax = fig()
    # cylindrical body (smart speaker), rounded top
    ax.add_patch(FancyBboxPatch((0.31, 0.07), 0.38, 0.82,
                 boxstyle="round,pad=0,rounding_size=0.14", color=BLACK))
    # top light-ring hint (white ellipse)
    ax.add_patch(Ellipse((0.5, 0.80), 0.20, 0.055, facecolor="white",
                         edgecolor="none"))
    # speaker grille (white horizontal slits, lower body)
    for yy in (0.24, 0.33, 0.42, 0.51):
        ax.add_patch(FancyBboxPatch((0.355, yy), 0.29, 0.028,
                     boxstyle="round,pad=0,rounding_size=0.014", color="white"))
    f.savefig(OUT / "speaker.png", transparent=True); plt.close(f)


watch(); speaker()
print("wrote watch.png, speaker.png")
