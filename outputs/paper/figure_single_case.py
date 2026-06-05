"""
Single-case panel (case (a) = reach_within_T) for the main text: one frozen
TeLoGraF flow-matching plan on the cluttered map, trajectory coloured by time,
with the NL task and STL window annotated.  Column-width.

    python3 outputs/paper/figure_single_case.py  ->  telograf_single.{png,pdf}
"""
from __future__ import annotations
import sys, textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from planner import get_case, plan_waypoints                       # noqa: E402
from keep_safe import keep_safe_robustness                          # noqa: E402

# same cluttered map as the 3-panel figure
EXTRA = [{"kind": "circle", "x": -1.8, "y": -0.4, "r": 0.40},
         {"kind": "circle", "x":  0.6, "y":  1.1, "r": 0.45},
         {"kind": "circle", "x":  1.9, "y": -2.1, "r": 0.40},
         {"kind": "circle", "x": -0.7, "y":  1.9, "r": 0.38},
         {"kind": "rect",   "x":  3.4, "y": -3.1, "w": 1.0, "h": 0.6}]


def main():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 12})
    case = get_case("reach_within_T")
    traj = np.asarray(plan_waypoints(case, n_steps=120, backend="telograf",
                                     obstacles=EXTRA), float)
    # representative per-replan planning latency for the figure (cf. Sec. 4.2,
    # mean ~3.9 s warm).  We use a stable value rather than the raw wall-clock of
    # this isolated one-shot run, whose model cold-start is not representative of
    # the warm closed-loop re-plans and would make the figure non-reproducible.
    plan_s = 4.0
    avoids = [{"kind": "circle", "x": g["x"], "y": g["y"], "r": g["r"]}
              for g in case["grounding"].values() if g.get("kind") == "avoid"]
    obs = avoids + EXTRA
    rho = keep_safe_robustness(obs, traj)
    reaches = [(g["x"], g["y"], g["r"]) for g in case["grounding"].values()
               if g.get("kind", "reach") == "reach"]

    # landscape layout: square plot (left) | colorbar | SPACER | NL+STL box (right).
    # the spacer column gives the colorbar's right-side tick labels room so they
    # don't collide with the text box.
    # give the square trajectory plot a bigger share of the width (it is the
    # paper's main trajectory figure now that planner panel (c) is gone).
    fig = plt.figure(figsize=(5.9, 3.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.75, 0.05, 0.34, 0.72], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0]); ax.set_aspect("equal")
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#999")
    for o in obs:
        if o.get("kind") == "rect":
            ax.add_patch(mp.Rectangle((o["x"]-o["w"]/2, o["y"]-o["h"]/2),
                         o["w"], o["h"], facecolor="#5b6675",
                         edgecolor="#3a4250", alpha=0.55, lw=0.8))
        else:
            ax.add_patch(mp.Circle((o["x"], o["y"]), o["r"], facecolor="#5b6675",
                         edgecolor="#3a4250", alpha=0.55, lw=0.8))
    for (gx, gy, gr) in reaches:
        ax.add_patch(mp.Circle((gx, gy), gr, facecolor="#f4d35e",
                     edgecolor="#caa83a", alpha=0.55, lw=1.0))
        ax.plot(gx, gy, marker="*", ms=16, color="#b8860b",
                markeredgecolor="k", markeredgewidth=0.5, zorder=6)
    sx, sy = case["map_hint"]["start"]
    ax.plot(sx, sy, marker="s", ms=10, color="k", zorder=6)
    ax.annotate("start", (sx, sy), textcoords="offset points", xytext=(6, 6),
                fontsize=10)

    # --- time anchoring (Route B): stamp each waypoint with REAL seconds at a
    # nominal tracking speed, so the colour axis is wall-clock time and the
    # deadline is a *feasible* real interval -- the old axis was normalised 0..1,
    # which is why a F_{[0,12s]} task looked like it finished in "1 unit". ---
    from stl_runtime import plan_reach_by_deadline
    V_NOM = float(case["map_hint"].get("nominal_speed", 0.18))   # tracking speed
    deadline = float(case["map_hint"].get("reach_deadline_s",
                     2.0 * np.hypot(reaches[0][0]-sx, reaches[0][1]-sy) / 0.22))
    seg_len = np.hypot(np.diff(traj[:, 0]), np.diff(traj[:, 1]))
    tcum = np.concatenate([[0.0], np.cumsum(seg_len)]) / V_NOM   # seconds/waypt
    t_arr = float(tcum[-1])                            # arrival time (s)
    rho_t, _ = plan_reach_by_deadline([tuple(p) for p in traj], reaches[0],
                                      deadline, V_NOM)
    # planning-latency-aware margin: debit the planner's own wall-clock so the
    # "in time" claim counts the time the planner itself spent thinking.
    rho_t_lat = (deadline - plan_s) - t_arr

    pts = traj.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_t = 0.5 * (tcum[:-1] + tcum[1:])              # segment-midpoint time (s)
    lc = LineCollection(segs, cmap="rainbow", norm=plt.Normalize(0, deadline),
                        linewidth=3.0, zorder=5)
    lc.set_array(seg_t)
    ax.add_collection(lc)
    # title: SHORT + left-aligned so it stays over the plot and clears the
    # colorbar's top "(deadline)" tick label (a longer title collides with it).
    # fontsize 11.5 here renders at the SAME on-page size as the (a)/(b) titles
    # (15 pt in planner.pdf), since this figure is displayed at a larger scale.
    ax.set_title(r"(c) TeLoGraF plan (no A*)",
                 fontsize=11.5, fontweight="bold", loc="left", pad=6)
    cax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(ScalarMappable(norm=plt.Normalize(0, deadline),
                                     cmap="rainbow"), cax=cax)
    cb.set_label("time (s)", fontsize=12, fontweight="bold")
    cb.set_ticks([0, round(t_arr), deadline])
    cb.set_ticklabels(["0", f"{t_arr:.0f}", f"{deadline:.0f}"])   # short: no overlap
    cax.tick_params(labelsize=11)
    # --- NL + STL task box: in the RIGHT column (gs[0,3]), large + bold ---
    tax = fig.add_subplot(gs[0, 3]); tax.axis("off")
    cap = ("NL: “" + "\n".join(textwrap.wrap(case["nl"], 18)) + "”\n\n"
           rf"$F_{{[0,{deadline:.0f}\,\mathrm{{s}}]}}(\mathrm{{reach}})\ \wedge\ "
           rf"G\,\neg\,\mathrm{{unsafe}}$" "\n\n"
           rf"exec ${t_arr:.0f}\,\mathrm{{s}}{{+}}{plan_s:.0f}\,\mathrm{{s}}$ plan "
           rf"({'in time' if rho_t_lat > 0 else 'late'})" "\n"
           rf"$\rho^{{\mathrm{{lat}}}}_{{\mathrm{{time}}}}{{=}}{rho_t_lat:+.0f}\,"
           rf"\mathrm{{s}}$,  $\rho_{{\mathrm{{safe}}}}{{=}}{rho:+.2f}\,\mathrm{{m}}$")
    # no bounding box around the text: a framed box's right edge gets clipped at
    # the figure margin, so we drop the frame entirely and keep just the (bold) text.
    tax.text(0.0, 0.5, cap, transform=tax.transAxes, ha="left", va="center",
             fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.05)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"telograf_single.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.04)
    print("wrote telograf_single; rho=%+.2f end=%s" %
          (rho, tuple(round(v, 2) for v in traj[-1])))


OUT_DIR = Path(__file__).resolve().parent
if __name__ == "__main__":
    main()
