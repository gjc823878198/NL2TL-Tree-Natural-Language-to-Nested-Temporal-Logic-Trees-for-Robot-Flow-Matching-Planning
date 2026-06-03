"""
Tier-3 self-check report + figure.

For every case, batch-sample the FROZEN TeLoGraF flow prior (best-of-N) and
measure how well the raw samples satisfy the STL spec BEFORE any guidance.
The spread of per-sample STL cost is a calibrated, training-free signal of
whether the spec is in- or out-of-distribution for the model:

    many samples satisfy  -> model is confident      -> use flow matching
    even the best is far  -> model is OOD/uncertain   -> defer to A*

This is the principled replacement for the hard-coded per-case `role`, and
the figure ("when does the planner know it doesn't know?") is a poster panel.

Usage:
    python3 tools/feasibility_report.py                 # all cases
    python3 tools/feasibility_report.py --cases reach_avoid seq_reach_ABC
    python3 tools/feasibility_report.py --n-samples 32

Outputs:
    outputs/feasibility/feasibility.json   (raw numbers)
    outputs/feasibility/feasibility.png    (figure, if matplotlib present)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner import CASES, get_case, telograf_available, telograf_feasibility
from planner.telograf_infer import SAT_COST_TOL


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None,
                    help="case ids (default: all)")
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=None,
                    help="override best-of-N batch size per case")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args(argv)

    if not telograf_available():
        print("ERROR: TeLoGraF not installed; run scripts/install_telograf.sh",
              file=sys.stderr)
        return 2

    case_ids = args.cases or [c["id"] for c in CASES]
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "feasibility"
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    print(f"{'case':18s} {'N':>3s} {'satisfy':>8s} {'best':>6s} {'mean':>6s} "
          f"{'decision':>14s}")
    print("-" * 62)
    for cid in case_ids:
        case = get_case(cid)
        rep = telograf_feasibility(case, n_steps=args.n_steps, ckpt=args.ckpt,
                                   n_samples=args.n_samples)
        decision = "flow matching" if rep["feasible"] else "defer to A*"
        print(f"{cid:18s} {rep['n_samples']:3d} {rep['satisfy_rate']:7.0%} "
              f"{rep['best_cost']:6.2f} {rep['mean_cost']:6.2f} {decision:>14s}")
        reports.append(rep)

    (out_dir / "feasibility.json").write_text(json.dumps(reports, indent=2))
    print(f"\nwrote {out_dir / 'feasibility.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for i, rep in enumerate(reports):
            costs = rep["costs"]
            xs = [i + 1] * len(costs)
            ax.scatter(xs, costs, s=14, alpha=0.5,
                       color="tab:green" if rep["feasible"] else "tab:red")
            ax.scatter([i + 1], [rep["best_cost"]], marker="_", s=400,
                       color="black", linewidths=2)
        ax.axhline(SAT_COST_TOL, ls="--", color="gray",
                   label=f"satisfaction threshold ({SAT_COST_TOL})")
        ax.set_xticks(range(1, len(reports) + 1))
        ax.set_xticklabels([r["case"] for r in reports], rotation=20, ha="right",
                           fontsize=8)
        ax.set_ylabel("per-sample STL cost (lower = satisfies)")
        ax.set_title("Best-of-N flow prior: feasibility self-check\n"
                     "green = in-distribution (use flow), red = OOD (defer to A*)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "feasibility.png", dpi=150)
        print(f"wrote {out_dir / 'feasibility.png'}")
    except ImportError:
        print("matplotlib not present; skipped figure (JSON still written)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
