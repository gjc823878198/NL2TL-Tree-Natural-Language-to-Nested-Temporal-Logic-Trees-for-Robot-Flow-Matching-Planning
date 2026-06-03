"""
Stage-2 repeated-run study: success rate + robustness variance.

The flow sampler is stochastic (best-of-N + guidance), so we re-plan each task
K times and report:
  * success rate  (fraction of runs whose trajectory passes the SAME validator
    the planner uses: starts right, reaches every goal, collision-free)
  * keep-safe robustness rho(G !unsafe): mean +/- std over the K runs
  * which engine each run used (flow / +decomp / A*)

    python3 scripts/stage2_robustness.py --k 8
"""
from __future__ import annotations
import argparse, io, contextlib, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner import CASES, plan_waypoints                      # noqa: E402
from planner.telograf_infer import _trajectory_validates        # noqa: E402
from keep_safe import keep_safe_robustness                      # noqa: E402
from sim2d.env_2d import build_scene                            # noqa: E402


def _engine(log: str) -> str:
    if "A* repair" in log or ("A*" in log and "no A*" not in log):
        return "A*"
    if "DECOMPOSE" in log or "decomposed stitch" in log:
        return "+decomp"
    return "flow"


def run(case, k):
    scene = build_scene(case, seed=0)
    obstacles = scene.obstacles_for_planner()
    backend = "telograf" if case.get("role") == "telograf" else "auto"
    succ, rhos, engines = 0, [], []
    for _ in range(k):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            traj = plan_waypoints(case, n_steps=96, backend=backend,
                                  obstacles=obstacles)
        ok, _why = _trajectory_validates(traj, case, obstacles)
        succ += int(ok)
        rhos.append(keep_safe_robustness(obstacles, traj))
        engines.append(_engine(buf.getvalue()))
    mean = statistics.mean(rhos)
    std = statistics.pstdev(rhos) if len(rhos) > 1 else 0.0
    eng = max(set(engines), key=engines.count)            # modal engine
    return succ, k, mean, std, eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    print(f"{'task':20s} {'success':>8s} {'rho_mean':>9s} {'rho_std':>8s} {'engine':>8s}")
    print("-" * 60)
    for c in CASES:
        s, k, mean, std, eng = run(c, args.k)
        print(f"{c['id']:20s} {f'{s}/{k}':>8s} {mean:+9.3f} {std:8.3f} {eng:>8s}")


if __name__ == "__main__":
    main()
