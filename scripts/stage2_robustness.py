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
import argparse, io, contextlib, math, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner import CASES, plan_waypoints                      # noqa: E402
from planner.telograf_infer import _trajectory_validates        # noqa: E402
from keep_safe import keep_safe_robustness                      # noqa: E402
from stl_runtime import PlanLatencyModel, latency_aware_reach_rho  # noqa: E402
from sim2d.env_2d import build_scene                            # noqa: E402


def _latency_margin(traj, case, plan_lat):
    """Latency-aware reach-by-deadline margin (our contribution): debit the
    measured planning wall-clock from the case's deadline. None if the case has
    no timed-reach deadline."""
    mh = case.get("map_hint", {})
    deadline = mh.get("reach_deadline_s")
    reaches = [(g["x"], g["y"], g.get("r", 0.4))
               for g in case["grounding"].values()
               if g.get("kind", "reach") == "reach"]
    if not deadline or not reaches:
        return None
    lat = PlanLatencyModel()
    lat.observe(plan_lat)                       # one representative plan's latency
    rho_lat, rho_naive, _est, _res = latency_aware_reach_rho(
        traj, reaches[-1], float(deadline), mh.get("nominal_speed", 0.18), lat)
    return rho_lat, rho_naive


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
    succ, rhos, engines, plat = 0, [], [], []
    last_traj = None
    for _ in range(k):
        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            traj = plan_waypoints(case, n_steps=96, backend=backend,
                                  obstacles=obstacles)
        plat.append(time.time() - t0)             # planning wall-clock (s)
        ok, _why = _trajectory_validates(traj, case, obstacles)
        succ += int(ok)
        rhos.append(keep_safe_robustness(obstacles, traj))
        engines.append(_engine(buf.getvalue()))
        last_traj = traj
    mean = statistics.mean(rhos)
    std = statistics.pstdev(rhos) if len(rhos) > 1 else 0.0
    eng = max(set(engines), key=engines.count)            # modal engine
    mean_plat = statistics.mean(plat)                     # mean inference time (s)
    lat_margin = _latency_margin(last_traj, case, mean_plat)
    return succ, k, mean, std, eng, mean_plat, lat_margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    print(f"{'task':20s} {'success':>8s} {'rho_mean':>9s} {'rho_std':>8s} "
          f"{'engine':>8s} {'plan_s':>7s} {'rho_time^lat':>13s}")
    print("-" * 78)
    plats = []
    for c in CASES:
        s, k, mean, std, eng, plat, lm = run(c, args.k)
        plats.append(plat)
        lat_s = "-" if lm is None else f"{lm[0]:+.0f}s(naive{lm[1]:+.0f})"
        print(f"{c['id']:20s} {f'{s}/{k}':>8s} {mean:+9.3f} {std:8.3f} "
              f"{eng:>8s} {plat:6.1f}s {lat_s:>13s}")
    print("-" * 78)
    print(f"mean planning latency = {statistics.mean(plats):.1f}s "
          f"(this wall-clock is debited from the deadline by the "
          f"planning-latency-aware robustness; see rho_time^lat above)")


if __name__ == "__main__":
    main()
