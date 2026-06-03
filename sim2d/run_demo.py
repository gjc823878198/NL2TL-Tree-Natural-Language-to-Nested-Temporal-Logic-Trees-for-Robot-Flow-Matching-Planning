"""
End-to-end 2D demo (matplotlib, no ROS required).

Stages:
  1. Pick one of the fixed cases (`planner.CASES`).
  2. Adapt our STL tree -> TeLoGraF graph (`planner.case_to_graph`).
     Hardware-agnostic; reports the graph that would be fed into TeLoGraF's
     GNN encoder.
  3. Build the 2D scene from `map_hint` (`sim2d.env_2d.build_scene`).
  4. Plan a trajectory by calling **the same** `planner.plan_waypoints`
     used by the ROS 2 trajectory follower -- so the diffusion-planner
     path is shared between 2D and 3D demos.

         --backend auto      (default) TeLoGraF if a checkpoint is on
                             disk, otherwise the robustness fallback.
         --backend telograf  Force TeLoGraF; error out if no checkpoint.
         --backend fallback  Always use the robustness-shaped planner.

  5. Render the trajectory as a PNG or animated GIF.

Usage:
    cd code
    python3 sim2d/run_demo.py                     # case 0, backend=auto
    python3 sim2d/run_demo.py --case reach_avoid --save out.gif
    python3 sim2d/run_demo.py --all --static
    python3 sim2d/run_demo.py --case reach_avoid --backend telograf \
                              --ckpt external/TeLoGraF/exps/.../model_last.ckpt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running from code/ root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner import (
    CASES,
    case_to_graph,
    get_case,
    plan_waypoints,
    telograf_available,
)
from sim2d.env_2d import build_scene, render


def run_case(case: dict, *, backend: str, ckpt: str | None,
             seed: int, n_steps: int, save: str | None, animate: bool):
    print(f"\n=== case: {case['id']}  (role={case.get('role','?')}) ===")
    print(f"  NL    : {case['nl']}")

    g = case_to_graph(case)
    print(f"  graph : {len(g.node_features)} nodes, "
          f"{len(g.edge_index)} edges; ops={g.op_names}")

    scene = build_scene(case, seed=seed)
    obstacles = scene.obstacles_for_planner()
    print(f"  scene : {len(scene.reach_disks())} reach, "
          f"{len(scene.avoid_disks())} avoid+obstacle disks, "
          f"{len(scene.rects)} rects "
          f"-> {len(obstacles)} obstacles routed to planner")

    # backend="role": telograf-primary cases use the pure TeLoGraF backend;
    # fallback cases use auto (telograf-then-A*).
    eff_backend = backend
    if backend == "role":
        eff_backend = "telograf" if case.get("role") == "telograf" else "auto"

    traj = plan_waypoints(case, n_steps=n_steps, ckpt=ckpt, backend=eff_backend,
                          obstacles=obstacles)
    print(f"  plan  : backend={eff_backend}  telograf_available={telograf_available()}  "
          f"waypoints={len(traj)}")

    # "always keep safe" as one STL predicate G(not unsafe), grounded by the
    # SAME obstacle set fed to the planner (the 2D analogue of live sensing).
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from keep_safe import keep_safe_robustness
    rho_ks = keep_safe_robustness(obstacles, traj)
    print(f"  safe  : keep-safe STL  rho(G !unsafe) = {rho_ks:+.2f} "
          f"({'SAFE' if rho_ks > 0 else 'VIOLATED'}; clearance to nearest "
          f"obstacle over the path)")

    title = f"{case['id']}  ({eff_backend})"
    render(scene, traj, title=title, save=save, animate=animate)
    if save:
        print(f"  saved : {save}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=CASES[0]["id"])
    ap.add_argument("--all", action="store_true", help="run every case")
    ap.add_argument("--backend",
                    choices=["role", "auto", "telograf", "fallback"],
                    default="role",
                    help="'role' (default): telograf-primary cases use pure "
                         "TeLoGraF, fallback cases use auto")
    ap.add_argument("--ckpt", default=None,
                    help="TeLoGraF checkpoint path (only used when "
                         "backend=telograf or backend=auto picks TeLoGraF)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--save", help="output file (.png or .gif). If omitted, "
                                    "writes under outputs/sim2d/.")
    ap.add_argument("--outdir", help="when used with --all, write one .gif "
                                      "per case here")
    ap.add_argument("--static", action="store_true",
                    help="render a single PNG instead of animation")
    args = ap.parse_args()

    # Default output dir lives inside the project: code/outputs/sim2d/
    DEFAULT_OUTDIR = Path(__file__).resolve().parent.parent / "outputs" / "sim2d"

    if args.all:
        outdir = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR
        outdir.mkdir(parents=True, exist_ok=True)
        for c in CASES:
            out = outdir / f"{c['id']}.{'png' if args.static else 'gif'}"
            run_case(c, backend=args.backend, ckpt=args.ckpt,
                     seed=args.seed, n_steps=args.n_steps,
                     save=str(out), animate=not args.static)
    else:
        save_path = args.save
        if save_path is None:
            DEFAULT_OUTDIR.mkdir(parents=True, exist_ok=True)
            ext  = "png" if args.static else "gif"
            save_path = str(DEFAULT_OUTDIR / f"{args.case}.{ext}")
        run_case(get_case(args.case),
                 backend=args.backend, ckpt=args.ckpt,
                 seed=args.seed, n_steps=args.n_steps,
                 save=save_path, animate=not args.static)


if __name__ == "__main__":
    main()
