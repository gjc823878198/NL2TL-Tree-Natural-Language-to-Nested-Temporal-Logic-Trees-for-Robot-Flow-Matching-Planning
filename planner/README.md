# planner/ — STL tree → grounding → diffusion trajectory

## Files

| File | Purpose |
|---|---|
| `case_examples.py`     | Five fixed cases, each carrying a `role` (telograf / fallback) |
| `telograf_adapter.py`  | tree JSON → the `(node_features, edge_index)` TeLoGraF expects |
| `telograf_infer.py`    | **Public entry point** `plan_waypoints(case, backend=…)` |
| `__init__.py`          | Exposes `CASES` / `get_case` / `case_to_graph` / `plan_waypoints` |

`plan_waypoints` is the **only** "case → (x, y) trajectory" entry point in the project:
both the 2-D simulation (`sim2d/run_demo.py`) and the ROS closed loop
(`sim_ros2/tb3_follower.py`, re-planning every 5 s) obtain trajectories through it, so
switching backends never requires touching any other code.

## Cases split into two roles (decided empirically)

| role | case | Notes |
|---|---|---|
| `telograf` | `reach_within_T` / `reach_avoid` / `reach_goal_north` | In-distribution (timed reach with obstacles beside the straight line); pure TeLoGraF best-of-N satisfies the STL directly and **never touches A\*** |
| `fallback` | `seq_reach_ABC` / `trigger_response` | Deliberately out of distribution (three sequential legs / imply / procedural walls); TeLoGraF fails and A* takes over |

With `backend="role"` (the default for verify and run_demo), telograf cases use pure
TeLoGraF and fallback cases use auto.

## Public API

```python
from planner import plan_waypoints, telograf_available

waypoints = plan_waypoints(
    case,             # case dict or case_id string
    n_steps=96,       # number of waypoints wanted
    ckpt=None,        # TeLoGraF checkpoint path (None = auto-discover)
    backend="auto",   # "auto" | "telograf" | "fallback"
    obstacles=None,   # extra obstacles (procedural walls/furniture/random); affects
                      # only the fallback, since TeLoGraF sees them via the STL graph
)
# -> [(x0, y0), (x1, y1), ...]
```

`obstacles` is a list of dicts, each in one of these two forms:

```python
{"kind": "circle", "x": ..., "y": ..., "r": ...}
{"kind": "rect",   "x": ..., "y": ..., "w": ..., "h": ...}   # centre + full size
```

`sim2d/env_2d.py::Scene.obstacles_for_planner()` emits every non-reach geometry in the
scene (walls, furniture, random disks, avoid atoms) in exactly this format, ready to
feed straight into `plan_waypoints`.

### How obstacles reach the diffusion planner: STLCG guidance

All obstacles (STL avoid atoms plus the procedural walls in `obstacles=`) are fed to the
diffusion planner through **STLCG-style differentiable test-time guidance**, matching the
CTG/LTLDoG setup of the TeLoGraF paper:

1. `_telograf_subprocess` puts `refine_obstacles` (the grounded avoid atoms plus
   `_obstacles_to_disks(extra_obstacles)`) into the payload.
2. The `tools/telograf_export.py` subprocess samples best-of-N from the GNN, picks the
   best candidate, and runs `_stl_refine` on it for 200 Adam steps with the loss
   `3·ReLU(obstacle violation)² + 2·distance-to-goal² + 5·start anchor + 0.05·smoothness`,
   pushing the trajectory clear of every obstacle and pulling it toward the goal.

> Why not put them into the GNN graph directly? Augmenting obstacles as `G(not(_obs))`
> nodes grows the graph from 6 to 40+ nodes, which leaves the training distribution of
> `simple_gnn_F` and collapses the GNN prior. Guidance adjusts the geometry after
> sampling without touching the graph, so many obstacles and obstacles directly on the
> path are both avoided reliably (see the §7C profile in the top-level README). The
> experimental switch `TELOGRAF_AUGMENT_OBSTACLES=1` still injects them into the graph,
> and is off by default.

### Backend behaviour

| `backend`  | What it uses | On failure |
|---|---|---|
| `role`     | telograf cases use `telograf`, fallback cases use `auto` (the verify/run_demo default) | — |
| `fallback` | A\* on a 0.15 m grid, 8-connected, 0.30 m clearance; reads `grounding[avoid]` plus `obstacles=` and visits every reach disk in order | Falls back to a straight line when no path is found |
| `telograf` | 1) the subprocess samples N flow trajectories in a batch (`telograf_samples`, test_muls); 2) best-of-N selection; 3) **differentiable STLCG guidance** pushes it clear of all obstacles and toward the goal; 4) if it verifies, **pure TeLoGraF + guidance is returned**; 5) otherwise (structurally out of distribution) A* repairs or takes over | Raises if the checkpoint or venv cannot be found |
| `auto`     | Tries `telograf` first and falls back silently to `fallback`, recording the reason in the log | Never fails |

The output of every backend finally passes through `_pin_reaches`, which pins the
trajectory to the centre of each reach atom (STL `reach` means entering the goal disk).

### TeLoGraF + repair verification (2-D)

```bash
python3 sim2d/run_demo.py --case reach_avoid --backend telograf
```

Run case by case, all report `planOK=True`, `goal_reached=True`, `hits=0`, and every STL
reach atom touched (at the planner level, through the same `plan_waypoints` entry point
the ROS closed loop uses). Details are in §7C of [code/README.md](../README.md).

### Environment-variable knobs

| Variable | Effect | Default |
|---|---|---|
| `TELOGRAF_NO_REPAIR=1`     | Skip STL verification and A* repair, showing the raw TeLoGraF diffusion output | repair enabled |
| `TELOGRAF_NO_SHIFT=1`      | Do not post-hoc translate the trajectory to pin the first waypoint to `case.start` | translation enabled |
| `TELOGRAF_AUGMENT_OBSTACLES=1` | Inject procedural obstacles into the STL graph as `globally(not(_obs))` (node count 6 → 40+) | OFF, because leaving the training distribution ruins every goal |
| `FAKE_OUTPUT=1`            | Make `tools/telograf_export.py` emit a straight fake trajectory, to test only the subprocess plumbing | OFF |

## TeLoGraF input format (this directory matches upstream)

Aligned directly with `external/TeLoGraF/code/stl_to_seq_utils.py::stl_to_seq`:

- Every node has 8 features: `[node_type_i, ts, te, x, y, z, r, n_child]`
- Missing fields are filled with `-1` (times) or `0` (coordinates and `n_child` for atoms)
- Edge direction: **child → parent**
- There is **no separate "avoid" atom**; avoidance is `not(reach)`, two nodes

OP_CODE lives in `telograf_adapter.py`:
```python
{ "and": 0, "or": 1, "not": 2, "finally": 5, "globally": 6, "until": 7, "reach": 8 }
```

## Normalization steps from tree to TeLoGraF

| Ours | TeLoGraF accepts | Normalization |
|---|---|---|
| `imply(A, B)` | not supported | Rewritten as `or(not(A), B)` |
| `iff(A, B)` | not supported | Rewritten as `and(imply(A,B), imply(B,A))`, then expanded |
| `not(atom)` | kept | Not folded — TeLoGraF expects a NOT node as the parent of a reach atom |
| `and/or` with many children | n-ary | Kept n-ary (the TeLoGraF GNN aggregates with scatter and does not require binary trees) |
| Abstract `prop_i` | must be geometrically grounded | Supplied as (x, y, z, r) from the `grounding` dict in the case |

## Installing TeLoGraF

Do not follow the conda commands in the TeLoGraF README step by step: that flow is
pinned to PyTorch 1.13 / CUDA 11.7 and will not install on recent GPUs (40/50 series).

Use the one-shot script at the project root:

```bash
cd /home/jiachen-tlab-ut/Conferences/UbiComp/code
bash scripts/install_telograf.sh
```

The script produces:

* `external/TeLoGraF/`  ← the TeLoGraF source
* `.venv-telograf/`     ← PyTorch 2.4.1 (CPU) + torch_geometric 2.5.3 + all TeLoGraF deps
* a built-in smoke test verifying that graph construction passes for all five cases

> CUDA users: `PYTORCH_CHANNEL=cu121 bash scripts/install_telograf.sh`

## Wiring TeLoGraF inference (after the checkpoint is downloaded)

Upstream TeLoGraF has no Python API for "load a checkpoint and then sample a graph"; all
inference paths live inside `train_gstl_v1.py --fix -T <run>`. We solve this with a
**subprocess bridge**:

```
sim_ros2/tb3_follower.py  (planner thread, re-plans every 5 s)
        ▼
planner/telograf_infer.py::plan_waypoints(backend="telograf")
        │  shell-out
        ▼
.venv-telograf/bin/python  tools/telograf_export.py  <stdin: case JSON>
        │  imports z_diffuser.GaussianFlow, loads ckpt, samples
        ▼
stdout: {"waypoints": [[x, y], ...]}
        │  parsed and returned
        ▼
MPPI control loop tracks the trajectory -> /cmd_vel
```

Concrete steps:

1. **Download a checkpoint**: fetch `g0128-075243_simple_gnn_F.zip` (or similar) from the
   Google Drive link in the TeLoGraF README. Unpack it to
   `external/TeLoGraF/exps/<run_id>/models/model_last.ckpt`.

2. **Fill in the stub**: `tools/telograf_export.py` contains a `TODO` comment. Open it and
   replace `_fail(...)` with roughly these 30 lines:

   ```python
   # 1) Rebuild args.Namespace, reading args.txt / cfg.json next to the checkpoint,
   #    or copying the command line used to train this run in run_icml2025_test.sh.
   args = build_args_from_checkpoint(ckpt)

   # 2) Rebuild the model (the same build_model as the --fix branch of train_gstl_v1.py)
   encoder, model = build_model(args)
   sd = torch.load(ckpt, map_location="cpu")
   model.load_state_dict(sd["model_state_dict"])
   encoder.load_state_dict(sd["encoder_state_dict"])
   model.eval(); encoder.eval()

   # 3) Wrap the (node_feats, edge_index) we pass in as a torch_geometric.Data
   #    -> call encoder.encode_graph(data) to obtain the cond embedding

   # 4) p_sample_loop:
   x = torch.randn(1, args.horizon, args.transition_dim)
   traj = model.p_sample_loop(x.shape, cond=cond_emb, args=args)

   # 5) Project out x and y
   xs = traj[0, :, 0].cpu().tolist()
   ys = traj[0, :, 1].cpu().tolist()
   print(json.dumps({"waypoints": list(zip(xs, ys)), "backend": "telograf"}))
   ```

3. **To test only the subprocess plumbing, without downloading a checkpoint**:
   ```bash
   FAKE_OUTPUT=1 python3 -c "
   from planner import plan_waypoints
   wp = plan_waypoints('reach_avoid', backend='telograf', n_steps=12)
   print(len(wp), 'fake waypoints from subprocess')"
   ```
   This makes `telograf_export.py` take its `if FAKE_OUTPUT == "1"` branch and emit a
   straight trajectory, proving that all three hops of the ROS ↔ venv ↔ TeLoGraF pipeline
   work.

## Quick offline checks

```bash
cd code

# 1. graph statistics for the five cases
python3 planner/telograf_adapter.py

# 2. fallback trajectory for a case
python3 planner/telograf_infer.py --case reach_avoid --backend fallback

# 3. the case descriptions
python3 planner/case_examples.py
```

## Still to do

* Fill in the TODO in `tools/telograf_export.py` (once the checkpoint is downloaded)
* Add STLCG++ robustness as inference-time guidance (`nabla rho`)
* Alternative: derivative-free guidance (SVDD-style, if the collision signal is
  non-differentiable)
