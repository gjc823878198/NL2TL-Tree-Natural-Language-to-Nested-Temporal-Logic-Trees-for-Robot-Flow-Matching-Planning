# NL2TL-Tree

**Natural Language → Nested Temporal-Logic Trees → Robustness-Guided Flow-Matching Planning**

Companion code for the UbiComp/ISWC 2026 Posters & Demos extended abstract
*“NL2TL-Tree: Natural Language to Nested Temporal-Logic Trees for
Robustness-Guided Flow-Matching Planning.”*

NL2TL-Tree is a **training-free** pipeline: no model is trained or fine-tuned — a
frozen LLM is few-shot-prompted and the motion planner is frozen. A phone-typed
task becomes a *checkable, executable spatiotemporal command* for a mobile robot.
Instead of translating language into a flat formula, we parse it into a **nested
Signal Temporal Logic (STL) tree** that a user can read, check, and correct at a
single sub-node, then drive a frozen graph-encoded flow-matching planner
(TeLoGraF) steered by a differentiable STL-robustness term — with an online
**sound** monitor that certifies keep-safe *and* reach-by-deadline before and
during execution.

> A detailed, file-by-file walkthrough (in Chinese) lives in
> [`README.zh.md`](README.zh.md).

---

## Pipeline (everything frozen)

```
 phone NL ──▶ few-shot LLM parse ──▶ render to differentiable ρ ──▶ frozen flow planner ──▶ executed trajectory
            (nested STL tree, JSON)   (sound monitor + STLCG)        (TeLoGraF + guidance)    (MPPI tracking, online shield)
```

1. **NL → STL tree.** A frozen LLM (Llama-3.3-70b via Groq, strict-JSON, 8
   few-shot exemplars) emits a nested STL tree as a JSON AST. A round-trip back to
   an STL string + schema validation makes every parse checkable; a
   robustness-feedback loop re-prompts the same model to repair mis-parses.
2. **Render to robustness.** The tree compiles to a canonical formula and two
   robustness functions: a **smooth** STLCG surrogate (softmin/softmax) that
   *steers* sampling, and a **separate exact (sound) monitor** that *certifies*.
   Time is anchored to wall-clock seconds, so `F_[0,D] reach` is a real deadline.
3. **Tree → trajectory.** The spec enters a frozen TeLoGraF GNN encoder natively;
   flow matching samples trajectories that the `∇ρ` gradient guides. Three
   test-time mechanisms — operator normalization, nested-tree decomposition, and a
   best-of-N feasibility self-check — let the frozen planner handle specs and
   obstacles it cannot alone, deferring to a classical A* backend only on
   out-of-distribution specs.
4. **Execution (closed loop).** The planner re-plans from the current pose every
   5 s; an MPPI controller tracks the latest trajectory at ~7 Hz. The sound
   monitor runs online — a non-positive ρ safe-stops and re-plans (a runtime
   shield), so a certificate failure is repaired, not merely logged.

---

## Repository layout

| Path | What it is |
|---|---|
| `stl_parser.py` | STL grammar (Lark) + AST ↔ STL round-trip + Graphviz rendering |
| `nl_to_tree_groq.py` | Groq Llama-3.3-70b client: NL → nested STL tree (strict-JSON, few-shot) |
| `nl_to_tree_selfcorrect.py` | Robustness-feedback self-correction loop |
| `tree_metrics.py` / `rescore.py` | Eval metrics (exact-match, op-F1, path-F1, TED, TED-norm) |
| `stl_robustness.py` | **Exact** (sound, non-smoothed) STL robustness — the certifier |
| `stl_runtime.py` | Time-anchored reach-by-deadline certificate (real seconds) |
| `keep_safe.py` | `G ¬unsafe` keep-safe predicate grounded by sensed disks |
| `planner/` | Grounding + TeLoGraF adapter + `plan_waypoints()` entry point |
| `sim2d/` | 2-D matplotlib simulation (PNG/GIF demos) |
| `sim_ros2/` | ROS 2 Humble + Gazebo Classic + TurtleBot3 **closed-loop** sim |
| `outputs/` | Generated figures (including the paper figures) |
| `scripts/` | Setup + evaluation helpers (`install_telograf.sh`, eval drivers) |
| `viz_app.py` / `mobile_app.py` | Streamlit inspection tool / phone-style entry UI |

Not committed (see **Setup**): `external/` (vendored TeLoGraF + checkpoint, 2.2 GB),
`.venv-telograf/` (1.7 GB), `data/` (NL2TL dataset, 297 MB).

---

## Setup

```bash
cd code
pip install -r requirements.txt          # lark, streamlit, groq, zss, matplotlib, numpy, networkx, ...
```

Three external pieces are fetched separately (they are `.gitignore`d):

```bash
# 1) Frozen LLM — get a free Groq key at https://console.groq.com/keys
export GROQ_API_KEY=gsk_...               # read from the environment only; never committed

# 2) Frozen planner — clone TeLoGraF + its simple_gnn_F checkpoint into external/
bash scripts/install_telograf.sh          # -> external/TeLoGraF/ + .venv-telograf/
#    (upstream: https://github.com/mengyuest/TeLoGraF)

# 3) Parsing dataset — NL2TL "lifted" dataset
python3 download_nl2tl.py                 # -> data/
```

The ROS 2 closed loop additionally needs **ROS 2 Humble**, **Gazebo Classic**, and
the **TurtleBot3** packages (`export TURTLEBOT3_MODEL=burger`). See
[`sim_ros2/README.md`](sim_ros2/README.md).

---

## Quickstart

```bash
# Parse one instruction to a nested STL tree (needs GROQ_API_KEY)
python3 nl_to_tree_groq.py --nl "reach the kitchen within 20 s but never enter the bedroom"

# Stage-1 parsing eval on the NL2TL lifted set
python3 scripts/record_eval.py            # writes predictions + the 5 metrics

# Stage-2 (a): 2-D plan for a case, planner auto-selected (flow / +decomp / A*)
python3 sim2d/run_demo.py --case reach_within_T --backend auto

# Stage-2 (b): closed-loop, multi-goal, sensor-driven re-planning (no ROS needed)
python3 sim_ros2/closed_loop_demo.py      # -> outputs/sim_ros2/closed_loop/*.png|.gif

# Regenerate the paper's single-case figure
python3 outputs/paper/figure_single_case.py

# Visual tree-inspection tool
streamlit run viz_app.py
```

### Closed-loop demo (sense → plan → re-plan)

There are two ways to run the closed loop; both use the **same** scenario in
[`sim_ros2/scenario.py`](sim_ros2/scenario.py) (start pose, three goals visited
**B → A → C**, a uniform cylinder field, deadline). Run
`python3 sim_ros2/scenario.py` to print the exact task (NL + STL + obstacles).

**(a) Pure 2-D, no ROS/Gazebo needed** — easiest; runs the identical
sense→plan→re-plan loop in a Python loop and writes a top-down PNG + GIF:

```bash
cd code
python3 sim_ros2/closed_loop_demo.py        # -> outputs/sim_ros2/closed_loop/*.png|.gif
```

**(b) Full ROS 2 + Gazebo Classic + TurtleBot3** — **two terminals, both must
run at the same time** (the robot moves only when the follower in terminal B is
running):

```bash
# --- terminal A: simulation (leave it running; do NOT Ctrl-C) ---
cd code
pkill -9 -f gzserver; pkill -9 -f gzclient        # clear any stale Gazebo on port 11345
export TURTLEBOT3_MODEL=burger
ros2 launch sim_ros2/launch/tb3_sim.launch.py     # add gui:=true to see the Gazebo 3-D window

# --- terminal B: follower (start while A is still up) ---
cd code
python3 sim_ros2/tb3_follower.py --multi-goal     # TeLoGraF re-plans every 5 s, MPPI -> /cmd_vel
```

Single-case variants: `ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=reach_avoid`
paired with `python3 sim_ros2/tb3_follower.py --case reach_avoid`.

**Gotchas** (these cost real debugging time):
- **Robot not moving?** You only started terminal A. The re-planning + control
  live in the follower (terminal B) — both terminals must run together.
- **`gzserver exit 255` / "Entity already exists":** a stale Gazebo holds port
  11345 — run `pkill -9 -f gzserver` before relaunching.
- **`rviz2: undefined symbol __libc_pthread_init`:** the snap VS Code terminal
  injects `/snap` libs — use a **system terminal** (the launch file also strips
  `/snap` from `LD_LIBRARY_PATH`).
- **Re-plans taking too long?** TeLoGraF is CPU-bound; run headless
  (`gui:=false`, the default) and/or cap threads with `TELOGRAF_THREADS=6`.

See [`sim_ros2/README.md`](sim_ros2/README.md) for the full troubleshooting list.

---

## Reproducibility

These are the exact settings used in the paper.

| Component | Setting |
|---|---|
| Parsing LLM | `llama-3.3-70b-versatile` via Groq, strict-JSON, **greedy** (temp 0) |
| Few-shot | `K_shot = 8` exemplars from the NL2TL lifted set |
| Self-consistency | sample at temperature **0.7**, majority-vote canonicalized trees |
| Planner | frozen TeLoGraF `simple_gnn_F` checkpoint, **64** flow steps |
| Best-of-N | `N ∈ [16, 64]` per-task (**8** in the closed loop) |
| Guidance | STLCG smooth robustness (softmin/softmax) |
| Feasibility self-check | defer to A* when best-of-N per-atom reach-shortfall > **τ = 2.5 m** |
| Closed loop | re-plan every **5 s**, MPPI tracking **~7 Hz**, `v_max = 0.22 m/s` |
| Scene | goal disks `r ≈ 0.45 m`, LiDAR-sensed cylinders `r = 0.22 m` |

Stage-1 parsing (preliminary, `n = 30`, no fine-tuning): exact-match **56.7%**
(95% Wilson CI [39, 73]%), op-F1 **0.95**, path-F1 **0.64**, TED **1.03**,
TED-norm **0.84**. The op-vs-path gap shows the residual error is almost entirely
*structural* (right operators, wrong nesting). These figures are early-stage and
indicative, pending a larger pooled evaluation.

---

## Citation

```bibtex
@inproceedings{nl2tltree2026,
  title     = {NL2TL-Tree: Natural Language to Nested Temporal-Logic Trees
               for Robustness-Guided Flow-Matching Planning},
  booktitle = {Companion of the 2026 ACM International Joint Conference on
               Pervasive and Ubiquitous Computing and the 2026 ACM International
               Symposium on Wearable Computers (UbiComp/ISWC '26)},
  year      = {2026}
}
```

## Acknowledgements

Built on **TeLoGraF** (Meng & Fan, ICML 2025) for graph-encoded flow-matching STL
planning, the **NL2TL** dataset for NL→STL parsing, and **STLCG** for smooth STL
robustness. The vendored TeLoGraF code under `external/` retains its upstream
license; this repository's own code is released for research use.
