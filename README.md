# NL2TL-Tree

**Natural Language → Nested Temporal-Logic Trees → Robustness-Guided Flow-Matching Planning**

Companion code for the UbiComp/ISWC 2026 Posters & Demos extended abstract
*“NL2TL-Tree: Natural Language to Nested Temporal-Logic Trees for
Robustness-Guided Flow-Matching Planning”* — accepted, to appear in the
*Companion of the 2026 ACM International Joint Conference on Pervasive and
Ubiquitous Computing (UbiComp Companion '26)*.
DOI: [10.1145/3798063.3837197](https://doi.org/10.1145/3798063.3837197)
(activates shortly after publication in the ACM Digital Library).

NL2TL-Tree is a **training-free** pipeline: no model is trained or fine-tuned — a
frozen LLM is few-shot-prompted and the motion planner is frozen. A phone-typed
task becomes a *checkable, executable spatiotemporal command* for a mobile robot.
Instead of translating language into a flat formula, we parse it into a **nested
Signal Temporal Logic (STL) tree** that a user can read, check, and correct at a
single sub-node, then drive a frozen graph-encoded flow-matching planner
(TeLoGraF) steered by a differentiable STL-robustness term — with an online
**sound** monitor that certifies keep-safe *and* reach-by-deadline before and
during execution.

> A detailed, file-by-file walkthrough lives in
> [`WALKTHROUGH.md`](WALKTHROUGH.md).
>
> 📄 **Documents in this repo:** the poster paper
> ([`ubicomp2026poster.pdf`](ubicomp2026poster.pdf)) and the demo supplement
> ([`demo/demo_supplement.pdf`](demo/demo_supplement.pdf)).

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
   shield), so a certificate failure is repaired, not merely logged. A
   **planning-latency-aware deadline** (our contribution) predicts the planner's
   own wall-clock and *debits it from the timed budget*
   (`ρ^lat_time = (D − t̂_plan) − t_exec`), so a reach-by-deadline check counts the
   time the planner itself spends thinking — the world does not freeze while we
   plan. It is wired through the core gate (`planner/telograf_infer.py`), the
   Stage-2 study, both closed-loop sims, and the demo.

---

## Repository layout

| Path | What it is |
|---|---|
| `stl_parser.py` | STL grammar (Lark) + AST ↔ STL round-trip + Graphviz rendering |
| `nl_to_tree_groq.py` | Groq Llama-3.3-70b client: NL → nested STL tree (strict-JSON, few-shot) |
| `nl_to_tree_selfcorrect.py` | Robustness-feedback self-correction loop |
| `tree_metrics.py` / `rescore.py` | Eval metrics (exact-match, op-F1, path-F1, TED, TED-norm) |
| `stl_robustness.py` | **Exact** (sound, non-smoothed) STL robustness — the certifier |
| `stl_runtime.py` | Reach-by-deadline certificate (real seconds) + **planning-latency-aware** deadline (`PlanLatencyModel`, `latency_aware_reach_rho`) |
| `keep_safe.py` | `G ¬unsafe` keep-safe predicate grounded by sensed disks |
| `planner/` | Grounding + TeLoGraF adapter + `plan_waypoints()` entry point (latency-aware temporal gate) |
| `sim2d/` | 2-D matplotlib simulation (PNG/GIF demos) |
| `sim_ros2/` | ROS 2 Humble + Gazebo Classic + TurtleBot3 **closed-loop** sim |
| `demo/` | **Natural-language demo** front-end (2D + ROS): type a task → STL tree → plan → robot, live |
| `outputs/` | Generated figures (including the paper figures) |
| `scripts/` | Setup + evaluation helpers (`install_telograf.sh`, `stage2_robustness.py`, `reviewer_exp.py`) |
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

## Natural-language demo (type a task → watch the robot)

The interactive demo in [`demo/`](demo/) lets you **type a plain-English task** for
the **same map** as the closed loop (regions **A / B / C** in a cylinder field) and
watch the full paper pipeline run: NL → frozen-LLM **STL tree** → grounded plan →
**TeLoGraF** → the robot executes, live. The live clock keeps advancing even while
the robot is stopped waiting for the planner, so you *see* the planning latency the
contribution debits from the deadline.

![NL2TL-Tree 2D demo: B→A→C, closed-loop, latency-aware deadline](demo/demo.gif)

```bash
cd code
export GROQ_API_KEY=gsk_...                  # optional; omit to use the offline keyword parser

# 2D front-end (no ROS): pops up an NL box, then a LIVE animated window
python3 demo/run_2d.py
python3 demo/run_2d.py --nl "Visit B, then A, then C, keeping safe, within 180 s" --no-llm

# ROS 2 + Gazebo front-end: ONE launch file opens the world + an NL input box;
# submitting a task opens RViz and drives the TurtleBot3.
ros2 launch demo/launch/demo.launch.py
```

Booth-safe: with no `GROQ_API_KEY` it falls back to a deterministic keyword parser
that yields the same STL tree for the example tasks, so the demo runs fully offline.
The longer write-up is the **Demo Supplement** (`demo_supplement.pdf`, built from the
LaTeX next to this repo).

---

## Reproducibility

> **This section is the paper's reproducibility reference.** The camera-ready
> keeps its four-page body per the ACM page budget and points here ("full
> settings in the released configs") instead of carrying an appendix — the
> table below is the authoritative record of every setting used in the paper.

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
| Monitor | sound monitor stops / re-plans on **ρ ≤ 0** (monitor-level) |
| Scene | goal disks `r ≈ 0.45 m`, LiDAR-sensed cylinders `r = 0.22 m` |

Stage-1 parsing (preliminary, `n = 30`, no fine-tuning): exact-match **56.7%**
(95% Wilson CI [39, 73]%), op-F1 **0.95**, path-F1 **0.64**, TED **1.03**,
TED-norm **0.84**. The op-vs-path gap shows the residual error is almost entirely
*structural* (right operators, wrong nesting). These figures are early-stage and
indicative, pending a larger pooled evaluation.

Reproduce the paper's comparison studies:

```bash
python3 scripts/reviewer_exp.py --n 30 --k 5     # (1) flat-STL vs nested tree (fair, tree_canon)
                                                 # (2) self-consistency / round-trip ablation
python3 scripts/stage2_robustness.py --k 8       # Stage-2 success/robustness + per-plan latency
                                                 #   (prints plan_s + the latency-aware rho_time^lat)
```

---

## Citation

```bibtex
@inproceedings{nl2tltree2026,
  author    = {Gong, Jiachen and Mao, Wencan and Javanmardi, Ehsan and
               Li, Yun and Zhou, Quanxi and Tsukada, Manabu},
  title     = {NL2TL-Tree: Natural Language to Nested Temporal-Logic Trees
               for Robustness-Guided Flow-Matching Planning},
  booktitle = {Companion of the 2026 ACM International Joint Conference on
               Pervasive and Ubiquitous Computing (UbiComp Companion '26)},
  year      = {2026},
  month     = oct,
  address   = {Shanghai, China},
  publisher = {ACM},
  doi       = {10.1145/3798063.3837197},
  isbn      = {979-8-4007-2533-3}
}
```

ACM Reference Format:

> Jiachen Gong, Wencan Mao, Ehsan Javanmardi, Yun Li, Quanxi Zhou, and Manabu
> Tsukada. 2026. NL2TL-Tree: Natural Language to Nested Temporal-Logic Trees
> for Robustness-Guided Flow-Matching Planning. In *Companion of the 2026 ACM
> International Joint Conference on Pervasive and Ubiquitous Computing
> (UbiComp Companion '26), October 11–15, 2026, Shanghai, China.* ACM, New
> York, NY, USA, 5 pages. https://doi.org/10.1145/3798063.3837197

## Paper copyright and license

The paper is published open access under a **Creative Commons Attribution 4.0
International (CC BY 4.0)** license:

> This work is licensed under a Creative Commons Attribution 4.0 International
> License.
> UbiComp Companion '26, October 11–15, 2026, Shanghai, China
> © 2026 Copyright held by the owner/author(s).
> ACM ISBN 979-8-4007-2533-3/2026/10
> https://doi.org/10.1145/3798063.3837197

The paper PDF in this repository (`ubicomp2026poster.pdf`) is the authors'
camera-ready version, shared under the same CC BY 4.0 license. The license of
the code in this repository is stated in the Acknowledgements section below
(the vendored TeLoGraF code retains its upstream license).

## Acknowledgements

Built on **TeLoGraF** (Meng & Fan, ICML 2025) for graph-encoded flow-matching STL
planning, the **NL2TL** dataset for NL→STL parsing, and **STLCG** for smooth STL
robustness. The vendored TeLoGraF code under `external/` retains its upstream
license; this repository's own code is released for research use.
