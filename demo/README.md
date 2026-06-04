# NL2TL-Tree — natural-language demo

Type a plain-English task and watch the full paper pipeline run on the closed-loop
map (regions **A / B / C** in a cylinder field): NL → frozen-LLM **nested STL tree**
→ grounded plan → **TeLoGraF** flow-matching → the robot executes, **live**.

![2D demo: B→A→C, closed-loop, latency-aware deadline](demo.gif)

The live clock keeps advancing even while the robot is **stopped waiting for the
planner**, so you see the planning latency that the *planning-latency-aware
deadline* (`stl_runtime.PlanLatencyModel` / `latency_aware_reach_rho`) debits from
the timed budget.

## Run

```bash
cd code
export GROQ_API_KEY=gsk_...                  # optional; omit for the offline keyword parser

# 2D front-end (no ROS): NL box, then a LIVE animated window
python3 demo/run_2d.py
python3 demo/run_2d.py --nl "Visit B, then A, then C, keeping safe, within 180 s" --no-llm
python3 demo/run_2d.py --render-only         # redraw the last run instantly (no planning)

# ROS 2 + Gazebo: ONE launch file opens the world + an NL input box; submitting a
# task opens RViz and drives the TurtleBot3.
ros2 launch demo/launch/demo.launch.py
```

## Files

| File | What it is |
|---|---|
| `nl_grounding.py` | NL → `nl_to_tree` (frozen LLM) → grounded planner case (shared core; offline fallback) |
| `nl_input.py` | tkinter natural-language input box |
| `run_2d.py` | 2D front-end: live window + PNG/GIF; latency-aware clock |
| `run_ros.py` | ROS GUI node: NL → case JSON → launches RViz + the TurtleBot3 follower |
| `launch/demo.launch.py` | one launch file: Gazebo world + TurtleBot3 + the NL box |

Booth-safe: with no `GROQ_API_KEY` it falls back to a deterministic keyword parser
that yields the same STL tree for the example tasks, so the demo runs fully offline.
The full write-up is the **Demo Supplement** (`demo_supplement.pdf`).
