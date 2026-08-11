# sim_ros2/ — ROS 2 Humble + Gazebo Classic + TurtleBot3 closed-loop simulation

A TurtleBot3 (burger) runs a **real closed loop** inside **Gazebo Classic**
(gazebo11 / `gazebo_ros`): the robot senses the obstacle cylinders online with its
360° LiDAR (`/scan`), TeLoGraF periodically re-plans a reference trajectory from the
current pose, and an MPPI controller tracks the latest trajectory, publishing
`/cmd_vel` and reading `/odom`, so the robot actually moves in RViz.

## Two processes

| Role | File | Purpose |
|---|---|---|
| Simulation | [`launch/tb3_sim.launch.py`](launch/tb3_sim.launch.py) | gzserver (open cylinder world) + TB3 spawn + static `map→odom` TF + RViz markers |
| Closed-loop control | [`tb3_follower.py`](tb3_follower.py) | TeLoGraF re-plans every 5 s + MPPI trajectory tracking → `/cmd_vel` |

A single `case:=<case_id>` argument decides everything: cases are defined in
[`planner/case_examples.py`](../planner/case_examples.py), `map_hint.start` decides
where the robot spawns, and the world is generated programmatically from the case by
[`tb3_world_gen.py`](tb3_world_gen.py).

## Open cylinder world (no walls)

The scene contains **only cylinder obstacles, no walls**:

- The **red obstacle cylinders** are physical bodies; the LiDAR senses them online and
  the robot steers around them.
- The **goal regions (green) are not spawned as physical cylinders** — otherwise the
  LiDAR would treat a goal as an obstacle and the robot would never reach it. Goals are
  shown only as RViz markers.
- Ground and sun are **inlined** into the SDF (no `model://` references to the online
  model database), so gzserver starts quickly and the `/spawn_entity` service does not
  time out.

## The two-stage closed loop (this is the loop described in §4 of the paper)

1. **Planning thread**: subscribes to `/scan`, projects the laser points into
   world-frame obstacle disks ([`sensor_obstacles.py`](sensor_obstacles.py)), and runs
   TeLoGraF (pure flow + STLCG guidance) on the spot to generate a full reference
   trajectory **from the current pose**; it re-plans on a fixed **5 s** period,
   MPC-style.
2. **MPPI control loop** (~7 Hz): sampling-based MPC tracking of the latest reference
   trajectory. The cost combines trajectory tracking, progress toward the goal,
   avoidance of the sensed cylinders (with a safety margin for the robot radius), and
   velocity/smoothness terms; the softmax-weighted control is published to `/cmd_vel`.
3. **RViz markers** (latched / transient-local, fixed frame `map`): the start pose,
   **all** goal regions, the executed trajectory history, the sensing range, and the
   detected obstacles are all visible.

## Environment pairing

| Component | Version |
|---|---|
| Ubuntu | **22.04 (Jammy)** |
| ROS 2 | **Humble** (`/opt/ros/humble`) |
| Gazebo | **Gazebo Classic 11** (`gazebo_ros`) |
| Robot | **TurtleBot3 burger** (`ros-humble-turtlebot3*`) |

## One-time package installation

```bash
sudo apt-get update
sudo apt-get install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-gazebo \
    ros-humble-turtlebot3-msgs
```

## Source ROS in every new terminal (deliberately not in ~/.bashrc)

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger        # the launch file sets this too; exporting helps manual debugging
```

> This is deliberately kept out of `~/.bashrc` so that ROS does not push its
> `site-packages` into `PYTHONPATH` and pollute the Streamlit / Groq / TeLoGraF
> environments.

## Usage: two terminals

```bash
# ====== Terminal A: Gazebo Classic + TB3 + RViz (/scan works headless too) ======
source /opt/ros/humble/setup.bash
cd /home/jiachen-tlab-ut/Conferences/UbiComp/code
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=cond_reach_either rviz:=true
#  gui:=false (default) starts only gzserver, with no 3D window; watch the markers + /scan in RViz
#  once up, /scan, /odom, /cmd_vel and /ubicomp/markers are available

# ====== Terminal B: the closed-loop follower ======
source /opt/ros/humble/setup.bash
cd /home/jiachen-tlab-ut/Conferences/UbiComp/code
python3 sim_ros2/tb3_follower.py --case cond_reach_either
#  --replan-period 5.0 (default): TeLoGraF re-plans every 5 s
#  --sense-range  3.0: maximum range at which LiDAR points become obstacles
```

**Launch overrides**:

```bash
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=reach_avoid            # switch case
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=reach_avoid gui:=true  # open the Gazebo 3D window
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=reach_avoid rviz:=false
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=reach_avoid model:=waffle
```

## Example follower output

```
[TeLoGraF re-plan #1] from (-3.20,-3.20) -> 18 waypts; sensed 12 cylinders; 3.58s
[TeLoGraF re-plan #2] from (-1.84,-1.91) -> 16 waypts; sensed 19 cylinders; 3.73s
...
DONE at (1.94,-2.49); TeLoGraF re-plans=6; sensed 31 cylinder cells; keep-safe rho(G !unsafe) = +1.21 m (SAFE)
```

> **Measured (2026-06, local ROS 2 Humble + Gazebo Classic 11)**: `/scan` returns points
> normally in headless mode (360 beams), the TB3 is genuinely driven through `/cmd_vel`,
> and `/odom` reports the world-frame pose. Over one complete episode, all six TeLoGraF
> re-plans finished **in under 5 s** (3.58 / 3.73 / 3.75 / 4.27 / 3.85 / 4.03 s, mean
> ~3.9 s) and the robot reached the goal without collision (keep-safe ρ = +1.21 m). The
> 5 s re-planning period is therefore reasonable: planning always finishes inside the
> period and never blocks the MPPI control loop.

## A 2-D closed-loop top view without Gazebo (no ROS, no GPU)

The closed-loop top view (`closed_loop.png`) comes from a pure matplotlib version that
does not depend on ROS or Gazebo:

```bash
python3 sim_ros2/closed_loop_demo.py        # produces closed_loop.png
```

## Troubleshooting (all observed locally)

| Symptom | Cause / fix |
|---|---|
| `/spawn_entity` times out, the robot never appears | The world references an online `model://` model, so gzserver stalls while downloading. The ground and sun in this repository's world are already inlined, so this should not recur; if it reappears in a custom world, replace `model://...` with inlined geometry |
| The robot avoids its own goal and never arrives | The goal was spawned as a physical obstacle. Check that `tb3_world_gen.py` **skips** disks with `kind=="reach"` and only turns the red obstacles into physical bodies |
| No markers visible in RViz | The fixed frame must be `map`; the launch file already loads [`gui/markers.rviz`](gui/markers.rviz) and publishes the static `map→odom` TF. The marker publisher is latched (transient-local), so RViz still receives them when it connects late |
| `/scan` has no data | The LiDAR is Classic Gazebo's CPU `type="ray"` sensor (`libgazebo_ros_ray_sensor`), which also produces points headless; check that `TURTLEBOT3_MODEL` is set |
| **rviz2 fails to start with `undefined symbol: __libc_pthread_init ... GLIBC_PRIVATE` (exit 127)** | **A snap terminal (for example the snap build of VS Code) injects `/snap/.../lib` into `LD_LIBRARY_PATH`, so rviz2 links against the wrong libpthread. `tb3_sim.launch.py` now strips `/snap/` paths from `LD_LIBRARY_PATH` automatically; if it still recurs, run from a system (non-snap) terminal, or first run `export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH \| tr ':' '\n' \| grep -v /snap/ \| paste -sd:)`** |
| gzserver suddenly reports `exit code -9` | An **external SIGKILL**, commonly another `pkill -9 gazebo` elsewhere, or an OOM kill. gzserver normally stays alive; check that no other cleanup script or process is killing it |
| The robot does not move / "nothing happens" | The launch file **only starts the simulation**; the robot waits for the follower in the **second terminal**: `python3 sim_ros2/tb3_follower.py --case <id>`. The first ~10–30 s of follower startup are silently spent waiting for sensors and the TeLoGraF cold start; the log reports each stage |
| **Stuck at `planning the first TeLoGraF trajectory`, with each frame taking minutes** | **CPU contention**: torch takes every core by default (11 of 16 observed here) and competes with the Gazebo GUI (`gui:=true`) plus RViz, so a single frame grows from ~5 s to minutes (oversubscription plus laptop P/E-core throttling). Fixed: the subprocess now **limits torch to 6 threads by default** (`TELOGRAF_THREADS=N` to change it). **If it is still slow, use `gui:=false`** (headless Gazebo, watching only RViz, which is the cheapest); the Gazebo 3D window is the most CPU-hungry component |
| `ros2: command not found` / `rclpy` import errors | ROS was not sourced → `source /opt/ros/humble/setup.bash` |

## Files

| File | Purpose |
|---|---|
| `launch/tb3_sim.launch.py` | **Entry point A**: gzserver (open cylinder world) + TB3 spawn + `map→odom` TF + RViz markers |
| `tb3_follower.py`          | **Entry point B**: TeLoGraF re-plans every 5 s + MPPI tracking → `/cmd_vel`, reading `/odom` and `/scan` and publishing markers |
| `tb3_world_gen.py`         | Generates a Classic Gazebo `.world` programmatically from a case (inlined ground/sun + obstacle cylinders; reach goals skipped) |
| `sensor_obstacles.py`      | LaserScan → world-frame obstacle disks (pure functions, unit-testable); the **live sensor** obstacle source |
| `closed_loop_demo.py`      | Pure matplotlib 2-D closed loop (produces `closed_loop.png`), independent of ROS and Gazebo |
| `gui/markers.rviz`         | RViz configuration (fixed frame `map`, MarkerArray `/ubicomp/markers` + LaserScan + RobotModel) |
| `../keep_safe.py`          | "always keep safe" as a single sensor-grounded STL predicate `G(¬unsafe)`; `ρ(¬unsafe)` is the distance to the nearest obstacle |

## Connecting the real TeLoGraF model

Closed-loop planning uses genuine flow-matching sampling. For checkpoint and venv
installation see the "TeLoGraF installation and use" section of
[`code/README.md`](../README.md); if it is not installed yet, run this from the project
root first:

```bash
bash scripts/install_telograf.sh
```
