# sim_ros2/ — ROS 2 Humble + Gazebo Classic + TurtleBot3 闭环仿真

TurtleBot3(burger)在 **Gazebo Classic**(gazebo11 / `gazebo_ros`)里跑一个**真闭环**:
机器人用 360° LiDAR(`/scan`)在线感知障碍圆柱,TeLoGraF 周期性地从当前位姿重规划参考
轨迹,MPPI 控制器跟踪最新轨迹、发 `/cmd_vel`、读 `/odom`,在 RViz 里真实移动。

## 两个进程

| 角色 | 文件 | 作用 |
|---|---|---|
| 仿真 | [`launch/tb3_sim.launch.py`](launch/tb3_sim.launch.py) | gzserver(开放圆柱 world)+ TB3 spawn + `map→odom` 静态 TF + RViz markers |
| 闭环控制 | [`tb3_follower.py`](tb3_follower.py) | TeLoGraF 每 5s 重规划 + MPPI 轨迹跟踪 → `/cmd_vel` |

`case:=<case_id>` 一个参数定全部:case 定义在 [`planner/case_examples.py`](../planner/case_examples.py),
`map_hint.start` 决定 spawn 在哪里;world 由 [`tb3_world_gen.py`](tb3_world_gen.py) 从 case 程序化生成。

## 开放圆柱 world(不用墙)

场景里**只放圆柱障碍、不用墙体**:

- **红色障碍圆柱**是物理实体,机器人的 LiDAR 在线感知、绕开。
- **目标区(green)不生成物理圆柱** —— 否则 LiDAR 会把目标当障碍绕开、永远到不了;
  目标只作为 RViz marker 显示。
- ground / sun **内联**进 SDF(不引用 `model://` 在线模型库),所以 gzserver 启动快、
  `/spawn_entity` 服务不超时。

## 两阶段闭环(论文 §4 / 附录就是这个回路)

1. **规划线程**:订阅 `/scan`,把激光点投影成世界系障碍圆盘
   ([`sensor_obstacles.py`](sensor_obstacles.py)),当场跑 TeLoGraF(pure flow + STLCG 引导)
   **从当前位姿**生成整条参考轨迹;按固定 **5s 周期**滚动重规划(MPC 式)。
2. **MPPI 控制环**(~7 Hz):对最新参考轨迹做采样式 MPC 跟踪 —— 代价 = 轨迹跟踪 + 趋近目标
   + 绕开感知到的圆柱(机器人半径感知的安全余量)+ 速度/平滑项 —— 取 softmax 加权控制量
   发 `/cmd_vel`。
3. **RViz markers**(latched / transient-local,Fixed Frame `map`):起点、**所有**目标区、
   已走历史轨迹、传感范围、探测到的障碍,全部可见。

## 环境配对

| 组件 | 版本 |
|---|---|
| Ubuntu | **22.04 (Jammy)** |
| ROS 2 | **Humble** (`/opt/ros/humble`) |
| Gazebo | **Gazebo Classic 11** (`gazebo_ros`) |
| Robot | **TurtleBot3 burger** (`ros-humble-turtlebot3*`) |

## 一次性装包

```bash
sudo apt-get update
sudo apt-get install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-gazebo \
    ros-humble-turtlebot3-msgs
```

## 每个新终端先 source(不写进 ~/.bashrc)

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger        # launch 内部也会设;export 方便手动调试
```

> 故意不写进 `~/.bashrc`,避免 ROS 把它的 `site-packages` 塞进 `PYTHONPATH` 污染
> Streamlit / Groq / TeLoGraF 的环境。

## 用法:两个终端

```bash
# ====== 终端 A:起 Gazebo Classic + TB3 + RViz(headless 也能出 /scan)======
source /opt/ros/humble/setup.bash
cd /home/jiachen-tlab-ut/Conferences/UbiComp/code
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=cond_reach_either rviz:=true
#  gui:=false(默认)只起 gzserver,不开 3D 窗口;在 RViz 看 markers + /scan
#  起来后有 /scan、/odom、/cmd_vel、/ubicomp/markers

# ====== 终端 B:起闭环 follower ======
source /opt/ros/humble/setup.bash
cd /home/jiachen-tlab-ut/Conferences/UbiComp/code
python3 sim_ros2/tb3_follower.py --case cond_reach_either
#  --replan-period 5.0(默认):TeLoGraF 每 5s 重规划一次
#  --sense-range  3.0:把 LiDAR 点投影成障碍的最大距离
```

**launch 覆盖参数**:

```bash
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=reach_avoid            # 换 case
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=reach_avoid gui:=true  # 开 Gazebo 3D 窗口
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=reach_avoid rviz:=false
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=reach_avoid model:=waffle
```

## follower 输出示例

```
[TeLoGraF re-plan #1] from (-3.20,-3.20) -> 18 waypts; sensed 12 cylinders; 3.58s
[TeLoGraF re-plan #2] from (-1.84,-1.91) -> 16 waypts; sensed 19 cylinders; 3.73s
...
DONE at (1.94,-2.49); TeLoGraF re-plans=6; sensed 31 cylinder cells; keep-safe rho(G !unsafe) = +1.21 m (SAFE)
```

> **实测(2026-06,本机 ROS 2 Humble + Gazebo Classic 11)**:headless 下 `/scan` 正常出点
> (360 beam),TB3 用 `/cmd_vel` 真驱动、`/odom` 报世界系位姿。一次完整 episode:6 次
> TeLoGraF 重规划**全部 < 5s**(3.58 / 3.73 / 3.75 / 4.27 / 3.85 / 4.03 s,均值 ~3.9s),
> 机器人无碰撞到达目标(keep-safe ρ = +1.21 m)。所以 5s 重规划周期合理:规划耗时始终
> 落在周期内,不会阻塞 MPPI 控制环。

## 不开 Gazebo 先看 2D 闭环俯视图(无 ROS / 无 GPU)

论文附录的闭环俯视图(`closed_loop.png`)来自纯 matplotlib 版本,不依赖 ROS / Gazebo:

```bash
python3 sim_ros2/closed_loop_demo.py        # 出 closed_loop.png(论文附录那张俯视图)
```

## 排查清单(本机实测)

| 现象 | 原因 / 修法 |
|---|---|
| `/spawn_entity` 超时、机器人没出现 | world 引用了 `model://` 在线模型 → gzserver 卡在下载。本仓库的 world ground/sun 已内联,不该再出现;若自定义 world 时复现,把 `model://...` 换成内联几何 |
| 机器人绕开自己的目标、到不了 | 目标被当成了物理障碍。确认 `tb3_world_gen.py` **跳过** `kind=="reach"` 的圆盘(只把红色 obstacle 变物理体) |
| RViz 里看不到 markers | Fixed Frame 要设成 `map`;launch 已加载 [`gui/markers.rviz`](gui/markers.rviz) 并发 `map→odom` 静态 TF;marker 发布器是 latched(transient-local),RViz 晚连也能拿到 |
| `/scan` 没数据 | 用 Classic Gazebo 的 CPU `type="ray"` LiDAR(`libgazebo_ros_ray_sensor`),headless 也出点;确认 `TURTLEBOT3_MODEL` 已设 |
| **rviz2 起不来,报 `undefined symbol: __libc_pthread_init ... GLIBC_PRIVATE`(exit 127)** | **snap 终端(如 snap 版 VS Code)把 `/snap/.../lib` 注入 `LD_LIBRARY_PATH`,rviz2 链到了错的 libpthread。`tb3_sim.launch.py` 现在会自动从 `LD_LIBRARY_PATH` 剔除 `/snap/` 路径;若仍复现,改用系统终端(非 snap)运行,或先 `export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH \| tr ':' '\n' \| grep -v /snap/ \| paste -sd:)`** |
| gzserver 突然 `exit code -9` | 被**外部 SIGKILL**(常见:别处在跑 `pkill -9 gazebo`,或内存 OOM)。gzserver 正常会一直存活;确认没有别的清理脚本/进程在杀它 |
| 机器人不动 / 「没反应」 | launch **只起仿真**,机器人要等**第二个终端**起 follower 才动:`python3 sim_ros2/tb3_follower.py --case <id>`。follower 启动头 ~10-30s 静默是正常的(等传感器 + TeLoGraF 冷启动),日志会逐阶段提示 |
| **卡在 `planning the first TeLoGraF trajectory`、每帧规划要几分钟** | **CPU 抢占**:torch 默认吃满所有核(实测 11/16),和 Gazebo GUI(`gui:=true`)+ RViz 抢核 → 单帧从 ~5s 暴涨到几分钟(超订 + 笔记本 P/E 核降频)。已修:subprocess **默认把 torch 限到 6 核**(`TELOGRAF_THREADS=N` 可调)。**仍慢就 `gui:=false`**(headless Gazebo,只用 RViz 看,最省 CPU);Gazebo 3D 窗口最吃 CPU |
| `ros2: command not found` / `rclpy` import 报错 | 没 source → `source /opt/ros/humble/setup.bash` |

## 文件

| 文件 | 作用 |
|---|---|
| `launch/tb3_sim.launch.py` | **入口 A**:gzserver(开放圆柱 world)+ TB3 spawn + `map→odom` TF + RViz markers |
| `tb3_follower.py`          | **入口 B**:TeLoGraF 每 5s 重规划 + MPPI 跟踪 → `/cmd_vel`,读 `/odom`/`/scan`,发 markers |
| `tb3_world_gen.py`         | 从 case 程序化生成 Classic Gazebo `.world`(内联 ground/sun + 障碍圆柱;跳过 reach 目标)|
| `sensor_obstacles.py`      | LaserScan → 世界系障碍圆盘(纯函数,可单测),**实时传感器**障碍源 |
| `closed_loop_demo.py`      | 纯 matplotlib 2D 闭环(出论文附录的 `closed_loop.png`),不依赖 ROS / Gazebo |
| `gui/markers.rviz`         | RViz 配置(Fixed Frame `map`,MarkerArray `/ubicomp/markers` + LaserScan + RobotModel)|
| `../keep_safe.py`          | "always keep safe" = 单个传感器接地 STL 谓词 `G(¬unsafe)`;`ρ(¬unsafe)`=到最近障碍距离 |

## 接 TeLoGraF 真模型

闭环规划走真正的 flow-matching 采样。checkpoint / venv 的安装见
[`code/README.md`](../README.md) 的 "TeLoGraF 安装与使用" 一节;没装的话先在项目根跑:

```bash
bash scripts/install_telograf.sh
```
