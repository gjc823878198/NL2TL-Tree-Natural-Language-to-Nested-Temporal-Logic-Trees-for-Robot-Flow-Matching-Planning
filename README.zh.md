# NL → STL-Tree 项目(UbiComp 2026 poster 配套代码)

这个目录里实现的是 poster 流水线的端到端 demo:

1. **NL → STL Tree**:把自然语言解析成嵌套 STL 语法树(JSON AST)
2. **可视化检查**:Streamlit 工具逐条验证转换正确性
3. **LLM 端到端**:Groq Llama-3.3-70b 做 NL → tree,带 5 项评估指标
4. **STL Tree → Plan**:`planner.plan_waypoints(case, backend=…)` 把 case dict 变成 (x,y) 轨迹
   - `backend=fallback`:鲁棒性最大化的解析路径(总能跑通)
   - `backend=telograf`:走 TeLoGraF flow-matching subprocess
5. **Plan → Sim**:2D matplotlib 和 ROS 2 Humble + Gazebo Classic + TurtleBot3 **共用同一个 planner 入口**
   - 2D 端:`python3 sim2d/run_demo.py --case <id> --backend <auto|telograf|fallback>`
   - ROS 端:**真闭环** —— `tb3_sim.launch.py`(Gazebo 起 TB3 + 圆柱 world) + `tb3_follower.py`(TeLoGraF 每 5s 重规划 + MPPI 跟踪)
   - backend 切换只改一个参数,两边语义一致

---

## 安装

```bash
cd code
pip install -r requirements.txt
```

依赖:`lark` / `streamlit` / `gdown` / `groq` / `zss`。可视化用 Streamlit 内置的 `st.graphviz_chart`,**不需要本地装 graphviz 二进制**。

---

## 文件说明

| 文件 | 作用 |
|---|---|
| **解析层** | |
| `stl_parser.py` | STL 文法 (Lark) + AST 转换 + 反向渲染(AST→STL)+ Graphviz DOT 渲染 |
| `tests/test_parser.py` | 解析器单元测试 + 往返一致性检查 |
| **数据层** | |
| `download_nl2tl.py` | 克隆 NL2TL 仓库 + 从 Google Drive 拉完整数据集 |
| `convert_dataset.py` | 批量把数据集里的 STL 公式转成 JSONL(含 tree 字段) |
| `gen_synthetic.py` | 模板合成 NL-STL 对(50 条 / 9 种结构,纯离线) |
| **LLM 层** | |
| `nl_to_tree_groq.py` | Groq 客户端——Llama-3.3-70b,免费 30 RPM / 14,400 RPD / 100k TPD |
| **评估层** | |
| `tree_metrics.py` | `exact_match` / `op_f1` / `path_f1` / `ted` / `ted_norm` 五项指标 |
| `rescore.py` | 复用已有 predictions JSONL 重算指标,**不烧 quota** |
| **可视化** | |
| `viz_app.py` | Streamlit 工具:Single formula / Browse / Groq 三个模式 |
| **规划 + 仿真** | |
| `planner/` | NL→Tree 之后的 grounding + TeLoGraF 适配 + `plan_waypoints()` 入口(详见 `planner/README.md`)|
| `sim2d/` | matplotlib 2D 仿真,生成 PNG/GIF 演示动画 |
| `sim_ros2/` | ROS 2 Humble + Gazebo Classic + TurtleBot3 **闭环**仿真(`tb3_sim.launch.py` + `tb3_follower.py`,TeLoGraF 重规划 + MPPI 跟踪)|
| `tools/telograf_export.py` | TeLoGraF subprocess bridge:venv 内载入 checkpoint、采样,JSON 出轨迹 |
| `scripts/install_telograf.sh` | 一键装 TeLoGraF 源 + venv + matching deps |
| `external/TeLoGraF/` | (脚本拉的)TeLoGraF 源码,不进 git |
| `.venv-telograf/` | (脚本建的)TeLoGraF 专用 venv,不进 git |
| `outputs/` | 自动生成的渲染结果(`sim2d/*.png` 和 `sim_ros2/*.sdf`)|
| **文档** | |
| `AST_NODES.md` | 每种 AST 节点的语义说明 |

---

## AST JSON Schema

```json
{
  "op":       "atom | not | and | or | imply | iff | globally | finally | until",
  "interval": [low, high] | null,
  "children": [...]              // non-atom 才有
  "name":     "prop_1"           // atom 才有
}
```

`and` / `or` 的同操作符链(`a & b & c`)会被**扁平化**为单个节点的多个 children。详细见 [AST_NODES.md](AST_NODES.md)。

---

## 完整流程

```
gen_synthetic.py     ──┐
                       ├──► (NL, STL) JSONL ──► convert_dataset.py ──► converted.jsonl
download_nl2tl.py    ──┘                                                   │
                                                                           ▼
                                ┌──────────────────────────────────┐
                                │  viz_app.py (Streamlit)          │
                                │  · Single formula                │
                                │  · Browse converted dataset      │
                                │  · Groq:   NL → tree             │◄─── nl_to_tree_groq.py
                                └──────────────────────────────────┘
                                                │
                                                ▼
                                       predictions.jsonl
                                                │
                                                ▼
                                          rescore.py
                                                │
                                                ▼
                                exact_match / op_f1 / path_f1 / ted / ted_norm
```

---

## 步骤 1 — 跑解析器测试(不联网、不用数据)

```bash
python3 tests/test_parser.py
```

应输出 `11 / 11 passed` 和 `11 / 11 round-trip OK`。

## 步骤 2 — 启动可视化工具(不联网,Single formula 模式)

```bash
streamlit run viz_app.py
```

三个 tab:
- **Single formula**:粘 STL 公式 → 看树形图 + JSON AST + 反向渲染
- **Browse converted dataset**:翻看 `data/*.jsonl`,挑错存到 feedback 文件
- **Groq: NL → tree**:输入 NL → Llama-3.3 实时生成树(需 `GROQ_API_KEY`)

## 步骤 3 — 下载 NL2TL 数据集

```bash
python3 download_nl2tl.py
```

会自动克隆 NL2TL repo + 用 gdown 拉 Google Drive 上的完整数据(39k 对)。下完会自动列出可用文件。

**没网或不想拉 Drive**:用 `gen_synthetic.py -n 50 -o data/synth.jsonl` 生成 50 条合成数据离线验证。

## 步骤 4 — 批量转换 STL 公式 → 树

```bash
python3 convert_dataset.py \
  data/nl2tl_dataset/Data_lifted_total39378_05_19/lifted_data.jsonl \
  data/nl2tl_converted.jsonl \
  --limit 2000
```

输出每行:
```json
{"natural": "...", "formula": "...", "tree": {...}, "ok": true}
```

脚本自动检测 CSV/TSV/JSON/JSONL,识别 `natural`/`nl`/`logic_sentence` 和 `formula`/`ltl`/`logic_ltl` 等字段。`--keep-failed` 把解析失败的也写出来(用于回头补语法)。

## 步骤 5 — 回到 viz_app 的 Browse 模式核对

逐条核对样本的 NL ↔ formula ↔ tree 是否吻合,发现错误存到 `*.feedback.jsonl`。

---

## 步骤 6 — LLM 端到端调用:NL → tree

文件:`nl_to_tree_groq.py`。Groq 免费层 **30 RPM / 14,400 RPD / 100k TPD**(70b 模型;8b 模型 TPD 是 500k),~500 tokens/sec,无需信用卡。

```bash
# 拿 key:https://console.groq.com/keys → Google/GitHub 登录 → Create API Key
echo 'export GROQ_API_KEY=gsk_你的key' >> ~/.bashrc
source ~/.bashrc
echo $GROQ_API_KEY | head -c 20 && echo "..."

# (a) 单句快测
python3 nl_to_tree_groq.py --nl "Always within time 5 to 10, prop_1 must hold."

# (b) 批量 + 评估(默认 25 RPM 安全冗余,100 条约 4 分钟)
python3 nl_to_tree_groq.py \
  --input  data/nl2tl_converted.jsonl \
  --output data/predictions.jsonl \
  --limit 100 --eval

# (c) 想跑更快:把 RPM 拉到 30(贴近免费层上限)
python3 nl_to_tree_groq.py --input ... --output ... --eval --rpm 30

# (d) 模型切换
python3 nl_to_tree_groq.py --nl "..." --model llama-3.1-8b-instant   # 更快但质量略低
python3 nl_to_tree_groq.py --nl "..." --model qwen2.5-32b            # 均衡
```

### 其他要点

- **Schema 和 few-shot**:从 `nl2tl_converted.jsonl` 抽 8 条覆盖各算子
- **指标**:`exact_match` / `op_f1` / `path_f1` / `ted` / `ted_norm`,每行写进输出 JSONL 的 `metrics` 字段,末尾汇总
- **Rate limit + 重试**:免费层 RPM 限速、429/5xx 自动指数退避、解析服务端 `retry-after` 提示;**TPD 耗尽 → fail-fast**(不会无意义重试一小时)
- **强 JSON 约束**:`response_format=json_object`(OpenAI 风格)

---

## 评估指标(看 `tree_metrics.py`)

| 指标 | 含义 | 读法 |
|---|---|---|
| `exact_match` | 严格相等(`and`/`or` 子节点已排序归一化) | 1.0 = 完全对 |
| `op_f1` | 算子多重集合 F1 | "用对了哪些算子吗" |
| `path_f1` | 根到叶路径多重集合 F1 | "嵌套/作用域对吗" |
| `ted` | 树编辑距离(Zhang-Shasha,通过 `zss` 库) | 几次编辑能改对 |
| `ted_norm` | 1 − TED / max(\|pred\|, \|gold\|) | 1.0 = 完全相同 |

**典型诊断:** `op_f1 = 0.94` 但 `path_f1 = 0.52` → 算子全对、嵌套一半错 → 走 chain-of-thought / 自一致性能修,不必微调。

## 每日评测协议(成功率 = 3 天 pooled exact-match)

70b 免费层每天约 **100k tokens ≈ 48 次调用**(实测 48 次用 98,680 tokens,卡的是每日 token 上限,不是请求数)。所以**实验设为每天 40 条**(压在配额线下留缓冲,不会跑一半撞 429),连跑 3 天对**不相交的行**累计 ~120 条,取 pooled exact-match 作为论文成功率。

> ⚠️ **`temperature=0` + 固定 few-shot seed 是确定性的**:每天跑同样的行会得到几乎一样的结果,"3 天平均"等于一次跑,没有统计意义。所以每天必须测**不同的行**——`daily_eval.sh` 自动把 offset 推进到"已测过的最远行"(day-1 = 行 0-49,day-2 = 50-89,day-3 = 90-129),保证不相交。

```bash
export GROQ_API_KEY=gsk_...
bash scripts/daily_eval.sh          # 自动:跑 40 条不相交 + 记录 + 更新 pooled 平均(offset 自动推进)
# 想开 #1/#4 提升(更费 token,每天能测的条数变少):
SC=5 ROUNDTRIP=1 bash scripts/daily_eval.sh
# 手动等价:
python3 nl_to_tree_groq.py --input data/nl2tl_converted.jsonl \
    --output data/predictions.jsonl --limit 40 --offset 50 --eval
python3 scripts/record_eval.py --offset 50 --window 40
```

结果存于 `outputs/nl2tl_tree_eval/`:`results.jsonl`(每日一条)、`SUMMARY.md`(每日表 + **pooled EM = 要写进论文的数**)、`predictions_<date>.jsonl`(当天原始预测,按日归档不被覆盖)。实测 day-1:n=48,EM 47.9%,op_f1 0.925,path_f1 0.557。

## 复用历史数据(不重新烧 quota)

```bash
python3 rescore.py data/predictions.jsonl --out data/predictions.rescored.jsonl
```

输出会打印 best/worst 案例,逐条带 NL + pred + gold 对比。

---

## 解析器支持的语法(STL)

| 类别 | 写法 |
|---|---|
| 谓词 | 任意标识符,典型 `prop_1`、`prop_42` |
| 否定 | `!`,`not`,`negation` |
| 与/或 | `&` / `and`,`\|` / `or` |
| 蕴含 / 等价 | `->` / `imply`,`<->` / `equal` / `iff` |
| 全局 | `G[a,b]`,`globally [a,b]`,**`globally <expr>`(无界)** |
| 终将 | `F[a,b]`,`finally [a,b]`,**`finally <expr>`(无界)** |
| 直到 | `prop_1 U[a,b] prop_2`,`prop_1 until [a,b] prop_2`,**无界形式** |
| 区间 | `[lo,hi]`,`[lo,infinite]` / `[lo,inf]` |
| 括号 | `(...)` |
| 优先级 | `!` > `G/F` > `U` > `&` > `\|` > `->` > `<->`(由高到低)|

NL2TL `lifted_data.jsonl` 2000 条全部通过解析(0 failure),包含大量无界 G/F/U。

---

## 当前数据状态(已下载)

```
data/
├── NL2TL_repo/                                            # NL2TL GitHub repo(代码)
├── nl2tl_dataset/                                         # Google Drive 完整数据(39k+)
│   ├── Data_lifted_total39378_05_19/lifted_data.jsonl     # 主数据集 39k 对
│   ├── Data_transfer_domain/                              # 迁移学习子集(28k+)
│   └── raw_data/                                          # 原始 span 标注
├── nl2tl_converted.jsonl                                  # 2000 条转换好的样本(含 tree)
├── synthetic_nl_stl.jsonl                                 # 50 条合成数据
├── synthetic_converted.jsonl                              # 上面的转换版
├── predictions.jsonl                                      # LLM 推理结果(--eval 自带 metrics)
└── predictions.rescored.jsonl                             # rescore.py 加完指标的版本
```

---

## 步骤 7 — 规划 + 仿真

### 7A — 2D matplotlib 仿真(立刻可跑,无需 ROS)

2D 仿真和 3D ROS 仿真**走同一个 planner 入口** —— `planner.plan_waypoints(case, backend=…, obstacles=…)`。

```bash
python3 sim2d/run_demo.py --all --static       # 5 个 case 全出 PNG(默认 backend=role)
python3 sim2d/run_demo.py --all                # 5 个 case 全出 GIF
python3 sim2d/run_demo.py --case reach_avoid   # 单个 case
```

输出默认落在 `outputs/sim2d/<case>.{png,gif}`。

#### 5 个 case 按 role 分两类(诚实区分 TeLoGraF 能 / 不能)

我们实测分析了 TeLoGraF `simple_gnn_F` checkpoint 的能力边界(见下 §7C),据此把 case 分成两类,每个 case 在 `case_examples.py` 里带 `role` 字段:

| case | role | STL | 障碍 | TeLoGraF 表现 |
|---|---|---|---|---|
| `reach_within_T`   | **telograf** | `F[0,12](g) ∧ G¬o1 ∧ G¬o2` | 2 | ✅ 纯 TeLoGraF+guidance |
| `reach_avoid`      | **telograf** | `F[20,55](g) ∧ G¬o1 ∧ G¬o2 ∧ G¬o3` | 3 | ✅ 纯 TeLoGraF+guidance |
| `reach_goal_north` | **telograf** | `F[20,55](g) ∧ G¬o1 ∧ G¬o2 ∧ G¬o3` | 3 | ✅ 纯 TeLoGraF+guidance |
| `seq_reach_ABC`    | fallback | `F(A ∧ F(B ∧ F(C)))` 三段顺序 + 程序化墙 | 8 | ❌ 结构 OOD → A* 兜底 |
| `trigger_response` | fallback | `G(trigger → F[0,5] safe)` imply + 程序化墙 | 9 | ❌ 结构 OOD → A* 兜底 |

- **`role=telograf`**:`backend=role` 时走**纯 TeLoGraF + STLCG guidance**(扩散给形状,可微 guidance 避开所有障碍),**不碰 A\***。每个 case 避 2-3 个障碍,NL 用通用 "keep safe / avoid all obstacles"。
- **`role=fallback`**:故意设计成 TeLoGraF 结构性跑不出的(3 段顺序到达 / imply),用来诚实展示"扩散失败 → A* 兜底"。注意失败原因是 STL **结构** OOD,**不是避障**——避障已经被 guidance 解决了。

`run_demo.py --backend role`(默认)按 role 自动选 backend。也可强制:`--backend telograf|fallback|auto`。

#### Backend 行为

| `--backend` | 用什么 |
|---|---|
| `role`     | **默认**:telograf-case 用纯 TeLoGraF,fallback-case 用 auto |
| `telograf` | 强制 TeLoGraF best-of-N + 验证 + (失败才)A* 修复 |
| `auto`     | 先试 TeLoGraF,失败/无 checkpoint 退 A* |
| `fallback` | 永远 A\* 网格规划,适合纯调试 |

### 7B — ROS 2 Humble + Gazebo Classic 仿真:TurtleBot3 闭环

ROS 端是一个**真闭环**:TurtleBot3(burger)在 **Gazebo Classic**(gazebo11 / `gazebo_ros`)里跑,
场景里**只放圆柱障碍、不用墙体**。机器人用 360° LiDAR(`/scan`)**在线**感知障碍圆柱,
TeLoGraF 周期性地从当前位姿重规划参考轨迹,MPPI 控制器跟踪最新轨迹、发 `/cmd_vel` 真驱动机器人。

| 角色 | 文件 | 作用 |
|---|---|---|
| 仿真 | [`sim_ros2/launch/tb3_sim.launch.py`](sim_ros2/launch/tb3_sim.launch.py) | gzserver(开放圆柱 world)+ TB3 spawn + RViz markers |
| 闭环控制 | [`sim_ros2/tb3_follower.py`](sim_ros2/tb3_follower.py) | TeLoGraF 每 5s 重规划 + MPPI 轨迹跟踪 → `/cmd_vel` |

**两阶段闭环**(论文 §4 / 附录描述的就是这个回路):

1. **规划线程**:订阅 `/scan`,把激光点投影成世界系障碍圆盘(`sensor_obstacles.py`),
   当场跑 TeLoGraF(pure flow + STLCG 引导)**从当前位姿**生成整条参考轨迹;按固定 **5s 周期**
   滚动重规划(MPC 式)。
2. **MPPI 控制环**(~7 Hz):对最新参考轨迹做采样式 MPC 跟踪 —— 代价 = 轨迹跟踪 + 趋近目标 +
   绕开感知到的圆柱(机器人半径感知的安全余量)+ 速度/平滑项 —— 取 softmax 加权控制量发 `/cmd_vel`。
3. **RViz markers**(latched / transient-local,Fixed Frame `map`):起点、**所有**目标区、
   已走历史轨迹、传感范围、探测到的障碍,全部可见。

#### 一次性装包(Classic Gazebo + TurtleBot3)

```bash
sudo apt-get install -y ros-humble-gazebo-ros-pkgs \
    ros-humble-turtlebot3 ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-msgs
```

#### 每个新终端先 source(不写进 ~/.bashrc,避免污染 Streamlit/Groq/TeLoGraF 的 PYTHONPATH)

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger        # launch 内部也会设;这里 export 方便手动调试
```

#### 启动:两个终端(**必须同时开着**)

> ⚠️ **三条铁律(都是踩过的坑)**
> 1. **每次启动前先清残留 Gazebo**:`pkill -9 -f gzserver; pkill -9 -f gzclient`。否则旧进程占着端口 11345,新 launch 直接 `gzserver exit 255` + `Entity [tb3_burger] already exists`。
> 2. **两个终端必须同时运行**:终端 A(仿真)**全程开着、别 Ctrl-C**;终端 B(follower)另开一个终端、A 还活着时再跑。关了 A,机器人就没了,B 自然没反应。
> 3. **用系统终端(GNOME Terminal),别用 snap 版 VS Code 的集成终端**:后者会让 rviz2 崩(`undefined symbol: __libc_pthread_init`)。launch 已自动从 `LD_LIBRARY_PATH` 剔除 `/snap/` 路径,若仍崩就换系统终端。

```bash
# ====== 终端 A:起 Gazebo + TB3 + RViz(开着别关)======
pkill -9 -f gzserver; pkill -9 -f gzclient        # 先清残留,避免 255
source /opt/ros/humble/setup.bash
cd /home/jiachen-tlab-ut/Conferences/UbiComp/code
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=cond_reach_either gui:=true rviz:=true
#  gui:=true  -> 开 Gazebo 3D 窗口(看灰地面 + 红圆柱 + TB3;首次 15-30s 编译着色器,别急)
#  gui:=false(默认)-> 只起 gzserver,轻量;在 RViz 看 markers + /scan
#  起来后有 /scan /odom /cmd_vel /ubicomp/markers;这个终端要一直开着!

# ====== 终端 B:另开一个终端,A 还开着时跑 follower ======
source /opt/ros/humble/setup.bash
cd /home/jiachen-tlab-ut/Conferences/UbiComp/code
python3 sim_ros2/tb3_follower.py --case cond_reach_either
#  启动头 ~10-30s 静默是正常的(等传感器 + TeLoGraF 模型冷启动);日志会逐阶段提示,别当卡死
#  --replan-period 5.0(默认):TeLoGraF 每 5s 重规划;--sense-range 3.0:LiDAR 投影最大距离
```

follower 会**逐阶段打印**整条 Task → TeLoGraF → Controller 链路:

```
Task: case='cond_reach_either', 2 reach goal(s); waiting for /odom + /scan ...
got /odom + /scan (robot at (-3.20,-3.20)); planning the first TeLoGraF trajectory
   -- model cold-start can take ~10-30 s ... (this is not a hang)
[Route B] reach deadline = 53s; exact-STL shield armed
goal locked at (2.60,-2.40); MPPI control loop running -> publishing /cmd_vel
  driving t=2s pose=(-2.95,-3.18) clearance=1.94m        # 每 ~2s 心跳,机器人在动
  [TeLoGraF re-plan #1] from (-2.61,-3.12) -> 64 waypts; sensed 9 cylinders; 4.25s
  ... 一路开向目标 ...
DONE at (2.09,-2.47); reached=True; TeLoGraF re-plans=6; safe-stops=0
[CERTIFY exact monitor] keep-safe rho(G!unsafe)=+1.24m (SAFE);
                        reach-by-deadline rho(F[0,53s])=+0.09m, t=34/53s (IN-TIME)
```

> **「终端 B 没反应」排查**:① 打印 `NO /odom or /scan after 10 s` → B 看不到 A 的话题:在 B 里查 `echo $ROS_DOMAIN_ID`(两端必须一致)、`ros2 topic echo /odom --once`(应有数据);② 连 `Task: ...` 都没有 → follower 没起来(目录 / `source` 不对);③ 有 `MPPI control loop running` 但 Gazebo 里不动 → 单独测 `ros2 topic pub -r 5 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.15}}'`。详见 [`sim_ros2/README.md`](sim_ros2/README.md) 排查清单。

> **实测状态(2026-06,本机 ROS 2 Humble + Gazebo Classic 11)**:`/scan` 正常出点(360 beam),
> TB3 用 `/cmd_vel` 真驱动、`/odom` 报世界系位姿。一次完整 episode:6 次 TeLoGraF 重规划**全部
> < 5s**(均值 ~3.9s),机器人**无碰撞到达目标**,且 Route B 的精确 STL 监控在**实走轨迹**上同时
> 认证 **keep-safe ρ=+1.24 m(SAFE)** 与 **reach-by-deadline ρ=+0.09 m, 34/53s(IN-TIME)**。
> 这证明 5s 重规划周期合理(规划耗时始终落在周期内,不阻塞 MPPI 控制环)。

> **为什么用圆柱、不用墙;为什么目标不是物理体。** world 由 [`sim_ros2/tb3_world_gen.py`](sim_ros2/tb3_world_gen.py)
> 从 case 生成:只有红色**障碍**圆柱是物理实体;**目标区(green)不生成物理圆柱** —— 否则机器人的
> LiDAR 会把目标当障碍绕开、永远到不了;目标只作为 RViz marker 显示。ground/sun 内联进 SDF
> (不碰在线模型库),所以 gzserver 启动快、`/spawn_entity` 不超时。

**常用参数**:

```bash
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=reach_avoid            # 换 case
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=reach_avoid gui:=true  # 顺便开 Gazebo 3D 窗口
ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=reach_avoid rviz:=false
python3 sim_ros2/tb3_follower.py --case reach_avoid --replan-period 5.0 --sense-range 3.0
```

#### 纯 2D 闭环俯视图(无 ROS / 无 GPU)

论文附录里的闭环俯视图(`closed_loop.png`)来自纯 matplotlib 版本,不依赖 ROS / Gazebo,
适合快速复现/迭代:

```bash
python3 sim_ros2/closed_loop_demo.py        # 出 closed_loop.png(论文附录那张俯视图)
```

---

### 7C — TeLoGraF 安装与使用(diffusion planner 真的接通了)

`backend=telograf` 跑的是**真正的 flow-matching 扩散采样 + best-of-N 选择 + STL 验证**。3 个 `role=telograf` case **纯 TeLoGraF**(不碰 A*)就满足 STL;2 个 `role=fallback` case 故意 OOD,TeLoGraF 失败后 A* 兜底。

#### 流程(每次 `plan_waypoints(case, backend="telograf")`)

```
1. case_to_graph(case)          # tree -> 8-d TeLoGraF node features
                                  # [node_type, ts, te, x, y, z, r, left_child]
                                  # 严格对齐 train_gstl_v1.py::get_graph_stl_embed_from_tree
2. 收集 refine_obstacles        # STL avoid 原子 + 程序化墙/家具(切成圆)
                                  #   —— 全部障碍, 喂给下一步的 guidance
3. tools/telograf_export.py     # .venv-telograf 子进程, 一次载模型:
   ├ args.npz / model_last.ckpt #   重建模型 + load_state_dict
   ├ GCN encoder(ego, batch)    #   graph -> 258-d cond emb
   ├ GaussianFlow                #   批量采 N 条(test_muls), 100 步 flow, [N,64,4]
   ├ 反归一化 ×5 + best-of-N     #   _sample_cost 选最优一条 (学到的轨迹"先验")
   └ _stl_refine (STLCG guidance)#   ★ 对最优轨迹做 200 步 Adam 可微优化:
                                  #     loss = ReLU(障碍违反)² + 到达goal² + start锚 + 平滑
                                  #     把轨迹推出每个 obstacle、拉到 goal
4. _trajectory_validates        # 验证 STL+碰撞。过 -> 纯 TeLoGraF(+guidance), 不碰 A*
5. (仅结构 OOD case) A* 修复/兜底
6. _pin_reaches                 # 把轨迹末端钉到 reach goal 圆心
```

#### 两个关键机制

1. **best-of-N**(对齐论文 `test_muls`):单条 flow 采样有噪声,论文采最多 1024 条选最优。我们一次 forward 批量采 N 条(`map_hint.telograf_samples`=48/64)。
2. **STLCG-style 可微 guidance**(对齐论文 CTG/LTLDoG):**这就是把地图障碍喂给扩散 planner 的通道**。GNN 给轨迹形状,guidance 用所有障碍(STL avoid 原子 + 程序化墙)做可微损失的梯度,把轨迹推出障碍 + 拉到 goal。实测连一条直穿 5 个障碍(含正中心)的轨迹都能被 refine 干净。
   - bare flow 模型本身避障很弱(2 个路径上的障碍只有 ~50% 采样能避开,加大 N 也救不了);**guidance 才是让多障碍避障稳定的关键**。

#### 验证结果

逐 case 跑 2D 仿真 `python3 sim2d/run_demo.py --case <id> --backend role`(纯 Python,无 ROS)得到(planner 层面的结果,与 ROS 闭环用的是同一个 `plan_waypoints` 入口):

| case | role | backend | 障碍数 | planOK | reach goal | obstacle hits | STL atoms |
|---|---|---|---|---|---|---|---|
| reach_within_T   | telograf | **telograf** | 2 | ✅ | ✅ | 0 | 1/1 |
| reach_avoid      | telograf | **telograf** | 3 | ✅ | ✅ | 0 | 1/1 |
| reach_goal_north | telograf | **telograf** | 3 | ✅ | ✅ | 0 | 1/1 |
| seq_reach_ABC    | fallback | auto (→A*) | 8 (墙) | ✅ | ✅ | 0 | 3/3 |
| trigger_response | fallback | auto (→A*) | 9 (墙) | ✅ | ✅ | 0 | 2/2 |

3 个 telograf case 用**纯 TeLoGraF + STLCG guidance**(日志 `best-of-N diffusion + STLCG guidance PASSES STL+collision check (pure TeLoGraF, no A*)`),每个避开 2-3 个障碍,轨迹是光滑扩散曲线(见 `outputs/sim2d/*.png`)。2 个 fallback case 因 STL **结构** OOD(3 段顺序 / imply)退到 A*。

#### TeLoGraF 能 / 不能(实测 profile)

| STL 形式 | bare flow best-of-N | + STLCG guidance |
|---|---|---|
| `F[t1,t2](goal)` 带时间窗的纯 reach | ✅ 24/24 | ✅ |
| reach + 1 障碍在直线旁 | ✅(N≥32)| ✅ |
| reach + 障碍正压直线 / 2-5 个障碍 | ❌ ~0-50% | **✅ 稳定**(guidance 推出去)|
| 裸 `F(goal)` 无时间窗 | ❌(必须带 `[t1,t2]`)| ❌(时间窗影响的是 GNN 先验)|
| 3 段顺序 `F(A∧F(B∧F(C)))` | ❌ 0/32 | ❌(顺序到达是结构问题,guidance 救不了)|

**结论**:guidance 解决了**避障**(多障碍、路径上的障碍都能避);但解决不了 **STL 结构 OOD**(顺序多目标、imply)——后者 GNN 先验本身就跑偏,guidance 只调几何不改访问结构。所以 telograf case = "reach + 任意避障",fallback case = "结构 OOD"。

#### STL 因果验证(STL-Tree 真的在驱动扩散吗?)

为回答"STL-Tree 有没有起作用",做了 3 个对照实验(`TELOGRAF_NO_REPAIR=1`,去掉一切后处理):

1. **改 goal 坐标 → 扩散终点跟着跑**(单 reach `F[0,12]`):

   | goal | 扩散终点 | 误差 |
   |---|---|---|
   | (+3,+3) | (+3.60,+3.49) | 0.78 |
   | (−3,+3) | (−3.22,+3.12) | 0.26 |
   | (+3,−3) | (+3.42,−3.85) | 0.95 |
   | (−3,−3) | (−3.25,−3.13) | 0.28 |
   | (0,+3.5)| (+0.23,+3.92) | 0.48 |

   4 个象限 + 中央都跟对了 → encoder 真读了 STL 里 reach 原子的几何。

2. **reach vs avoid 同一个点 → 行为相反**(`TELOGRAF_NO_SHIFT=1`,平移不污染形状):
   - `F[0,12] reach(p@2.5,2.5)`:轨迹均距 p = **1.59**,终点落 (2.88,2.87) → 拉到点上
   - `G ¬(p@2.5,2.5)`:轨迹均距 p = **7.26**,终点跑 (−1.46,−5.71) → 推开

3. **改 start → 纯模型起点跟着动**(NO_SHIFT,纯模型):起点误差仅 0.05–0.13(ego conditioning 真生效;平移只做最后 ~0.1m 收尾)。

结论:**STL-Tree 不是摆设**——它的几何、算子(reach/avoid)、ego 都因果地控制扩散输出。复现:
```bash
# 在 README 历史里有完整脚本;核心是对 _telograf_subprocess 改 grounding 再看终点
TELOGRAF_NO_REPAIR=1 TELOGRAF_NO_SHIFT=1 python3 -c "..."
```

#### 修了的几个 upstream / project bug

| Bug | 修法 |
|---|---|
| 8-d node feature 的第 8 列 我写的是 `n_child`,upstream 训练用的是 `left_child` (+1 for until's first child, -1 otherwise) | 改写 `planner/telograf_adapter.py` 对齐 |
| 非 reach 节点的 (x,y,z,r) 我填了 `0`,upstream 训练时填 `-1` | 同上 |
| `case_to_graph` 把 `not(reach)` 折叠成 `avoid` 原子,TeLoGraF 没有 avoid 原子,期望保留 NOT 节点 | normalise 不再折叠 |
| 把 procedural 障碍 augment 进 STL graph (节点数 6 → 40+),把模型推出训练分布 | 默认 OFF(`TELOGRAF_AUGMENT_OBSTACLES=1` 可启) |
| `tools/telograf_export.py` 的 model rebuild 是 stub | 完整重建,处理 `args.npz` 里可能缺的属性 (flow_pattern, guidance_*) |
| Procedural 墙横穿 reach 原子 | `sim2d/env_2d.py::_gap_aligned_with_atoms` 让墙缝自动对齐 reach 位置 |
| 反归一化 stat 用近似值 | hardcoded `[0,0,0,0]/[5,5,1,1]` ←直接抄 `train_gstl_v1.py` 的 simple env 路径(line 1342-1343) |
| 单条 flow 采样噪声大,避障/到达不稳 | **best-of-N 批量采样**(`telograf_samples`),`_traj_stl_cost` 选最优,对齐论文 `test_muls` |
| 裸 `F(goal)` OOD、3 段顺序 OOD、障碍压直线 OOD | case 重设计:reach 带时间窗 + 障碍放直线旁 + 按 `role` 分 telograf/fallback |
| reach 轨迹擦 goal 边缘不进圈 | `_pin_reaches` 把轨迹钉到 reach 圆心(对所有 backend)|

#### 给 TeLoGraF 的信息(对齐 upstream 训练格式)

| 信息 | 怎么传 |
|---|---|
| reach 原子的 (x, y, r) | `case.grounding` → 8-d node feature 第 4-7 维 |
| STL 树结构 (and/or/not/G/F/U/reach) | 8-d feature 第 1 维 = node type code (0/1/2/5/6/7/8) |
| 时间区间 (ts, te) | 8-d feature 第 2, 3 维 |
| Until 子节点序 | 8-d feature 第 8 维 (`left_child`: +1 = until's first child) |
| Ego state (case start) | encoder forward 的第一个参数 (`encoder(ego, batch)`) |
| 反归一化 mean/std | hardcoded `[0,0,0,0]/[5,5,1,1]` 对齐 train_gstl_v1.py:1342 |

#### 障碍怎么喂给扩散 planner(STLCG guidance,不是塞 graph)

地图障碍(STL avoid 原子 + 程序化墙)**不直接塞进 GNN graph**(实测塞进去节点数 6→40+ 会把 GNN 先验推出训练分布)。而是通过 **STLCG-style 可微 guidance** 喂:

| 信息 | 怎么喂给扩散 |
|---|---|
| STL avoid 原子(在 grounding 里)| 既进 GNN graph(`G(not(atom))` 节点),又进 guidance 的可微障碍损失 |
| 程序化墙/家具(`obstacles=` 参数)| 切成 bounding 圆,只进 guidance 损失;`_stl_refine` 把轨迹推出去 |
| reach goal | guidance 损失里的 "拉到 goal" 项 + `_pin_reaches` 末端钉死 |

guidance 在 `tools/telograf_export.py::_stl_refine`:对 best-of-N 最优轨迹做 200 步 Adam,`loss = 3·ReLU(障碍违反)² + 2·到goal² + 5·start锚 + 0.05·平滑`。

#### 其余对齐细节

| 信息 | 怎么处理 |
|---|---|
| Start 点 | encoder 把 ego 编进 cond(纯模型起点误差 0.05–0.13),guidance 的 start 锚 + post-hoc 平移钉死 |
| 反归一化 | hardcoded `[0,0,0,0]/[5,5,1,1]` 对齐 `train_gstl_v1.py` simple env(完整数据集已下但 simple env 用硬编码 stat)|

如果你要在 poster 里把 TeLoGraF 当 "已接通" 来讲,这是诚实的描述:
> "We integrate TeLoGraF's pretrained GNN+flow-matching checkpoint, run its
> conditional sampler with best-of-N selection (the paper's `test_muls`), and
> apply STLCG-style differentiable test-time guidance (the paper's CTG/LTLDoG
> idea) so the diffusion trajectory is pushed out of every specified obstacle
> and onto the goal.  On three cases — timed reach and reach-while-avoiding
> multiple obstacles — the **diffusion planner satisfies the STL specification
> with no A\* repair**.  For cases whose STL *structure* is out of
> distribution (3-stage sequential visits, conditional response), the
> diffusion planner fails and an A\* fallback recovers a valid trajectory."

STL-Tree 真的在驱动扩散(实测,见 §下方"STL 因果验证"):改 goal 坐标 → 扩散终点跟着跑对象限(误差 0.26–0.95);reach vs avoid 同一点 → reach 把轨迹拉到点上(均距 1.59)、avoid 推开(均距 7.26);改 start → 纯模型起点误差仅 0.05–0.13。这些都不被任何后处理污染。

#### 安装

`scripts/install_telograf.sh` 全自动做安装 —— 不动系统 Python,也不污染 ROS。

#### (1) 一键装

```bash
cd /home/jiachen-tlab-ut/Conferences/UbiComp/code
bash scripts/install_telograf.sh
```

脚本会:

1. 把 TeLoGraF 仓库 clone 到 `code/external/TeLoGraF/`
2. 在 `code/.venv-telograf/` 新建独立 venv
3. 装 PyTorch 2.4.1 (CPU) + torch_geometric 2.5.3 + networkx / einops / pandas / matplotlib / pybullet / gymnasium / pytorch_kinematics / gurobipy
4. 跑 smoke test:对 5 个 case 都构图 + 拿到 fallback 轨迹,验证打通

> **GPU 用户**:默认装 CPU 版(TeLoGraF 推理图只有 ~30 个节点,CPU 跑 <1s)。你想用 GPU 就改环境变量:
> ```bash
> PYTORCH_CHANNEL=cu121 bash scripts/install_telograf.sh   # cu118 / cu121 / cu124
> ```
> RTX 50 系最新的 sm_120 至少需要 PyTorch 2.5+ 才完全支持,默认 CPU 是最稳的。

#### (2) 下载 checkpoint(已经下完了)

```bash
source .venv-telograf/bin/activate
mkdir -p external/TeLoGraF/exps/pretrained_models
cd external/TeLoGraF/exps/pretrained_models
gdown --folder https://drive.google.com/drive/folders/1DYqgMYrg0zfkkhtUfVhLlmOmXXu0pQki -O ./
# 这一步会拉 exps_telograf.zip(2.2 GB),所有 36 个预训练模型的总和

# 只解出我们用的那个 ckpt(simple + GNN + flow-matching)
unzip -q exps_telograf.zip "g0128-075243_simple_gnn_F/*" -d ../
```

结果会在 `code/external/TeLoGraF/exps/g0128-075243_simple_gnn_F/` 下,包含:

| 文件 | 作用 |
|---|---|
| `args.npz`             | 训练时的 argparse Namespace,告诉我们用哪个 encoder / horizon / data_dim |
| `models/model_last.ckpt` | 18 MB 的 state_dict |
| `log-0128-075243.txt`  | 1000 epoch 训练日志(可忽略) |

下别的 checkpoint 同理,改 `unzip` 那一行的子串即可(`simple_gru_F`、`pointmaze_gnn_F`、`panda_gnn_F` 等)。

#### (3) Backend 行为

`planner.plan_waypoints(case, backend=...)` 支持三种 backend:

| backend | 行为 |
|---|---|
| `fallback` | 不调 TeLoGraF,用 `_robustness_path`:绕开避障圆盘,顺序到达每个 reach 区(总能跑通,适合调试 ROS) |
| `telograf` | 走 `tools/telograf_export.py` subprocess,载入 venv → 载 checkpoint → flow-matching 采样。**找不到 checkpoint 就报错** |
| `auto`     | 优先 TeLoGraF,出错就静默回退到 fallback,日志里会写明 |

backend 由 case 的 `role` 自动选(`telograf` case 走 TeLoGraF + A* 兜底,`fallback` case 直接 A*),ROS 闭环 `tb3_follower.py` 用的就是这个自动选择。想手动指定 backend 验证,用 2D 仿真:

```bash
python3 sim2d/run_demo.py --case reach_avoid --backend auto
python3 sim2d/run_demo.py --case reach_avoid --backend telograf      # 没 checkpoint 会报错
```

#### (4) Subprocess bridge

TeLoGraF 上游没暴露 "load checkpoint + sample one graph" 的 Python API,推理路径整个挂在 `train_gstl_v1.py --fix -T <run>` 里。所以我们用 subprocess bridge —— [`tools/telograf_export.py`](tools/telograf_export.py):

* 父进程(系统 Python / ROS Python)调用它,**走 venv 里的 python**
* stdin 传 case + 已编码的图(`planner.case_to_graph`)
* stdout 最后一行是 `{"waypoints": [[x, y], ...]}`,父进程解析并返回

这样:

* ROS 端不需要装 TeLoGraF 的 PyTorch
* TeLoGraF 端不需要装 ROS / rclpy
* 两边只通过 JSON 沟通,边界清楚

> `telograf_export.py` 默认是 stub:对应 model-rebuild 段写着 `TODO`,你 download checkpoint 之后填上 30 行 `build_model + load_state_dict + sample` 就行(参考 `train_gstl_v1.py` 里 `--fix` 分支)。打通后**没有其他代码需要改**,因为父进程接口已经定好。
>
> 不想接 checkpoint、只想验证 subprocess 通的:`FAKE_OUTPUT=1 python3 sim2d/run_demo.py --case reach_avoid --backend telograf`,export 会吐一条假轨迹回来,证明 主进程 ↔ venv ↔ TeLoGraF 三者已经能跑同一个进程链。

#### (5) 验证装好了

```bash
source code/.venv-telograf/bin/activate
PYTHONPATH=code python -c "
from planner import CASES, plan_waypoints, telograf_available
print('telograf_available:', telograf_available())
for c in CASES:
    wp = plan_waypoints(c, backend='fallback', n_steps=32)
    print(f'  {c[\"id\"]:18s}  {len(wp)} waypoints')
"
```

应输出:

```
telograf_available: True
  reach_within_T      33 waypoints
  reach_avoid         33 waypoints
  reach_goal_north    33 waypoints
  seq_reach_ABC       33 waypoints
  trigger_response    33 waypoints
```

---

## 接下来的方向(等当前数字落地后再上)

1. **Chain-of-thought**:让模型先输出解析理由再输出 JSON,通常 +5–15% `path_f1`
2. **n-shots 拉到 16**:更多模板覆盖,+5–10%
3. **错例回流**:viz_app 标记的错样本补到 few-shot,主动学习
4. **微调开源 7B-8B(Llama / Qwen)**:走 NL2TL 论文那条 >95% 的路线
5. **接 TeLoGraF**:把 `sim2d/run_demo.py::_telograf_plan` stub 填上,让扩散轨迹替代直线
