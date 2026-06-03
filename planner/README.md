# planner/ — STL tree → grounding → diffusion 轨迹

## 文件

| 文件 | 作用 |
|---|---|
| `case_examples.py`     | 5 个固定 case,每个带 `role`(telograf / fallback)|
| `telograf_adapter.py`  | tree JSON → TeLoGraF 期望的 `(node_features, edge_index)` |
| `telograf_infer.py`    | **公共入口** `plan_waypoints(case, backend=…)` |
| `__init__.py`          | 暴露 `CASES` / `get_case` / `case_to_graph` / `plan_waypoints` |

`plan_waypoints` 是项目里**唯一**的"case → (x, y) 轨迹"入口:2D 仿真(sim2d/run_demo.py)和 ROS 闭环(sim_ros2/tb3_follower.py 每 5s 重规划)都通过它拿轨迹,backend 切换不需要碰任何其他代码。

## case 按 role 分两类(实测决定)

| role | case | 说明 |
|---|---|---|
| `telograf` | `reach_within_T` / `reach_avoid` / `reach_goal_north` | in-distribution(带时间窗的 reach + 障碍在直线旁),纯 TeLoGraF best-of-N 直接满足 STL,**不碰 A\*** |
| `fallback` | `seq_reach_ABC` / `trigger_response` | 故意 OOD(3 段顺序 / imply / 程序化墙),TeLoGraF 失败 → A* 兜底 |

`backend="role"`(verify / run_demo 默认):telograf-case 走纯 TeLoGraF,fallback-case 走 auto。

## 公共 API

```python
from planner import plan_waypoints, telograf_available

waypoints = plan_waypoints(
    case,             # case 字典 or case_id 字符串
    n_steps=96,       # 想要的 waypoint 数
    ckpt=None,        # TeLoGraF checkpoint 路径 (None=自动找)
    backend="auto",   # "auto" | "telograf" | "fallback"
    obstacles=None,   # 额外障碍(procedural walls/furniture/random),只
                      # 影响 fallback;TeLoGraF 从 STL graph 编码里看
)
# -> [(x0, y0), (x1, y1), ...]
```

`obstacles` 是一个 dict 列表,每个元素任选其一:

```python
{"kind": "circle", "x": ..., "y": ..., "r": ...}
{"kind": "rect",   "x": ..., "y": ..., "w": ..., "h": ...}   # 中心 + 全尺寸
```

`sim2d/env_2d.py::Scene.obstacles_for_planner()` 就把场景里所有非 reach 几何(walls / furniture / random disks / 避障原子)按这个格式吐出来,直接喂给 `plan_waypoints`。

### 障碍怎么喂进扩散 planner —— STLCG guidance

所有障碍(STL avoid 原子 + `obstacles=` 程序化墙)都喂给扩散 planner,通过 **STLCG-style 可微 test-time guidance**(对齐 TeLoGraF 论文的 CTG/LTLDoG):

1. `_telograf_subprocess` 把 `refine_obstacles` = grounding 的 avoid 原子 + `_obstacles_to_disks(extra_obstacles)` 一起放进 payload。
2. 子进程 `tools/telograf_export.py`:GNN 采 best-of-N → 选最优 → `_stl_refine` 对它做 200 步 Adam,损失 = `3·ReLU(障碍违反)² + 2·到goal² + 5·start锚 + 0.05·平滑`,把轨迹推出每个障碍、拉到 goal。

> 为什么不直接塞进 GNN graph?实测把障碍 augment 成 `G(not(_obs))` 节点会让图 6→40+,推出 simple_gnn_F 训练分布,GNN 先验崩。guidance 在采样后调几何,不动 graph,所以多障碍/路径上障碍都能稳定避开(见 README §7C profile)。实验开关 `TELOGRAF_AUGMENT_OBSTACLES=1` 仍可注入 graph,默认关。

### Backend 行为

| `backend`  | 用什么 | 出错时 |
|---|---|---|
| `role`     | telograf-case 用 `telograf`,fallback-case 用 `auto`(verify/run_demo 默认) | — |
| `fallback` | A\* on 0.15m grid,8 连通,clearance 0.30m;读 `grounding[avoid]` + `obstacles=`,顺序到达每个 reach 圆 | 路径找不到时退回直线 |
| `telograf` | 1) 子进程批量采 N 条 flow 轨迹(`telograf_samples`,test_muls) 2) best-of-N 选最优 3) **STLCG 可微 guidance** 把它推出所有障碍+拉到 goal 4) 验证过 → **纯 TeLoGraF+guidance 返回** 5) 否则(结构 OOD)A* 修复/兜底 | 找不到 checkpoint / venv 时报错 |
| `auto`     | 先试 `telograf`,失败就静默退到 `fallback`,日志会写明 | 永远不出错 |

所有 backend 的输出最后都经 `_pin_reaches`,把轨迹钉到每个 reach 原子圆心(STL `reach` = 进 goal 圈)。

### TeLoGraF + repair 验证(2D)

```bash
python3 sim2d/run_demo.py --case reach_avoid --backend telograf
```

逐 case 跑出来:都 `planOK=True` + `goal_reached=True` + `hits=0` + 所有 STL reach 原子都触达(planner 层面,与 ROS 闭环用同一个 `plan_waypoints` 入口)。详细在 [code/README.md](../README.md) 7C 节。

### 环境变量旋钮

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `TELOGRAF_NO_REPAIR=1`     | 跳过 STL 验证 + A* 修复,看 TeLoGraF 原始扩散输出 | 启用修复 |
| `TELOGRAF_NO_SHIFT=1`      | 不做 post-hoc 平移把 first waypoint 钉到 case.start | 启用平移 |
| `TELOGRAF_AUGMENT_OBSTACLES=1` | 把 procedural 障碍当 `globally(not(_obs))` 塞进 STL graph(节点数 6→40+) | OFF,因为推出训练分布会让 goal 全废 |
| `FAKE_OUTPUT=1`            | `tools/telograf_export.py` 出直线假轨迹,只验证 subprocess 通信 | OFF |

## TeLoGraF 输入格式(本目录已对齐 upstream)

直接对齐 `external/TeLoGraF/code/stl_to_seq_utils.py::stl_to_seq`:

- 每个节点 8 维特征:`[node_type_i, ts, te, x, y, z, r, n_child]`
- 缺失字段填 `-1`(时间)或 `0`(坐标和 n_child for atoms)
- 边方向:**child → parent**
- **没有独立的 "avoid" 原子**,避障是 `not(reach)` 两个节点

OP_CODE 在 `telograf_adapter.py`:
```python
{ "and": 0, "or": 1, "not": 2, "finally": 5, "globally": 6, "until": 7, "reach": 8 }
```

## tree → TeLoGraF 的规范化步骤

| 我们的 | TeLoGraF 接受 | 规范化方法 |
|---|---|---|
| `imply(A, B)` | 无 | 重写为 `or(not(A), B)` |
| `iff(A, B)` | 无 | 重写为 `and(imply(A,B), imply(B,A))` 再展开 |
| `not(atom)` | 保留 | 不折叠 —— TeLoGraF 期望 NOT 节点作为 reach 原子的父节点 |
| `and/or` 多 children | n 叉 | 保留 n 叉(TeLoGraF GNN 用 scatter 聚合,不要求二叉) |
| 抽象 `prop_i` | 必须几何 grounded | 用 case 里 `grounding` 字典提供 (x, y, z, r) |

## 安装 TeLoGraF

不要按 TeLoGraF README 的 conda 命令一步步装 —— 那个流程绑定 PyTorch 1.13/CUDA 11.7,在新 GPU(40/50 系)上装不下。

用项目根的一键脚本:

```bash
cd /home/jiachen-tlab-ut/Conferences/UbiComp/code
bash scripts/install_telograf.sh
```

脚本结果:

* `external/TeLoGraF/`  ← TeLoGraF 源码
* `.venv-telograf/`     ← PyTorch 2.4.1 (CPU) + torch_geometric 2.5.3 + 全部 TeLoGraF deps
* 自带 smoke test 验证 5 个 case 的图构造都通

> CUDA 用户:`PYTORCH_CHANNEL=cu121 bash scripts/install_telograf.sh`

## Wiring TeLoGraF 推理(等 checkpoint 下了再做)

TeLoGraF 上游没有"载 checkpoint 然后 sample 一个 graph"的 Python API,所有推理路径都在 `train_gstl_v1.py --fix -T <run>` 里。我们用 **subprocess bridge** 解决:

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

具体打通步骤:

1. **下 checkpoint**:从 TeLoGraF README 里的 Google Drive 链接拉 `g0128-075243_simple_gnn_F.zip` 等。展开到 `external/TeLoGraF/exps/<run_id>/models/model_last.ckpt`。

2. **填 stub**:`tools/telograf_export.py` 里有一段 `TODO` 注释。打开它,把以下 30 行替换 `_fail(...)`:

   ```python
   # 1) 重建 args.Namespace。从 checkpoint 同目录的 args.txt / cfg.json 读;
   #    或者复制 run_icml2025_test.sh 里训练这个 run 用的命令行参数。
   args = build_args_from_checkpoint(ckpt)

   # 2) 重建模型(同 train_gstl_v1.py 里 --fix 分支的 build_model)
   encoder, model = build_model(args)
   sd = torch.load(ckpt, map_location="cpu")
   model.load_state_dict(sd["model_state_dict"])
   encoder.load_state_dict(sd["encoder_state_dict"])
   model.eval(); encoder.eval()

   # 3) 把我们传进来的 (node_feats, edge_index) 包成 torch_geometric.Data
   #    -> 调 encoder.encode_graph(data) 拿 cond embedding

   # 4) p_sample_loop:
   x = torch.randn(1, args.horizon, args.transition_dim)
   traj = model.p_sample_loop(x.shape, cond=cond_emb, args=args)

   # 5) 拿 x, y 投影输出
   xs = traj[0, :, 0].cpu().tolist()
   ys = traj[0, :, 1].cpu().tolist()
   print(json.dumps({"waypoints": list(zip(xs, ys)), "backend": "telograf"}))
   ```

3. **不想下 checkpoint,只想验证 subprocess 通**:
   ```bash
   FAKE_OUTPUT=1 python3 -c "
   from planner import plan_waypoints
   wp = plan_waypoints('reach_avoid', backend='telograf', n_steps=12)
   print(len(wp), 'fake waypoints from subprocess')"
   ```
   会让 `telograf_export.py` 走 `if FAKE_OUTPUT == "1"` 分支吐一条直线轨迹,证明 ROS ↔ venv ↔ TeLoGraF 三段管道都通了。

## 离线快测

```bash
cd code

# 1. 看 5 个 case 的图统计
python3 planner/telograf_adapter.py

# 2. 看 5 个 case 的 fallback 轨迹
python3 planner/telograf_infer.py --case reach_avoid --backend fallback

# 3. 看 5 个 case 的 case 描述
python3 planner/case_examples.py
```

## 后续要做的

* 把 `tools/telograf_export.py` 的 TODO 填掉(等 checkpoint 下完)
* 加 STLCG++ 鲁棒度作推理时引导(`nabla rho`)
* 备选:导数自由引导(SVDD 风格,如果 collision 信号是非可微的)
