# diffusion_simple_reacher.py 算法步骤详解

本文档根据 `examples/diffusion_sampling/diffusion_simple_reacher.py` 的代码实现，
逐步说明每个阶段的输入/输出。

---

## 概览

脚本流程:

```
配置加载 → 创建任务 → 设置目标 → 主控制循环(200步) → 绘图
```

核心函数: `holonomic_robot(args)` 和 `plot_traj(traj_log, ...)`.

---

## Step 1: 命令行参数与配置加载

### 1.1 命令行参数

```python
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--control_dt', type=float, default=0.02)
parser.add_argument('--config', type=str, default='diffusion_simple_reacher.yml')
```

- **输入**: 命令行参数
- **输出**: `args` 对象，含 `cuda=True`, `control_dt=0.02`, `config='diffusion_simple_reacher.yml'`

### 1.2 YAML 配置加载

```python
config_path = join_path('config/' + args.config)
with open(config_path) as f:
    cfg = yaml.safe_load(f)
```

- **输入**: `config/diffusion_simple_reacher.yml` 文件路径
- **输出**: `cfg` 字典，包含以下关键节：
  - `mppi`: STORM 控制器参数（`horizon=30`, `num_particles=500`, `init_cov=0.01` 等）
  - `diffusion`: 扩散参数（`n_diffuse=4`, `beta_1=1.0`, `beta_2=1.0`, `sigma_base=1.0`）
  - `cost`: 代价函数配置（`goal_state`, `zero_vel`, `stop_cost`, `smooth` 等）
  - `model`: 模型参数（`state_dim=4`, `dt=0.02`, `control_space=2`）

---

## Step 2: 创建 DiffusionSimpleTask

```python
task = DiffusionSimpleTask(cfg, tensor_args)
```

### 内部流程

#### 2.1 DiffusionSimpleTask.__init__

- **输入**: `cfg` (完整配置), `tensor_args` (dtype=float32, device=cuda:0)
- **处理**: 调用 `init_diffusion_mppi(robot_file=None)`
- **输出**: 初始化的 task 对象

#### 2.2 init_diffusion_mppi 内部

```python
# ① 创建 rollout 函数
simple_task = SimpleReacher(
    dt=0.02, control_space=2, state_dim=4,
    batch_size=num_particles, horizon=horizon, ...)
```

- **输入**: `cfg['model']`, `cfg['cost']`, `cfg['mppi']` 中的维度/粒子数参数
- **输出**: `simple_task` — 一个可微前向模拟器，接收 (state, actions) → (state_seqs, costs)

```python
# ② 构建 mppi_params 并注入扩散参数
mppi_params = {
    'd_action': 2,          # 动作维度
    'horizon': 30,          # 时域长度
    'num_particles': 500,   # 总粒子数
    'init_cov': 0.01,       # 初始协方差
    'step_size_mean': 0.9,  # 均值更新步长
    'step_size_cov': 0.6,   # 协方差更新步长
    'kappa': 0.0001,        # 协方差膨胀常数
    'n_iters': 1,           # 外部迭代次数 (这里设1)
    'update_cov': True,     # 是否更新协方差
    'cov_type': 'diag_AxA', # 对角协方差
    'base_action': 'repeat',
    'knot_scale': 5,        # B样条节点缩放
    'sample_mode': 'halton',
    ...
    # 扩散参数注入
    'n_diffuse': 4,
    'n_diffuse_init': 10,
    'beta_1': 1.0,
    'beta_2': 1.0,
    'sigma_base': 1.0,
}
```

```python
# ③ 创建 DiffusionMPPI 控制器
controller = DiffusionMPPI(
    mppi_params,
    dynamics_model=simple_task,
    sampling_method=HaltonSampleLib,
    tensor_args=tensor_args)
```

- **输入**: `mppi_params`, `simple_task` (rollout_fn)
- **输出**: `DiffusionMPPI` 控制器实例

```python
# ④ 创建 ControlProcess
self.controller = ControlProcess(controller, ...)
```

- **输入**: DiffusionMPPI, `n_iters=1`
- **输出**: `self.controller` 包装器（内部按 n_iters 调用 optimize）

---

## Step 3: 设置目标和初始状态

```python
# 目标: [x, y, vx, vy] = [0.4, 0.3, 0.0, 0.0]
goal = [0.4, 0.3, 0.0, 0.0]
task.update_params(goal_state=goal)
```

- **输入**: 4维目标状态 [位置x, 位置y, 速度x, 速度y]
- **处理**: `DiffusionTaskBase.update_params` → `ControlProcess.update_params` → `DiffusionMPPI.update_params`
  → `SimpleReacher.update_params` → 更新 `goal_state` tensor、`goal_ee_pos`
- **输出**: 控制器内部代价函数的目标已更新

```python
# 初始位置: [x, y] = [0.05, 0.2]
start_state = [0.05, 0.2]
current_state = torch.tensor(
    [start_state[0], start_state[1], 0.0, 0.0],
    **tensor_args)
# shape: (4,) = [0.05, 0.2, 0.0, 0.0]
```

- **输入**: 起始位置坐标
- **输出**: `current_state` tensor, shape (4,), 含位置+零速度

---

## Step 4: 初始化日志结构

```python
traj_log = {
    'position': [],        # 每步的 [x, y]
    'velocity': [],        # 每步的 [vx, vy]
    'error': [],           # 到目标的距离
    'command': [],         # 控制指令 [ax, ay]
    'noise_scale': [],     # Phase 1 最后迭代的 noise_scale (Eq.7)
    'storm_scale_tril': [],# Phase 2 的 scale_tril 对角线
    'iteration_costs': [], # 每次迭代的最佳代价
    'best_cost': [],       # 每步最终最佳代价
    'variance_schedule': [],# 所有迭代的 noise_scale 历史
}
```

---

## Step 5: 主控制循环（200步）

```python
for i in range(200):
```

### 5.1 计算当前误差

```python
ee_error = task.get_current_error(current_state)
```

- **输入**: `current_state` (4,) — 当前完整状态 [x, y, vx, vy]
- **处理**: `SimpleReacher.current_cost(current_state)` → 提取位置误差
  - 内部: `position_error = goal_ee_pos - current_state[:2]`
  - `ee_error = torch.norm(position_error)`
- **输出**: `ee_error` — 标量，欧几里得距离 ‖goal_pos - current_pos‖

### 5.2 获取控制指令 (核心)

```python
command = task.get_command(current_state, control_dt=0.02, WAIT=True)
```

内部调用链:

#### 5.2.1 状态预处理

```python
# DiffusionTaskBase.get_command()
filtered_state = self.filter_joint_state(current_state)
curr_state = self._state_to_tensor(filtered_state)
# shape: (4,) — [x, y, vx, vy]
```

- **输入**: `current_state` (4,)
- **输出**: `curr_state` (4,) — 预处理后的状态张量（对 SimpleReacher 场景无变换）

#### 5.2.2 控制器优化

```python
# ControlProcess.get_command_debug()
# 内部调用 DiffusionMPPI.optimize(curr_state)
```

**DiffusionMPPI.optimize() 详细流程**：

```python
# ① 确定总迭代次数
if self._first_call:
    n_total = self.n_diffuse_init  # 首次: 10
    self._first_call = False
else:
    n_total = self.n_diffuse       # 后续: 4
```

```python
# ② Phase 1: 扩散迭代 (i = n_total-1 → 1)
for i in range(n_total - 1, 0, -1):
    # a. 计算 Eq.7 噪声调度
    noise_scale = self.compute_variance_schedule(i, n_total)
    # 输入: iteration=i, n_total — 输出: (30,) per-horizon σ

    # b. 扩散采样
    act_seq = self._diffusion_sample_actions(curr_state, noise_scale)
    # 输入: curr_state (4,), noise_scale (30,)
    # 输出: act_seq (500, 30, 2) — 500个粒子的动作序列

    # c. Rollout (前向模拟)
    trajectories = self._rollout_fn(act_seq)
    # 输入: act_seq (500, 30, 2)
    # 输出: trajectories 含 costs (500,) 和 actions (500, 30, 2)

    # d. 均值更新 (无协方差更新)
    self._diffusion_update_mean(trajectories)
    # 输入: trajectories
    # 输出: 更新 self.mean_action (30, 2)，记录 iteration_costs
```

```python
# ③ Phase 2: STORM 原生迭代 (i = 0)
noise_scale = self.compute_variance_schedule(0, n_total)
# 记录最终 noise_scale

trajectory = self.generate_rollouts(curr_state)
# 输入: curr_state (4,)
# 处理: STORM 的 sample_actions (用 scale_tril 缩放) → rollout
# 输出: trajectory 含 costs, actions

self._update_distribution(trajectory)
# 输入: trajectory
# 输出: 更新 mean_action + cov_action + scale_tril
```

```python
# ④ 构建返回值
action_seq = self.mean_action  # (30, 2) 完整动作序列
value = self.best_cost          # 标量, 最低代价
info = {
    'best_cost': value,
    'iteration_costs': [...],    # 每次迭代的最低代价
    'variance_schedule': [...],  # Phase 1 各迭代的 noise_scale
    'mean_action': mean_action,  # 当前均值
}
```

- **optimize 输入**: `curr_state` (4,) — 当前状态
- **optimize 输出**: `action_seq` (30, 2), `value` 标量, `info` 字典

#### 5.2.3 积分得到控制指令

```python
# DiffusionTaskBase.get_command() 中
# command = action_seq[0]  (取第一步)
command = self.integrate_acc(action_seq)
# 输入: action_seq (30, 2) — 加速度序列
# 输出: command (2,) — 当前步的加速度指令 [ax, ay]
```

#### 5.2.4 暴露诊断信息

```python
self._last_opt_info = info
self._last_scale_tril = controller.full_scale_tril
```

- **get_command 最终输出**: `command` (2,) — 加速度指令

### 5.3 提取诊断数据

```python
opt_info = task._last_opt_info

# Eq.7 噪声调度记录
noise_scale = opt_info.get('variance_schedule', [None])[-1]
# 最后一个 Phase 1 迭代的 noise_scale, shape (30,)

# STORM scale_tril 记录
storm_scale_tril = task._last_scale_tril
# shape 取决于 cov_type='diag_AxA' → (2,) 对角线

# 每次迭代代价
iteration_costs = opt_info.get('iteration_costs', [])
# list of scalars, 长度 = n_total

# 最终最佳代价
best_cost = opt_info.get('best_cost', None)
# 标量
```

### 5.4 状态更新 (前向欧拉积分)

```python
cmd = command.cpu().numpy()  # (2,) — [ax, ay]
dt = 0.02

# 速度更新: v += a * dt
curr_vel = current_state[2:].cpu().numpy()  # [vx, vy]
new_vel = curr_vel + cmd * dt               # [vx', vy']

# 位置更新: x += v * dt
curr_pos = current_state[:2].cpu().numpy()  # [x, y]
new_pos = curr_pos + new_vel * dt           # [x', y']

# 组合新状态
current_state = torch.tensor(
    [new_pos[0], new_pos[1], new_vel[0], new_vel[1]],
    **tensor_args)
# shape: (4,) = [x', y', vx', vy']
```

- **输入**: `current_state` (4,), `command` (2,), `dt=0.02`
- **输出**: 更新后的 `current_state` (4,)

### 5.5 记录日志

```python
traj_log['position'].append(new_pos.tolist())
traj_log['velocity'].append(new_vel.tolist())
traj_log['error'].append(ee_error.item())
traj_log['command'].append(cmd.tolist())
traj_log['noise_scale'].append(noise_scale)
traj_log['storm_scale_tril'].append(storm_scale_tril)
traj_log['iteration_costs'].append(iteration_costs)
traj_log['best_cost'].append(best_cost)
traj_log['variance_schedule'].append(
    opt_info.get('variance_schedule', []))
```

---

## Step 6: 绘图 `plot_traj`

```python
plot_traj(traj_log, goal, start_state, save_path='traj_log.png')
```

### 输入

- `traj_log`: 字典，包含 200 步的全部记录
- `goal`: [0.4, 0.3, 0.0, 0.0]
- `start_state`: [0.05, 0.2]
- `save_path`: 图片保存路径

### 8 个子图布局 (4×2)

| 子图位置 | 内容 | 数据来源 |
|---------|------|---------|
| (0,0) | **Position** — x/y 位置随时间变化 + 目标虚线 | `traj_log['position']` |
| (0,1) | **Error** — 到目标距离随时间变化 | `traj_log['error']` |
| (1,0) | **Velocity** — vx/vy 速度随时间变化 | `traj_log['velocity']` |
| (1,1) | **Noise Scale (log)** — Eq.7 噪声 per-horizon (对数刻度) | `traj_log['noise_scale']` |
| (2,0) | **Best Cost (log)** — 每步最佳代价 (对数刻度) | `traj_log['best_cost']` |
| (2,1) | **Acceleration** — ax/ay 控制指令随时间变化 | `traj_log['command']` |
| (3,0) | **2D Trajectory** — xy 平面轨迹 + 起点/终点标记 | `traj_log['position']` |
| (3,1) | **Per-iteration Cost** — 每次迭代的最低代价 | `traj_log['iteration_costs']` |

### 输出

- 保存图片到 `save_path` (默认 `traj_log.png`)
- 终端输出 `Trajectory plot saved to ...`

---

## 完整数据流总结

```
┌─────────────────────────────────────────────────────┐
│ diffusion_simple_reacher.py                         │
│                                                     │
│  cfg ──→ DiffusionSimpleTask                        │
│            ├─ SimpleReacher (rollout_fn)             │
│            ├─ DiffusionMPPI (controller)             │
│            └─ ControlProcess (wrapper)               │
│                                                     │
│  每步循环:                                           │
│  current_state (4,) ──→ get_current_error ──→ error │
│  current_state (4,) ──→ get_command                 │
│                           │                         │
│                    DiffusionMPPI.optimize            │
│                    ├─ Phase 1 (x N-1)               │
│                    │  Eq.7 → noise_scale (30,)      │
│                    │  sample (500,30,2)              │
│                    │  rollout → costs (500,)         │
│                    │  update mean (30,2)             │
│                    └─ Phase 2 (x 1)                 │
│                       STORM sample (500,30,2)       │
│                       rollout → costs (500,)        │
│                       update mean+cov               │
│                           │                         │
│                    ←── command (2,)                  │
│                                                     │
│  command (2,) ──→ 欧拉积分 ──→ new_state (4,)       │
│                                                     │
│  200步后 ──→ plot_traj ──→ traj_log.png             │
└─────────────────────────────────────────────────────┘
```

---

## 关键张量形状参考

| 变量 | 形状 | 说明 |
|------|------|------|
| `current_state` | (4,) | [x, y, vx, vy] |
| `mean_action` | (30, 2) | 当前动作序列均值 |
| `noise_scale` | (30,) | Eq.7 per-horizon 标准差 |
| `scale_tril` | (2,) | STORM 协方差对角线 (diag_AxA) |
| `act_seq` (采样后) | (500, 30, 2) | 全部粒子的动作序列 |
| `costs` | (500,) | 每个粒子的总代价 |
| `command` | (2,) | 当前步加速度指令 [ax, ay] |
| `ee_error` | 标量 | 位置误差距离 |
