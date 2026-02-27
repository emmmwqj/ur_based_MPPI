# STORM MPPI 完整工作原理

本文档根据代码实现，完整说明 STORM 如何从接收当前状态到输出最终动作的全部过程。

---

## 1. 整体架构

```
用户/ROS节点
    │
    ▼
BaseTask.get_command(t_step, curr_state, control_dt)
    │
    ├── 1. 状态滤波 (JointStateFilter)
    ├── 2. ControlProcess 优化
    │       └── Controller.optimize()
    │             ├── _shift()          ← 热启动
    │             ├── generate_rollouts() ← 采样+仿真+成本
    │             └── _update_distribution() ← 权重+更新
    ├── 3. truncate_command()  ← 截断到当前时刻
    └── 4. integrate_acc()     ← 加速度积分为位置/速度
    │
    ▼
cmd_des = {position, velocity, acceleration}
```

**类继承链**: `Controller → OLGaussianMPC → MPPI`

---

## 2. 控制流程（逐步说明）

### 步骤 1: 状态滤波

**代码**: `task_base.py → BaseTask.get_command()`

```python
filt_state = self.state_filter.filter_joint_state(curr_state)
state_tensor = self._state_to_tensor(filt_state)  # 拼接 [position, velocity, acceleration]
```

- 输入: 原始关节状态 `{position, velocity, acceleration}`
- `JointStateFilter` 对速度/加速度做低通滤波，消除传感器噪声
- 首次调用时将速度置零: `curr_state['velocity'] *= 0.0`
- 输出: 滤波后的状态张量，形状 `(n_dofs * 3,)`

### 步骤 2: ControlProcess 调度

**代码**: `mpc_process_wrapper.py → ControlProcess.get_command_debug()`

```python
# 补偿优化延迟: 用上一步命令将状态前推 mpc_dt
curr_state = self.predict_next_state(t_step, curr_state)

# 计算需要前移的步数
shift_steps = find_first_idx(self.command_tstep, t_step + self.mpc_dt)

# 调用 MPPI 优化
command = controller.optimize(state_tensor, shift_steps=shift_steps)
```

- `predict_next_state`: 用上一步的命令序列做前向积分，补偿从上次优化到现在的时间延迟
- `shift_steps`: 根据时间偏移量计算 mean_action 需要前移多少步
- 支持两种模式:
  - `WAIT=True` → `get_command_debug`: 同进程阻塞优化
  - `WAIT=False` → `get_command`: 独立进程异步优化（实时控制用）

### 步骤 3: MPPI 优化 (核心)

**代码**: `control_base.py → Controller.optimize()`

```python
def optimize(self, state, calc_val=False, shift_steps=1, n_iters=None):
    # 3a. 热启动: 将上一步的解前移
    if self.hotstart:
        self._shift(shift_steps)
    else:
        self.reset_distribution()

    # 3b. 迭代优化 (默认 n_iters=1)
    for _ in range(n_iters):
        trajectory = self.generate_rollouts(state)   # 采样 + rollout
        self._update_distribution(trajectory)         # 加权更新

    # 3c. 提取最优动作序列
    curr_action_seq = self._get_action_seq(mode='mean')  # 返回 mean_action
    return curr_action_seq, value, info
```

下面逐步展开。

---

### 3a. 热启动 `_shift()`

**代码**: `olgaussian_mpc.py → _shift()` + `mppi.py → _shift()`

每个控制步开始时，将上一步的解前移:

```python
# OLGaussianMPC._shift:
self.mean_action = self.mean_action.roll(-shift_steps, 0)  # 均值前移
self.best_traj = self.best_traj.roll(-shift_steps, 0)
self.mean_action[-shift_steps:].zero_()  # 末尾补零 (base_action='null')
# 或 repeat: self.mean_action[-shift_steps:] = self.mean_action[-shift_steps-1]

# MPPI._shift (在 update_cov=True 时):
self.cov_action += self.kappa    # 协方差微增长，防止塌缩
self.scale_tril = sqrt(cov_action)  # 更新 Cholesky 因子
```

- `roll(-1, 0)`: 将动作序列向前移一步，因为时间前进了一步
- `base_action`: 控制末尾填充方式 (`null`=零, `repeat`=重复最后值, `random`=随机)
- `kappa`: 每步给协方差加一个小常数，防止协方差被 `_update_distribution` 压到零

---

### 3b-①: 采样动作序列 `sample_actions()`

**代码**: `olgaussian_mpc.py → sample_actions()`

```python
def sample_actions(self, state=None):
    # ① 从采样库获取标准正态噪声 (已经过 B 样条平滑)
    delta = self.sample_lib.get_samples(sample_shape, base_seed)
    # delta: (num_nonzero-2, H, A) 例如 (493, 30, 6)

    # ② 追加零噪声样本 (确保 mean 本身在候选中)
    delta = torch.cat((delta, self.Z_seq), dim=0)  # +1 → (494, 30, 6)

    # ③ 协方差缩放: δ_scaled = δ @ L^T
    scaled_delta = torch.matmul(delta, self.full_scale_tril)
    # full_scale_tril 根据 cov_type:
    #   diag_AxA → torch.diag(scale_tril), 即 [A, A] 对角矩阵
    #   full_AxA → Cholesky 下三角矩阵 [A, A]

    # ④ 加到均值: a = μ + δ_scaled
    act_seq = self.mean_action.unsqueeze(0) + scaled_delta

    # ⑤ 裁剪到合法范围
    act_seq = scale_ctrl(act_seq, action_lows, action_highs, squash_fn='clamp')

    # ⑥ 拼接特殊粒子
    act_seq = cat(act_seq,           # 494 个采样粒子
                  best_traj,          # 1 个上次最优轨迹
                  null_act_seqs,      # N_null 个零动作 (制动)
                  neg_act_seqs)       # N_neg 个反向动作
    # 总计 500 个粒子
    return act_seq  # (500, 30, 6)
```

**粒子组成** (以 `num_particles=500, null_act_frac=0.01` 为例):

| 类型 | 数量 | 说明 |
|------|------|------|
| Halton-knot B样条采样 | 493 | 低差异 + B 样条平滑 |
| 零噪声 (Z_seq) | 1 | 保证均值动作在候选中 |
| 上次最优 (best_traj) | 1 | 保留历史最优解 |
| 零动作 (null) | 5 | 用于目标附近制动/停止 |
| **总计** | **500** | |

**采样噪声生成** (halton-knot 模式):

```
Halton均匀序列 [0,1]     逆CDF → N(0,1)     B样条拟合      最终噪声
(493, 7×6=42)维     →   (493, 42)     →   (493, 30, 6)  →  delta
    低差异序列            标准正态            平滑轨迹
```

1. 在 42 维空间 (7个knot × 6个关节) 生成 Halton 低差异序列
2. 通过 $\epsilon = \sqrt{2} \cdot \text{erfinv}(2u - 1)$ 转换为标准正态
3. 每个粒子、每个关节的 7 个值通过 `splrep(degree=2, s=0.5)` 拟合为 30 步平滑曲线
4. 采样维度从 180 降到 42，同时保证时间平滑性

---

### 3b-②: 前向仿真 `generate_rollouts()`

**代码**: `olgaussian_mpc.py → generate_rollouts()`

```python
def generate_rollouts(self, state):
    act_seq = self.sample_actions(state)           # 上述采样
    trajectories = self._rollout_fn(state, act_seq)  # 前向积分 + 成本计算
    return trajectories
```

`_rollout_fn` 内部 (以 `arm_base.py` 为例):
1. **动力学积分**: 根据动作序列 (加速度) 逐步积分得到位置/速度轨迹
   - 支持变时间步 `dt_traj`: 近期小 dt (精细)，远期大 dt (扩展预测窗口)
2. **正运动学**: 关节角度 → 末端执行器位姿 (可微分 PyTorch 实现)
3. **成本计算**: 对每条轨迹的每个时间步计算各项成本之和

输出: `trajectories = {actions: (500,30,6), costs: (500,30), state_seq: ..., ee_pos_seq: ...}`

---

### 3b-③: 计算权重与更新分布 `_update_distribution()`

**代码**: `mppi.py → _update_distribution()`

**权重计算** (`_exp_util`):

```python
# 1. 折扣累积成本 (cost-to-go)
traj_costs = cost_to_go(costs, gamma_seq)[:, 0]
# costs: (500, 30) → traj_costs: (500,) 每条轨迹一个标量

# cost_to_go 做的事:
#   discounted = gamma^t * cost_t
#   traj_costs[i, t] = sum_{t'=t}^{H} gamma^{t'-t} * cost_{t'}
#   取 [:, 0] 即从 t=0 开始的总折扣成本

# 2. Softmax 权重
w = softmax(-traj_costs / beta)  # beta 是温度参数
# beta 小 → 只有最低成本轨迹有权重 (贪婪)
# beta 大 → 权重更均匀 (探索)
```

**分布更新**:

```python
# 3. 记录最优轨迹
best_idx = argmax(w)
self.best_traj = actions[best_idx]  # 保留为下一步的 best_traj 粒子

# 4. 加权平均计算新均值
new_mean = sum(w * actions, dim=0)  # (30, 6)

# 5. 平滑更新均值 (step_size_mean 控制新旧比例)
self.mean_action = (1 - step_size_mean) * self.mean_action + step_size_mean * new_mean
# 例如 step_size_mean=0.98: 98% 新值 + 2% 旧值

# 6. (可选) 更新协方差 (update_cov=True 时)
delta = actions - mean_action
if cov_type == 'diag_AxA':
    cov_update = mean(sum(w * delta^2, dim=粒子), dim=时间)  # (A,)
self.cov_action = (1 - step_size_cov) * cov_action + step_size_cov * cov_update
```

---

### 动作均值 `mean_action` 的完整生命周期

`mean_action` 形状为 `(H, A)` 即 `(30, 6)`，是 MPPI 的核心状态量。它在三个阶段被修改:

#### 阶段 A: 初始化 (`reset_mean`)

```python
self.mean_action = self.init_mean.clone()  # 全零 (30, 6)
self.best_traj = self.mean_action.clone()
```

首次运行或重置时，`mean_action` 为全零——表示"不动"的初始猜测。

#### 阶段 B: 每步开头的热启动 (`_shift`)

每个控制步开始时，时间前进了 `shift_steps` 步（通常为 1），需要将动作序列对齐:

```
shift 前:  mean_action = [a₀, a₁, a₂, ..., a₂₈, a₂₉]
                          ↑ 这个已执行过了

shift 后:  mean_action = [a₁, a₂, a₃, ..., a₂₉, a_new]
                                                   ↑ 末尾填充
```

代码 (`olgaussian_mpc.py`):
```python
self.mean_action = self.mean_action.roll(-shift_steps, 0)  # 整体前移

# 末尾填充策略 (由 base_action 配置决定):
if base_action == 'null':
    self.mean_action[-shift_steps:].zero_()              # 填零
elif base_action == 'repeat':
    self.mean_action[-shift_steps:] = self.mean_action[-shift_steps-1].clone()  # 复制倒数第二步
elif base_action == 'random':
    self.mean_action[-1] = generate_noise(...)           # 随机
```

同时 `best_traj` 也做相同的 shift。

如果 `update_cov=True`，`_shift` 还会让协方差微增长: `cov_action += kappa`。

#### 阶段 C: 优化迭代中的加权更新 (`_update_distribution`)

这是 `mean_action` 真正被优化的地方。每次迭代中:

**第 1 步 — 计算每条轨迹的权重:**

$$w_i = \text{softmax}\!\left(\frac{-1}{\beta} \sum_{t=0}^{H-1} \gamma^t \cdot c_{i,t}\right)$$

- $c_{i,t}$: 第 $i$ 条轨迹在时间步 $t$ 的即时成本
- $\gamma$: 折扣因子 (0.98)，远期成本权重递减
- $\beta$: 温度参数，$\beta$ 越小权重越集中于低成本轨迹

**第 2 步 — 加权平均得到候选均值:**

$$\mu_{\text{new}}[t, j] = \sum_{i=1}^{N} w_i \cdot a_i[t, j]$$

- $a_i[t, j]$: 第 $i$ 个粒子、时间步 $t$、关节 $j$ 的动作值
- 结果: $\mu_{\text{new}}$ 形状 `(30, 6)`，是所有粒子按权重的加权平均

**第 3 步 — 平滑混合更新:**

$$\mu \leftarrow (1 - \alpha_\mu) \cdot \mu_{\text{old}} + \alpha_\mu \cdot \mu_{\text{new}}$$

- $\alpha_\mu$ = `step_size_mean`，例如 0.98
- 含义: 98% 用新的加权平均，2% 保留旧均值
- 作用: 防止单次采样噪声导致均值剧烈跳变

代码 (`mppi.py`):
```python
weighted_seq = w.unsqueeze(-1).unsqueeze(-1) * actions  # (500,30,6)
new_mean = torch.sum(weighted_seq, dim=0)                # (30,6)
self.mean_action = (1 - step_size_mean) * self.mean_action + step_size_mean * new_mean
```

#### 完整时间线示例

以连续 3 个控制步为例 (`n_iters=1`, `base_action='repeat'`, `step_size_mean=0.98`):

```
┌─ 控制步 0 (首次) ─────────────────────────────────────────────────┐
│ reset_mean:  μ = [0, 0, 0, ..., 0]     ← 全零初始化              │
│ optimize:                                                         │
│   generate_rollouts → 500条轨迹 → costs                           │
│   _update_distribution:                                           │
│     new_mean = Σ wᵢ·aᵢ = [0.02, 0.05, 0.03, ..., -0.01]         │
│     μ = 0.02·[0,...,0] + 0.98·new_mean                           │
│       = [0.020, 0.049, 0.029, ..., -0.010]                       │
│ 输出: mean_action[0] 作为本步动作                                 │
└───────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─ 控制步 1 ────────────────────────────────────────────────────────┐
│ _shift(1):                                                        │
│   μ = [0.049, 0.029, ..., -0.010, -0.010]  ← 前移+repeat末尾     │
│ optimize:                                                         │
│   generate_rollouts → 500条轨迹 (围绕 shifted μ 采样)             │
│   _update_distribution:                                           │
│     new_mean = Σ wᵢ·aᵢ = [0.06, 0.04, ..., 0.01]                │
│     μ = 0.02·μ_shifted + 0.98·new_mean                           │
│       ≈ new_mean (因为 step_size_mean=0.98 几乎全用新值)           │
│ 输出: mean_action[0]                                              │
└───────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─ 控制步 2 ────────────────────────────────────────────────────────┐
│ _shift(1): 前移 + 末尾填充                                        │
│ optimize:  采样 → rollout → 加权更新                              │
│ 输出: mean_action[0]                                              │
│                                                                   │
│ ... 随着步数推进，mean_action 逐渐收敛到低成本轨迹 ...             │
└───────────────────────────────────────────────────────────────────┘
```

#### 关键理解

1. **`mean_action` 是整条 H 步序列**，不是单个动作。每步只执行 `mean_action[0]`，其余步为未来规划。
2. **Shift 实现了滚动时域**: 上一步的 `mean_action[1:]` 变成本步的 `mean_action[:-1]`，只需优化末尾新增的一步。
3. **`step_size_mean` 的效果**: 接近 1.0 时响应快（几乎全用新值），接近 0.0 时保守（保留旧值更多）。典型值 0.98 意味着每次迭代几乎完全采纳加权平均结果。
4. **收敛机制**: 低成本粒子获得高权重 → 加权平均朝低成本方向移动 → 下一步以更新后的均值为中心采样 → 进一步向最优收敛。

---

### 3c: 提取最终动作序列

**代码**: `olgaussian_mpc.py → _get_action_seq()`

```python
def _get_action_seq(self, mode='mean'):
    act_seq = self.mean_action.clone()  # (30, 6) 完整动作序列
    act_seq = scale_ctrl(act_seq, action_lows, action_highs, squash_fn='clamp')
    return act_seq
```

- `mode='mean'`: 直接返回更新后的 `mean_action`（默认，最稳定）
- `mode='sample'`: 从当前分布再采样一条（引入额外随机性）

---

### 步骤 4: 命令截断与积分

**代码**: `mpc_process_wrapper.py → truncate_command()` + `state_filter.py → integrate_acc()`

```python
# 4a. 截断: 从完整 H 步序列中取当前时刻对应的动作
command_buffer = command[f_idx:]  # 截掉已过时的前几步

# 4b. 积分: 加速度 → 位置/速度 (在 ControlProcess 内部)
act = dynamics_model.integrate_action_step(command_buffer[0], control_dt)
# 对于 action_order=2 (加速度控制): act = acc * dt * dt

# 4c. 积分: 加速度 → 完整状态命令 (在 BaseTask 内部)
# state_filter.integrate_acc(qdd_des):
#   velocity = velocity + qdd * dt
#   position = position + velocity * dt
cmd_des = {
    'position':     position,      # 关节位置指令
    'velocity':     velocity,      # 关节速度指令
    'acceleration': qdd_des        # 关节加速度
}
```

**注意**: `integrate_acc` 是简单欧拉积分:
$$\dot{q}_{cmd} = \dot{q}_{prev} + \ddot{q}_{des} \cdot dt$$
$$q_{cmd} = q_{prev} + \dot{q}_{cmd} \cdot dt$$

这里 $q_{prev}, \dot{q}_{prev}$ 来自上一步的 `cmd_joint_state`（命令状态，非传感器状态）。

---

## 3. 关键设计

### 3.1 为什么默认只迭代 1 次 (`n_iters=1`)?

- 热启动 (`hotstart=True`) 使得每步的初始解已经很好（上一步的解前移一步）
- 实时性要求: 每个控制周期只有约 20ms，多次迭代会增加延迟
- 500 个粒子 + B 样条平滑已经提供了充分的搜索能力

### 3.2 B 样条采样的作用

- **降维**: 从 H×A=180 维降到 M×A=42 维 (M=7 个 knot, A=6 个关节)
- **平滑**: B 样条 (degree=2) 保证 $C^1$ 连续，相邻时间步动作平滑
- **高效覆盖**: Halton 低差异序列在 42 维空间比随机采样覆盖更均匀
- 实现: `splrep(data, k=2, s=0.5)` 拟合（非直接控制点），然后 `splev` 在 30 个时间步上采样

### 3.3 协方差管理

当 `update_cov=True` 时:
- `_update_distribution`: 根据粒子权重计算新协方差，按 `step_size_cov` 混合
- `_shift`: 每步加 `kappa` 防止协方差塌缩到零
- `scale_tril = sqrt(cov_action)` (对角情况) 用于缩放采样噪声

当 `update_cov=False` (UR7e 默认) 时:
- 协方差始终保持 `init_cov`，不变
- `scale_tril` 固定，采样范围恒定

### 3.4 特殊粒子

| 粒子类型 | 作用 |
|----------|------|
| **Z_seq** (零噪声) | 保证 `mean_action` 本身参与评估，防止好的均值被噪声掩盖 |
| **best_traj** | 保留上一步最优轨迹，防止丢失已找到的好解 |
| **null_act_seqs** (零动作) | 提供"不动"选项，目标附近用于制动/停止 |
| **neg_act_seqs** (反向动作) | 提供反向运动选项，帮助减速 |

---

## 4. 数据流总结

```
输入: curr_state = {position: (6,), velocity: (6,), acceleration: (6,)}
      t_step = 当前时间戳
                │
                ▼
        ┌───────────────────┐
        │   状态滤波         │ JointStateFilter.filter_joint_state
        │   速度/加速度低通  │
        └───────┬───────────┘
                │ state_tensor: (18,)
                ▼
        ┌───────────────────┐
        │   状态前推补偿      │ predict_next_state (补偿优化延迟)
        └───────┬───────────┘
                │
                ▼
        ┌───────────────────┐
        │   _shift           │ mean_action.roll + cov += kappa
        │   (热启动)         │
        └───────┬───────────┘
                │
                ▼
        ┌───────────────────┐
        │  sample_actions    │ Halton+B样条 → 噪声 → ×scale_tril → +mean
        │  (500条动作序列)   │ → clamp → 拼接特殊粒子
        └───────┬───────────┘
                │ act_seq: (500, 30, 6)
                ▼
        ┌───────────────────┐
        │  _rollout_fn       │ 动力学积分 + 正运动学 + 成本计算
        │  (前向仿真)        │
        └───────┬───────────┘
                │ costs: (500, 30)
                ▼
        ┌───────────────────┐
        │ _update_distribution│ cost_to_go → softmax权重 → 加权平均
        │ (更新 mean + cov)  │ mean = (1-α)*mean + α*new_mean
        └───────┬───────────┘
                │
                ▼
        ┌───────────────────┐
        │  _get_action_seq   │ 返回 mean_action (30, 6)
        │  (提取最优解)      │
        └───────┬───────────┘
                │ action_seq: (30, 6)
                ▼
        ┌───────────────────┐
        │  truncate_command  │ 取当前时刻对应的动作
        │  + integrate_acc   │ 加速度 → 积分 → 位置/速度
        └───────┬───────────┘
                │
                ▼
输出: cmd_des = {position: (6,), velocity: (6,), acceleration: (6,)}
```

---

## 5. 核心代码文件索引

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `control_base.py` | `Controller.optimize()` | 主优化循环: shift → rollout → update |
| `olgaussian_mpc.py` | `OLGaussianMPC` | 采样 `sample_actions`、分布管理、协方差 |
| `mppi.py` | `MPPI` | 权重 `_exp_util`、更新 `_update_distribution`、shift |
| `sample_libs.py` | `MultipleSampleLib` | 组合采样; `KnotSampleLib` B样条采样 |
| `sample_libs.py` | `bspline()` | `splrep` + `splev` B样条拟合 |
| `control_utils.py` | `generate_gaussian_halton_samples` | Halton → 逆CDF → N(0,1) |
| `control_utils.py` | `cost_to_go()` | 折扣累积成本计算 |
| `mpc_process_wrapper.py` | `ControlProcess` | 多进程/同步调度、状态前推、命令截断 |
| `task_base.py` | `BaseTask.get_command()` | 外层接口: 滤波 → 优化 → 积分 |
| `state_filter.py` | `JointStateFilter` | 状态滤波、加速度积分为位置/速度 |
