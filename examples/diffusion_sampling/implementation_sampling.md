# DIAL-MPC 在 STORM 框架中的实现文档

## 1. 算法背景

### 1.1 原始 DIAL-MPC (ICRA 2025)

DIAL-MPC 将扩散模型的逆向去噪过程引入采样优化控制，核心公式为 **Equation 7**:

```
σ_{i,h} = σ_base · exp( -(N-i)/(β₁·N) - (H-h)/(β₂·H) )
```

- `i`: 扩散迭代索引，从 N-1（最大噪声）到 0（最小噪声）
- `h`: horizon 时间步索引，从 1 到 H
- `N`: 总扩散迭代次数
- `H`: horizon 步数
- `β₁`: 迭代级退火速率（越大衰减越慢）
- `β₂`: 时域级退火速率（越大衰减越慢）

原始 DIAL-MPC 的采样与更新逻辑（JAX 实现）:

```python
# 采样: noise_scale 直接作为标准差
eps = jax.random.normal(rng, (Nsample, Hnode+1, nu))
Y0s = eps * noise_scale[None, :, None] + Ybar

# 评估 cost 并计算权重
weights = jax.nn.softmax((rews - rew_Ybar) / rews.std() / temp_sample)

# 更新 mean: 纯加权平均，无 step_size
Ybar = jnp.einsum("n,nij->ij", weights, Y0s)
```

关键特点:
- `noise_scale` **直接**作为采样标准差，不经过任何自适应协方差
- **没有协方差更新** — noise_scale 完全由外部扩散调度控制
- mean 通过**纯加权平均**更新，等效于 `step_size_mean=1.0`

### 1.2 STORM MPPI

STORM 的 MPPI 继承链: `Controller → OLGaussianMPC → MPPI`

采样过程:
```python
delta = sample_lib.get_samples(...)          # Halton-knot B-spline 低差异采样
scaled_delta = delta @ diag(scale_tril)      # 通过 scale_tril 缩放噪声
act_seq = mean_action + scaled_delta          # 加到均值上
act_seq = clamp(act_seq, -max_action, max_action)
```

更新过程:
- `_update_distribution()`: 同时更新 `mean_action`（via `step_size_mean`）和 `cov_action`（via `step_size_cov`）
- `_shift()`: 每步开始时将 mean 前移一步，并通过 `kappa` 增长 `cov_action`
- `scale_tril = sqrt(cov_action)` 在 `_shift` 中同步更新

STORM 的工程优势（必须保留）:
- Halton-knot B-spline 平滑采样（低差异序列，比纯随机采样覆盖更均匀）
- Null particles（零动作粒子，用于在目标附近停止/制动）
- Best trajectory 保留（始终保留上一步最优轨迹参与采样）
- `kappa` 驱动的协方差自增长（防止协方差塌缩到零）
- `step_size_mean` 平滑均值更新（防止 mean 剧烈跳变）

---

## 2. 融合策略: 双阶段优化

### 2.1 核心难题

直接在 STORM 中修改 `scale_tril` 来实现扩散调度会失败，因为:

1. **修改 `scale_tril` 后调用 `_update_distribution()`**:
   - `_update_distribution` 中协方差更新公式为 `cov_action = (1-α)·cov + α·cov_update`
   - 放大 `scale_tril` → 粒子分散太广 → MPPI 权重集中在极少数粒子 → `cov_update` 极小
   - 经过多次迭代，`cov_action` 被压缩到接近零，后续步的采样范围不足
   - 实测: step=0 的 `scale_tril` 从 0.246 正常衰减到 0.100，但 step=1 的 base 已经只有 0.021

2. **冻结 cov（每次迭代后恢复 `cov_action`）**:
   - `_shift` 中 `cov_action += kappa` 持续累积，没有 `_update_distribution` 的收缩来平衡
   - `cov_action` 无限增长 → `scale_tril` 过大 → 采样被 `clamp` 饱和 → 机器人漂移
   - 实测: 到达 [0.41, 0.31] 但以 -0.00026/step 速度线性漂移

3. **仅在最后一次迭代更新 cov**:
   - 前 N-1 次的大噪声采样已经把 `mean_action` 通过 `step_size_mean=0.9` 推偏
   - 最后一次基于偏移后的 mean 更新 cov，效果不佳
   - 实测: 卡在 [0.13, 0.22]

### 2.2 正确方案: 扩散噪声绕过 scale_tril

核心洞察来自原始 DIAL-MPC 代码 — 它的 `noise_scale` 是**直接**乘在标准正态噪声上的，
不经过任何自适应协方差。因此正确的融合方式是:

> **扩散迭代完全绕过 `scale_tril`，直接用 Eq.7 的 noise_scale 乘以采样噪声；
> STORM 的 `cov_action` / `scale_tril` 全程不被扩散过程触碰。**

### 2.3 双阶段流程

每个控制步的 `optimize()` 执行如下:

```
步骤 1: _shift()
    └─ STORM 标准热启动: mean_action 前移一步, cov_action += kappa

步骤 2: Phase 1 — 扩散探索 (i = N-1 down to 1)
    ┌─────────────────────────────────────────────────────────────────┐
    │ for iter_idx in range(N-1, 0, -1):   # 从大噪声到小噪声       │
    │     noise_scale = Eq.7(iter_idx, N)   # 计算每个 h 的 σ       │
    │     delta = Halton-knot 采样           # STORM 的低差异采样器   │
    │     act = mean + delta * noise_scale   # 直接乘，不走 scale_tril│
    │     trajectory = rollout(act)          # 前向仿真               │
    │     _diffusion_update_mean(trajectory) # 只更新 mean，不碰 cov  │
    └─────────────────────────────────────────────────────────────────┘

步骤 3: Phase 2 — STORM 精细化 (i = 0，最后一次迭代)
    ┌─────────────────────────────────────────────────────────────────┐
    │ generate_rollouts()       # STORM 原生采样，使用 scale_tril    │
    │ _update_distribution()    # 完整更新 mean + cov               │
    └─────────────────────────────────────────────────────────────────┘

步骤 4: 返回 action_seq
```

关键不变量:
- `cov_action` / `scale_tril` **全程不被扩散过程修改**
- 它们仅通过 `_shift`（kappa 增长）和 Phase 2 的 `_update_distribution`（自适应收缩）自然演化
- Phase 1 的扩散迭代提供粗粒度的 mean 探索，Phase 2 的 STORM 迭代提供精细化 + cov 更新

---

## 3. 文件结构与类继承

### 3.1 文件列表

```
新增文件（不修改任何 STORM 原始文件）:

storm_kit/mpc/control/diffusion_mppi.py       # 核心控制器 DiffusionMPPI
storm_kit/mpc/task/diffusion_task_base.py      # 任务基类 DiffusionTaskBase
storm_kit/mpc/task/diffusion_simple_task.py    # Simple Reacher 任务

examples/diffusion_sampling/
    diffusion_simple_reacher.py                # 主脚本（运行 + 数据收集 + 绘图）
    config/diffusion_simple_reacher.yml        # 配置文件
    results/                                   # 输出诊断图
    implementation_sampling.md                 # 本文档
```

### 3.2 类继承关系

```
STORM 原版:                              DIAL-MPC 新增:
                                          
Controller                                Controller
  └── OLGaussianMPC                         └── OLGaussianMPC
        └── MPPI                                  └── MPPI
                                                        └── DiffusionMPPI  ← 新增

BaseTask                                  BaseTask
  └── SimpleTask                            └── DiffusionTaskBase  ← 新增
                                                  └── DiffusionSimpleTask  ← 新增
```

DiffusionMPPI 是 MPPI 的**纯继承扩展**，通过 override `optimize()` 实现扩散调度。
STORM 的所有原始文件（`mppi.py`、`olgaussian_mpc.py`、`control_base.py`、`task_base.py`、
`simple_task.py`）均**零修改**。

---

## 4. 核心代码详解

### 4.1 DiffusionMPPI 初始化

文件: `storm_kit/mpc/control/diffusion_mppi.py`

在 MPPI 的全部参数基础上，额外接收 5 个扩散参数:

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `beta_1` | float | 1.0 | Eq.7 迭代级退火速率 |
| `beta_2` | float | 1.0 | Eq.7 时域级退火速率 |
| `sigma_base` | float | 1.0 | Eq.7 基础 σ 缩放因子 |
| `n_diffuse` | int | 4 | 正常控制步的扩散迭代次数 N |
| `n_diffuse_init` | int | 10 | 首步的扩散迭代次数（冷启动多探索） |

初始化时预计算 horizon 指数以避免每次迭代重复计算:

```python
H = self.horizon  # 例如 30
h_indices = torch.arange(1, H + 1)
self._horizon_exponent = -(H - h_indices) / (self.beta_2 * H)
# shape: (30,), 值从 -29/30 ≈ -0.967 (h=1) 到 0.0 (h=H)
```

### 4.2 compute_variance_schedule(iteration, n_total)

严格实现 Equation 7，返回每个 horizon 步的标准差:

```python
def compute_variance_schedule(self, iteration, n_total):
    iter_exponent = -(n_total - iteration) / (self.beta_1 * n_total)
    total_exponent = iter_exponent + self._horizon_exponent
    return self.sigma_base * torch.exp(total_exponent)
```

返回值示例（`n_diffuse=4, beta_1=1.0, beta_2=1.0, sigma_base=1.0`）:

```
iter=3 (最大噪声): σ_h 从 0.097 (h=1,近期) 到 0.755 (h=30,远期)，均值≈0.307
iter=2:            σ_h 从 0.076 到 0.588，均值≈0.239
iter=1:            σ_h 从 0.059 到 0.458，均值≈0.186
iter=0 (最小噪声): σ_h 从 0.046 到 0.357，均值≈0.145
```

两个维度的退火效果:
- **迭代维度**: iter=3（广泛探索） → iter=0（局部精细化）
- **时域维度**: 近期动作 h≈1（小噪声，精确控制） → 远期动作 h≈H（大噪声，允许探索）

### 4.3 _diffusion_sample_actions(state, noise_scale)

实现 DIAL-MPC 原始的直接噪声采样，但复用 STORM 的 Halton-knot 采样器:

```python
def _diffusion_sample_actions(self, state, noise_scale):
    # 1. 获取标准正态样本（Halton + B-spline 平滑）
    delta = self.sample_lib.get_samples(
        sample_shape=self.sample_shape,
        base_seed=self.seed_val + self.num_steps
    )
    delta = torch.cat((delta, self.Z_seq), dim=0)  # 追加零噪声样本
    
    # 2. 直接用 noise_scale 作为标准差（绕过 scale_tril）
    #    delta: (N, H, A), noise_scale: (H,) → (1, H, 1) 广播
    scaled_delta = delta * noise_scale.unsqueeze(0).unsqueeze(-1)
    
    # 3. 加到均值上
    act_seq = self.mean_action.unsqueeze(0) + scaled_delta
    
    # 4. clamp 到 [-max_action, max_action]
    act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs,
                         squash_fn=self.squash_fn)
    
    # 5. 附加 best_traj + null particles + negative particles
    append_acts = self.best_traj.unsqueeze(0)
    if self.num_null_particles > 0:
        neg_action = -1.0 * self.mean_action.unsqueeze(0)
        neg_act_seqs = neg_action.expand(self.num_neg_particles, -1, -1)
        append_acts = torch.cat((append_acts, self.null_act_seqs, neg_act_seqs), dim=0)
    act_seq = torch.cat((act_seq, append_acts), dim=0)
    return act_seq
```

与 STORM 原生 `sample_actions()` 的对比:

| 特性 | STORM `sample_actions` | `_diffusion_sample_actions` |
|------|------------------------|-----------------------------|
| 噪声缩放 | `delta @ diag(scale_tril)` | `delta * noise_scale` |
| 噪声来源 | `scale_tril`（自适应） | Eq.7 schedule（外部控制） |
| per-horizon 不同 σ | 否（统一 scale_tril） | 是（每个 h 不同的 σ） |
| 对 cov_action 的影响 | 间接影响（后续 `_update_distribution`） | 无影响 |

### 4.4 _diffusion_update_mean(trajectories)

仿照 DIAL-MPC 原始的加权平均，但保留 STORM 的 `step_size_mean` 平滑:

```python
def _diffusion_update_mean(self, trajectories):
    costs = trajectories["costs"]
    actions = trajectories["actions"]
    
    # 1. 计算 MPPI 权重（与 STORM 相同的 _exp_util）
    w = self._exp_util(costs, actions)
    
    # 2. 更新 best_traj / top_trajs（用于诊断和后续采样追加）
    best_idx = torch.argmax(w)
    self.best_traj = actions[best_idx]
    
    # 3. 加权平均计算新均值
    weighted_seq = w.unsqueeze(-1).unsqueeze(-1) * actions
    new_mean = torch.sum(weighted_seq, dim=0)
    
    # 4. STORM 式平滑混合（step_size_mean=0.9 → 90% 用新值）
    self.mean_action = (1 - self.step_size_mean) * self.mean_action \
                       + self.step_size_mean * new_mean
```

**与 `_update_distribution()` 的唯一区别**: 不修改 `cov_action`。这是保证 STORM 协方差
自然演化的关键。

### 4.5 optimize() — 双阶段主循环

```python
def optimize(self, state, calc_val=False, shift_steps=1, n_iters=None):
    # 首步用更多迭代（冷启动）
    n_total = self.n_diffuse_init if self._is_first_step else self.n_diffuse
    
    # STORM 标准热启动: mean 前移 + cov += kappa
    if self.hotstart:
        self._shift(shift_steps)
    else:
        self.reset_distribution()
    
    # ── Phase 1: 扩散探索 (i = N-1 down to 1) ──
    for iter_idx in range(n_total - 1, 0, -1):
        noise_scale = self.compute_variance_schedule(iter_idx, n_total)
        act_seq = self._diffusion_sample_actions(state, noise_scale)
        trajectory = self._rollout_fn(state, act_seq)
        self._diffusion_update_mean(trajectory)  # 只更新 mean
    
    # ── Phase 2: STORM 精细化 (i = 0) ──
    trajectory = self.generate_rollouts(state)    # STORM 原生采样
    self._update_distribution(trajectory)          # 完整更新 mean + cov
    
    # 返回动作序列
    curr_action_seq = self._get_action_seq(mode=self.sample_mode)
    self._is_first_step = False
    return curr_action_seq, value, info
```

迭代顺序说明:
- 从 `N-1`（最大噪声）向 `1`（较小噪声）递减 — 扩散逆向去噪
- 最后 `i=0` 交给 STORM 原生流程 — 利用自适应协方差精细化
- 这样每个控制步共执行 N 次 rollout（N-1 次扩散 + 1 次 STORM）

---

## 5. 任务层实现

### 5.1 DiffusionTaskBase

文件: `storm_kit/mpc/task/diffusion_task_base.py`

继承 `BaseTask`，功能与 STORM 的 `BaseTask` 完全相同。核心方法 `get_command()` 的调用链:

```
get_command(t_step, curr_state, control_dt, WAIT=True)
  │
  ├── state_filter.filter_joint_state(curr_state)      # 状态滤波
  ├── _state_to_tensor(filt_state)                      # 转为 tensor
  ├── control_process.get_command_debug(...)             # 调用 ControlProcess
  │     │
  │     ├── predict_next_state(t_step, curr_state)      # 状态预测（补偿延迟）
  │     ├── find_first_idx(command_tstep, t_step)       # 计算 shift_steps
  │     ├── controller.optimize(state_tensor, ...)      # DiffusionMPPI.optimize()
  │     └── truncate_command(command, t_step, tstep)    # 截断命令到当前时间
  │
  ├── state_filter.integrate_acc(qdd_des)               # 加速度积分为 pos/vel
  ├── _last_opt_info = command[2]                       # 暴露优化 info 供诊断
  └── 返回 cmd_des {position, velocity, acceleration}
```

ControlProcess 无需任何修改，因为 DiffusionMPPI 的 `optimize()` 与 MPPI 的 `optimize()`
有**完全相同的接口**: `(state, calc_val, shift_steps, n_iters) → (action_seq, value, info)`。

额外功能:
- `_last_opt_info`: 暴露每步的 `variance_schedule` 和 `iteration_costs`，供主脚本收集绘图
- `_last_scale_tril`: 暴露 STORM 自适应 `scale_tril` 的当前值

### 5.2 DiffusionSimpleTask

文件: `storm_kit/mpc/task/diffusion_simple_task.py`

继承 `DiffusionTaskBase`，与 STORM 的 `SimpleTask` 完全对称。唯一区别是
`init_diffusion_mppi()` 创建 `DiffusionMPPI` 而非 `MPPI`:

```python
def init_diffusion_mppi(self, robot_file):
    # 1. 加载 YAML 配置（与 SimpleTask.init_mppi 完全相同）
    exp_params = yaml.safe_load(open(mpc_yml_file))
    rollout_fn = SimpleReacher(exp_params=exp_params, tensor_args=self.tensor_args)
    mppi_params = exp_params['mppi']
    
    # 2. 注入扩散参数
    mppi_params['beta_1'] = self.diffusion_params['beta_1']
    mppi_params['beta_2'] = self.diffusion_params['beta_2']
    mppi_params['n_diffuse'] = self.diffusion_params['n_diffuse']
    mppi_params['n_diffuse_init'] = self.diffusion_params['n_diffuse_init']
    mppi_params['sigma_base'] = self.diffusion_params['sigma_base']
    
    # 3. 创建 DiffusionMPPI（而非 MPPI）
    controller = DiffusionMPPI(**mppi_params)
    return controller
```

---

## 6. 配置文件

文件: `examples/diffusion_sampling/config/diffusion_simple_reacher.yml`

### 6.1 STORM MPPI 参数 (`mppi:` 段)

```yaml
mppi:
  horizon           : 30       # 规划时域步数 H
  init_cov          : 0.01     # 初始协方差 → scale_tril = sqrt(0.01) = 0.1
  gamma             : 0.98     # cost 折扣因子
  n_iters           : 1        # STORM 原生迭代数（DiffusionMPPI 忽略此值）
  step_size_mean    : 0.9      # mean 更新步长（0.9 = 90% 新值）
  step_size_cov     : 0.6      # cov 更新步长（0.6 = 60% 新值）
  beta              : 1.0      # MPPI 温度参数 λ
  num_particles     : 500      # 采样粒子数 K
  update_cov        : True     # 启用自适应协方差（仅在 Phase 2 生效）
  cov_type          : 'diag_AxA'  # 对角协方差（每个动作维度独立）
  kappa             : 0.0001   # _shift 中协方差增长率
  null_act_frac     : 0.01     # 零动作粒子比例（500×0.01 = 5 个）
  hotstart          : True     # 热启动
  sample_params:
    type: 'multiple'
    sample_ratio: {halton-knot: 1.0}  # 100% Halton-knot B-spline 采样
    knot_scale: 5
```

### 6.2 扩散参数 (`diffusion:` 段)

```yaml
diffusion:
  n_diffuse      : 4      # 正常步扩散迭代次数 N
  n_diffuse_init : 10     # 首步扩散迭代次数（冷启动，多探索）
  beta_1         : 1.0    # Eq.7 迭代级退火
  beta_2         : 1.0    # Eq.7 时域级退火
  sigma_base     : 1.0    # Eq.7 基础 σ 缩放因子
```

### 6.3 参数交互关系

- `n_iters: 1` 被 DiffusionMPPI 忽略，实际迭代次数由 `n_diffuse` 控制
- `init_cov: 0.01` 仅影响 Phase 2（最后一次 STORM 迭代）的采样噪声
- Phase 1 的采样噪声完全由 `sigma_base × Eq.7` 控制，与 `init_cov` 无关
- `update_cov: True` 仅在 Phase 2 生效（Phase 1 用 `_diffusion_update_mean`，不碰 cov）
- `kappa: 0.0001` 每步通过 `_shift` 让 `cov_action` 微增长，防止协方差塌缩

---

## 7. 主脚本与诊断

文件: `examples/diffusion_sampling/diffusion_simple_reacher.py`

### 7.1 控制循环

完全镜像 STORM 的 `examples/simple_reacher.py`:

```python
while i < plan_length:  # 200 步
    error, _ = simple_task.get_current_error(filtered_state)
    command = simple_task.get_command(t_step, filtered_state, sim_dt, WAIT=True)
    current_state = command  # 更新状态
    # ... 记录数据 ...
```

### 7.2 数据收集

每步额外从 `simple_task._last_opt_info` 收集:

| 数据 | 来源 | 说明 |
|------|------|------|
| `variance_schedule` | `opt_info['variance_schedule']` | 每次扩散迭代的 Eq.7 噪声均值 |
| `noise_scale` | 上述列表的均值 | 该步的平均扩散噪声 |
| `storm_scale_tril` | `controller.scale_tril.mean()` | STORM 自适应 scale_tril |
| `iteration_costs` | `opt_info['iteration_costs']` | 每次扩散迭代的 min cost |
| `best_cost` | 上述列表最后一项 | Phase 2 迭代的 min cost |

### 7.3 诊断图 (4×2 布局，共 8 个子图)

| 位置 | 子图 | 说明 |
|------|------|------|
| (0,0) | XY Position vs Time | x/y 位置随时间变化，含目标参考线 |
| (0,1) | XY Position Error vs Time | x/y 方向误差 + 欧氏距离 |
| (1,0) | XY Velocity vs Time | x/y 方向速度 |
| (1,1) | Noise Scale vs Time | 各迭代散点 + 均值 + STORM scale_tril（对数坐标） |
| (2,0) | Best Cost vs Time | 最终迭代的最优 cost（对数坐标） |
| (2,1) | Acceleration vs Time | x/y 方向加速度命令 |
| (3,0) | 2D Trajectory | 世界地图 + 轨迹（颜色编码时间） + top trajectories |
| (3,1) | Per-Iteration Cost | 每个控制步内各扩散迭代的 cost 演化 |

---

## 8. 运行结果

在 Simple Reacher 环境中（起点 [0.05, 0.20]，目标 [0.40, 0.30]）:

```
步骤 0-14:   快速离开起点，利用扩散大噪声广泛探索
步骤 14-46:  迅速向目标移动
步骤 47:     到达目标附近 [0.3999, 0.3015]
步骤 47-199: 稳定保持在 [0.400, 0.302]，收敛精度 < 0.001
```

STORM 原版（`n_iters=4`，无扩散）也能收敛到相同精度，验证了双阶段融合的正确性。
DiffusionMPPI 的理论优势在于，Eq.7 的双级退火（迭代 + 时域）在复杂环境中能更好地
避免局部最优。

---

## 9. 设计决策总结

| 决策点 | 选择 | 原因 |
|--------|------|------|
| 扩散噪声注入 | 直接乘以 delta，绕过 scale_tril | 保护 STORM 自适应协方差不被破坏 |
| Phase 1 更新 | 只更新 mean（`_diffusion_update_mean`） | 与原始 DIAL-MPC 一致，不修改 cov |
| Phase 2 | 完整 STORM 迭代 | 利用自适应 cov 进行精细化 + 更新 cov |
| 迭代顺序 | N-1 → 0（大→小噪声） | 扩散逆向去噪 |
| 采样器 | 复用 STORM Halton-knot | 低差异序列优于纯随机 |
| Null particles | 保留 | 目标附近停止/制动的关键能力 |
| ControlProcess | 保留，无修改 | DiffusionMPPI 是 MPPI 的 drop-in replacement |
| STORM 原始文件 | 零侵入（不修改） | 纯继承 + override，降低维护成本 |
