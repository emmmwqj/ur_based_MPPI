# Diffusion MPPI 采样过程说明

本文档根据 `diffusion_mppi.py` 的代码实现，说明扩散 MPPI 的采样过程。

---

## 1. 双阶段采样架构

`DiffusionMPPI.optimize()` 中，每个控制步有 N 次迭代（rollout），分为两个阶段：

- **Phase 1**（迭代 i = N-1 → 1）：扩散采样 `_diffusion_sample_actions`，噪声由 Eq.7 调度控制
- **Phase 2**（迭代 i = 0）：STORM 原生采样 `sample_actions`，噪声由 `scale_tril` 控制

两个阶段使用**不同的噪声注入方式**，这是核心设计。

---

## 2. Phase 1: 扩散采样 `_diffusion_sample_actions`

### 2.1 噪声调度 (Equation 7)

每次迭代先计算当前迭代的 per-horizon-step 标准差：

```python
# diffusion_mppi.py → compute_variance_schedule()
def compute_variance_schedule(self, iteration, n_total):
    iter_exponent = -(n_total - iteration) / (self.beta_1 * n_total)
    total_exponent = iter_exponent + self._horizon_exponent
    return self.sigma_base * torch.exp(total_exponent)
```

其中 `_horizon_exponent` 在初始化时预计算：

```python
H = self.horizon  # 30
h_indices = torch.arange(1, H + 1)
self._horizon_exponent = -(H - h_indices) / (self.beta_2 * H)
# shape: (30,)，值从 -(H-1)/(β₂H) (h=1, 近期) 到 0 (h=H, 远期)
```

最终公式：

$$\sigma_{i,h} = \sigma_{\text{base}} \cdot \exp\!\left(-\frac{N-i}{\beta_1 N} - \frac{H-h}{\beta_2 H}\right)$$

返回 `noise_scale` 形状 `(H,)` = `(30,)`，每个 horizon 步有不同的 σ。

**两个维度的退火效果**：
- 迭代维度：i 从 N-1 (大噪声) 递减到 1 (小噪声) — 先粗后细
- 时域维度：h=1 (近期, 小噪声) 到 h=H (远期, 大噪声) — 近期精确, 远期宽松

### 2.2 采样过程

```python
# diffusion_mppi.py → _diffusion_sample_actions()

# ① 获取标准正态噪声 (Halton + B样条平滑, 与 STORM 相同的采样器)
delta = self.sample_lib.get_samples(sample_shape, base_seed)
delta = torch.cat((delta, self.Z_seq), dim=0)  # 追加零噪声样本
# delta: (494, 30, 2)

# ② 用 Eq.7 的 noise_scale 直接缩放 (绕过 scale_tril)
scaled_delta = delta * noise_scale.unsqueeze(0).unsqueeze(-1)
# noise_scale: (30,) → (1, 30, 1), 对每个 horizon 步用不同的 σ

# ③ 加到均值
act_seq = self.mean_action.unsqueeze(0) + scaled_delta

# ④ clamp 到动作范围
act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn='clamp')

# ⑤ 拼接特殊粒子 (best_traj + null + neg)
act_seq = torch.cat((act_seq, best_traj, null_act_seqs, neg_act_seqs), dim=0)
# 最终: (500, 30, 2)
```

### 2.3 与 STORM 原生采样的关键区别

| 步骤 | STORM `sample_actions` | 扩散 `_diffusion_sample_actions` |
|------|----------------------|--------------------------------|
| 噪声缩放 | `delta @ diag(scale_tril)` | `delta * noise_scale` |
| 缩放来源 | `scale_tril`（自适应，受 `_update_distribution` 影响） | Eq.7 计算（外部调度，与 cov_action 无关） |
| per-horizon 不同 σ | 否（所有 h 用相同 scale_tril） | **是**（每个 h 有不同的 σ_{i,h}） |
| 对 cov_action 的影响 | 间接（后续 `_update_distribution` 会更新 cov） | **无**（不触碰 cov_action） |

**这个区别是核心设计**——扩散迭代的噪声完全绕过 STORM 的自适应协方差。

### 2.4 扩散迭代的分布更新

Phase 1 每次迭代后调用 `_diffusion_update_mean`（而非 `_update_distribution`）：

```python
# diffusion_mppi.py → _diffusion_update_mean()

w = self._exp_util(costs, actions)          # 与 STORM 相同的 softmax 权重
weighted_seq = w.unsqueeze(-1).unsqueeze(-1) * actions
new_mean = torch.sum(weighted_seq, dim=0)   # 加权平均

self.mean_action = (1 - step_size_mean) * self.mean_action + step_size_mean * new_mean
```

**与 `_update_distribution` 的唯一区别**: 不更新 `cov_action`。

---

## 3. Phase 2: STORM 原生采样 `sample_actions`

最后一次迭代（i=0）使用 STORM 的标准采样流程：

```python
# diffusion_mppi.py → optimize() 中的 Phase 2

trajectory = self.generate_rollouts(state)    # 调用 STORM 的 sample_actions
self._update_distribution(trajectory)          # 完整更新 mean + cov
```

`sample_actions` 中噪声通过 `scale_tril` 缩放：

```python
# olgaussian_mpc.py → sample_actions()
scaled_delta = torch.matmul(delta, self.full_scale_tril)
act_seq = self.mean_action.unsqueeze(0) + scaled_delta
```

`_update_distribution` 同时更新均值和协方差（若 `update_cov=True`）：

```python
# 均值更新
self.mean_action = (1 - step_size_mean) * mean_action + step_size_mean * new_mean

# 协方差更新 (update_cov=True 时)
cov_update = mean(sum(w * delta^2, dim=粒子), dim=时间)
self.cov_action = (1 - step_size_cov) * cov_action + step_size_cov * cov_update
```

---

## 4. 完整采样时序

以 `n_diffuse=4` 为例，一个控制步的采样时序：

```
_shift()  ← 热启动: mean 前移, cov += kappa

迭代 i=3 (Phase 1):
  noise_scale = Eq.7(i=3, N=4)           ← 最大噪声
  act_seq = mean + delta * noise_scale    ← 扩散采样
  rollout → costs → weights
  mean_action ← 加权更新 (只更新 mean)

迭代 i=2 (Phase 1):
  noise_scale = Eq.7(i=2, N=4)           ← 噪声缩小
  act_seq = mean + delta * noise_scale
  rollout → costs → weights
  mean_action ← 加权更新 (只更新 mean)

迭代 i=1 (Phase 1):
  noise_scale = Eq.7(i=1, N=4)           ← 更小噪声
  act_seq = mean + delta * noise_scale
  rollout → costs → weights
  mean_action ← 加权更新 (只更新 mean)

迭代 i=0 (Phase 2):
  act_seq = mean + delta @ scale_tril    ← STORM 原生采样
  rollout → costs → weights
  mean_action ← 加权更新                  ← 更新 mean + cov

输出: mean_action
```

---

## 5. 噪声来源汇总

| 来源 | 使用阶段 | 说明 |
|------|---------|------|
| `sample_lib` (Halton + B样条) | Phase 1 + 2 | 标准正态噪声 δ ~N(0,1)，B样条保证平滑 |
| `noise_scale` (Eq.7) | Phase 1 | 直接乘以 δ，per-horizon 不同 σ |
| `scale_tril` (自适应) | Phase 2 | 通过矩阵乘法缩放 δ，受 `_update_distribution` 驱动 |
| `kappa` | `_shift` | 每步给 cov_action 加常数，防止协方差塌缩 |

关键不变量：**Phase 1 的扩散噪声不影响 `cov_action` / `scale_tril`**。它们仅通过 `_shift`（kappa 增长）和 Phase 2（`_update_distribution` 收缩）自然演化。

---

## 6. 常见问题

### Q1: 两个阶段采样的噪声 delta 和 `mean_action`、`cov_action` 有没有关系？

**delta 本身与 `mean_action` 和 `cov_action` 都没有关系。**

根据代码，两个阶段的 delta 都来自同一个采样器 `self.sample_lib.get_samples()`：

```python
# HaltonSampleLib.get_samples() (sample_libs.py 第88行)
def get_samples(self, sample_shape, base_seed=None, ...):
    self.samples = generate_gaussian_halton_samples(
        sample_shape[0], self.ndims,
        use_ghalton=True, seed_val=self.seed_val, ...)
    self.samples = self.samples.view(..., self.horizon, self.d_action)
    self.samples = self.filter_samples(self.samples)  # B样条平滑
    return self.samples
```

`generate_gaussian_halton_samples` 使用 Halton 低差异序列生成标准正态噪声，再通过 B 样条插值做平滑。这个过程**仅由 `sample_shape`（粒子数）和 `seed_val`（随机种子）决定**，不读取 `mean_action` 或 `cov_action` 的值。

具体来说：

- **Phase 1** (`_diffusion_sample_actions`):
  ```python
  delta = self.sample_lib.get_samples(sample_shape=self.sample_shape,
                                       base_seed=self.seed_val + self.num_steps)
  ```

- **Phase 2** (`sample_actions` in `olgaussian_mpc.py`):
  ```python
  delta = self.sample_lib.get_samples(sample_shape=self.sample_shape,
                                       base_seed=self.seed_val + self.num_steps)
  ```

两者调用方式完全相同，delta 都是从 Halton + B 样条生成的**标准正态噪声**，与控制器内部的 `mean_action`、`cov_action` 无关。

delta 和 `mean_action`、`cov_action` 产生关联的地方在**后续步骤**：
- `mean_action` 在两个阶段都作为采样中心：`act_seq = mean_action + scaled_delta`
- `cov_action`（通过 `scale_tril`）只在 Phase 2 中参与噪声缩放：`scaled_delta = delta @ scale_tril`

但 delta **本身**是预先生成的标准正态样本，不依赖任何分布参数。

---

### Q2: `mean_action` 和 `cov_action` 对动作生成分别有什么影响？

根据 `olgaussian_mpc.py` 中 `sample_actions()` 和 `_get_action_seq()` 的代码：

#### `mean_action` 的影响：采样中心

`mean_action` 是所有采样粒子的**中心点**，在两个阶段都参与：

```python
# Phase 1 (_diffusion_sample_actions, diffusion_mppi.py 第191行):
act_seq = self.mean_action.unsqueeze(0) + scaled_delta

# Phase 2 (sample_actions, olgaussian_mpc.py 第187行):
act_seq = self.mean_action.unsqueeze(0) + scaled_delta
```

同时，最终输出动作也直接取自 `mean_action`：

```python
# _get_action_seq (olgaussian_mpc.py 第148行):
def _get_action_seq(self, mode='mean'):
    if mode == 'mean':
        act_seq = self.mean_action.clone()  # 直接返回 mean_action
```

所以 **`mean_action` 既是采样中心，也是最终输出动作**。它的质量直接决定控制性能。

#### `cov_action` 的影响：Phase 2 采样的噪声幅度

`cov_action` 通过 `scale_tril`（其 Cholesky 分解/平方根）控制 Phase 2 中噪声的缩放幅度。以本项目使用的 `cov_type='diag_AxA'` 为例：

```python
# reset_covariance (olgaussian_mpc.py 第274行):
self.cov_action = torch.tensor([init_cov] * d_action)  # (2,) = [0.01, 0.01]
self.scale_tril = torch.sqrt(self.cov_action)           # (2,) = [0.1, 0.1]

# full_scale_tril 属性 (olgaussian_mpc.py 第333行):
return torch.diag(self.scale_tril)  # (2,2) 对角矩阵

# sample_actions (olgaussian_mpc.py 第183行):
scaled_delta = torch.matmul(delta, self.full_scale_tril)
# delta (N,H,A) @ diag(scale_tril) (A,A) = scaled_delta (N,H,A)
# 每个动作维度的噪声幅度 = delta * sqrt(cov_action[dim])
```

`cov_action` 越大 → `scale_tril` 越大 → 采样扩散范围越广 → 探索性越强。
`cov_action` 越小 → `scale_tril` 越小 → 采样集中在 mean 附近 → 利用性越强。

**注意一个代码细节**：`_update_distribution` 更新了 `cov_action` 但**没有**同步更新 `scale_tril`：

```python
# mppi.py _update_distribution (第199行):
self.cov_action = (1.0 - self.step_size_cov) * self.cov_action + \
    self.step_size_cov * cov_update
# ← 此处没有 self.scale_tril = torch.sqrt(self.cov_action)
```

`scale_tril` 只在 `_shift()` 中被更新：

```python
# mppi.py _shift (第224行):
self.cov_action += self.kappa
self.scale_tril = torch.sqrt(self.cov_action)  # ← 这里才更新 scale_tril
```

这意味着 **`cov_action` 的变化要到下一个控制步的 `_shift()` 才会体现到 `scale_tril`（进而影响 Phase 2 采样）**。在同一个控制步内，即使 Phase 2 的 `_update_distribution` 改变了 `cov_action`，当前步的 Phase 2 采样使用的 `scale_tril` 仍然是本步开头 `_shift()` 时设定的值。

#### 总结对比

| 变量 | 角色 | 影响 Phase 1 | 影响 Phase 2 | 影响最终输出 |
|------|------|:----------:|:----------:|:----------:|
| `mean_action` | 采样中心 + 输出动作 | ✅ (采样中心) | ✅ (采样中心) | ✅ (直接输出) |
| `cov_action` | 噪声幅度（通过 `scale_tril`） | ❌ | ✅ (缩放 delta) | ❌ (不直接影响) |

---

### Q3: 为什么 Phase 1 不更新 `cov_action`？

根据代码注释和实际行为，有以下原因：

#### 原因 1：与 DIAL-MPC 原始设计一致

代码文件头部注释明确说明了设计意图（`diffusion_mppi.py` 第35-44行）：

```python
# Original DIAL-MPC samples as: Y0s = eps * noise_scale + Ybar
# where noise_scale is the DIRECT standard deviation, controlled externally
# by traj_diffuse_factor^i (geometric decay across iterations).
# Mean is updated via pure weighted average: Ybar = sum(w * Y0s).
# There is NO adaptive covariance — noise_scale is never updated from data.
```

原始 DIAL-MPC **完全没有自适应协方差**的概念。Phase 1 忠实复现了这一设计：噪声幅度完全由 Eq.7 的外部调度控制，不从数据中学习。

#### 原因 2：防止 `cov_action` 被扩散噪声污染

`_update_distribution` 计算协方差的方式是（`mppi.py` 第189行，`diag_AxA` 情况）：

```python
delta = actions - self.mean_action.unsqueeze(0)
weighted_delta = w * (delta ** 2).T
cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0)
self.cov_action = (1 - step_size_cov) * self.cov_action + step_size_cov * cov_update
```

这里的 `delta = actions - mean_action` 反映的是**采样粒子相对于均值的离散程度**。

在 Phase 1 中，噪声由 Eq.7 控制，早期迭代的 `noise_scale` 很大（例如 `sigma_base * exp(-1/4) ≈ 0.78`），远大于 STORM 正常的 `scale_tril`（约 `sqrt(0.01) = 0.1`）。如果此时计算 `cov_update`：

```
cov_update ≈ mean(sum(w * (大噪声)^2)) >> 当前 cov_action
```

这会导致 `cov_action` 被大幅膨胀。而到 Phase 2 时，`scale_tril = sqrt(cov_action)` 也会变得很大，使 Phase 2 的 STORM 采样失去精度，违背了"Phase 2 做精细局部优化"的设计目标。

#### 原因 3：保持 STORM 协方差的自然演化

代码注释（`diffusion_mppi.py` 第46-48行）：

```python
# This ensures STORM's cov_action sees only "normal" scale samples and
# evolves correctly, while diffusion iterations provide coarse exploration.
```

`cov_action` / `scale_tril` 在整个系统中的演化路径是：

```
_shift():  cov_action += kappa        → 微量膨胀（防止塌缩）
           scale_tril = sqrt(cov_action) → 更新采样幅度

Phase 1:   不触碰 cov_action           → 保持不变

Phase 2:   cov_action = blend(old, data_driven_update) → 根据粒子分布自适应调整
           (但 scale_tril 不立即更新，等下一步 _shift)
```

如果 Phase 1 也更新 `cov_action`，会破坏这个"小幅膨胀 → 数据驱动收缩"的自然平衡，因为 Phase 1 的大噪声会引入错误的协方差估计。

#### 代码对比佐证

Phase 1 的 `_diffusion_update_mean` 与 Phase 2 的 `_update_distribution` 唯一区别：

```python
# _diffusion_update_mean (diffusion_mppi.py 第218-237行):
w = self._exp_util(costs, actions)
weighted_seq = w.unsqueeze(-1).unsqueeze(-1) * actions
new_mean = torch.sum(weighted_seq, dim=0)
self.mean_action = (1 - step_size_mean) * self.mean_action + step_size_mean * new_mean
# ← 结束。没有 delta、cov_update 的计算。

# _update_distribution (mppi.py 第105-205行):
# ... 同样的 mean 更新 ...
# 然后额外执行:
delta = actions - self.mean_action.unsqueeze(0)
weighted_delta = w * (delta ** 2).T
cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0)
self.cov_action = (1 - step_size_cov) * self.cov_action + step_size_cov * cov_update
```

`_diffusion_update_mean` 是 `_update_distribution` 去掉协方差更新部分后的精简版本，这是有意为之的设计。
