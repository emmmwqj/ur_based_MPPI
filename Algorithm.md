# STORM MPPI 算法原理

本文档详细说明 STORM (Stochastic Tensor Optimization for Robot Motion) 中 MPPI (Model Predictive Path Integral) 控制器的算法原理。

---

## 📖 算法概述

MPPI 是一种基于采样的模型预测控制 (MPC) 算法，源自随机最优控制理论。其核心思想是：
1. 从当前动作分布中采样多条轨迹
2. 通过前向仿真评估每条轨迹的成本
3. 使用指数效用函数加权平均得到最优动作
4. 滚动时域执行第一个动作

**论文**: Williams et al., "Information Theoretic MPC for Model-Based Reinforcement Learning"

---

## 🧠 MPPI 控制器输入与输出

### 输入

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STORM MPPI 控制器 输入/输出                              │
└─────────────────────────────────────────────────────────────────────────────┘

                           ┌─────────────────────────┐
         输入              │                         │              输出
        ─────────────>     │     MPPI Controller     │     ─────────────>
                           │                         │
                           └─────────────────────────┘

  ┌──────────────────┐                                    ┌──────────────────┐
  │ 当前状态 (state) │                                    │ 最优动作 (action)│
  │                  │                                    │                  │
  │ • q  (关节角度)  │                                    │ • a* (加速度)    │
  │ • q̇  (关节速度)  │           ───────────>             │   或             │
  │ • q̈  (关节加速度)│                                    │ • q* (位置增量)  │
  └──────────────────┘                                    └──────────────────┘
  
  ┌──────────────────┐
  │ 目标 (goal)      │
  │                  │
  │ • 目标末端位置   │
  │ • 目标末端姿态   │
  └──────────────────┘
```

### 输入详解

#### 1. 当前状态 `state`

```python
# 状态向量维度: [n_dofs * 3] = [6 * 3] = 18 维
state = {
    'position':     [q1, q2, q3, q4, q5, q6],      # 关节角度 (rad), 6维
    'velocity':     [v1, v2, v3, v4, v5, v6],      # 关节速度 (rad/s), 6维
    'acceleration': [a1, a2, a3, a4, a5, a6]       # 关节加速度 (rad/s²), 6维
}

# 拼接成完整状态向量
full_state = np.concatenate([position, velocity, acceleration])  # shape: (18,)
```

#### 2. 目标 `goal`

```python
# 末端执行器目标
goal_ee_pos = [x, y, z]           # 目标位置 (m), 3维
goal_ee_quat = [qx, qy, qz, qw]   # 目标姿态 (四元数), 4维

# 或者关节空间目标
goal_state = [q1, ..., q6, 0, 0, 0, 0, 0, 0]  # 目标关节角度 + 零速度
```

### 输出详解

#### 最优动作序列 `optimal_action`

```python
# MPPI 输出: 最优动作序列
# 维度: [horizon, d_action] = [30, 6]

# 控制空间为 'acc' (加速度控制):
optimal_action = [
    [a1_t0, a2_t0, ..., a6_t0],   # t=0 时刻的 6 个关节加速度
    [a1_t1, a2_t1, ..., a6_t1],   # t=1 时刻
    ...
    [a1_t29, a2_t29, ..., a6_t29] # t=29 时刻 (horizon=30)
]

# 只使用第一个动作 (MPC 的滚动时域策略)
action_to_execute = optimal_action[0]  # shape: (6,)
```

---

## 🔄 MPPI 算法流程

### 总体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MPPI 优化算法流程                                     │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ 1.采样  │ ──> │ 2.仿真  │ ──> │ 3.成本  │ ──> │ 4.权重  │ ──> │ 5.更新  │
  │ 动作序列│     │ 前向积分│     │ 计算    │     │ Softmax │     │ 分布    │
  └─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
       │               │               │               │               │
       ▼               ▼               ▼               ▼               ▼
   N×H×A 轨迹      状态序列        成本向量         权重向量        最优动作
   (500×30×6)     (含碰撞检测)     [500,]          [500,]         [30, 6]
```

### 步骤 1: 采样动作序列

从当前动作分布采样 N 条轨迹：

```
   mean_action [H×A]     噪声 δ [N×H×A]          采样的动作序列 [N×H×A]
   ┌─────────────┐      ┌─────────────┐          ┌─────────────┐
   │ μ (当前均值) │  +   │ ε ~ N(0,Σ)  │   =      │ 500条轨迹   │
   └─────────────┘      └─────────────┘          └─────────────┘
                                                  N=500, H=30, A=6
```

**采样方式** (配置项 `sample_params.type`):
- `multiple`: 混合采样 (**推荐，默认使用**)
- `halton`: 纯 Halton 低差异序列
- `random`: 标准高斯随机采样
- `stomp`: STOMP 算法的相关采样

---

## 🎲 采样过程详解

### 采样方式配置

STORM 使用 `MultipleSampleLib` 支持多种采样策略的混合，配置示例：

```yaml
# 文件: examples/HIL/config/ur7e_reacher_hil.yml
sample_params:
  type: 'multiple'
  fixed_samples: True
  sample_ratio: {'halton': 0.0, 'halton-knot': 1.0, 'random': 0.0, 'random-knot': 0.0}
  seed: 0
  filter_coeffs: None
  knot_scale: 4
```

**采样比例说明**:
| 采样方法 | 比例 | 说明 |
|----------|------|------|
| `halton` | 0.0 | 纯 Halton 序列采样 |
| `halton-knot` | **1.0** | **B 样条 + Halton (默认使用)** |
| `random` | 0.0 | 纯随机高斯采样 |
| `random-knot` | 0.0 | B 样条 + 随机采样 |

> ⚠️ **重要**: STORM 默认使用 **`halton-knot`** 采样，即 **B 样条插值 + Halton 序列**，而非纯 Halton 序列！

### 采样流程总览（正确流程）

> 📝 **用户确认**: 采样流程是"先生成均匀 Halton，再逆 CDF 变换，然后时间滤波，接着协方差缩放，加到均值，裁剪"对吗？
>
> ✅ **对于 `halton-knot` 模式（默认）**: 流程是 **"均匀 Halton → 逆 CDF → B 样条插值 → 协方差缩放 → 加到均值 → 裁剪"**
> - **没有时间滤波步骤**，因为 B 样条插值本身已提供时间平滑性
>
> ❌ **对于纯 `halton` 模式**: 流程才是 "均匀 Halton → 逆 CDF → 时间滤波 → 协方差缩放 → 加到均值 → 裁剪"

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    动作采样完整流程 (halton-knot 模式) ⭐推荐                  │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │ 1.均匀Halton│ ──> │ 2.逆CDF变换 │ ──> │ 3.B样条插值 │ ──> │ 4.协方差缩放│ ──> │ 5.加到均值  │
  │ 在控制点空间│     │ → 高斯分布  │     │ → 平滑轨迹  │     │ (σ缩放)     │     │ 并裁剪      │
  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼                   ▼
   均匀 [0,1]           标准正态            平滑轨迹噪声         缩放后噪声          最终动作
   [N, K×A]             N(0,1)              [N, H, A]           δ = σ·ε            a = μ + δ
   K=7控制点            [N, K×A]            H=30时间步          [N, H, A]          [N, H, A]
```

**详细步骤说明**:

| 步骤 | 操作 | 维度变化 | 说明 |
|------|------|----------|------|
| 1 | Halton 序列采样 | → [N, K×A] = [498, 42] | 在低维控制点空间采样 |
| 2 | 逆 CDF 变换 | [N, 42] → [N, 42] | $\sqrt{2} \cdot \text{erfinv}(2u - 1)$ |
| 3 | B 样条插值 | [N, 7, 6] → [N, 30, 6] | 7 个控制点 → 30 个时间步 |
| 4 | 协方差缩放 | [N, 30, 6] → [N, 30, 6] | $\delta = \sigma \cdot \epsilon$ |
| 5 | 加到均值并裁剪 | [N, 30, 6] → [N, 30, 6] | $a = \text{clamp}(\mu + \delta)$ |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    动作采样完整流程 (halton 模式) 用于对比                     │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │ 1.均匀Halton│ ──> │ 2.逆CDF变换 │ ──> │ 3.时间滤波  │ ──> │ 4.协方差缩放│ ──> │ 5.加到均值  │
  │ 全时间步空间│     │ → 高斯分布  │     │ (自回归)    │     │ (σ缩放)     │     │ 并裁剪      │
  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼                   ▼
   均匀 [0,1]           标准正态            平滑后噪声          缩放后噪声          最终动作
   [N, H×A]             N(0,1)              [N, H, A]           δ = σ·ε            a = μ + δ
   H×A=180维            [N, H×A]                                [N, H, A]          [N, H, A]
```

---

## 🔷 B 样条采样详解 (Knot Sampling)

### 为什么使用 B 样条？

论文中指出，直接在每个时间步采样独立的噪声会导致轨迹不平滑、抖动。B 样条采样的核心思想是：

1. **在控制点空间采样**：仅采样少量控制点 (knots)
2. **B 样条插值**：通过 B 样条将控制点插值为完整的平滑轨迹
3. **天然平滑性**：B 样条的连续性保证了轨迹的平滑

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    B 样条采样 vs 直接采样对比                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  直接采样 (每个时间步独立)            B 样条采样 (控制点 + 插值)
  
  ┌─────────────────────────┐         ┌─────────────────────────┐
  │   *                     │         │   *                     │
  │      *                  │         │      *                  │
  │ *       *               │         │ ~~~~~~*~~~~~            │ ← 平滑曲线
  │           *  *          │         │              ~~~*~~~    │
  │    *          *         │         │    *              ~*~   │
  │                  *      │         │                      *  │
  └─────────────────────────┘         └─────────────────────────┘
      (抖动、不连续)                       (平滑、连续)
```

### B 样条采样实现

```python
# 文件: storm_kit/mpc/control/sample_libs.py

class KnotSampleLib(object):
    """
    B 样条控制点采样库
    
    通过在低维控制点空间采样，然后使用 B 样条插值生成完整轨迹
    """
    def __init__(self, horizon=0, d_action=0, n_knots=0, degree=3, 
                 sample_method='halton', **kwargs):
        """
        参数:
            horizon: 时间步数 (30)
            d_action: 动作维度 (6)
            n_knots: 控制点数量 (horizon // knot_scale = 30 // 4 = 7)
            degree: B 样条阶数 (默认 3 次样条)
            sample_method: 'halton' 或 'random'
        """
        self.ndims = n_knots * d_action  # 7 * 6 = 42 维
        self.n_knots = n_knots           # 7 个控制点
        self.horizon = horizon           # 30 个时间步
        self.d_action = d_action         # 6 个关节
        self.degree = degree             # 3 次 B 样条
        
    def get_samples(self, sample_shape, **kwargs):
        """
        采样过程:
        1. 采样控制点 (Halton 或 Random)
        2. B 样条插值生成完整轨迹
        """
        
        # Step 1: 采样控制点
        if self.sample_method == 'halton':
            # 使用 Halton 序列采样控制点
            # 采样维度: N × (n_knots × d_action) = N × 42
            self.knot_points = generate_gaussian_halton_samples(
                sample_shape[0],         # 采样数量
                self.ndims,              # 42 维 (7个控制点 × 6个关节)
                use_ghalton=True,
                seed_val=self.seed_val,
                device=self.tensor_args['device'],
                float_dtype=self.tensor_args['dtype']
            )
        elif self.sample_method == 'random':
            # 使用随机高斯采样控制点
            self.knot_points = self.mvn.sample(sample_shape=sample_shape)
        
        # Step 2: 重塑为 [N, d_action, n_knots]
        knot_samples = self.knot_points.view(
            sample_shape[0],   # N
            self.d_action,     # 6
            self.n_knots       # 7
        )
        
        # Step 3: 初始化输出张量 [N, horizon, d_action]
        self.samples = torch.zeros(
            (sample_shape[0], self.horizon, self.d_action), 
            **self.tensor_args
        )
        
        # Step 4: 对每个粒子的每个动作维度进行 B 样条插值
        for i in range(sample_shape[0]):       # 遍历 N 个粒子
            for j in range(self.d_action):     # 遍历 6 个关节
                # 从 7 个控制点插值生成 30 个时间步
                self.samples[i, :, j] = bspline(
                    knot_samples[i, j, :],     # 7 个控制点
                    n=self.horizon,            # 插值到 30 个点
                    degree=self.degree         # 3 次 B 样条
                )
        
        return self.samples  # [N, 30, 6]
```

### B 样条插值函数

```python
# 文件: storm_kit/mpc/control/sample_libs.py

from scipy.interpolate import BSpline
import scipy.interpolate as si

def bspline(c_arr, t_arr=None, n=100, degree=3):
    """
    使用 SciPy 进行 B 样条插值
    
    参数:
        c_arr: 控制点数组 (7 个点)
        t_arr: 控制点对应的时间参数 (默认均匀分布)
        n: 输出采样点数 (30)
        degree: 样条阶数 (3)
    
    返回:
        插值后的平滑曲线 (30 个点)
    """
    sample_device = c_arr.device
    sample_dtype = c_arr.dtype
    cv = c_arr.cpu().numpy()
    count = len(cv)

    # 默认在控制点处均匀分布时间参数
    if t_arr is None:
        t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
    else:
        t_arr = t_arr.cpu().numpy()
    
    # 使用 SciPy 的 splrep 进行样条拟合
    spl = si.splrep(t_arr, cv, k=degree, s=0.5)
    
    # 在完整时间范围内均匀采样
    xx = np.linspace(0, cv.shape[0], n)
    samples = si.splev(xx, spl, ext=3)
    
    samples = torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)
    
    return samples
```

### B 样条采样数学公式

B 样条曲线的数学定义：

$$S(t) = \sum_{i=0}^{n} P_i B_{i,k}(t)$$

其中：
- $S(t)$: 时间 $t$ 处的曲线值
- $P_i$: 第 $i$ 个控制点 (从 Halton 序列采样)
- $B_{i,k}(t)$: $k$ 阶 B 样条基函数
- $n$: 控制点数量 (7 个)
- $k$: 样条阶数 (3 次)

**采样维度变化**:

```
采样控制点空间          B 样条插值           完整轨迹空间
  [N × K × A]     ───────────────>      [N × H × A]
  [500 × 7 × 6]                        [500 × 30 × 6]
     42 维                                180 维
```

> 💡 **优势**: 采样发生在 42 维空间而非 180 维空间，同时 B 样条保证了时间平滑性！

### 采样方法总结对比

| 方法 | 采样空间维度 | 时间平滑 | 空间覆盖 | 计算开销 | 推荐场景 |
|------|--------------|----------|----------|----------|----------|
| `halton` | H×A = 180 | ❌ (需滤波) | ✅ 均匀 | 中 | 快速测试 |
| `random` | H×A = 180 | ❌ (需滤波) | ❌ 随机 | 低 | 基线对比 |
| **`halton-knot`** | **K×A = 42** | **✅ B样条** | **✅ 均匀** | **中** | **⭐ 推荐** |
| `random-knot` | K×A = 42 | ✅ B样条 | ❌ 随机 | 低 | 对比实验 |
| `stomp` | H×A = 180 | ✅ 协方差 | ❌ 随机 | 高 | 特殊场景 |

**关键参数**:
- `knot_scale = 4`: 控制点数量 = horizon / knot_scale = 30 / 4 ≈ 7
- `degree = 3`: B 样条阶数 (3 次样条，C² 连续)

---

## 🎯 Halton 低差异序列

### Halton 序列原理

STORM 在 B 样条控制点采样时使用 Halton 低差异序列：

```python
# 文件: storm_kit/mpc/control/control_utils.py

def generate_gaussian_halton_samples(num_samples, ndims, seed_val, device, float_dtype):
    """
    生成高斯分布的 Halton 样本
    
    Halton 序列是一种准随机序列，比纯随机采样有更好的空间覆盖性
    """
    # 1. 生成均匀分布的 Halton 样本 [0, 1]
    uniform_halton_samples = generate_halton_samples(
        num_samples, ndims, use_ghalton=True, seed_val=seed_val, device=device
    )
    
    # 2. 通过逆误差函数转换为高斯分布
    # 逆 CDF 变换: U[0,1] -> N(0,1)
    gaussian_halton_samples = sqrt(2) * erfinv(2 * uniform_halton_samples - 1)
    
    return gaussian_halton_samples  # shape: [num_samples, ndims]
```

**Halton 序列 vs 随机采样对比**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    采样覆盖性对比 (2D 示意)                                   │
└─────────────────────────────────────────────────────────────────────────────┘

     随机采样 (Random)                    Halton 低差异序列
     ┌────────────────┐                  ┌────────────────┐
     │  *    *   *    │                  │  *    *    *   │
     │     *   *      │                  │    *    *    * │
     │  *  *   * *    │  ← 有聚集和空洞  │  *    *    *   │  ← 均匀覆盖
     │    *    *   *  │                  │    *    *    * │
     │  *   * *    *  │                  │  *    *    *   │
     └────────────────┘                  └────────────────┘
```

**为什么使用 Halton?**
- 更均匀地覆盖控制点空间
- 减少采样数量也能保持良好覆盖
- 确定性序列，便于调试和复现
- 结合 B 样条，在低维空间采样高维平滑轨迹

---

## 📊 可选的时间滤波器

当使用纯 `halton` 或 `random` 采样（非 B 样条）时，可选择使用自回归滤波器增加时间平滑性：

```python
# 文件: storm_kit/mpc/control/sample_libs.py

def filter_samples(self, eps):
    """
    自回归滤波器，增加时间相关性
    
    eps[t] = β₀ * eps[t] + β₁ * eps[t-1] + β₂ * eps[t-2]
    
    注意: 使用 B 样条采样时通常设置 filter_coeffs: None，
    因为 B 样条本身已保证平滑性
    """
    if self.filter_coeffs is not None:
        beta_0, beta_1, beta_2 = self.filter_coeffs
        
        for i in range(2, eps.shape[1]):
            eps[:, i, :] = (beta_0 * eps[:, i, :] + 
                            beta_1 * eps[:, i-1, :] + 
                            beta_2 * eps[:, i-2, :])
    return eps
```

> 💡 **注意**: 使用 `halton-knot` 采样时，`filter_coeffs` 通常设为 `None`，因为 B 样条插值已经保证了轨迹平滑性。

---

## 🔧 协方差缩放

B 样条采样后得到的噪声轨迹通过协方差矩阵的 Cholesky 分解进行缩放：

```python
# 文件: storm_kit/mpc/control/olgaussian_mpc.py

def sample_actions(self, state=None):
    """
    完整的动作采样函数
    
    对于 halton-knot 采样模式:
    1. MultipleSampleLib 调用 KnotSampleLib.get_samples()
    2. KnotSampleLib 内部: Halton采样控制点 -> B样条插值 -> 平滑轨迹
    3. 返回的 delta 已经是平滑的噪声轨迹 [N, H, A]
    """
    
    # Step 1: 从采样库获取噪声轨迹
    # 对于 halton-knot: 返回的是 B 样条插值后的平滑噪声
    # delta: [N-2, H, A] 其中 N-2 = 498
    delta = self.sample_lib.get_samples(
        sample_shape=self.sample_shape,  # [498]
        base_seed=self.seed_val + self.num_steps
    )
    
    # Step 2: 添加零噪声序列 (确保均值动作在采样中)
    # Z_seq: [1, H, A] 全零
    delta = torch.cat((delta, self.Z_seq), dim=0)  # [499, H, A]
    
    # Step 3: 协方差缩放 (可选，取决于 cov_type)
    # 对于 halton-knot + diag_AxA: 按关节维度缩放
    if self.cov_type == 'diag_AxA':
        # scale_tril: [A] = [6]
        scaled_delta = delta * self.scale_tril  # 逐元素相乘
    elif self.cov_type == 'full_HAxHA':
        # 完整时空协方差矩阵
        delta = delta.view(delta.shape[0], self.horizon * self.d_action)
        scaled_delta = torch.matmul(delta, self.full_scale_tril)
        scaled_delta = scaled_delta.view(delta.shape[0], self.horizon, self.d_action)
    else:
        scaled_delta = delta * self.scale_tril
    
    # Step 4: 加到均值动作
    # mean_action: [H, A] = [30, 6]
    # act_seq = μ + σ * δ
    act_seq = self.mean_action.unsqueeze(0) + scaled_delta  # [N, H, A]
    
    # Step 5: 裁剪到动作范围
    act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn='clamp')
    
    # Step 6: 添加特殊粒子
    append_acts = self.best_traj.unsqueeze(0)  # [1, H, A] 上一次最优轨迹
    
    if self.num_null_particles > 0:
        null_act_seqs = torch.zeros(self.num_null_particles, self.horizon, self.d_action)
        append_acts = torch.cat((append_acts, null_act_seqs), dim=0)
    
    # 最终拼接
    act_seq = torch.cat((act_seq, append_acts), dim=0)  # [500, H, A]
    
    return act_seq
```

### 协方差类型详解

STORM 支持多种协方差矩阵类型：

```python
# 文件: storm_kit/mpc/control/olgaussian_mpc.py

def reset_covariance(self):
    """初始化协方差矩阵"""
    
    if self.cov_type == 'sigma_I':
        # 标量协方差: Σ = σ² * I
        # 所有维度使用相同方差
        self.cov_action = torch.tensor(self.init_cov)  # 标量
        self.scale_tril = torch.sqrt(self.cov_action)  # L = σ
        
    elif self.cov_type == 'diag_AxA':
        # 对角协方差: Σ = diag(σ₁², σ₂², ..., σ_A²)
        # 每个动作维度独立方差 (UR7e 配置使用此类型)
        self.cov_action = torch.tensor([self.init_cov] * self.d_action)  # [A]
        self.scale_tril = torch.sqrt(self.cov_action)  # [A]
        
    elif self.cov_type == 'full_AxA':
        # 完整 A×A 协方差矩阵
        # 动作维度之间有相关性
        self.cov_action = torch.diag(torch.tensor([self.init_cov] * self.d_action))  # [A, A]
        self.scale_tril = torch.linalg.cholesky(self.cov_action)  # [A, A]
        
    elif self.cov_type == 'full_HAxHA':
        # 完整 (H*A) × (H*A) 协方差矩阵
        # 时间和动作维度都有相关性
        self.cov_action = torch.diag(torch.tensor(
            [self.init_cov] * (self.horizon * self.d_action)
        ))  # [H*A, H*A]
        self.scale_tril = torch.linalg.cholesky(self.cov_action)  # [H*A, H*A]
```

**协方差类型对比**:

| 类型 | 矩阵大小 | 时间相关 | 动作相关 | 计算开销 | 推荐场景 |
|------|----------|----------|----------|----------|----------|
| `sigma_I` | 标量 | ❌ | ❌ | 最低 | 快速原型 |
| `diag_AxA` | [A] = [6] | ❌ | ❌ | 低 | **UR7e 默认** |
| `full_AxA` | [A×A] = [6×6] | ❌ | ✅ | 中 | 关节相关性强 |
| `full_HAxHA` | [H*A × H*A] = [180×180] | ✅ | ✅ | 高 | 需完整时空相关 |

> 💡 **UR7e HIL 配置使用 `diag_AxA`**，因为 B 样条已经提供了时间平滑性，无需协方差矩阵引入时间相关。

---

## 📐 完整采样数学公式

### 对于 B 样条采样 (halton-knot)

完整的采样过程可表示为：

$$\mathbf{a}_i = \boldsymbol{\mu} + \sigma \cdot \text{BSpline}(\text{Halton}(\mathbf{z}_i))$$

展开为：

1. **控制点采样**: $\mathbf{P}_i = \text{InvCDF}(\text{Halton}(i))$，其中 $\mathbf{P}_i \in \mathbb{R}^{K \times A}$，$K=7$ 个控制点
2. **B 样条插值**: $\boldsymbol{\epsilon}_i = \text{BSpline}(\mathbf{P}_i) \in \mathbb{R}^{H \times A}$
3. **协方差缩放**: $\boldsymbol{\delta}_i = \sigma \cdot \boldsymbol{\epsilon}_i$
4. **加到均值**: $\mathbf{a}_i = \boldsymbol{\mu} + \boldsymbol{\delta}_i$

其中：
- $\mathbf{a}_i \in \mathbb{R}^{H \times A}$: 第 $i$ 条采样的动作序列 (30×6)
- $\boldsymbol{\mu} \in \mathbb{R}^{H \times A}$: 当前均值动作
- $\sigma$: 协方差缩放因子 (对于 `diag_AxA` 是 [6] 向量)
- $\text{Halton}$: Halton 准随机序列 [0,1]
- $\text{InvCDF}$: 逆误差函数变换到 $\mathcal{N}(0,1)$

---

## 📊 采样组成分析

最终的 500 个采样粒子包含以下组成：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         500 个采样粒子的组成                                  │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────────┐
   │                                                                          │
   │  ┌─────────────────────────────────────────────────┐                     │
   │  │  B 样条 + Halton 采样粒子                         │  498 个            │
   │  │  (Halton 控制点 → B样条插值 → 平滑轨迹)            │                    │
   │  └─────────────────────────────────────────────────┘                     │
   │                                                                          │
   │  ┌─────────────────────────────────────────────────┐                     │
   │  │  零噪声粒子 (均值动作)                            │  1 个              │
   │  │  a = μ + 0 = μ (确保均值在采样中)                │                    │
   │  └─────────────────────────────────────────────────┘                     │
   │                                                                          │
   │  ┌─────────────────────────────────────────────────┐                     │
   │  │  上一次最优轨迹                                   │  1 个              │
   │  │  a = best_traj (保留历史最优解)                  │                    │
   │  └─────────────────────────────────────────────────┘                     │
   │                                                                          │
   │  ┌─────────────────────────────────────────────────┐                     │
   │  │  零动作粒子 (可选)                                │  0 个 (默认)       │
   │  │  a = 0 (用于紧急停止场景)                        │                    │
   │  └─────────────────────────────────────────────────┘                     │
   │                                                                          │
   └──────────────────────────────────────────────────────────────────────────┘
                                                              总计: 500 个
```

---

## 🔗 采样代码调用链 (halton-knot 模式)

```python
# 完整的采样调用链 (halton-knot 模式)

# 1. 控制器主循环调用
trajectories = controller.generate_rollouts(state)

    # 2. generate_rollouts 内部调用 sample_actions
    def generate_rollouts(self, state):
        act_seq = self.sample_actions(state=state)  # ← 采样入口
        trajectories = self._rollout_fn(state, act_seq)
        return trajectories

        # 3. sample_actions 调用 MultipleSampleLib
        def sample_actions(self, state=None):
            # sample_lib 是 MultipleSampleLib 实例
            delta = self.sample_lib.get_samples(...)  # ← 获取 B 样条噪声
            ...
            scaled_delta = delta * self.scale_tril     # ← 协方差缩放 (diag)
            act_seq = self.mean_action + scaled_delta  # ← 加到均值
            act_seq = scale_ctrl(act_seq, ...)         # ← 裁剪
            return act_seq

            # 4. MultipleSampleLib.get_samples (sample_ratio: halton-knot=1.0)
            def get_samples(self, sample_shape, **kwargs):
                # 调用 KnotSampleLib
                samples = self.knot_halton_sample_lib.get_samples(sample_shape)
                return samples

                # 5. KnotSampleLib.get_samples (核心采样逻辑)
                def get_samples(self, sample_shape):
                    # 5a. Halton 采样控制点 (42维 = 7控制点 × 6关节)
                    knot_points = generate_gaussian_halton_samples(
                        N, self.n_knots * self.d_action, ...
                    )  # [N, 42]
                    
                    # 5b. 重塑为 [N, d_action, n_knots]
                    knot_samples = knot_points.view(N, 6, 7)
                    
                    # 5c. 对每个关节进行 B 样条插值
                    for i in range(N):
                        for j in range(6):
                            samples[i,:,j] = bspline(
                                knot_samples[i,j,:],  # 7 个控制点
                                n=30,                 # 插值到 30 个时间步
                                degree=3              # 3 次 B 样条
                            )
                    return samples  # [N, 30, 6]

                    # 6. bspline (使用 SciPy)
                    def bspline(c_arr, n, degree):
                        spl = si.splrep(t, c_arr, k=degree, s=0.5)
                        samples = si.splev(xx, spl)
                        return samples
```

---

### 步骤 2: 前向仿真 (Rollout)

对每条采样的动作序列，通过动力学模型前向积分：

```
┌─────────────────────────────────────────────────────────────┐
│  for each particle i in [0, 499]:                          │
│      state[0] = current_state                               │
│      for t in [0, horizon-1]:                               │
│          # 动力学积分 (dt = 0.02s)                          │
│          q̈[t] = action[i, t]           # 加速度控制        │
│          q̇[t+1] = q̇[t] + q̈[t] * dt                        │
│          q[t+1] = q[t] + q̇[t] * dt + 0.5 * q̈[t] * dt²      │
│                                                             │
│          # 正运动学: 关节空间 → 笛卡尔空间                   │
│          ee_pos[t], ee_rot[t] = FK(q[t])                    │
│          link_pos[t] = FK_links(q[t])                       │
└─────────────────────────────────────────────────────────────┘
```

**动力学模型**:
- 使用 URDF 定义的运动学链
- 可微分正运动学 (PyTorch 实现)
- 批量并行计算 (GPU 加速)

### 步骤 3: 计算成本

对每条轨迹计算总成本：

```
┌─────────────────────────────────────────────────────────────┐
│  Cost = Σ [ w_goal × ‖ee_pos - goal_pos‖²                  │  目标跟踪
│           + w_orient × ‖ee_rot - goal_rot‖²                │  姿态跟踪
│           + w_coll × collision_cost(link_pos)              │  碰撞避免
│           + w_self × self_collision_cost(q)                │  自碰撞
│           + w_smooth × ‖Δq̈‖²                               │  平滑性
│           + w_bound × bound_violation(q, q̇, q̈)            │  关节限制
│           + w_vel × ‖q̇‖²  (near goal)                      │  到达时减速
│         ]                                                   │
└─────────────────────────────────────────────────────────────┘
```

输出: `costs: [500,]` 每条轨迹一个标量成本

### 步骤 4: 计算权重 (Softmax)

使用指数效用函数将成本转换为权重：

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   w_i = softmax( -cost_i / β )                              │
│                                                             │
│       = exp(-cost_i / β) / Σ exp(-cost_j / β)               │
│                                                             │
│   β (temperature): 控制权重分布的锐度                        │
│   - β 小: 只有最优轨迹有高权重 (贪婪)                        │
│   - β 大: 权重更均匀 (探索性)                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

输出: `weights: [500,]` 每条轨迹一个权重, $\sum w_i = 1$

### 步骤 5: 加权平均得到最优动作

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   optimal_action = Σ w_i × action_i                         │
│                                                             │
│   (加权平均所有采样的动作序列)                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

输出: `optimal_action: [30, 6]` 最优动作序列

### 步骤 6: 积分得到位置指令

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   # 使用第一个动作                                          │
│   a_cmd = optimal_action[0]      # [6,] 加速度              │
│                                                             │
│   # 积分得到期望位置                                        │
│   q_cmd = q_current + q̇_current * dt + 0.5 * a_cmd * dt²   │
│                                                             │
│   # 或直接用 MPC 预测的下一步位置                            │
│   q_cmd = predicted_state[1, :6]                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

输出: `q_cmd: [6,]` 发送给机械臂的关节位置指令

---

## 📊 成本函数详解

### 成本函数组成

| 成本项 | 公式 | 作用 | 权重 (典型值) |
|--------|------|------|--------------|
| `goal_pose` | $\|p_{ee} - p_{goal}\|^2 + w_r\|R_{ee} - R_{goal}\|^2$ | 目标跟踪 | 15.0, 100.0 |
| `primitive_collision` | $\sum_i \max(0, r_i - d_i)^2$ | 障碍物碰撞 | 5000.0 |
| `robot_self_collision` | $\sum_{ij} \max(0, r_{ij} - d_{ij})^2$ | 自碰撞 | 5000.0 |
| `smooth` | $\|\ddot{q}_{t+1} - \ddot{q}_t\|^2$ | 轨迹平滑 | 1.0 |
| `state_bound` | $\max(0, q - q_{max})^2 + \max(0, q_{min} - q)^2$ | 关节限制 | 1000.0 |
| `zero_vel` | $\|\dot{q}\|^2$ (near goal) | 到达减速 | 0.1 |
| `null_space` | $\|q - q_{retract}\|^2$ | 冗余优化 | 0.1 |

### 目标跟踪成本

$$C_{goal} = w_p \|p_{ee} - p_{goal}\|^2 + w_r \|R_{ee} \ominus R_{goal}\|^2$$

其中:
- $p_{ee}$: 末端执行器位置
- $R_{ee}$: 末端执行器旋转矩阵
- $\ominus$: 旋转误差运算

### 碰撞成本

$$C_{coll} = \sum_{i} \max(0, r_i + r_{obs} - d_i)^2$$

其中:
- $r_i$: 机器人连杆碰撞球半径
- $r_{obs}$: 障碍物半径
- $d_i$: 碰撞球到障碍物中心距离

---

## 💻 代码对应关系

### MPPI 核心类

```python
# 文件: storm_kit/mpc/control/mppi.py

class MPPI(OLGaussianMPC):
    
    def _update_distribution(self, trajectories):
        """步骤 4-5: 计算权重并更新分布"""
        costs = trajectories["costs"]
        actions = trajectories["actions"]
        
        # 步骤 4: Softmax 计算权重
        w = self._exp_util(costs, actions)
        
        # 步骤 5: 加权平均
        weighted_seq = w.unsqueeze(-1).unsqueeze(-1) * actions
        new_mean = torch.sum(weighted_seq, dim=0)
        
        # 更新均值 (带步长)
        self.mean_action = (1-α) * self.mean_action + α * new_mean
    
    def _exp_util(self, costs, actions):
        """指数效用函数 (Softmax)"""
        traj_costs = cost_to_go(costs, self.gamma_seq)[:, 0]
        w = torch.softmax((-1.0/self.beta) * traj_costs, dim=0)
        return w
```

### Rollout 函数

```python
# 文件: storm_kit/mpc/rollout/arm_base.py

class ArmBase(RolloutBase):
    
    def rollout_fn(self, start_state, act_seq):
        """步骤 2-3: 前向仿真 + 成本计算"""
        
        # 步骤 2: 动力学前向积分
        state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        
        # 步骤 3: 计算成本
        cost_seq = self.cost_fn(state_dict, act_seq)
        
        return {
            'actions': act_seq,
            'costs': cost_seq,
            'ee_pos_seq': state_dict['ee_pos_seq'],
            ...
        }
```

### 采样函数

```python
# 文件: storm_kit/mpc/control/olgaussian_mpc.py

class OLGaussianMPC(Controller):
    
    def sample_actions(self, state=None):
        """步骤 1: 采样动作序列"""
        # 生成噪声
        delta = self.sample_lib.get_samples(sample_shape=self.sample_shape)
        
        # 缩放噪声
        scaled_delta = torch.matmul(delta, self.full_scale_tril)
        
        # 添加到均值
        act_seq = self.mean_action.unsqueeze(0) + scaled_delta
        
        # 裁剪到动作范围
        act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs)
        
        return act_seq
```

---

## 🔁 完整控制循环

```python
# 伪代码: 完整的 MPC 控制循环

while running:
    # 1. 获取当前状态 (输入)
    state = robot.get_state()
    full_state = np.concatenate([
        state['position'],      # q
        state['velocity'],      # q̇  
        state['acceleration']   # q̈
    ])
    
    # 2. MPPI 优化 (核心计算)
    for _ in range(n_iters):  # 通常 1-3 次迭代
        # 步骤 1: 采样动作序列
        actions = mpc.sample_actions(full_state)  # [500, 30, 6]
        
        # 步骤 2-3: 前向仿真 + 成本计算
        trajectories = mpc.rollout_fn(full_state, actions)
        
        # 步骤 4-5: 更新动作分布
        mpc._update_distribution(trajectories)
    
    # 3. 获取最优动作 (输出)
    optimal_action = mpc.mean_action[0]  # 第一步动作
    
    # 4. 积分得到位置指令
    dt = 0.02
    q_cmd = state['position'] + state['velocity'] * dt + 0.5 * optimal_action * dt**2
    
    # 5. 安全限制
    q_cmd = apply_velocity_limit(q_cmd, max_vel=0.5)
    q_cmd = apply_smoothing(q_cmd)
    
    # 6. 发送到机械臂
    robot.send_position_command(q_cmd)
    
    # 7. Hotstart: 移动时域窗口
    mpc._shift(shift_steps=1)
    
    # 8. 等待下一个控制周期
    rate.sleep()  # 50Hz
```

---

## ⚙️ 关键参数

### MPPI 参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `horizon` | 预测步数 | 30 |
| `num_particles` | 采样粒子数 | 500 |
| `beta` | 温度参数 | 0.01 |
| `gamma` | 折扣因子 | 0.99 |
| `n_iters` | 迭代次数 | 1 |
| `step_size_mean` | 均值更新步长 | 1.0 |

### 动力学参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `dt` | 仿真时间步 | 0.02s |
| `control_dt` | 控制周期 | 0.02s |
| `max_acc` | 最大加速度 | 5.0 rad/s² |

---

## 📈 性能优化

### GPU 加速

STORM 使用 PyTorch 在 GPU 上批量并行计算：
- 500 条轨迹同时前向积分
- 批量正运动学计算
- 批量碰撞检测

### Hotstart

每个控制周期不从零开始优化，而是使用上一步的解作为初始猜测：
```python
# 移动时域窗口
self.mean_action = self.mean_action.roll(-1, 0)
self.mean_action[-1] = 0  # 或重复最后一个动作
```

---

## 📚 参考资料

- [STORM MPC 论文](https://arxiv.org/abs/2104.13542)
- [MPPI 原始论文](https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf)
- [STORM GitHub](https://github.com/NVlabs/storm)
