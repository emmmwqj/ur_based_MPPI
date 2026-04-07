# 当前实现的 SAGE_MPPI 与 STORM 原生控制器的区别

本文档的目的不是复述论文，而是明确说明**当前仓库里已经落地的实现**，相对于 STORM 原生 controller，到底改了什么、没改什么、额外又叠加了什么运行时逻辑。这样后续你在分析算法增益、做 ablation、或者继续改进实现时，不会把“控制器本体差异”和“Gazebo 部署侧补丁”混在一起。

---

## 1. 对比对象

这里的“STORM 控制器”指当前仓库中的原生 MPPI 实现：

- `storm_kit/mpc/control/control_base.py`
- `storm_kit/mpc/control/olgaussian_mpc.py`
- `storm_kit/mpc/control/mppi.py`

这里的“当前实现的 SAGE_MPPI”主要指：

- `storm_kit/mpc/control/sage_mppi.py`

但如果你实际运行的是：

- `examples/SAGE_MPPI/reach_static_ur7e_tall.py`

那么还必须额外考虑这个脚本里为了 Gazebo tall 场景稳定运行而加上的**近目标精修 / Jacobian 末端精修 / 到达后保持 / stall recovery / RViz 刷新**等逻辑。这些东西**不是 `sage_mppi.py` 本体的一部分**，而是部署层额外策略。

因此，下面会分四层来讲：

1. STORM 原生控制器本体
2. SAGE_MPPI 控制器本体
3. SAGE task 装配层
4. SAGE Gazebo tall 主脚本的额外运行时策略

---

## 2. 先说不变的部分

当前 SAGE 实现并不是推翻 STORM 整个控制链，而是在可信主链上替换 proposal/covariance 更新逻辑。

### 2.1 控制主链没有变

两者都仍然遵循：

`task -> ControlProcess -> controller.optimize()`

也就是说：

- task 仍然负责装配 rollout、控制器、滤波器、ControlProcess
- ControlProcess 仍然负责异步/同步求解、时间基准管理、命令恢复
- controller 仍然只做“给定当前状态，优化一段 horizon 动作序列”

### 2.2 rollout / dynamics / cost 链路没有变

当前 SAGE 没有改：

- `storm_kit/mpc/rollout/arm_base.py`
- `storm_kit/mpc/rollout/arm_reacher.py`
- `storm_kit/mpc/model/urdf_kinematic_model.py`
- `storm_kit/differentiable_robot_model/...`
- goal / collision / self-collision / smoothness / retract 等 cost 定义

因此，baseline 和 SAGE 的**动力学模型、FK/Jacobian、环境碰撞模型、cost 组成**是一致的。真正变化的核心在于：

- 如何采样 proposal
- 如何根据 rollout 结果更新动作分布

### 2.3 许多工程接口仍然保持 STORM 风格

虽然 `SAGE_MPPI` 没有继承原 controller，但它刻意保持了相似的外部使用体验：

- 仍有 `rollout_fn`
- 仍有 `optimize(state, shift_steps=...)`
- 仍有 `reset()` / `reset_covariance()`
- 仍维护 `mean_action` / `best_traj` / `top_trajs`
- 仍支持 `hotstart` / `base_action='repeat'|'null'|'random'`
- 仍支持 `sample_mode='mean'|'sample'`
- 仍复用同一套 sample libs：`HaltonSampleLib` / `MultipleSampleLib` / `RandomSampleLib` / `StompSampleLib`

换句话说，当前 SAGE 实现是“**在 STORM 控制链协议不变的前提下，重写控制器内部更新机制**”。

---

## 3. 控制器层的根本区别

这一节只讨论：

- `mppi.py + olgaussian_mpc.py`
- `sage_mppi.py`

不讨论 Gazebo 主脚本里额外加的 refinement / hold。

### 3.1 继承结构不同

#### STORM 原生 MPPI

继承关系是：

`Controller -> OLGaussianMPC -> MPPI`

特点：

- `Controller.optimize()` 定义统一主循环
- `OLGaussianMPC` 负责开环高斯动作分布、采样、hotstart、shift
- `MPPI` 只重写 `_update_distribution()`，用 MPPI 权重更新均值/协方差

#### 当前 SAGE_MPPI

`SAGE_MPPI` 是一个**完全独立的类**，不继承上述任何 controller。

原因是当时的实现目标是：

1. 不修改旧 controller
2. 不继承 `mppi.py`
3. 只参考 STORM 的接口风格和张量约定

所以当前 `sage_mppi.py` 是一套“独立实现，但外部接口兼容”的 controller。

这意味着：

- 好处：改动隔离，不污染原 baseline
- 代价：它没有自动复用 `Controller.optimize()` 的所有细节，而是自己重写了一套 optimize 主循环

### 3.2 baseline 是“固定/可选更新高斯协方差”，SAGE 是“scale × shape 分解”

#### STORM 原生 MPPI

baseline 的分布是标准开环高斯分布，核心状态是：

- `mean_action`
- `cov_action` / `scale_tril`

协方差类型由 `cov_type` 决定，支持：

- `sigma_I`
- `diag_AxA`
- `full_AxA`
- `full_HAxHA`

在当前 whole_control 的 UR7e baseline 常用配置下：

- `cov_type='diag_AxA'`
- `update_cov=False`

所以实际上它更接近：

- **均值在更新**
- **协方差基本固定**

#### 当前 SAGE_MPPI

SAGE 不直接维护一个“单一协方差矩阵”作为 proposal，而是显式做了：

\[
\Sigma_{k,h} = s_{k,h} \cdot C_{k,h}
\]

其中：

- `s_{k,h}`：每个 inner iteration、每个 horizon step 的标量 scale
- `C_{k,h}`：归一化 shape matrix

也就是说，当前 SAGE proposal 的变化被拆成两部分：

1. proposal 总体“放大/缩小多少”
2. proposal 在各动作维度上的“形状朝向如何各向异性变化”

这是和 baseline 最本质的结构差异。

### 3.3 stage-scaled proposal 是 SAGE 的第一个核心变化

当前实现里，stage scale 在 `_compute_stage_scale()` 中显式写成：

\[
s_{k,h} = \sigma_0 \exp\left( \sigma_1 \frac{h-H}{H} - \sigma_2 \frac{k}{K} \right)
\]

含义：

- horizon 越靠前，proposal 可以更大或更小，取决于 `sigma_1`
- inner iteration 越往后，proposal 通常衰减，取决于 `sigma_2`

而 baseline STORM 里没有这一层显式的 per-stage scalar schedule。baseline 的 proposal 尺度主要由固定/更新后的协方差统一决定。

### 3.4 stagnation-triggered amplification 是第二个核心变化

当前 SAGE 会先在每次 `optimize()` 开始时，利用当前状态和目标计算：

\[
\Delta goal_t = d_{t-1} - d_t
\]

若：

\[
\Delta goal_t < \tau_p
\]

则认为停滞，记为 `z_t = 1`，并放大 stage scale：

\[
s_{k,h} \leftarrow (1 + \alpha z_t) s_{k,h}
\]

当前实现对应：

- `tau_p`
- `stagnation_alpha`
- `z_t`

baseline STORM 原生 MPPI 没有这一机制。它不会根据“最近一拍是否几乎没朝目标前进”去主动放大 proposal。

### 3.5 safe elite anisotropic covariance 是第三个核心变化

baseline STORM 的均值更新使用所有 rollout 的 softmax 权重；协方差更新如果打开，也是在整体样本上做。

当前 SAGE 多了一层“安全精英集合”：

\[
E_t = \{ n \mid J_n \le Q_\eta(J), \; \delta_n > 0 \}
\]

也就是：

1. rollout cost 落在当前 batch 的前 `eta` 分位
2. 同时整个 horizon 上的最小 safety margin 为正

然后只用这组 safe elite 来估 shape matrix：

\[
\hat C_{k,h} = \sum_n \bar w_n (u_{k,h}^{(n)} - \mu_{k,h})(u_{k,h}^{(n)} - \mu_{k,h})^T
\]

再做 trace normalization：

\[
\tilde C_{k,h} = \hat C_{k,h} / (\mathrm{trace}(\hat C_{k,h}) / d)
\]

然后用 trust coefficient：

\[
\rho_k = (k/K)\sum_n w_n \mathbf 1[n \in E_t]
\]

混合成最终 shape：

\[
C_{k,h} = (1-\rho_k)I + \rho_k \tilde C_{k,h}
\]

这和 baseline 的差异非常大：

- baseline 没有 safe elite 这层筛选
- baseline 没有 trace-normalized shape
- baseline 没有 `rho_k` 这种“前期不太信任 anisotropic covariance，后期再逐步信任”的机制

### 3.6 fallback 机制比 baseline 更强

当前 SAGE 有两层重要 fallback：

1. 如果 safe elite 为空，则 `C_{k,h}=I`
2. 如果 `trace(Chat)` 非正或太小，也退回 `I`

所以 SAGE 的各向异性协方差不是“永远强制打开”，而是：

- 有足够好且足够安全的样本时才相信它
- 否则退回更保守的 isotropic 形状

baseline 的逻辑更简单：要么固定协方差，要么直接按样本整体更新。

### 3.7 权重本身仍然是 MPPI 风格，不是彻底换了控制理论

这一点非常重要。

当前 SAGE **没有**改变 MPPI 的基本权重形式，仍然用：

\[
w_n = \frac{\exp(-J_n / \lambda)}{\sum_m \exp(-J_m / \lambda)}
\]

也就是说：

- 均值更新仍然是 MPPI 风格的 softmax reweighting
- 不是把 controller 完全改成 CEM / CMA-ES / diffusion policy

当前 SAGE 的新增，主要集中在**proposal covariance 的构造与更新逻辑**，而不是整个优化器框架全部推倒重来。

### 3.8 当前 SAGE 的 total cost 仍然基本沿用 baseline cost-to-go

当前 `sage_mppi.py` 的 `_compute_total_costs()` 是：

- 直接对 rollout 的 per-step costs 做 `cost_to_go`
- 取每条轨迹的 `[:, 0]`

这与 baseline `mppi.py` 的主路径是一致的。当前实现没有另外引入一大堆新 cost 项，也没有把安全 margin 直接并进主 cost 里；安全 margin 主要用于 safe elite 选择和 covariance 更新，而不是直接重写总代价。

### 3.9 采样器来源没换，但 proposal 映射方式变了

两者都复用 STORM 的 sample libs。

但 baseline 是：

- 标准噪声样本
- 乘上统一 `full_scale_tril`

而当前 SAGE 是：

- 先用 sample lib 拿到标准噪声
- 每个 stage 用自己的 `proposal_scale_tril[h]`
- `proposal_scale_tril[h] = sqrt(s_{k,h}) * chol(C_{k,h})`

因此，采样源相同，但映射到动作扰动的方式不同。

### 3.10 optimize 主循环节奏相似，但不是同一个实现

#### baseline

由 `Controller.optimize()` 统一控制：

1. hotstart shift
2. inner loop:
   - `generate_rollouts`
   - `_update_distribution`
3. 输出 `mean_action` 或 sample

#### current SAGE

虽然没继承 `Controller.optimize()`，但它自己重写的 `optimize()` 仍然保持相似节奏：

1. hotstart shift
2. 计算当前 `goal_progress` 与 `stagnated`
3. 每个 inner iteration：
   - 先构建 stage-dependent proposal
   - 再 rollout
   - 再更新均值和 shape
4. 最后输出 `mean_action` 或 `sample`

所以从外部看，它和 baseline 仍然是同一种 receding horizon 调用模式；区别在于 inner loop 内部每一轮如何构造 proposal。

---

## 4. 当前实现里，SAGE 是如何获得“安全裕度”的

这是一个非常关键、也最容易被忽略的实现差异。

### 4.1 论文里需要的 `delta_n`，当前 rollout 并没有直接提供

现有 `arm_base.py / arm_reacher.py` 并不会直接在 rollout 输出中给出：

- `delta_safe`
- `safety_margin_seq`
- `collision_margin_seq`

所以当前 `SAGE_MPPI` 必须在 controller 里自己恢复这个量。

### 4.2 当前实现采用“基于已有 collision 模块的保守重建”

在 `_compute_rollout_safety_margin()` 中，当前实现会优先尝试：

1. primitive collision
2. self-collision

具体做法是：

- 从 rollout 的 `state_dict` 里拿 `link_pos_seq / link_rot_seq / state_seq`
- 直接调用 rollout 已经挂好的 collision module
- 获取 raw signed distance
- 再转成 paper 里 safe elite 需要的“正数代表安全”的 safety margin

即：

\[
m = -(d_{raw} + distance\_threshold)
\]

所以当前实现的 `delta_n` 不是 rollout 原生字段，而是 controller 内部“二次计算”的结果。

### 4.3 这带来的后果

优点：

- 不用改旧 rollout 文件
- 在当前 whole_control 分支下就能直接工作

代价：

- controller 和 rollout 的耦合更深
- `delta_n` 是“重建量”，不是原生 rollout 输出
- 如果某些场景下拿不到这些中间量，就会触发 margin fallback，使 safe elite 退空，shape 回到单位阵

所以你后续如果要继续优化算法，最值得优先做的工程改进之一就是：

**直接在 rollout 输出里显式加 `delta_safe` 或 `safety_margin_seq`。**

---

## 5. 当前实现里，SAGE 是如何检测停滞的

### 5.1 goal progress 不是靠 task 传进来的，而是 controller 自己算的

当前 `SAGE_MPPI` 不假设 task 会显式提供：

- `x_{t-1}`
- `x_t`

它自己在内部缓存：

- `prev_goal_dist`

每次 `optimize()` 时，根据当前状态重新算：

\[
g(x) = ||p_{ee}(q) - p_{goal}||_2
\]

然后得到：

\[
\Delta goal_t = prev\_goal\_dist - current\_goal\_dist
\]

### 5.2 优先用末端位置误差，不行再回退到关节空间

优先路径：

- 如果 rollout 有 `get_ee_pose()` 且有 `goal_ee_pos`
- 就用末端位置距离

回退路径：

- 如果只拿到了 `goal_state`
- 则退回到 joint-space distance

baseline STORM controller 里没有这一整套“controller 自己维护 goal progress cache 并判定 stagnation”的逻辑。

---

## 6. 当前实现新增了哪些统计量

baseline controller 原生不会给出一套统一实验统计。

当前 `SAGE_MPPI` 则额外维护并导出：

- `success`
- `failure`
- `final_goal_distance`
- `minimum_safety_margin`
- `safe_elite_fraction`
- `safe_weight_mass`
- `rho_k`
- `z_t`
- `covariance_fallback`
- `margin_fallback`

这些量会写进：

- `info`
- `info["stats"]`
- `controller.latest_stats`

意义是：

1. 便于实验日志统一
2. 便于后续分析“性能来自哪里”
3. 能判断 SAGE 的各个新机制是否真的被触发

这部分是 baseline STORM controller 明显没有的实验友好增强。

---

## 7. task 装配层的区别

当前除了 controller 之外，还额外新建了：

- `storm_kit/mpc/task/sage_arm_task.py`
- `storm_kit/mpc/task/sage_reacher_task.py`

### 7.1 这不是改 baseline task，而是并列新建

它们的作用是：

- 不动旧 task
- 单独实例化 `SAGE_MPPI`
- 继续沿用原 Arm rollout / ControlProcess / state filter

所以 task 层的本质变化是：

- 原 baseline task 实例化 `MPPI`
- 当前 SAGE task 实例化 `SAGE_MPPI`

### 7.2 task 层还做了统计量补齐

`SageArmTask` 额外做了：

- 从 controller 取 `latest_stats`
- 根据 `success_threshold` 推断 success/failure
- 暴露 `get_command_and_stats()`

这让 SAGE 更方便被后面的 benchmark / batch runner / CSV logger 直接使用。

这部分也不是 baseline task 默认具有的。

---

## 8. Gazebo tall 项目里，哪些额外逻辑并不属于“纯 SAGE_MPPI 控制器”

这一节非常重要。

如果你运行的是：

- `examples/SAGE_MPPI/reach_static_ur7e_tall.py`

那么最终效果并不只是 `sage_mppi.py` 单独带来的。这个脚本里叠加了多层部署侧策略。

### 8.1 显式绑定 ros2_control forward_position_controller

当前 SAGE tall 脚本和启动器会明确要求：

- Gazebo 控制必须通过 `ros2_control`
- 命令话题必须是 `/forward_position_controller/commands`

这属于部署约束，不是 SAGE 理论本体。

### 8.2 预测轨迹和黄色碰撞球的 RViz 刷新被提高到每控制周期一次

这只是可视化实时性优化，不改变控制律。

### 8.3 预测轨迹起点会显式 prepend 当前末端位置

这修的是可视化一致性问题，不改变 controller 内部 rollout。

### 8.4 近目标 refinement controller

脚本里有 `_NearGoalRefinementController`，它会在近目标时：

- 缩小 `sigma_0`
- 把 `stagnation_alpha` 调低或关掉
- 提高 goal position weight
- 临时把 `retract_state` 改为当前关节位置

这不是 `sage_mppi.py` 论文核心三点的一部分，而是一个**部署层的 near-goal heuristic**。

### 8.5 Jacobian 末端精修器

脚本里还有 `_CartesianGoalRefiner`：

- 利用当前 Jacobian
- 做阻尼最小二乘 Cartesian position correction

这已经不是采样式 MPC proposal 更新逻辑，而是一个局部解析 refinement。

因此，如果你之后要做“纯算法 ablation”，必须把它和 `SAGE_MPPI` 本体分开看。

### 8.6 到达后保持器

脚本里有 `_GoalHoldController` 和后续 `cart_hold` 锁定逻辑：

- 稳定到达后锁定当前关节位姿
- 防止目标附近反复重规划导致回弹

这属于部署层稳态处理，不是 SAGE 核心协方差更新机制。

### 8.7 stall monitor + reset recovery

脚本里还有 `_StallMonitor`：

- 如果离目标还远
- 一段时间内几乎没动
- 且速度也很小

则会：

- reset SAGE distribution
- reset control process timing
- 让下一轮以放大 proposal 重新探索

这本质上是“把论文的 stagnation-aware exploration 和工程恢复机制结合起来”的部署增强，也不是 baseline STORM 自带的标准行为。

---

## 9. 所以，当前“可运行的 SAGE 系统”实际上比论文核心控制器多了哪些东西

如果严格区分，当前项目里有三层：

### 9.1 论文核心最接近的部分

这部分主要在 `storm_kit/mpc/control/sage_mppi.py`：

1. stage-scaled proposal
2. safe-elite anisotropic covariance
3. stagnation-triggered amplification

### 9.2 为了接入 STORM/whole_control 结构而做的实现性改造

这部分也在 `sage_mppi.py` / `sage_arm_task.py`：

1. 不继承 baseline controller，而是独立重写类
2. 保持外部接口兼容
3. 从现有 collision 模块重建 safety margin
4. 内部缓存 goal progress
5. 输出实验统计量

这些是“为了让它在当前仓库里能跑起来”的实现层改造。

### 9.3 为了 Gazebo tall 场景稳定运行而叠加的部署层策略

这部分在 `examples/SAGE_MPPI/reach_static_ur7e_tall.py`：

1. near-goal parameter refinement
2. Jacobian Cartesian refinement
3. hold / cart hold
4. stall recovery
5. 高频 RViz 刷新
6. 强制 forward_position_controller

这些不是论文核心 SAGE 协方差逻辑本身。

---

## 10. 你在分析结果时，应该怎样归因

后面如果你要继续做算法分析，建议把性能来源分成三类来归因：

### A. 真正属于 SAGE 核心控制器的增益

主要看：

- 更强的全局搜索/跨障碍探索能力
- 早期不盲信 anisotropic covariance，后期逐步利用 safe elite shape
- 停滞时的自适应 exploration amplification

### B. 属于当前 whole_control 实现细节的增益

主要看：

- 通过已有 collision 模块重建 safety margin
- goal progress cache
- 更强的 fallback / stats / debug 机制

这些会影响可运行性和可分析性，但不完全等同于论文算法本体。

### C. 属于 Gazebo 部署脚本附加策略的增益

主要看：

- 末端最后几厘米的 Jacobian 精修
- 到达后的稳定保持
- stall recovery

这些对“最终能不能稳稳停住、会不会回弹、最后能不能压到毫米级误差”影响非常大。

因此：

- 如果你比较的是“paper-level controller core”，就不能把 C 类也一起算成 SAGE 算法本体
- 如果你比较的是“实际可部署系统效果”，那 C 类当然也应该算进整体系统性能

---

## 11. 当前实现相对 baseline 的主要优点

总结成最核心的几条：

1. **proposal 不再是固定尺度的统一高斯**，而是 stage-dependent、iteration-dependent 的 scale-shape proposal
2. **安全信息不只进入 cost，而是直接参与 proposal shape 更新**
3. **停滞时会主动增大探索**，不再完全依赖固定协方差硬撞
4. **实验统计量更完整**，更容易做 benchmark 和 ablation
5. **在 Gazebo tall 部署里又叠加了近目标精修与稳定保持**，所以实机/仿真闭环更稳

---

## 12. 当前实现相对 baseline 的主要代价与局限

### 12.1 控制器不是继承式扩展，而是并列重写

这让隔离性更好，但也意味着：

- 代码复用少一些
- baseline 后续修 bug 时，SAGE 不会自动继承到

### 12.2 safety margin 不是 rollout 原生字段

这会带来：

- controller 和 rollout 耦合加深
- 某些场景下可能触发 margin fallback
- 论文公式和工程实现之间多了一层“重建近似”

### 12.3 Gazebo tall 项目里的最终性能并不全来自 `sage_mppi.py`

这是最需要注意的。

如果你看到：

- 最终误差被压到毫米级
- 到达后几乎不回弹
- 近目标阶段非常稳

这些并不只是 SAGE covariance 更新本身的贡献，还包含：

- Jacobian 精修
- hold 锁定
- near-goal 参数切换

### 12.4 当前 success 判定仍然主要基于距离阈值

目前 success/failure 的判定主要还是：

- 目标距离阈值

而不是一个更严格的任务完成定义，例如：

- 位置 + 姿态都满足
- 持续若干步满足
- 同时满足安全裕度约束

后面如果你要做更严格论文实验，建议把 success 口径进一步固定。

---

## 13. 后续最值得优先改进的方向

如果你的目标是继续把当前实现推进成更干净、更强、更适合论文分析的版本，我建议优先顺序如下：

### 13.1 把 safety margin 直接下沉到 rollout 输出

这是最重要的一条。

让 rollout 原生输出：

- `delta_safe`
或
- `safety_margin_seq`

这样：

- controller 就不用再二次推导
- safe elite 逻辑更干净
- ablation 也更容易做

### 13.2 把“纯 SAGE controller”与“Gazebo 部署增强器”彻底分离

例如把：

- near-goal refinement
- cartesian refiner
- hold controller

做成明确可开关的 deployment module，而不是混在主脚本里。

这样你后面做实验时可以更清楚地比较：

1. baseline MPPI
2. pure SAGE_MPPI
3. SAGE_MPPI + deployment refinements

### 13.3 明确哪些参数属于核心算法，哪些属于场景调参

当前 `ur7e_reacher_gazebo_tall_sage.yml` 里既有：

- `sigma_0 / sigma_1 / sigma_2 / eta / tau_p / stagnation_alpha`

也有：

- `refine_*`
- `cart_refine_*`
- `hold_*`

建议后续把它们拆成两个命名层次：

1. `sage_core`
2. `deployment_refinement`

### 13.4 做更干净的 ablation

建议至少做：

1. baseline MPPI
2. baseline + more particles
3. pure SAGE core
4. SAGE core + stagnation only
5. SAGE core + safe elite only
6. SAGE core + deployment refinements

这样才更容易回答：

- 提升到底主要来自 safe-elite covariance 还是 stagnation amplification
- 还是主要来自近目标 Jacobian 精修与 hold

---

## 14. 一句话总结

当前仓库中的 SAGE 系统不是“在 STORM MPPI 上改了几行参数”，而是：

1. 在 `sage_mppi.py` 中，独立实现了一套**兼容 STORM 接口但不继承原 controller** 的 SAGE proposal 更新器；
2. 在 `sage_arm_task.py / sage_reacher_task.py` 中，单独接入了这套 controller；
3. 在 `reach_static_ur7e_tall.py` 中，又额外叠加了一层**面向 Gazebo tall 场景部署稳定性的局部精修和保持逻辑**。

所以后续分析时一定要把：

- **核心控制器差异**
- **实现层兼容性改造**
- **部署层附加策略**

这三层严格分开。
