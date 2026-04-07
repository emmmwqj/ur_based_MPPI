# 当前实现与论文理想形式之间的差距

这份文档专门回答一个问题：

**当前仓库里已经跑通的 SAGE_MPPI，实现上距离论文里最理想、最干净、最容易做学术归因的版本，还差哪些东西？**

这里的“差距”不是说当前实现不能用，而是说：

- 哪些地方是工程妥协
- 哪些地方是当前仓库结构限制
- 哪些地方会影响论文里的算法归因
- 哪些地方最值得后续优先修

---

## 1. 先给一个总判断

当前实现已经足够做到：

1. 在 whole_control 分支里真实跑通
2. 接入 task / ControlProcess / Gazebo
3. 做 baseline vs SAGE benchmark
4. 在 harder scenes 上得到有信息量的实验结果

但它距离“论文理想实现”仍然存在四类差距：

1. **算法接口层差距**
2. **rollout / safety margin 数据层差距**
3. **评测归因层差距**
4. **部署系统层差距**

下面逐条展开。

---

## 2. 算法接口层差距

### Gap 1. SAGE 不是在原 controller 框架上做最小扩展，而是并列重写

#### 理想形式

从论文实现整洁度看，更理想的形式是：

- 直接基于 STORM 原 controller 层做最小差异扩展
- 尽量复用 `Controller.optimize()` 与 `OLGaussianMPC` 的公共逻辑
- 只替换 proposal covariance 相关部分

#### 当前实现

当前 `storm_kit/mpc/control/sage_mppi.py`：

- 完全独立
- 不继承 `Controller`
- 不继承 `OLGaussianMPC`
- 不继承 `MPPI`

而是自己重写了：

- 采样
- rollout 调用
- optimize 主循环
- 分布更新

#### 影响

优点：

- 不污染 baseline
- 风险隔离很强

缺点：

- 和 baseline controller 存在逻辑重复
- baseline 后续改 bug 时，SAGE 不会自动继承
- 做严格算法比较时，很难保证两边除了 proposal 逻辑外其余实现完全一致

#### 建议修复

后续更理想的方向是：

1. 把 SAGE 重构成“尽量复用 baseline controller 主循环”的版本
2. 只把 proposal covariance / safe-elite update 抽成可替换模块

如果暂时不想动现有代码，至少应在实验分析里明确写：

- 当前 SAGE 是**接口兼容的独立重实现**
- 不是 baseline MPPI 的继承式微小修改

---

### Gap 2. 当前实现仍然保留了一些 baseline 兼容字段，但它们对 SAGE 本体并不真正生效

#### 典型字段

- `alpha`
- `step_size_cov`
- `cov_type`
- `update_cov`
- `kappa`

#### 理想形式

理想上应该有更干净的参数接口：

- 核心 SAGE 参数
- 与 baseline 兼容但不生效的遗留参数，最好不要再暴露或至少显式标注无效

#### 当前实现

这些字段现在保留下来，主要是为了：

- 和现有 config / task 构造方式兼容

但其中一部分在 `sage_mppi.py` 里只是“保留签名”，并不会像 baseline 那样驱动核心行为。

#### 影响

- 读配置时容易误判哪些参数真的起作用
- 后续调参时容易出现“改了一个其实不生效的字段”

#### 建议修复

把 SAGE config 进一步拆成：

1. `baseline_compat`
2. `sage_core`
3. `deployment_refinement`

并明确列出：

- 哪些参数仅为接口兼容保留
- 哪些参数真的进入控制律

---

### Gap 3. 目前 SAGE 仍然主要是位置 reaching 口径，并没有把“姿态到达”纳入统一完成判定

#### 理想形式

如果论文要强调通用 manipulator reaching/controller improvement，更理想的成功定义应是：

- 位置
- 姿态
- 安全
- 持续稳定

联合满足

#### 当前实现

当前主要围绕：

- `goal_ee_pos`
- `final_goal_distance`
- `success_threshold`

success/failure 主要还是按**位置误差阈值**判定。

#### 影响

- 论文结果更像“position-reaching benchmark”
- 如果后面切到姿态要求更强的任务，现有 success 定义不够严格

#### 建议修复

后续如果要写更完整的操控实验，应把 success 口径升级为：

- `position_error <= threshold`
- `orientation_error <= threshold`
- `minimum_safety_margin >= threshold`
- 持续 `N` 个 control cycles

---

## 3. rollout / safety margin 数据层差距

### Gap 4. 论文理想上需要直接可用的 `delta_n`，当前实现只能 controller 内重建

#### 理想形式

论文里 safe-elite 定义依赖：

\[
\delta_n = \min_h \text{signed safety margin along rollout } n
\]

最理想的工程实现是：

- rollout 本身就显式输出 `delta_safe`
或
- 输出 `safety_margin_seq`

controller 只消费这个字段。

#### 当前实现

当前 `arm_base.py / arm_reacher.py` 不直接提供该字段。  
`sage_mppi.py` 只能在 `_compute_rollout_safety_margin()` 中：

1. 取 `state_dict`
2. 调用 rollout 已挂好的 collision modules
3. 二次重建 signed safety margin

#### 影响

这是当前实现和论文理想形式之间**最重要的工程差距之一**：

- controller 与 rollout 耦合过深
- `delta_n` 不是 rollout 原生语义，而是 controller 内部重建量
- 使 safe-elite 逻辑不够“单纯”

#### 建议修复

最高优先级建议：

直接在 rollout 输出中增加：

- `delta_safe`
或
- `safety_margin_seq`

这样：

- controller 可以更简洁
- safe-elite 逻辑更接近论文理想形式
- 也更容易做消融和 debug

---

### Gap 5. 当前 safety margin 是“保守近似”，不是任务层显式定义的统一安全指标

#### 理想形式

理想上 `delta_n` 应该有统一定义，并在所有实验中严格一致：

- environment clearance
- self-collision clearance
- 若有其他 safety constraints，也应合并成同一种 signed margin 语义

#### 当前实现

当前 safety margin 的来源是：

1. primitive collision 距离
2. self-collision NN 距离

然后按 controller 内部约定变成“正值代表安全”的 margin。

#### 影响

- 安全指标定义仍然偏工程实现导向
- 如果后续引入 voxel/world SDF、mesh distance、传感器障碍，这个定义可能不够统一

#### 建议修复

后续最好在 rollout 层明确建立一个统一的 safety margin 接口：

- 环境
- 自碰撞
- 其他安全约束

最终统一输出同一语义的 signed margin。

---

### Gap 6. margin fallback 的存在说明当前数据链还不够完整

#### 理想形式

理想上 safe-elite shape 更新不应该经常因为“拿不到 margin”而退化到 identity。

#### 当前实现

如果 rollout 返回的信息不足，当前实现会：

- 让 `delta_safe` 统一为负
- safe elite 为空
- shape fallback 到 `I`

#### 影响

- 控制器在某些任务/场景下会悄悄退化成“只有 stage scale / stagnation，没有 safe-elite shape”
- 如果实验里不监控 `margin_fallback`，很可能误以为 safe-elite 正常起作用

#### 建议修复

后续 benchmark 一定要把下面两项纳入常规日志和主附录表：

- `margin_fallback_rate`
- `covariance_fallback_rate`

同时优先修复上一个 gap，让 rollout 原生提供 margin。

---

## 4. 算法归因层差距

### Gap 7. 当前可运行系统的最终性能，不完全来自 SAGE core

#### 理想形式

从论文算法归因角度，理想情况是：

- benchmark 中的性能提升，尽量只来自 SAGE proposal 设计本身

#### 当前实现

实际运行的 Gazebo tall 项目在 `examples/SAGE_MPPI/reach_static_ur7e_tall.py` 里又叠加了：

- `_NearGoalRefinementController`
- `_CartesianGoalRefiner`
- `_GoalHoldController`
- `_StallMonitor`

这些都不属于 `sage_mppi.py` 控制器本体。

#### 影响

如果不区分：

- controller core gain
- deployment stabilizer gain

那么你很难在论文里准确说：

- “SAGE proposal 本身提升了多少”
- “额外局部精修与保持又贡献了多少”

#### 建议修复

后续必须做清晰的双版本划分：

1. `Pure SAGE Core`
2. `Deployed SAGE System`

并在论文里明确：

- 核心 benchmark 结果来自哪个版本
- Gazebo 联机闭环展示来自哪个版本

---

### Gap 8. 当前 benchmark 结果与 Gazebo 部署结果还不是完全同一种系统

#### 理想形式

理想上：

- headless benchmark
- Gazebo closed-loop validation

应该尽量使用同一系统配置，只改变仿真后端，而不是改变控制栈结构。

#### 当前实现

当前 round4 harder benchmark 的结果主要基于 headless 基础设施；而 Gazebo tall 项目则叠加了多层部署逻辑。

#### 影响

- benchmark 强调 controller proposal 的统计提升
- Gazebo 强调实际闭环稳定运行

这两者的目标不同，但如果不写清楚，读者会以为它们是完全同一个系统设置。

#### 建议修复

建议后续在文稿里明确分两块：

1. **Benchmark section**：paired harder benchmark
2. **Deployment recheck section**：Gazebo closed-loop validation

并说明 Gazebo 结果包含额外部署增强器。

---

### Gap 9. 当前还缺少“固定算力预算”的公平性归因

#### 理想形式

如果论文要说 proposal 设计优于 baseline，就应控制计算预算：

- 同等粒子数 × 迭代数
或
- 同等总 rollout 数

#### 当前实现

当前 SAGE 常用：

- `n_iters = 3`
- `num_particles = 1000`

而 baseline 的常用设置未必完全预算匹配。

#### 影响

这会让审稿或后续自我分析时出现一个常见质疑：

- “是不是只是因为 SAGE 多做了优化迭代”

#### 建议修复

后续一定补：

- fixed-compute-budget ablation

这是论文级说服力很关键的一项。

---

## 5. 统计与评测层差距

### Gap 10. 当前 success 定义仍然偏单阈值，且不同实验层级的完成条件还不够统一

#### 理想形式

希望所有实验都用一致成功定义：

- 位置误差
- 姿态误差
- 安全约束
- 时间预算
- 稳态保持

#### 当前实现

当前 success 更接近：

- `final_goal_distance <= success_threshold`

有些 Gazebo 脚本里又叠加了：

- 连续多步进入阈值
- 或 Jacobian 精修达到更小阈值后锁定

#### 影响

- headless benchmark 成功定义
- Gazebo 主脚本保持逻辑

两者不完全等价

#### 建议修复

后续应给出正式统一口径，例如：

1. 达到位置阈值
2. 满足最小安全裕度
3. 在连续 `N` 步内保持

然后 benchmark 与 Gazebo 都统一使用。

---

### Gap 11. 当前统计已经不错，但“停滞发生率”还没有成为主汇总指标

#### 理想形式

既然论文核心机制之一是 stagnation-triggered amplification，那么理想上应显式报告：

- 每个场景的 stagnation incidence
- SAGE 是否减少了不可恢复停滞

#### 当前实现

日志里已经有：

- `z_t`

但当前主表更多聚焦：

- success rate
- steps to success
- minimum safety margin
- final goal distance

#### 影响

虽然主指标合理，但缺少一个能直接对应论文机制的诊断量。

#### 建议修复

后续附录或机制分析表里补：

- mean `z_t`
- stagnation episode rate
- stall recovery trigger count

---

## 6. Gazebo / 部署系统层差距

### Gap 12. 真实 Gazebo 障碍物注入链还不够稳定

#### 理想形式

理想的 Gazebo 复核应做到：

- planner world
- Gazebo physical obstacles

两边完全一致

#### 当前实现

之前联机复核时已经发现：

- `spawn_entity` / obstacle 注入并不总是稳定

有时会变成：

- 真实 ROS2/Gazebo 闭环控制是通的
- 但物理障碍物没有完整镜像进去

#### 影响

这使 Gazebo 复核更像：

- “真实闭环控制复核”

而不是：

- “完全与 planner world 对齐的 physical world 复核”

#### 建议修复

后续最好单独补一个更稳的 obstacle mirroring 层，确保：

- world yaml
- planner collision world
- Gazebo spawned obstacles

三者严格一致。

---

### Gap 13. 当前到达后稳定性很大程度依赖额外 local heuristic，而不是纯 MPC 收敛

#### 理想形式

理想上 controller 自身就能：

- 收敛到目标
- 保持稳定

而不是强依赖额外 latch / hold / Jacobian patch

#### 当前实现

当前为了解决 tall Gazebo 的实际问题，加入了：

- near-goal parameter shrink
- Jacobian Cartesian refinement
- hold latch

#### 影响

从部署视角，这是合理的。  
但从算法纯度视角，这说明：

- 当前纯 SAGE core 还不是最终部署闭环的全部答案

#### 建议修复

后续要么：

1. 明确承认这是“controller + local stabilizer”的系统

要么：

2. 做更纯的 controller-only 版本，并说明部署版额外多了哪些补丁

---

### Gap 14. 当前 RViz / Gazebo 可视化已经很实用，但还不是标准实验记录系统

#### 理想形式

理想上实验系统应支持：

- 自动保存轨迹回放
- 自动保存关键 marker / trajectory snapshot
- 自动对齐 scene / controller / pair id

#### 当前实现

当前可视化主要服务在线调试：

- 预测轨迹
- 黄色碰撞球
- 当前末端和目标

#### 影响

对调试很好，但对论文复现实验记录还不够体系化。

#### 建议修复

如果后续要做更完整的论文附录，建议补：

- 关键 episode 自动导出 screenshot / rosbag / marker dump

这不是核心算法问题，但有助于结果展示。

---

## 7. 代码结构层差距

### Gap 15. 当前 SAGE 相关逻辑分散在 controller / task / example 多处

#### 理想形式

理想上应有比较清晰的模块边界：

1. `controller core`
2. `task wiring`
3. `deployment refinement`
4. `experiment tooling`

#### 当前实现

当前逻辑分散在：

- `storm_kit/mpc/control/sage_mppi.py`
- `storm_kit/mpc/task/sage_arm_task.py`
- `storm_kit/mpc/task/sage_reacher_task.py`
- `examples/SAGE_MPPI/reach_static_ur7e_tall.py`
- 多个 benchmark / analysis / paper scripts

#### 影响

虽然都能跑，但对后续继续演化不够理想：

- 新人上手难
- 做 clean ablation 时容易遗漏某些开关

#### 建议修复

后续可以把与部署增强器相关的逻辑单独抽到：

- `examples/SAGE_MPPI/deployment/`
或
- `storm_kit/mpc/deployment/`

这样 controller core 会更清晰。

---

### Gap 16. 目前缺少统一的“纯 SAGE / 部署增强版 SAGE”配置切换层

#### 理想形式

理想上通过 config 就能明确控制：

- 是否只启用 core SAGE
- 是否启用 refinement / hold / stall recovery

#### 当前实现

虽然已有一部分 `sage:` 配置项，但部署层逻辑开关还不够系统化。

#### 影响

做系统性 ablation 会比较麻烦。

#### 建议修复

后续建议把配置拆成：

- `sage_core`
- `sage_deployment_refinement`

并确保每个部署增强器都可以显式开关。

---

## 8. 从论文角度看，哪些 gap 最值得优先修

如果只按论文价值排序，我建议优先级如下：

### P1. 让 rollout 原生输出 `delta_safe`

这是最值得优先修的 gap。  
原因：

- 它直接决定 safe-elite 机制是否足够干净
- 也决定算法描述和工程实现之间是否还隔着一层重建近似

### P2. 做 fixed-compute-budget 公平性比较

这是说服别人“提升来自 proposal 设计而不是算力堆叠”的关键。

### P3. 把 Pure SAGE Core 与 Deployed SAGE System 严格分离

否则后面很难清楚回答：

- 哪些提升是算法本身
- 哪些提升来自部署层局部精修

### P4. 统一 success 定义

这是让 benchmark、Gazebo 复核、论文表格三者口径一致的关键。

### P5. 提高 Gazebo 障碍物注入一致性

这会提升联机复核的可信度，但优先级低于前四项。

---

## 9. 一句话总结

当前实现已经不是“想法级原型”，而是**可以真实跑 benchmark 和 Gazebo 闭环的工程版本**。  
但它离论文最理想形式仍然有三条主线差距：

1. `delta_n` 还不是 rollout 原生字段，而是 controller 内重建
2. `Pure SAGE Core` 与 `Deployed SAGE System` 还没有被完全分开评估
3. 还缺固定算力预算与更统一成功定义的严格归因

如果后续优先把这三条补齐，当前这套实现就会从“工程上能跑、效果也不错”进一步升级为“算法归因更干净、论文说服力更强”的版本。
