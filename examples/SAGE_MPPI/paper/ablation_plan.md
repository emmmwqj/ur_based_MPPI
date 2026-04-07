# SAGE_MPPI 后续消融实验计划

这份文档的目标是把后续最值得做的消融实验拆成**可执行清单**，而不是停留在“以后可以做 ablation”这种泛泛建议。

文档默认基于当前仓库已经具备的实验基础设施：

- round2 / round3 / round4 benchmark runner
- round4 harder pair 数据集
- baseline 与 SAGE 统一日志 schema
- paired statistics 分析脚本

因此，下面的计划会尽量复用现有实验链，而不是另起一套完全新的框架。

---

## 1. 消融实验的总体原则

后续消融必须始终区分三类因素：

1. **SAGE 核心控制器因素**  
   指 `storm_kit/mpc/control/sage_mppi.py` 中的 proposal 设计本体。

2. **whole_control 实现因素**  
   指为了在当前仓库中落地而做的兼容性实现，比如 controller 内重建 safety margin、额外的实验统计量输出等。

3. **Gazebo 部署附加因素**  
   指 `examples/SAGE_MPPI/reach_static_ur7e_tall.py` 里加的 near-goal refinement、Jacobian 末端精修、hold、stall recovery 等。

如果后续消融不把这三层拆开，最后很难回答：

- 提升到底来自 SAGE 的 covariance 设计本身
- 还是来自 near-goal 局部精修
- 或只是因为多加了重置 / 保持 / 恢复策略

---

## 2. 消融实验的主数据集建议

### 2.1 主数据集

后续消融优先复用：

- round4 harder pairs

原因：

1. 这套数据已经证明能重新激活 failure mode
2. baseline 与 SAGE 在相同 pair 上可做配对统计
3. `obstacle_medium / obstacle_hard / narrow_medium / narrow_hard` 已经能稳定拉开差异

### 2.2 不建议再以 round1/round2 为主消融集

原因：

- round1 基本是 deterministic fixed-goal smoke benchmark
- round2 虽然有多目标，但任务难度和 failure mode 仍不够强
- 真正对 controller proposal 设计有辨识度的，仍然是 round4 harder pairs

### 2.3 推荐的主评价场景

优先级建议：

1. `narrow_hard`
2. `obstacle_hard`
3. `narrow_medium`
4. `obstacle_medium`

其中：

- `hard` 更适合看 success/failure 与停滞恢复
- `medium` 更适合看收敛效率和安全裕度

---

## 3. 论文最小必做消融

根据 `main.pdf` 的核心设计，最小必须回答的问题是：

1. stage scale 单独是否有效
2. safe-elite anisotropic covariance 单独是否有效
3. stagnation-triggered amplification 单独是否有效
4. full SAGE 是否比各个部分单开更强

因此，最小必做 ablation 集应为：

### A0. Baseline fixed-covariance MPPI

定义：

- 当前 whole_control baseline
- `mppi.py + olgaussian_mpc.py`
- 固定 `diag_AxA` 协方差
- `update_cov=False`

作用：

- 所有消融的统一参照组

### A1. Stage Scale Only

定义：

- 保留 `s_{k,h}` 的 stage-scaled schedule
- 但 `C_{k,h} = I`
- 不启用 safe-elite anisotropic covariance
- 不启用 stagnation amplification

回答的问题：

- 单独只做 coarse-to-fine scale schedule，是否已经比固定协方差更好

### A2. Stage Scale + Safe-Elite Shape

定义：

- 启用 stage-scaled schedule
- 启用 safe-elite anisotropic covariance
- 关闭 stagnation amplification

回答的问题：

- geometry-aware anisotropic covariance 是否能单独带来收益

### A3. Stage Scale + Stagnation Amplification

定义：

- 启用 stage-scaled schedule
- 保持 `C_{k,h}=I`
- 启用 stagnation-triggered amplification

回答的问题：

- 单独只加“停滞就放大探索”能否显著减少 local minimum failure

### A4. Full SAGE

定义：

- stage scale
- safe-elite anisotropic covariance
- stagnation amplification

回答的问题：

- 三者结合是否优于任意单项或双项组合

---

## 4. 推荐的第二层消融：实现敏感性消融

上面的 A0-A4 能回答“核心设计是否有效”。但如果你后面还要继续改算法，还需要做第二层消融，搞清楚“当前实现里哪些细节最敏感”。

### B1. 直接安全裕度 vs 当前 controller 内重建安全裕度

当前现状：

- `delta_n` 不是 rollout 原生输出
- `sage_mppi.py` 内部从 collision module 重建

建议消融：

1. 当前版本：controller 内重建 margin
2. 未来版本：rollout 原生输出 `delta_safe`

回答的问题：

- 当前 safety margin reconstruction 是否引入了额外噪声或偏差
- safe-elite 选择是否受该近似影响明显

### B2. `eta` 敏感性

当前默认：

- `eta = 0.2`

建议测试：

- `0.1`
- `0.2`
- `0.3`
- `0.4`

回答的问题：

- safe elite 太少时，anisotropic covariance 是否不稳定
- safe elite 太多时，是否又退化成普通整体样本统计

### B3. `stagnation_alpha` 敏感性

当前 tall 场景 SAGE 配置里默认较大：

- `stagnation_alpha = 8.0`

建议测试：

- `0`
- `1`
- `3`
- `5`
- `8`

回答的问题：

- amplification 的阈值和强度多大才最合适
- 是“有就行”还是“必须足够大才显著”

### B4. `sigma_1 / sigma_2` 敏感性

当前默认：

- `sigma_1 = 1.0`
- `sigma_2 = 0.5`

建议测试：

- `sigma_1`: `0.0 / 0.5 / 1.0 / 1.5`
- `sigma_2`: `0.0 / 0.25 / 0.5 / 1.0`

回答的问题：

- horizon-wise broadening 和 iteration-wise cooling 各自的贡献
- 当前 stage schedule 是否过强或过弱

### B5. `rho_k` 混合策略敏感性

当前实现：

\[
\rho_k = (k/K)\sum_n w_n \mathbf 1[n \in E_t]
\]

建议测试：

1. 当前实现
2. 去掉 `(k/K)`，只保留 safe weight mass
3. 固定 `rho_k = 1`

回答的问题：

- “前期别太相信 anisotropic covariance” 这件事到底有多重要

### B6. `n_iters` 与搜索质量

当前 SAGE 常用：

- `n_iters = 3`

建议测试：

- `1`
- `2`
- `3`
- `5`

回答的问题：

- SAGE 的收益是不是依赖足够多的 inner iterations
- 如果 `n_iters=1`，safe-elite covariance 几乎是否还来不及发挥作用

### B7. 固定计算预算下的公平性消融

建议：

- baseline: `500 particles × 1 iter`
- SAGE: `500 particles × 3 iter`
- baseline budget-matched: `1500 particles × 1 iter`
- 或 baseline: `500 × 3` / SAGE: `500 × 3`

回答的问题：

- SAGE 的收益到底来自“更好的 proposal 设计”
- 还是只是因为多做了 inner iterations

这一项对论文说服力非常重要。

---

## 5. 推荐的第三层消融：部署层附加策略消融

如果你要把 `examples/SAGE_MPPI/reach_static_ur7e_tall.py` 的实际闭环效果写进系统结果，就必须把这些部署层策略单独消融出来。

### C1. Pure SAGE Core

定义：

- 只保留 `SAGE_MPPI`
- 不启用 near-goal refinement
- 不启用 Jacobian Cartesian refiner
- 不启用 hold
- 不启用 stall monitor reset

用途：

- 作为“纯 controller core”版本

### C2. SAGE + Near-Goal Refinement

定义：

- 在 Pure SAGE Core 基础上
- 只开启 `_NearGoalRefinementController`

回答的问题：

- 单独只做近目标参数切换，能提升多少

### C3. SAGE + Cartesian Refiner

定义：

- 在 Pure SAGE Core 基础上
- 只开启 `_CartesianGoalRefiner`

回答的问题：

- 最后几厘米 / 毫米级收敛主要是不是来自 Jacobian 局部解析精修

### C4. SAGE + Hold

定义：

- 在 Pure SAGE Core 基础上
- 只开启 `_GoalHoldController`

回答的问题：

- “达到后不回弹”主要是不是 hold 带来的，而不是控制器本身稳态优化更好

### C5. SAGE + Stall Recovery

定义：

- 在 Pure SAGE Core 基础上
- 只开启 `_StallMonitor` 触发的 reset/recover

回答的问题：

- tall / obstacle 场景里，局部最小值跳出能力有多少来自 runtime recovery，而不是 proposal 自身

### C6. Full Deployed SAGE

定义：

- 当前 `reach_static_ur7e_tall.py` 全部启用

作用：

- 与 Pure SAGE Core 对比，量化“部署层增强器”总体加成

---

## 6. 每个消融要看哪些主指标

为了让后面结果表更一致，建议固定主指标层级。

### 一级主指标

这些应该进入论文主表：

- `success_rate`
- `mean_steps_to_success`
- `mean_minimum_safety_margin`
- `mean_final_goal_distance`

### 二级诊断指标

这些更适合放附录或 ablation 分析表：

- `stagnation incidence`
- `mean_safe_elite_fraction`
- `mean_safe_weight_mass`
- `mean_rho_k`
- `covariance_fallback_rate`
- `margin_fallback_rate`
- 平均控制频率
- `avg rollout_time`

### 三级实现诊断指标

这些建议只在调参期用：

- 每次优化的 `stage_scale_mean`
- `rho_k_seq`
- `covariance_fallback_seq`
- stall recovery 触发次数
- near-goal refinement 进入/退出次数
- cartesian refiner 启动次数
- hold 进入/退出次数

---

## 7. 统计方法建议

既然后续主 benchmark 已经是相同 pair 的 paired design，那么 ablation 也应尽量延续配对统计。

### 二元成功率

建议：

- McNemar test

适用对象：

- baseline vs 某个 ablation
- ablation A1 vs A4
- Pure SAGE Core vs Full Deployed SAGE

### 连续型指标

建议：

- Wilcoxon signed-rank test

适用对象：

- `steps_to_success`
- `minimum_safety_margin`
- `final_goal_distance`

### 额外建议

除了 p-value，建议始终同时记录：

- paired mean difference
- median difference
- 成功样本数量
- 失败样本数量

不要只报显著性，不报 effect size。

---

## 8. 建议的执行顺序

为了控制工作量，建议按下面顺序推进，而不是一次铺满所有 ablation。

### Phase 1: 论文最小核心消融

先做：

1. `A0 Baseline`
2. `A1 Stage Scale Only`
3. `A2 Stage Scale + Safe-Elite Shape`
4. `A3 Stage Scale + Stagnation`
5. `A4 Full SAGE`

目的：

- 回答论文最核心问题：三部分分别贡献了什么

### Phase 2: 关键实现敏感性

只补最重要的三项：

1. `B2 eta`
2. `B3 stagnation_alpha`
3. `B6 n_iters / fixed compute budget`

目的：

- 弄清楚当前实现的收益对哪些超参数最敏感

### Phase 3: 部署层归因

在 Gazebo tall 项目里补：

1. `C1 Pure SAGE Core`
2. `C3 SAGE + Cartesian Refiner`
3. `C4 SAGE + Hold`
4. `C6 Full Deployed SAGE`

目的：

- 回答“最后几厘米精度和到达后稳定性到底来自哪里”

---

## 9. 建议优先新增的实验入口

虽然这份文档不直接写代码，但从仓库结构上看，后续最自然的新增方式是：

### 9.1 控制器变体

建议并列新增，而不是改现有 `sage_mppi.py`：

- `sage_stage_only.py`
- `sage_stage_safe.py`
- `sage_stage_stagnation.py`

或者更轻一点：

- 保持 `sage_mppi.py`
- 通过 config 中的开关禁用某些模块

但如果走 config 开关路线，建议显式支持：

- `enable_stage_scale`
- `enable_safe_elite_shape`
- `enable_stagnation_amplification`

### 9.2 部署层开关

对于 `reach_static_ur7e_tall.py` 这种脚本，建议后续增加明确开关：

- `enable_near_goal_refinement`
- `enable_cartesian_refiner`
- `enable_hold`
- `enable_stall_recovery`

否则部署层消融会很难做干净。

---

## 10. 最值得优先回答的五个研究问题

如果时间有限，我建议后续消融优先回答下面五个问题：

### Q1. SAGE 的真正核心增益主要来自哪一项？

最关键对比：

- `A0` vs `A1`
- `A1` vs `A2`
- `A1` vs `A3`
- `A2` / `A3` vs `A4`

### Q2. 当前更难场景下，safe-elite geometry shaping 还是 stagnation amplification 更重要？

预期：

- `obstacle_hard` 更看放大探索
- `narrow_hard` 更可能更依赖 geometry-aware shape

### Q3. SAGE 的收益在固定计算预算下是否仍然成立？

如果这个问题回答不好，审稿时很容易被质疑：

- “你只是多做了迭代，不是 proposal 更好”

### Q4. 末端毫米级收敛到底来自 controller core，还是来自 Jacobian refiner？

这个问题对你后续写“算法贡献”非常关键。

### Q5. 当前实现中的 safety margin reconstruction 会不会扭曲 safe-elite set？

这决定了你后续是否应该优先把 `delta_safe` 下沉到 rollout。

---

## 11. 结论：推荐的最小可执行消融包

如果只想用最小额外工作量拿到最有价值的结论，我建议下一轮至少完成：

1. `A0` baseline
2. `A1` stage scale only
3. `A2` stage scale + safe-elite shape
4. `A3` stage scale + stagnation
5. `A4` full SAGE
6. `B6` fixed compute budget fairness
7. `C1` pure SAGE core
8. `C6` full deployed SAGE

这样你就能同时回答：

- 论文核心三部分分别贡献了什么
- 收益是否只是算力带来的
- Gazebo 实际闭环增益里，有多少来自额外部署策略

这会比继续无穷扩 benchmark 更有研究价值。
