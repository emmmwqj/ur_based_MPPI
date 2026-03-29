# STORM Tall Debug Analysis

## 调试入口
- Gazebo: `~/storm/examples/sim_gazebo/run_gazebo.sh`
- Debug controller: `~/storm/examples/sim_gazebo/debug/run_reach_static_tall_debug.sh`
- Debug script: `~/storm/examples/sim_gazebo/debug/reach_static_ur7e_tall_debug.py`
- 日志: `~/storm/examples/sim_gazebo/debug/logs/run_20260328_203254.log`
- Captures: `~/storm/examples/sim_gazebo/debug/captures/`

## 我实际做的复现
1. 保持 Gazebo 启动方式不变。
2. 用 debug 版控制器接管高墙场景。
3. 先让机械臂到默认目标 `world=[0.5, -0.45, 0.4]`。
4. 发布跨墙目标 `world=[0.403, 0.400, 0.500]`。
5. 再发布回原目标 `world=[0.5, -0.45, 0.4]`。
6. 在“末端离目标较远且基本不动”时自动保存控制器采样数据。

## 新增的 debug 数据
每次触发停滞保存一份 `npz + json`，内容包括：
- 当前 `q / dq / ddq`
- 当前 `ee_pos_world` 和 `goal_world`
- `controller.cov_action`
- `controller.scale_tril`
- `controller.mean_action`
- `controller.best_traj`
- 所有粒子的 `sample_actions`
- 所有粒子的 `sample_cost_seq`
- 所有粒子的 `sample_ee_pos_seq`
- 所有粒子的 `total_costs`

## 结论
这次看到的“机械臂不动”不是单一原因，而是两件事叠加：

### 1. 目标切换瞬间有一个控制流程 bug
日志里反复出现：
- `后台 MPC 进程未在超时内退出，强制终止...`
- `MPC异常] 目标更新后的同步重规划失败: index 0 is out of bounds for dimension 0 with size 0`
- `MPC恢复] 同步取命令失败 ...`

这说明：
- 目标切换时脚本调用了 `_restart_control_process(...)`
- 之后立刻同步求解
- 但 `ControlProcess` 的时域状态没有稳定衔接好，先触发一次 `index 0 is out of bounds ...`

这会导致目标切换后的**第一段短暂停滞**。

这个问题是控制流程 bug，不是 local minimum。

### 2. 随后确实进入了一个真实的 local minimum / sampling basin
这一点在回切目标后最明显。

代表性 capture：
- `stall_capture_20260328_203459_04.npz`
- `stall_capture_20260328_203507_05.npz`
- `stall_capture_20260328_203518_06.npz`
- `stall_capture_20260328_203526_07.npz`
- `stall_capture_20260328_203534_08.npz`

共同现象：
- `ee_error` 长时间卡在 `0.30m ~ 0.32m`
- `history_motion` 只有 `0.002m ~ 0.010m`
- `velocity_norm` 很小，机械臂基本不动
- 但控制器每轮仍在正常优化，`opt_dt` 正常

这说明不是控制器死掉，而是**优化器一直在同一个坏 basin 里打转**。

## 为什么我判断是真正的局部极小

### 证据 A：采样协方差几乎固定不变，探索半径太小
从多份 capture 看，始终是：
- `cov_action = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]`
- `scale_tril = [0.07071, ..., 0.07071]`

说明：
- 当前 MPPI 一直用固定对角协方差
- 没有随着“跨墙目标切换”自动放大探索
- 分布没有自适应扩张去跳到另一条 homotopy

### 证据 B：在停滞时，1000 条采样轨迹几乎没有一条真正接近目标
以 `stall_capture_20260328_203459_04.npz` 为例：
- 当前 `ee_error = 0.3065 m`
- 所有采样轨迹对目标的 `path_min_dist`:
  - `min = 0.1742 m`
  - `mean = 0.2914 m`
  - `max = 0.3066 m`
- 进入 `0.10m` 范围的轨迹数量：`0 / 1000`
- 进入 `0.20m` 范围的轨迹数量：`4 / 1000`

以 `stall_capture_20260328_203518_06.npz` 为例：
- 当前 `ee_error = 0.3094 m`
- 进入 `0.10m` 范围的轨迹数量：`0 / 1000`
- 进入 `0.20m` 范围的轨迹数量：`0 / 1000`

这已经不是“有好轨迹但执行不出来”，而是**采样本身就没有覆盖到目标附近**。

### 证据 C：最优 5 条轨迹的终点仍然停在错误一侧
例如 `stall_capture_20260328_203526_07.npz`：
- 目标：`[0.5, -0.45, 0.4]`
- 最优 5 条轨迹终点大致都在：
  - `x≈0.43~0.52`
  - `y≈-0.19`
  - `z≈0.59~0.62`
- 它们距离目标仍有 `0.32m ~ 0.35m`

也就是说：
- 代价最低的那几条轨迹，仍然没有“绕到墙另一侧再下探”
- 最优粒子本身就还留在当前这一侧的 basin 里

### 证据 D：top cost 经常几乎一样，说明采样集中在同一类轨迹上
多份 capture 里都有这种情况：
- top5 cost 几乎相同
- 有时甚至 5 条 cost 完全一样

这通常意味着：
- 有效样本都落在同一类几何路径附近
- 分布已经塌缩到一个很窄的局部区域
- MPPI 没有真正探索到另一条跨墙绕障路径

## 根因分析
结合当前配置，真正导致 local minimum 的主因是这几个配置一起作用：

### 1. `hotstart: True`
控制器强烈继承上一个目标时刻的均值轨迹。

在“目标从墙一侧跳到另一侧”时，这会把采样中心继续留在旧目标对应的那条 basin 上。

### 2. `sample_mode: mean`
当前执行的是**采样分布的均值轨迹**，不是最优样本轨迹。

在多模态绕障问题里：
- 均值轨迹很容易落在两个可行 homotopy 中间的坏区域
- 或者始终被困在当前 basin 的平均动作里

### 3. `update_cov: False`
协方差不会根据当前困难程度扩张。

这意味着：
- 一旦 warm start 把均值带进一个坏 basin
- 采样半径不够大，就很难跳出去

### 4. `init_cov: 0.005` 太保守
对应的 `scale_tril ≈ 0.0707`。

对于“需要跨墙换 homotopy”的目标切换，这个探索尺度偏小。

### 5. `n_iters: 1`
每个控制周期只更新一次采样分布，重新组织到另一条绕障模式的能力很弱。

### 6. `step_size_mean: 0.98`
均值更新太激进，但前提是它只能看见当前这一个 basin 的样本。

结果就是：
- 它不是往正确新 basin 大跳
- 而是快速收缩到当前错误 basin 的均值附近

## 最终判断
### 不是环境几何算错
这次调试用的是 `sim_gazebo` 的 primitive world，环境距离来自 STORM 自己的解析几何，不是 ESDF snapshot。

所以这次卡住不能归因于 ESDF 精度问题。

### 真正原因
真正原因是：
- **目标切换瞬间**先有一个 `ControlProcess` 重启/时域衔接 bug，造成短暂停滞
- 随后在高墙这种**多模态绕障**问题里，MPPI 因为
  - `hotstart=True`
  - `sample_mode=mean`
  - `update_cov=False`
  - `init_cov` 固定且偏小
  - `n_iters=1`
  进入了一个**窄而稳定的局部 basin**，采样根本没有覆盖到另一侧可行路径

所以机械臂“看起来不动”，本质上是：
- 控制器仍在工作
- 但它每轮都只在错误 basin 周围做小扰动
- 无法跳到穿墙另一侧的那条正确绕障路径上

## 如果下一步要修
最直接有效的改法优先级：
1. 目标切换时不要重启 `ControlProcess`，而是只重置时间基准或直接前台同步求解
2. 目标切换时显式 `reset_mean()` 和 `reset_covariance()`，不要沿用旧目标的 warm start
3. 增大 `init_cov`
4. 把 `update_cov` 打开，或者在停滞检测触发时临时放大协方差
5. 增加 `n_iters`
6. 不要执行 `mean`，改成执行 best sample / top-k 中最优样本轨迹
