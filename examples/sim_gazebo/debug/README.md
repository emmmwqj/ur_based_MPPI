# sim_gazebo/debug

这个目录用于调试 `sim_gazebo` 高墙场景下的 STORM MPPI 控制器，重点分析两类问题：

- 目标跨墙切换后为什么会短暂停住
- 控制器为什么会落入 local minimum，导致末端远离目标但机械臂基本不动

## 目录结构

- [reach_static_ur7e_tall_debug.py](/home/wqj/storm/examples/sim_gazebo/debug/reach_static_ur7e_tall_debug.py)
  - debug 版主程序
  - 基于 `examples/sim_gazebo/reach_static_ur7e_tall.py`
  - 额外加入：
    - 停滞检测
    - 采样分布放大
    - 时间基准重置
    - 调试数据抓取与保存

- [run_reach_static_tall_debug.sh](/home/wqj/storm/examples/sim_gazebo/debug/run_reach_static_tall_debug.sh)
  - 推荐使用的 debug 启动脚本
  - 作用：
    - 启动 `reach_static_ur7e_tall_debug.py`
    - 可选启动 RViz
    - 把控制台输出保存到 `logs/`
    - 显示 `captures/` 的输出位置

- [run_reach_static_tall_debug_main.sh](/home/wqj/storm/examples/sim_gazebo/debug/run_reach_static_tall_debug_main.sh)
  - 结构更接近主目录下 `run_reach_static_tall.sh`
  - 功能更简单
  - 默认不做日志落盘
  - 主要用于保留一个“主脚本风格”的调试入口

- `captures/`
  - 自动保存的停滞现场数据
  - 每次触发保存一组 `json + npz`
  - 命名形式：`stall_capture_<timestamp>_<idx>.*`

- `logs/`
  - debug 启动脚本的运行日志
  - 用于回看目标切换、恢复触发、异常信息

- [local_minimum_analysis.md](/home/wqj/storm/examples/sim_gazebo/debug/local_minimum_analysis.md)
  - 第一轮调试分析结论
  - 说明：
    - 当时为什么卡住
    - 为什么判断是 local minimum
    - 当时的控制流程 bug 是什么

- [recovery_validation.md](/home/wqj/storm/examples/sim_gazebo/debug/recovery_validation.md)
  - 第二轮验证报告
  - 说明：
    - “放大采样分布 + 重置时间基准”这套恢复策略是否有效
    - 实测跨墙切换后的误差下降情况

## 推荐运行方式

终端 1：

```bash
cd ~/storm/examples/sim_gazebo
./run_gazebo.sh
```

终端 2：

```bash
cd ~/storm/examples/sim_gazebo/debug
./run_reach_static_tall_debug.sh
```

如果不需要 RViz：

```bash
./run_reach_static_tall_debug.sh --no-rviz
```

## captures 里保存了什么

每次检测到“末端离目标较远且基本不动”时，会保存一份 `npz + json`。

主要字段包括：

- 当前机器人状态：
  - `q`
  - `dq`
  - `ddq`
- 当前目标与末端：
  - `goal_world`
  - `ee_pos_world`
  - `ee_error`
- 当前控制器分布参数：
  - `cov_action`
  - `scale_tril`
  - `mean_action`
  - `best_traj`
- 本轮采样数据：
  - `sample_actions`
  - `sample_cost_seq`
  - `sample_ee_pos_seq`
  - `sample_ee_world_seq`
  - `total_costs`

这些数据用于回答两个问题：

- 当前是不是已经掉进了一个局部 basin
- 采样分布到底有没有覆盖到目标附近的可行轨迹

## “放大采样分布 + 重置时间基准”是什么意思

这是 debug 版里用于跳出局部极小的一套恢复策略。

### 1. 放大采样分布

MPPI 每轮都会围绕当前均值轨迹采样控制序列。

如果当前分布太窄：

- 所有样本都只在当前坏 basin 附近小幅扰动
- 根本采不到“绕到墙另一侧”的轨迹

debug 版里的做法是：

- 目标切换时，把 `init_cov` 放大到原来的 `9x`
- 检测到停滞时，再放大到原来的 `16x`
- 同时把 `step_size_mean` 调小一点，避免均值更新过猛但探索不足

这样做的效果是：

- 扩大每轮采样半径
- 让采样覆盖到新的 homotopy / 新的可行绕障路径
- 不再只围绕旧目标的 warm-start 轨迹小范围抖动

### 2. 重置时间基准

`ControlProcess` 内部会记录：

- 当前命令的时间轴
- 上一次 MPC 的时间步
- 当前控制周期对应的命令索引

目标切换后，如果还沿用旧的时间基准，就可能出现：

- 旧轨迹和新目标不一致
- 时域索引越界
- `index 0 is out of bounds ...`
- `MPC command horizon exhausted ...`

debug 版里的做法是：

- 不重启整个 `ControlProcess`
- 只把 `command / command_tstep / prev_mpc_tstep / mpc_dt` 重置到当前时刻

这样做的效果是：

- 旧的时域缓存被清掉
- 新目标从当前时刻重新同步规划
- 避免目标切换瞬间的控制流程异常

## 这套恢复策略为什么有效

它本质上同时修两个问题：

### 控制流程问题

重置时间基准解决的是：

- 目标切换瞬间的时域衔接错误

### 优化问题

放大采样分布解决的是：

- 采样只停留在旧 basin
- 无法探索到新的绕障路径

所以这不是单纯“调大噪声”，而是：

- 先保证时间轴和当前目标一致
- 再保证采样足够大，能跳出旧局部极小

## debug 版里的恢复逻辑

### 目标切换时

会做：

1. 更新新的 `goal_ee_pos`
2. 放大采样分布
3. 重置时间基准
4. 立刻同步重规划一次

### 检测到停滞时

当满足以下条件时会触发恢复：

- `ee_error` 仍然较大
- 一段时间内末端几乎不动
- 当前关节速度很小

触发后会：

1. 保存一份调试数据到 `captures/`
2. 再次放大采样分布
3. 重置时间基准
4. 继续规划

### 接近目标后

会把分布恢复成默认值，避免一直保持“大协方差”导致末端附近抖动。

## 这个目录的用途边界

这个目录主要用于：

- 复现实验问题
- 保存现场数据
- 验证恢复策略

不建议直接把这里的所有逻辑无差别复制到主入口。

更合理的做法是：

1. 先在这里验证机制有效
2. 再把必要的最小修复回写到主脚本
