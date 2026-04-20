# SAGE_MPPI Launchers

## 项目结构

`examples/SAGE_MPPI` 现在按“控制器项目”和“依赖角色”拆成三块：

- `./`
  - 传统 SAGE 高墙 Gazebo 项目主入口
- `clean_SAGE/`
  - clean SAGE core 高墙 Gazebo 项目主入口
- `benchmark/`
  - baseline / diffusion / SAGE 对照实验与统计分析脚本
- `support/`
  - 单独的 SAGE 支撑脚本，不属于高墙主入口

共享依赖仍然放在 `examples/sim_gazebo`：

- Gazebo 场景与机器人公共配置
- RViz 配置
- Gazebo 障碍物 spawn 工具
- 通用 Gazebo 接口与可视化节点

这样拆分的原因是：

- SAGE/clean_SAGE 的控制器项目归到 `SAGE_MPPI`
- Gazebo 基础设施继续由 `sim_gazebo` 提供
- benchmark/统计脚本和 SAGE 对照实验强耦合，不再放在 `sim_gazebo`

## 启动文件

### 1. 传统 SAGE 高墙项目

- [run_reach_static_tall.sh](/home/wqj/storm/examples/SAGE_MPPI/run_reach_static_tall.sh)
  - 只启动传统 SAGE 控制器
  - 需要你先单独启动 Gazebo

- [run_all_reach_static_tall.sh](/home/wqj/storm/examples/SAGE_MPPI/run_all_reach_static_tall.sh)
  - 一键启动 Gazebo + 传统 SAGE 控制器
  - 默认 headless Gazebo，只开控制器侧 RViz

### 2. clean SAGE 高墙项目

- [clean_SAGE/run_reach_static_tall.sh](/home/wqj/storm/examples/SAGE_MPPI/clean_SAGE/run_reach_static_tall.sh)
  - 只启动 clean SAGE 控制器
  - 需要你先单独启动 Gazebo

- [clean_SAGE/run_all_reach_static_tall.sh](/home/wqj/storm/examples/SAGE_MPPI/clean_SAGE/run_all_reach_static_tall.sh)
  - 一键启动 Gazebo + clean SAGE 控制器
  - 默认 headless Gazebo，只开控制器侧 RViz

### 3. 支撑脚本

- [support/reach_static_ur7e_sage.py](/home/wqj/storm/examples/SAGE_MPPI/support/reach_static_ur7e_sage.py)
  - 独立 SAGE 入口
  - 更偏向单独实验/记录，不是高墙项目的主 launcher

## benchmark 脚本

`benchmark/` 下的脚本不负责 Gazebo 启动，它们属于论文/对照实验工具链：

- [benchmark/run_controller_batch.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/run_controller_batch.py)
  - baseline / diffusion / SAGE 批量运行入口
- [benchmark/generate_round2_targets.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/generate_round2_targets.py)
  - round2 目标集生成
- [benchmark/generate_round3_pairs.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/generate_round3_pairs.py)
  - round3 配对状态生成
- [benchmark/generate_round4_pairs.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/generate_round4_pairs.py)
  - round4 harder 配对状态生成
- [benchmark/analyze_round3_statistics.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/analyze_round3_statistics.py)
  - round3 统计分析
- [benchmark/analyze_round4_paired_statistics.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/analyze_round4_paired_statistics.py)
  - round4 paired 统计分析
- [benchmark/make_round3_tables_and_plots.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/make_round3_tables_and_plots.py)
  - round3 表格和图
- [benchmark/make_round4_tables_and_plots.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/make_round4_tables_and_plots.py)
  - round4 表格和图
- [benchmark/finalize_round4_paper_outputs.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/finalize_round4_paper_outputs.py)
  - round4 论文输出整理
- [benchmark/run_baseline_vs_sage_round1.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/run_baseline_vs_sage_round1.py)
  - round1 baseline vs SAGE 批量运行
- [benchmark/run_baseline_vs_sage_round2.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/run_baseline_vs_sage_round2.py)
  - round2 baseline vs SAGE 批量运行
- [benchmark/run_baseline_vs_sage_round3.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/run_baseline_vs_sage_round3.py)
  - round3 baseline vs SAGE 批量运行
- [benchmark/run_baseline_vs_sage_round4.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/run_baseline_vs_sage_round4.py)
  - round4 baseline vs SAGE harder benchmark 运行
- [benchmark/run_round4_gazebo_recheck.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/run_round4_gazebo_recheck.py)
  - round4 Gazebo 真实复核运行
- [benchmark/summarize_experiments.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/summarize_experiments.py)
  - round1/通用汇总
- [benchmark/summarize_round2_benchmark.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/summarize_round2_benchmark.py)
  - round2 汇总
- [benchmark/summarize_round3_benchmark.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/summarize_round3_benchmark.py)
  - round3 汇总
- [benchmark/summarize_round4_gazebo_recheck.py](/home/wqj/storm/examples/SAGE_MPPI/benchmark/summarize_round4_gazebo_recheck.py)
  - round4 Gazebo 复核汇总
