# sim_gazebo Launchers

## 当前目录职责

`examples/sim_gazebo` 现在只保留两类项目：

- STORM baseline 控制器项目
- diffusion 控制器项目

SAGE、clean_SAGE 以及和 SAGE 强耦合的 benchmark/统计脚本已经整理到：

- [examples/SAGE_MPPI](/home/wqj/storm/examples/SAGE_MPPI)

`sim_gazebo` 继续保留这些公共依赖，因为它们是所有 Gazebo 控制项目共享的基础设施：

- `config/`
  - Gazebo 机器人配置、世界配置、RViz 配置
- [reach_static_ur7e.py](/home/wqj/storm/examples/sim_gazebo/reach_static_ur7e.py)
  - 公共 Gazebo 接口与基础可视化逻辑
- [reach_static_ur7e_tall.py](/home/wqj/storm/examples/sim_gazebo/reach_static_ur7e_tall.py)
  - baseline 高墙场景主脚本
- [gazebo_obstacle_utils.py](/home/wqj/storm/examples/sim_gazebo/gazebo_obstacle_utils.py)
  - Gazebo 障碍物 spawn/delete 工具

## 启动文件

### Gazebo 基础启动

- [run_gazebo.sh](/home/wqj/storm/examples/sim_gazebo/run_gazebo.sh)
  - 只启动 UR7e Gazebo + ros2_control + `forward_position_controller`
  - 默认不打开 Gazebo 侧 RViz

- [run_rviz.sh](/home/wqj/storm/examples/sim_gazebo/run_rviz.sh)
  - 单独打开 RViz，读取共享的 `config/reach_static.rviz`

### STORM baseline 项目

- [run_reach_static.sh](/home/wqj/storm/examples/sim_gazebo/run_reach_static.sh)
  - baseline 普通场景入口
  - 需要先手动启动 Gazebo

- [run_reach_static_tall.sh](/home/wqj/storm/examples/sim_gazebo/run_reach_static_tall.sh)
  - baseline 高墙场景入口
  - 需要先手动启动 Gazebo

- [bash/run_all_reach_static_tall.sh](/home/wqj/storm/examples/sim_gazebo/bash/run_all_reach_static_tall.sh)
  - baseline 高墙场景一键启动
  - 会先启动 Gazebo，再启动 baseline STORM 控制器

- [test_mpc.sh](/home/wqj/storm/examples/sim_gazebo/test_mpc.sh)
  - baseline 非交互测试入口

- [test_mpc_tall.sh](/home/wqj/storm/examples/sim_gazebo/test_mpc_tall.sh)
  - baseline 高墙场景非交互测试入口

### diffusion 项目

- [run_reach_static_tall_diffusion.sh](/home/wqj/storm/examples/sim_gazebo/run_reach_static_tall_diffusion.sh)
  - diffusion 高墙场景入口
  - 需要先手动启动 Gazebo

## debug

`debug/` 目录只服务于 baseline 高墙项目调试：

- [debug/run_reach_static_tall_debug.sh](/home/wqj/storm/examples/sim_gazebo/debug/run_reach_static_tall_debug.sh)
  - 推荐的 debug 入口
  - 会保存日志和 stall capture

- [debug/run_reach_static_tall_debug_main.sh](/home/wqj/storm/examples/sim_gazebo/debug/run_reach_static_tall_debug_main.sh)
  - 更接近主 launcher 风格的 debug 入口

更详细的 debug 说明见：

- [debug/README.md](/home/wqj/storm/examples/sim_gazebo/debug/README.md)
