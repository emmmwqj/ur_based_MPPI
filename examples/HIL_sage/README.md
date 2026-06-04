# UR7e SAGE-MPPI HIL

本目录是 `examples/HIL` 的 SAGE-MPPI 版本，用于将 `examples/SAGE_MPPI/clean_SAGE` 中的 clean SAGE 控制链迁移到真实 UR7e 硬件在环模式。

关键区别：
- 控制器：`SAGE_MPPI`
- Task：`SageReacherTask`
- Rollout：clean SAGE rollout path
- 真实机器人接口：沿用 HIL 的 `/joint_states` 和 `/forward_position_controller/commands`
- 障碍物：使用 Gazebo tall 高墙场景的 HIL 镜像版本，只在 RViz 中可视化，但 SAGE-MPPI 会按配置避障
- RViz：显示目标、末端和虚拟障碍物

## 文件结构

```text
HIL_sage/
├── ur7e_hil_sage_mpc.py          # HIL SAGE-MPPI 主控制入口
├── run_hil_sage_mpc.sh           # 控制器启动脚本
├── run_ur_driver.sh              # UR ROS2 Driver 启动脚本
├── run_rviz.sh                   # RViz 启动脚本
├── test_connection.py            # 连接测试
└── config/
    ├── ur7e_reacher_hil_sage.yml # SAGE-MPPI 控制参数
    ├── ur7e_robot_hil_sage.yml   # 真实机器人和安全参数
    ├── collision_world_hil.yml   # tall 高墙虚拟障碍物
    ├── ur7e_collision_spheres.yml
    └── hil_rviz.rviz
```

## 快速启动

终端 1，启动 UR ROS2 Driver：

```bash
cd ~/storm/examples/HIL_sage
./run_ur_driver.sh
```

终端 2，测试连接：

```bash
cd ~/storm/examples/HIL_sage
python3 test_connection.py
```

终端 2，启动 SAGE-MPPI HIL 控制器：

```bash
cd ~/storm/examples/HIL_sage
./run_hil_sage_mpc.sh --safe-mode --rate 20
```

默认 conda 环境是 `whole_control`。如果你的 HIL 依赖在 `storm_py310` 中：

```bash
STORM_CONDA_ENV=storm_py310 ./run_hil_sage_mpc.sh --safe-mode --rate 20
```

不自动启动 RViz：

```bash
./run_hil_sage_mpc.sh --no-rviz --safe-mode
```

## 动态目标

运行时通过 `/target_pose` 发布新目标。控制器只读取 position，orientation 不参与目标更新。

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub /target_pose geometry_msgs/PoseStamped \
  "{header: {frame_id: 'base_link'}, pose: {position: {x: 0.5, y: -0.2, z: 0.45}, orientation: {w: 1.0}}}" -1
```

## 主要参数

SAGE 参数在 `config/ur7e_reacher_hil_sage.yml`：
- `sage_controller_core`: SAGE proposal shape/stage scaling/stagnation 参数
- `sage_deployment_refinement`: 目标跳变 boost、near-goal、local refinement、stall recovery
- `mppi.execution_mode`: `best_sample` 或 `mean`
- 控制器参数与 `examples/SAGE_MPPI/clean_SAGE/config/ur7e_reacher_gazebo_tall_sage_clean.yml` 对齐；HIL 专属路径和默认目标保留在本目录配置中
- `default_goal`: 启动默认目标，机器人基座坐标系

HIL 安全参数在 `config/ur7e_robot_hil_sage.yml`：
- `sim_params.hil.safety.max_velocity`
- `sim_params.hil.safety.max_acceleration`

tall 高墙场景在 `config/collision_world_hil.yml`。它对应 `examples/sim_gazebo/config/collision_world_gazebo_tall.yml`，但 Y 轴坐标按 HIL 真实机器人坐标约定做了镜像。这些障碍物只用于 RViz 显示和 MPC 避障代价，真实工作台上不要放置对应实体。

## 安全说明

这会控制真实机械臂。首次运行建议：

```bash
./run_hil_sage_mpc.sh --safe-mode --rate 20
```

启动前确认真实工作区无人员和障碍物，急停按钮在可触及范围内，UR 示教器已运行 External Control。
