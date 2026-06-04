# HIL_sage 配置说明

本目录是 UR7e SAGE-MPPI 硬件在环控制的配置集合。

| 文件 | 作用 |
| --- | --- |
| `ur7e_robot_hil_sage.yml` | 真实 UR7e 位姿、ROS2 话题、关节名、安全限速 |
| `ur7e_reacher_hil_sage.yml` | SAGE-MPPI、代价函数、默认目标、deployment refinement |
| `collision_world_hil.yml` | RViz 中显示并由控制器避让的 tall 高墙虚拟障碍物 |
| `ur7e_collision_spheres.yml` | 机器人碰撞球 |
| `hil_rviz.rviz` | RViz 显示配置 |

## SAGE 控制配置

`ur7e_reacher_hil_sage.yml` 的控制器参数与下面的仿真 tall 配置对齐：

```text
examples/SAGE_MPPI/clean_SAGE/config/ur7e_reacher_gazebo_tall_sage_clean.yml
```

保留的 HIL 差异包括本目录的碰撞球路径和 `default_goal`。

## Tall 高墙场景

`collision_world_hil.yml` 使用 Gazebo tall 高墙场景：

- 参考源：`examples/sim_gazebo/config/collision_world_gazebo_tall.yml`
- HIL 坐标：按真实机器人坐标约定使用 Y 轴镜像后的障碍物位置
- 墙体：两面高墙尺寸均为 `[0.3, 0.1, 0.6]`

这些障碍物只在 RViz 中可视化，真实工作台上不要放置对应实体。

`ur7e_reacher_hil_sage.yml` 相比旧 HIL MPPI 配置新增：

- `task_metrics.success_threshold`
- `sage_controller_core`
- `sage_deployment_refinement`
- `mppi.execution_mode`

其中 `sage_controller_core` 只配置 controller core；`sage_deployment_refinement` 是运行时部署启发式，可用启动参数覆盖：

```bash
./run_hil_sage_mpc.sh --disable-deployment-refinement
./run_hil_sage_mpc.sh --enable-cartesian-refinement
./run_hil_sage_mpc.sh --disable-cartesian-refinement
```

## 默认目标

`default_goal.position` 使用机器人基座坐标系：

```yaml
default_goal:
  position: [0.503, -0.427, 0.459]
  orientation: [0.0, 0.707, 0.0, 0.707]
```

运行时可通过 `/target_pose` 更新目标，控制器只读取 position。

## 安全参数

真实机器人限速在 `ur7e_robot_hil_sage.yml`：

```yaml
sim_params:
  hil:
    safety:
      max_velocity: 0.5
      max_acceleration: 1.0
```

`--safe-mode` 会临时使用 `max_velocity=0.3` 和 `max_acceleration=0.5`。
