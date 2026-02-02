# HIL 配置文件说明

本目录包含 UR7e 硬件在环 (HIL) 控制所需的所有配置文件。

---

## 📁 配置文件列表

| 文件名 | 功能 | 说明 |
|--------|------|------|
| `ur7e_robot_hil.yml` | 机器人基座配置 | 机器人位姿、初始关节角度、ROS2 话题、安全限制 |
| `ur7e_reacher_hil.yml` | MPC 控制器参数 | MPPI 算法参数、成本函数权重、默认目标位置 |
| `ur7e_collision_spheres.yml` | 碰撞球定义 | 机器人各连杆的包裹球，用于自碰撞检测 |
| `collision_world_hil.yml` | 虚拟障碍物配置 | 场景中的虚拟障碍物（球体、立方体）位置和尺寸 |
| `hil_rviz.rviz` | RViz 可视化配置 | RViz 显示设置，包括机器人模型、标记、TF 等 |

---

## 📄 配置文件详解

### 1. `ur7e_robot_hil.yml` - 机器人基座配置

定义真实机器人的基本参数和 ROS2 接口。

```yaml
sim_params:
  robot_pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # [x, y, z, qx, qy, qz, qw]
  init_state: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]  # 初始关节角度 (rad)
  hil:
    robot_ip: "192.168.56.100"                      # 机器人 IP 地址
    joint_state_topic: "/joint_states"              # 关节状态话题
    position_cmd_topic: "/forward_position_controller/commands"  # 适合高频 MPC
    safety:
      max_velocity: 0.5       # 最大速度 (rad/s)
      max_acceleration: 1.0   # 最大加速度 (rad/s^2)
```

> ⚠️ **重要**: 启动 UR 驱动时必须添加 `initial_joint_controller:=forward_position_controller` 参数，
> 否则默认会激活 `scaled_joint_trajectory_controller`，无法用于高频 MPC 控制。

### 2. `ur7e_reacher_hil.yml` - MPC 控制器参数

STORM MPPI 算法的核心配置，包括成本函数和默认目标。

```yaml
control_dt: 0.02              # 控制周期 (50Hz)

model:
  max_acc: 5.0                # 最大加速度
  ee_link_name: "tool0"       # 末端连杆名称

cost:
  goal_pose:                  # 目标位姿跟踪
    weight: [15.0, 100.0]     # [位置权重, 姿态权重]
  primitive_collision:        # 障碍物碰撞
    weight: 5000.0
  robot_self_collision:       # 自碰撞
    weight: 5000.0

default_goal:                 # 默认目标位置 (笛卡尔空间，Y轴已镜像)
  position: [0.503, -0.427, 0.459]          # [x, y, z] 米 (Y轴镜像后)
  orientation: [0.0, 0.707, 0.0, 0.707]     # [qx, qy, qz, qw] 四元数

mppi:
  horizon: 30                 # 预测步数
  num_particles: 500          # 采样粒子数
```

### 3. `ur7e_collision_spheres.yml` - 碰撞球定义

定义机器人各连杆的包裹球，用于自碰撞和障碍物碰撞检测。

```yaml
robot_collision_params:
  urdf: "urdf/ur7e/ur7e.urdf"
  collision_spheres:
    - link_name: "shoulder_link"
      radius: 0.08
      center: [0.0, 0.0, 0.0]
    - link_name: "upper_arm_link"
      radius: 0.06
      center: [0.0, 0.0, 0.2]
    # ... 更多连杆
```

### 4. `collision_world_hil.yml` - 虚拟障碍物配置

定义虚拟障碍物，仅在 RViz 中可视化，MPC 会避开这些障碍物。

⚠️ **注意**: 这些障碍物是虚拟的，真实工作台上没有！

```yaml
world_model:
  coll_objs:
    sphere:
      sphere1:
        radius: 0.1
        position: [0.4, -0.4, 0.1]  # Y轴镜像后
    cube:
      cube1:
        dims: [0.3, 0.1, 0.4]
        pose: [0.4, -0.2, 0.2, 0, 0, 0, 1.0]  # Y轴镜像后
      cube2:
        dims: [0.3, 0.1, 0.5]
        pose: [0.4, 0.3, 0.2, 0, 0, 0, 1.0]  # Y轴镜像后
      ground:
        dims: [2.0, 2.0, 0.2]
        pose: [0.0, 0.0, -0.1, 0, 0, 0, 1.0]
```

### 5. `hil_rviz.rviz` - RViz 可视化配置

RViz 的预设配置文件，包括：
- 机器人模型显示
- TF 坐标系
- 目标/末端/障碍物标记 (`/visualization_marker_array`)
- 末端位姿 (`/ee_pose`)

---

## 🔗 配置文件引用关系

```
ur7e_hil_mpc.py
    │
    ├── ur7e_robot_hil.yml        ← 机器人基座位姿、安全参数
    │
    ├── ur7e_reacher_hil.yml      ← MPC 控制器参数、默认目标
    │       │
    │       └── ur7e_collision_spheres.yml  ← 碰撞球定义
    │
    └── collision_world_hil.yml   ← 虚拟障碍物定义
```

---

## ⚙️ 与 Gazebo 仿真的对应关系

| HIL 配置 | Gazebo 配置 | 说明 |
|----------|-------------|------|
| `ur7e_robot_hil.yml` | `ur7e_robot_gazebo.yml` | 机器人基座位姿 |
| `ur7e_reacher_hil.yml` | `ur7e_reacher_gazebo.yml` | MPC 参数 |
| `collision_world_hil.yml` | `collision_world_gazebo.yml` | 障碍物配置 |
| `ur7e_collision_spheres.yml` | `ur7e_collision_spheres.yml` | 碰撞球 (共用) |

**重要**: HIL 和 Gazebo 的障碍物配置和目标位置已同步，确保两个环境行为一致。

---

## 📝 修改建议

1. **调整目标位置**: 修改 `ur7e_reacher_hil.yml` 中的 `default_goal`
2. **添加障碍物**: 修改 `collision_world_hil.yml`
3. **调整安全限制**: 修改 `ur7e_robot_hil.yml` 中的 `safety` 参数
4. **调整 MPC 行为**: 修改 `ur7e_reacher_hil.yml` 中的 `cost` 权重

---

## 🎯 如何修改目标位置

### 方法 1: 修改配置文件（静态目标）

编辑 `ur7e_reacher_hil.yml` 文件，找到 `default_goal` 部分：

```yaml
# 文件: ur7e_reacher_hil.yml (约第 154 行)
default_goal:
  position: [0.503, -0.427, 0.459]      # [x, y, z] 米 - 修改这里 (Y轴镜像后)
  orientation: [0.0, 0.707, 0.0, 0.707]  # [qx, qy, qz, qw] 四元数
```

**坐标系说明**:
- `x`: 机器人前方为正
- `y`: 机器人左侧为正
- `z`: 向上为正
- 原点在机器人基座

**示例**: 将目标改为正前方 0.5m，高度 0.3m
```yaml
default_goal:
  position: [0.5, 0.0, 0.3]
  orientation: [0.0, 0.707, 0.0, 0.707]
```

### 方法 2: ROS2 话题动态更新（运行时）

程序运行后，可通过 `/target_pose` 话题动态更新目标位置：

```bash
# 发布新目标位置 (x=0.4, y=0.2, z=0.5)
ros2 topic pub /target_pose geometry_msgs/PoseStamped \
  "{header: {frame_id: 'base_link'}, pose: {position: {x: 0.4, y: 0.2, z: 0.5}, orientation: {w: 1.0}}}" --once
```

---

## ⚠️ 安全提示

- `collision_world_hil.yml` 中的障碍物是**虚拟**的，真实环境中不存在
- 使用前务必确认真实工作区域安全
- 建议先使用 `--safe-mode` 参数降低速度
