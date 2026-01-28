# UR7e STORM MPC - Gazebo 仿真验证

本目录包含使用 **STORM MPPI** 算法在 **Gazebo Ignition** 中控制 **UR7e** 机械臂的完整代码。

---

## 📁 目录结构

```
sim_gazebo/
├── README.md                    # 本文档
├── run_gazebo.sh               # 启动 Gazebo 仿真
├── test_mpc.sh                 # 测试 MPC 控制器 (关节目标)
├── run_reach_static.sh         # 启动静态目标到达任务 (自带 RViz)
├── run_rviz.sh                 # 单独启动 RViz 可视化
├── monitor_topics.sh           # 监控 ROS2 话题
├── test_joint_control.py       # 关节控制测试
├── ur7e_mpc_gazebo.py          # 基础关节位置控制节点
├── reach_static_ur7e.py        # 静态目标到达任务 (笛卡尔空间)
└── config/                     # 配置文件目录
    ├── ur7e_robot_gazebo.yml       # 机器人基座位姿配置
    ├── ur7e_reacher_gazebo.yml     # MPC 控制器参数
    ├── ur7e_collision_spheres.yml  # 机器人球包裹描述
    ├── collision_world_gazebo.yml  # 场景障碍物配置
    ├── ur7e_gazebo.yml             # 简化配置 (旧版)
    └── reach_static.rviz           # RViz 可视化配置
```

---

## 📄 配置文件说明

### config/ 目录下的配置文件

| 配置文件 | 原始文件 | 说明 |
|----------|----------|------|
| `ur7e_robot_gazebo.yml` | `content/configs/gym/ur7e_isaacsim.yml` | 机器人基座位置与姿态，初始关节角度，Gazebo 控制话题 |
| `ur7e_reacher_gazebo.yml` | `content/configs/mpc/ur7e_reacher_isaacsim.yml` | MPPI 算法参数，成本函数权重，控制周期，URDF 路径 |
| `ur7e_collision_spheres.yml` | `content/configs/robot/ur7e.yml` | 机器人各连杆的碰撞球定义，用于自碰撞检测 |
| `collision_world_gazebo.yml` | `content/configs/gym/collision_primitives_3d.yml` | 场景中的障碍物（球体、立方体）位置和尺寸 |
| `reach_static.rviz` | - | RViz 可视化配置，显示机器人、目标、末端和障碍物 |

### 配置文件引用关系

```
reach_static_ur7e.py
    │
    ├── ur7e_robot_gazebo.yml        ← 机器人基座位姿
    │
    ├── ur7e_reacher_gazebo.yml      ← MPC 控制器参数
    │       │
    │       └── ur7e_collision_spheres.yml  ← 碰撞球定义
    │
    └── collision_world_gazebo.yml   ← 障碍物定义
```

### 配置文件详解

#### 1. `ur7e_robot_gazebo.yml` - 机器人配置
```yaml
sim_params:
  robot_pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # [x, y, z, qx, qy, qz, qw]
  init_state: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]  # 初始关节角度
  control_mode: 'position'
  gazebo:
    joint_state_topic: '/joint_states'
    position_cmd_topic: '/forward_position_controller/commands'
```

#### 2. `ur7e_reacher_gazebo.yml` - MPC 参数
```yaml
control_dt: 0.02          # 控制周期 (50Hz)
model:
  max_acc: 5.0            # 最大加速度
  ee_link_name: "tool0"   # 末端连杆名称
cost:
  primitive_collision:
    weight: 5000.0        # 障碍物碰撞权重
  robot_self_collision:
    weight: 5000.0        # 自碰撞权重
mppi:
  horizon: 30             # 预测步数
  num_particles: 500      # 采样粒子数
```

#### 3. `collision_world_gazebo.yml` - 障碍物配置
```yaml
world_model:
  coll_objs:
    sphere:
      sphere1:
        radius: 0.1
        position: [0.4, 0.4, 0.3]
    cube:
      cube1:
        dims: [0.3, 0.1, 0.4]
        pose: [0.4, 0.25, 0.2, 0, 0, 0, 1.0]
```

---

## 🚀 快速开始

### 前置条件

1. **ROS2 Humble** 已安装
2. **Gazebo Ignition** 已安装
3. **STORM** 环境已配置 (`conda activate storm_py310`)
4. **UR Gazebo 仿真包** 已编译 (`~/ur_arm/gazebo_ur_sim`)

### 步骤 1: 启动 Gazebo 仿真

```bash
# 终端 1
cd ~/storm/examples/sim_gazebo
./run_gazebo.sh
```

或者手动启动:

```bash
source /opt/ros/humble/setup.bash
source ~/ur_arm/ros_ur_driver/install/setup.bash
source ~/ur_arm/gazebo_ur_sim/install/setup.bash

ros2 launch ur_simulation_gazebo ur_sim_control.launch.py \
    ur_type:=ur7e \
    initial_joint_controller:=forward_position_controller
```

### 步骤 2: 验证控制器（可选）

```bash
# 终端 2
source /opt/ros/humble/setup.bash

# 检查控制器状态
ros2 control list_controllers
# 应该看到: forward_position_controller [active]

# 测试发送指令
ros2 topic pub --once /forward_position_controller/commands \
    std_msgs/Float64MultiArray \
    "{data: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]}"
```

### 步骤 3: 启动 MPC 控制器

**方式 A: 关节目标控制 (Joint-Space)**
```bash
# 终端 2 (激活 STORM 环境)
conda activate storm_py310
cd ~/storm/examples/sim_gazebo
./test_mpc.sh
```

**方式 B: 静态目标到达任务 (Task-Space)** 🆕
```bash
# 终端 2 (激活 STORM 环境)
conda activate storm_py310
cd ~/storm/examples/sim_gazebo
./run_reach_static.sh
```

或者手动运行:

```bash
conda activate storm_py310
source /opt/ros/humble/setup.bash
python3 ur7e_mpc_gazebo.py --cuda --rate 50
```

---

## 🎛️ 命令行参数

### ur7e_mpc_gazebo.py (关节空间控制)

```bash
python3 ur7e_mpc_gazebo.py [OPTIONS]

选项:
  --cuda          使用 GPU 加速 (默认: True)
  --no-cuda       禁用 GPU
  --rate RATE     控制频率 Hz (默认: 50)
  --goal Q1..Q6   目标关节角度 (6个值，单位: rad)

示例:
  # 默认运行
  python3 ur7e_mpc_gazebo.py
  
  # CPU 模式，100Hz
  python3 ur7e_mpc_gazebo.py --no-cuda --rate 100
  
  # 指定目标位置
  python3 ur7e_mpc_gazebo.py --goal 0.5 -1.2 1.2 -1.57 -1.57 0.0
```

### reach_static_ur7e.py (笛卡尔空间目标到达) 🆕

```bash
python3 reach_static_ur7e.py [OPTIONS]

选项:
  --cuda          使用 GPU 加速 (默认: True)
  --no-cuda       禁用 GPU
  --rate RATE     控制频率 Hz (默认: 50)
  --target X Y Z  目标位置 (米，相对于机器人基座)

示例:
  # 默认目标位置
  python3 reach_static_ur7e.py
  
  # 指定目标位置 (x=0.5, y=0.3, z=0.4)
  python3 reach_static_ur7e.py --target 0.5 0.3 0.4
  
  # 运行时动态修改目标
  ros2 topic pub /target_pose geometry_msgs/PoseStamped \
    "{header: {frame_id: 'base_link'}, pose: {position: {x: 0.6, y: 0.2, z: 0.5}, orientation: {w: 1.0}}}" --once
```

#### 可视化 (需要 RViz)
```bash
# 启动 RViz 并添加以下话题:
ros2 run rviz2 rviz2

# 添加 Marker 显示:
#   - /target_marker (红球 - 目标位置)
#   - /ee_marker (绿球 - 末端执行器位置)
#   - /obstacle_markers (蓝色 - 障碍物)
```

---

## 🎯 两种控制模式对比

| 特性 | `ur7e_mpc_gazebo.py` | `reach_static_ur7e.py` |
|------|----------------------|------------------------|
| **控制空间** | 关节空间 (Joint-Space) | 笛卡尔空间 (Task-Space) |
| **目标类型** | 关节角度 (6 个 rad 值) | 末端位置 (x, y, z) |
| **成本函数** | 关节位置跟踪 | 位姿跟踪 + 碰撞避障 |
| **动态目标** | ❌ | ✅ 通过 `/target_pose` |
| **障碍物** | ❌ | ✅ 球形障碍物 |
| **可视化** | 基础日志 | RViz 标记 |
| **适用场景** | 简单测试 | 完整任务验证 |

---

## 🔧 技术细节

### 控制器选择

| 控制器 | 话题 | 消息类型 | 延迟 | 适用场景 |
|--------|------|----------|------|----------|
| `JointTrajectoryController` | `/joint_trajectory_controller/joint_trajectory` | `JointTrajectory` | 高 | 离线轨迹执行 |
| **`ForwardPositionController`** | `/forward_position_controller/commands` | `Float64MultiArray` | **低** | **实时 MPC** ✅ |

### 接口说明

- **输入**: `/joint_states` (sensor_msgs/JointState)
  - 关节位置、速度
  - 频率: 100 Hz

- **输出**: `/forward_position_controller/commands` (std_msgs/Float64MultiArray)
  - 6 个关节位置指令 (rad)
  - 格式: `[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]`

### 关节顺序

```
0: shoulder_pan_joint   (底座旋转)
1: shoulder_lift_joint  (肩关节俯仰)
2: elbow_joint          (肘关节)
3: wrist_1_joint        (腕关节 1)
4: wrist_2_joint        (腕关节 2)
5: wrist_3_joint        (腕关节 3)
```

---

## 📊 调试工具

### 监控话题

```bash
./monitor_topics.sh
```

### 关节控制测试

```bash
source /opt/ros/humble/setup.bash
python3 test_joint_control.py
```

### 实时绘图

```bash
# 查看关节状态
ros2 topic echo /joint_states

# 查看发送的指令
ros2 topic echo /forward_position_controller/commands

# 话题频率
ros2 topic hz /joint_states
ros2 topic hz /forward_position_controller/commands
```

---

## ⚠️ 常见问题

### 1. 无法接收关节状态

**症状**: `等待 Gazebo 关节状态...` 卡住

**解决方案**:
```bash
# 确认 Gazebo 正在运行
ros2 topic list | grep joint_states

# 确认话题有数据
ros2 topic echo /joint_states --once
```

### 2. 控制器未激活

**症状**: 发送指令无响应

**解决方案**:
```bash
# 检查控制器状态
ros2 control list_controllers

# 手动切换控制器
ros2 control switch_controllers \
    --deactivate joint_trajectory_controller \
    --activate forward_position_controller
```

### 3. CUDA 内存不足

**症状**: `CUDA out of memory`

**解决方案**:
```bash
# 使用 CPU 模式
python3 ur7e_mpc_gazebo.py --no-cuda
```

### 4. MPC 计算太慢

**症状**: 控制频率达不到设定值

**解决方案**:
- 降低控制频率: `--rate 20`
- 使用 CUDA: `--cuda`
- 减少 MPC horizon/samples (修改配置文件)

---

## 📝 配置文件说明

### STORM 配置

- `content/configs/gym/ur7e_isaacsim.yml` - 机器人参数
- `content/configs/mpc/ur7e_reacher_isaacsim.yml` - MPC 参数
- `content/configs/gym/collision_primitives_3d.yml` - 障碍物定义

### Gazebo 专用配置

- `config/ur7e_gazebo.yml` - Gazebo 特定参数

---

## 🔗 相关链接

- [STORM 项目](https://github.com/emmmwqj/ur_based_MPPI)
- [UR ROS2 Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)
- [UR Gazebo Simulation](https://github.com/UniversalRobots/Universal_Robots_ROS2_Gazebo_Simulation)

---

## 📄 License

MIT License
