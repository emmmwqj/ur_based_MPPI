# UR7e STORM MPC - 硬件在环仿真 (HIL)

本目录包含 **硬件在环仿真 (Hardware-In-Loop)** 的完整代码：
- **真实机械臂**: UR7e (通过 ROS2 UR Driver 控制)
- **虚拟场景**: 障碍物在 RViz 中可视化 (真实工作台上无障碍物)

---

## ⚠️ 安全警告

**真实机械臂操作，请确保：**
1. 工作区域内无人员和障碍物
2. 急停按钮在可触及范围内
3. 首次运行时使用低速模式 (`--rate 20`)
4. 理解并遵守 UR 机器人安全规范

---

## 📁 目录结构

```
HIL/
├── README.md                      # 本文档
├── run_ur_driver.sh              # 启动 UR ROS2 驱动
├── run_hil_mpc.sh                # 启动 HIL MPC 控制器
├── run_rviz.sh                   # 启动 RViz 可视化
├── test_connection.py            # 测试机器人连接
├── ur7e_hil_mpc.py               # HIL MPC 主控制节点
└── config/
    ├── ur7e_robot_hil.yml        # 机器人配置 (真实机器人)
    ├── ur7e_reacher_hil.yml      # MPC 控制器参数
    ├── ur7e_collision_spheres.yml # 机器人碰撞球
    ├── collision_world_hil.yml   # 虚拟障碍物配置
    └── hil_rviz.rviz             # RViz 配置
```

---

## 🚀 快速开始

### 前置条件

1. **UR ROS2 Driver** 已安装 (`~/ur_arm/ros_ur_driver`)
2. **UR7e 机器人** 已连接 (IP: 192.168.131.38)
3. **STORM** 环境已配置 (`conda activate storm_py310`)
4. **机器人标定文件** 已生成 (`~/ur_arm/my_robot_calibration.yaml`)

### 步骤 1: 启动 UR ROS2 驱动

```bash
# 终端 1
cd ~/storm/examples/HIL
./run_ur_driver.sh
```

或手动启动:
```bash
source /opt/ros/humble/setup.bash
source ~/ur_arm/ros_ur_driver/install/setup.bash

ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur7e \
  robot_ip:=192.168.131.38 \
  kinematics_params_file:="${HOME}/ur_arm/my_robot_calibration.yaml" \
  launch_rviz:=false
```

### 步骤 2: 测试连接

```bash
# 终端 2
cd ~/storm/examples/HIL
python3 test_connection.py
```

### 步骤 3: 启动 HIL MPC 控制器

```bash
# 终端 2
cd ~/storm/examples/HIL
./run_hil_mpc.sh
```

### 步骤 4: 启动 RViz 可视化 (可选)

```bash
# 终端 3
cd ~/storm/examples/HIL
./run_rviz.sh
```

---

## 🎛️ 命令行参数

```bash
python3 ur7e_hil_mpc.py [OPTIONS]

选项:
  --cuda          使用 GPU 加速 (默认: True)
  --no-cuda       禁用 GPU
  --rate RATE     控制频率 Hz (默认: 50, 安全起见可用 20)
  --safe-mode     安全模式：降低速度和加速度限制

示例:
  # 安全模式运行 (推荐首次测试)
  python3 ur7e_hil_mpc.py --safe-mode --rate 20
  
  # 正常模式
  python3 ur7e_hil_mpc.py --cuda --rate 50
  
  # 动态修改目标位置
  ros2 topic pub /target_pose geometry_msgs/PoseStamped \
    "{header: {frame_id: 'base_link'}, pose: {position: {x: 0.5, y: 0.2, z: 0.4}, orientation: {w: 1.0}}}" --once
```

---

## 📡 ROS2 话题

### 订阅

| 话题 | 类型 | 说明 |
|------|------|------|
| `/joint_states` | `sensor_msgs/JointState` | 真实机器人关节状态 |
| `/target_pose` | `geometry_msgs/PoseStamped` | 动态目标位置 |

### 发布

| 话题 | 类型 | 说明 |
|------|------|------|
| `/scaled_joint_trajectory_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | 关节轨迹指令 |
| `/ee_pose` | `geometry_msgs/PoseStamped` | 末端位置 |
| `/visualization_marker_array` | `visualization_msgs/MarkerArray` | RViz 可视化 |

---

## 🔧 与 Gazebo 仿真的区别

| 项目 | Gazebo (sim_gazebo) | HIL (真实机器人) |
|------|---------------------|------------------|
| 机器人 | 仿真模型 | 真实 UR7e |
| 障碍物 | 仿真 + RViz | **仅 RViz 可视化** |
| 控制器 | forward_position_controller | **scaled_joint_trajectory_controller** |
| 话题 | `/forward_position_controller/commands` | `/scaled_joint_trajectory_controller/joint_trajectory` |
| 安全性 | 无限制 | **速度/加速度限制** |

---

## ⚙️ 配置文件说明

### `ur7e_robot_hil.yml` - 机器人配置

```yaml
sim_params:
  robot_pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
  init_state: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
  
  # HIL 专用参数
  hil:
    robot_ip: "192.168.131.38"
    joint_state_topic: "/joint_states"
    trajectory_topic: "/scaled_joint_trajectory_controller/joint_trajectory"
```

### `collision_world_hil.yml` - 虚拟障碍物

```yaml
world_model:
  coll_objs:
    sphere:
      virtual_sphere1:
        radius: 0.1
        position: [0.4, 0.3, 0.4]
    cube:
      virtual_wall:
        dims: [0.3, 0.1, 0.4]
        pose: [0.4, 0.25, 0.2, 0, 0, 0, 1.0]
```

---

## ⚠️ 常见问题

### 1. 无法连接机器人

```bash
# 检查网络连接
ping 192.168.131.38

# 检查 ROS2 话题
ros2 topic list | grep joint_states
ros2 topic echo /joint_states --once
```

### 2. 控制器未激活

```bash
# 查看控制器状态
ros2 control list_controllers

# 如果需要切换控制器
ros2 control switch_controllers \
    --deactivate forward_position_controller \
    --activate scaled_joint_trajectory_controller
```

### 3. 机器人不响应指令

- 确保 UR 示教器上选择了 "External Control" 程序
- 检查 UR ROS2 Driver 是否正常连接

---

## 🎯 目标位置设置

### 默认目标位置

配置文件 `config/ur7e_reacher_hil.yml` 中定义了默认目标：

```yaml
# 默认目标位置 (末端执行器位置，机器人基座坐标系)
default_goal:
  position: [0.4, 0.0, 0.4]      # [x, y, z] 米
  orientation: [0.0, 0.707, 0.0, 0.707]  # [x, y, z, w] 四元数 (末端朝下)
```

目标位置示意：
```
        Z
        ↑
        │    目标位置
        │    ★ (0.4, 0.0, 0.4)
        │   
        │      ↓ 末端朝下
        │   
   ─────┼────────→ X
        │
        Y (进入屏幕)
       [base]
```

### 运行时行为

| 情况 | 行为 |
|------|------|
| 有 `default_goal` | 启动后机器人自动移动到默认目标位置 |
| 无 `default_goal` | 启动后机器人保持当前位置不动 |
| 发布 `/target_pose` | 实时更新目标位置 |

### 动态更新目标

运行时可通过 ROS2 话题发布新目标：

```bash
# 发布新目标位置 (世界坐标)
ros2 topic pub --once /target_pose geometry_msgs/PoseStamped \
  '{header: {frame_id: "base_link"}, pose: {position: {x: 0.4, y: 0.2, z: 0.5}}}'

# 左前方位置
ros2 topic pub --once /target_pose geometry_msgs/PoseStamped \
  '{pose: {position: {x: 0.35, y: 0.25, z: 0.35}}}'
```

### 安全目标范围

对于空旷工作台上的 UR7e，建议的安全目标范围：

| 参数 | 安全范围 |
|------|----------|
| X | 0.2 ~ 0.6 m |
| Y | -0.4 ~ 0.4 m |
| Z | 0.1 ~ 0.6 m |

### 禁用默认目标 (启动时保持不动)

如果希望启动时机器人保持当前位置，注释掉配置文件中的 `default_goal`：

```yaml
# default_goal:
#   position: [0.4, 0.0, 0.4]
#   orientation: [0.0, 0.707, 0.0, 0.707]
```

---

## 📄 License

MIT License
