# UR7e MPC ROS2 软件在环 (SIL) 运行指南

## 概述

本文档说明如何运行 UR7e 机械臂的 MPC 控制器与 Isaac Sim 仿真的 ROS2 软件在环测试。

**功能特性：**
- 🔴 **红色目标球**：可在 Isaac Sim 中拖动，实时更新 MPC 目标位置
- 🟢 **绿色末端球**：显示机械臂实际末端位置
- 通过 ROS2 话题进行通信，为后续硬件实验做准备

## 系统要求

- Isaac Sim 5.1.0 (Python 3.11)
- ROS2 Humble (Python 3.10)
- CUDA GPU (推荐)
- 两个终端窗口

## 文件说明

| 文件 | 说明 | Python 环境 |
|------|------|-------------|
| `ur7e_ros2_sim.py` | Isaac Sim 仿真 + OmniGraph ROS2 桥接 | Isaac Sim Python 3.11 (env_isaaclab) |
| `ur7e_mpc_ros2.py` | MPC 控制器 | storm_py310 (Python 3.10 + ROS2) |
| `run_ur7e_ros2_sim.sh` | Isaac Sim 启动脚本 | env_isaaclab |
| `run_ur7e_mpc_ros2.sh` | MPC 控制器启动脚本 | storm_py310 |
| `test_ros2_topics.py` | ROS2 话题监控工具 | storm_py310 |
| `../ur7e_mpc_main.py` | 纯 Isaac Sim 仿真 (无 ROS2) | env_isaaclab |

## 运行步骤

### 步骤 1: 启动 Isaac Sim + ROS2 桥接 (终端 1)

```bash
cd ~/storm/examples/SIL
./run_ur7e_ros2_sim.sh
```

这会：
- 设置 Isaac Sim 内置 ROS2 环境变量
- 使用 Isaac Sim 的 Python 运行仿真
- 创建红色目标球（可拖动）和绿色末端球
- 发布 `/joint_states` 和 `/target_pose` 话题
- 订阅 `/joint_command` 和 `/ee_pose` 话题

等待看到 "仿真已启动!" 消息。

### 步骤 2: 启动 MPC 控制器 (终端 2)

```bash
cd ~/storm/examples/SIL
./run_ur7e_mpc_ros2.sh
```

这会：
- 激活 storm_py310 conda 环境
- Source ROS2 Humble 环境
- 启动 MPC 控制器
- 订阅 `/joint_states`，发布 `/joint_command`

### 步骤 3: (可选) 监控 ROS2 话题 (终端 3)

```bash
conda activate storm_py310
source /opt/ros/humble/setup.bash
python3 ~/storm/examples/SIL/test_ros2_topics.py
```

或者使用标准 ROS2 工具：
```bash
ros2 topic list
ros2 topic echo /joint_states
ros2 topic echo /joint_command
```

## 故障排除

### 问题 1: MPC 控制器连接超时

**症状：** "错误: 超时等待机器人状态!"

**原因：** Isaac Sim 没有发布 `/joint_states` 话题

**解决：**
1. 确保 Isaac Sim 已完全启动并显示 "仿真已启动!"
2. 检查 ROS2 话题：`ros2 topic list`
3. 检查是否能看到关节状态：`ros2 topic echo /joint_states`

### 问题 2: Isaac Sim 导入 rclpy 失败

**症状：** "警告: 无法导入 ROS2 库"

**原因：** Isaac Sim 内置 ROS2 库未正确加载

**解决：**
1. 确保使用 `run_ur7e_ros2_sim.sh` 启动（不是直接运行 python）
2. 检查环境变量：
   ```bash
   export ROS_DISTRO=humble
   export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wqj/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/exts/isaacsim.ros2.bridge/humble/lib
   ```

### 问题 3: 控制循环卡住

**症状：** 只显示一次误差就不动了

**原因：** 
- MPC 优化线程阻塞
- 或没有持续收到关节状态

**解决：**
1. 检查 rx/tx 计数是否在增加
2. 确认 Isaac Sim 仿真正在运行（看到帧计数增加）

### 问题 4: 机器人不动

**症状：** MPC 运行正常，但仿真中机器人不动

**原因：** 关节指令没有被 Isaac Sim 接收

**解决：**
1. 使用 `test_ros2_topics.py` 检查 `/joint_command` 消息数量
2. 确认 Isaac Sim 端显示 "收到关节指令" 日志

## ROS2 话题

| 话题 | 类型 | 方向 | 说明 |
|------|------|------|------|
| `/joint_states` | sensor_msgs/JointState | Isaac Sim → MPC | 当前关节位置和速度 |
| `/joint_command` | sensor_msgs/JointState | MPC → Isaac Sim | 目标关节位置 |
| `/target_pose` | geometry_msgs/PoseStamped | Isaac Sim → MPC | 目标位置（拖动红球时发布） |
| `/ee_pose` | geometry_msgs/PoseStamped | MPC → Isaac Sim | 末端位置（更新绿球） |

## 动态目标设置

在 Isaac Sim 仿真窗口中：
1. 选中红色目标球
2. 使用移动工具拖动到新位置
3. MPC 控制器会自动接收新目标并让机械臂跟踪

## 手动运行方式

如果脚本不工作，可以手动运行：

### 终端 1 (Isaac Sim):
```bash
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wqj/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/exts/isaacsim.ros2.bridge/humble/lib

cd /home/wqj/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64
./python.sh ~/storm/examples/SIL/ur7e_ros2_sim.py
```

### 终端 2 (MPC):
```bash
conda activate storm_py310
source /opt/ros/humble/setup.bash
python3 ~/storm/examples/SIL/ur7e_mpc_ros2.py
```

## 与纯仿真版本的对比

| 特性 | ur7e_mpc_main.py | ur7e_mpc_ros2.py + ur7e_ros2_sim.py |
|------|------------------|----------------------------------------|
| 通信方式 | 直接 Python 调用 | ROS2 话题 |
| 进程数 | 1 | 2 |
| Python 版本 | 3.11 | 3.11 (仿真) + 3.10 (MPC) |
| 延迟 | 最低 | 增加约 1-5ms |
| 用途 | 快速测试 | 硬件在环准备 |

---

## 数据传输机制详解

### 架构概览

软件在环 (SIL) 仿真使用 **两个独立进程** 通过 **ROS2 DDS 中间件** 进行通信：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              软件在环 (SIL) 架构                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────┐       ┌─────────────────────────────────┐  │
│  │   Isaac Sim 仿真进程         │       │      MPC 控制器进程              │  │
│  │   (Python 3.11)             │       │      (Python 3.10)              │  │
│  │                             │       │                                 │  │
│  │  ┌───────────────────────┐  │       │  ┌───────────────────────────┐  │  │
│  │  │  物理仿真引擎          │  │       │  │  STORM MPC 控制器          │  │  │
│  │  │  (PhysX)              │  │       │  │  (MPPI 优化)              │  │  │
│  │  └───────────┬───────────┘  │       │  └─────────────┬─────────────┘  │  │
│  │              │              │       │                │                │  │
│  │              ▼              │       │                ▼                │  │
│  │  ┌───────────────────────┐  │       │  ┌───────────────────────────┐  │  │
│  │  │  OmniGraph ROS2 桥接   │  │       │  │  ROS2RobotInterface       │  │  │
│  │  │  (Isaac Sim 内置)      │  │       │  │  (rclpy 节点)             │  │  │
│  │  └───────────┬───────────┘  │       │  └─────────────┬─────────────┘  │  │
│  │              │              │       │                │                │  │
│  └──────────────┼──────────────┘       └────────────────┼────────────────┘  │
│                 │                                       │                   │
│                 │         ┌─────────────────┐           │                   │
│                 │         │   FastDDS       │           │                   │
│                 └────────►│   中间件        │◄──────────┘                   │
│                           │  (共享内存)     │                               │
│                           └─────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 数据流详解

#### 1. 关节状态发布 (Isaac Sim → MPC)

```
Isaac Sim 物理引擎
        │
        ▼ (每帧 ~60Hz)
┌───────────────────────────────────┐
│ Robot.get_joint_positions()       │
│ Robot.get_joint_velocities()      │
└───────────────────┬───────────────┘
                    │
                    ▼
┌───────────────────────────────────┐
│ OmniGraph: ROS2PublishJointState  │
│ - 节点路径: /World/ROS2_Joint...  │
│ - 目标: /World/UR7e/root_joint    │
└───────────────────┬───────────────┘
                    │
                    ▼ (sensor_msgs/JointState)
┌───────────────────────────────────┐
│ 话题: /joint_states               │
│ 内容:                             │
│   - header.stamp: 仿真时间        │
│   - name: [关节名称列表]          │
│   - position: [6个关节位置]       │
│   - velocity: [6个关节速度]       │
└───────────────────┬───────────────┘
                    │
        ┌───────────┴───────────┐
        │      FastDDS          │
        │   (共享内存传输)       │
        └───────────┬───────────┘
                    │
                    ▼
┌───────────────────────────────────┐
│ ROS2RobotInterface._state_callback│
│ - 解析 JointState 消息            │
│ - 更新 self._q, self._dq          │
└───────────────────┬───────────────┘
                    │
                    ▼
┌───────────────────────────────────┐
│ MPC 控制器获取状态                 │
│ robot.get_joint_positions()       │
│ robot.get_joint_velocities()      │
└───────────────────────────────────┘
```

#### 2. 关节指令发送 (MPC → Isaac Sim)

```
┌───────────────────────────────────┐
│ MPC 优化计算                       │
│ mpc.get_command(t, state)         │
│ 输出: cmd['position'] (目标位置)   │
└───────────────────┬───────────────┘
                    │
                    ▼
┌───────────────────────────────────┐
│ robot.apply_action(cmd['position'])│
│ - 创建 JointState 消息            │
│ - 发布到 /joint_command           │
└───────────────────┬───────────────┘
                    │
                    ▼ (sensor_msgs/JointState)
┌───────────────────────────────────┐
│ 话题: /joint_command              │
│ 内容:                             │
│   - header.stamp: 当前时间        │
│   - name: [关节名称列表]          │
│   - position: [6个目标位置]       │
└───────────────────┬───────────────┘
                    │
        ┌───────────┴───────────┐
        │      FastDDS          │
        │   (共享内存传输)       │
        └───────────┬───────────┘
                    │
                    ▼
┌───────────────────────────────────┐
│ OmniGraph: ROS2SubscribeJointState│
│ - 接收 /joint_command 消息        │
│ - 输出 positionCommand            │
└───────────────────┬───────────────┘
                    │
                    ▼
┌───────────────────────────────────┐
│ OmniGraph: ArticulationController │
│ - 接收 positionCommand            │
│ - 应用到 /World/UR7e/root_joint   │
└───────────────────┬───────────────┘
                    │
                    ▼
┌───────────────────────────────────┐
│ Isaac Sim 物理引擎                 │
│ - 关节驱动器 (Stiffness=400)      │
│ - 执行位置控制                    │
└───────────────────────────────────┘
```

### Python 版本不兼容问题的解决方案

Isaac Sim 使用 **Python 3.11**，而 ROS2 Humble 默认使用 **Python 3.10**。这导致无法在同一进程中同时使用两者。

**解决方案：DDS 中间件通信**

```
┌─────────────────────┐              ┌─────────────────────┐
│   Isaac Sim         │              │   MPC 控制器         │
│   Python 3.11       │              │   Python 3.10       │
│                     │              │                     │
│   使用:             │              │   使用:             │
│   - Isaac Sim 内置  │◄────────────►│   - 系统 ROS2       │
│     ROS2 库         │   FastDDS    │     Humble          │
│   - OmniGraph 节点  │   (共享内存)  │   - rclpy           │
└─────────────────────┘              └─────────────────────┘
```

**关键配置：**

1. **Isaac Sim 端** (不要 source 系统 ROS2):
   ```bash
   export ROS_DISTRO=humble
   export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<isaac_sim>/exts/isaacsim.ros2.bridge/humble/lib
   ```

2. **MPC 端** (使用系统 ROS2):
   ```bash
   source /opt/ros/humble/setup.bash
   ```

3. **DDS 发现**：两端使用相同的 `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`，FastDDS 自动发现并建立连接。

### OmniGraph ROS2 桥接结构

Isaac Sim 中创建了两个 OmniGraph：

#### 发布器图 (`/World/ROS2_JointStatePublisher`)
```
OnPlaybackTick ──────► PublishJointState ◄── targetPrim: /World/UR7e/root_joint
       │                      ▲
       │                      │
ReadSimTime ─────► timeStamp  │
                              │
ROS2Context ──────► context ──┘
```

#### 订阅器图 (`/World/ROS2_JointCommandSubscriber`)
```
OnPlaybackTick ──► SubscribeJointState ──► ArticulationController
                          │                        ▲
                          │                        │
ROS2Context ──► context ──┘        targetPrim: /World/UR7e/root_joint
```

### 时序图

```
时间 ─────────────────────────────────────────────────────────────────────────►

Isaac Sim (60 Hz):
  │     │     │     │     │     │     │     │     │
  ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
 [S1]  [S2]  [S3]  [S4]  [S5]  [S6]  [S7]  [S8]  [S9]  ← 物理步进 + 发布状态

                    ▼           ▼           ▼
FastDDS:    ───[状态]───[状态]───[状态]───[指令]───[指令]───►

MPC (50 Hz):
       │           │           │           │
       ▼           ▼           ▼           ▼
      [M1]        [M2]        [M3]        [M4]  ← MPC 计算 + 发布指令
       │           │           │           │
       └───[CMD1]──┘───[CMD2]──┘───[CMD3]──┘

延迟: ~1-5ms (共享内存传输) + MPC 计算时间 (~7-20ms)
```

### 性能指标

| 指标 | 典型值 | 说明 |
|------|--------|------|
| Isaac Sim 物理频率 | 60 Hz | 仿真步进 |
| MPC 控制频率 | 50 Hz | 控制周期 20ms |
| MPC 优化时间 | 7-20 ms | MPPI 优化 |
| DDS 传输延迟 | 1-5 ms | 共享内存 |
| rx (接收状态) | ~60/s | 来自 Isaac Sim |
| tx (发送指令) | ~50/s | 发送到 Isaac Sim |

