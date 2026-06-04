# HIL 数据链路说明

本文档详细说明 UR7e 硬件在环 (HIL_sage) 系统中，SAGE-MPPI 与真实机械臂之间的数据链路。

---

## 📊 数据链路总览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              HIL 系统数据链路                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────┐                                    ┌──────────────────┐
 │   SAGE-MPPI      │                                    │   真实 UR7e      │
 │(ur7e_hil_sage_mpc)│                                   │   机械臂         │
 └────────┬─────────┘                                    └────────┬─────────┘
          │                                                       │
          │  Float64MultiArray                                    │
          │  [q1, q2, q3, q4, q5, q6]                              │
          │  ─────────────────────────────────────────────────>   │
          │  /forward_position_controller/commands                │
          │                                                       │
          │  JointState                                           │
          │  {position, velocity, effort}                         │
          │  <─────────────────────────────────────────────────   │
          │  /joint_states                                        │
          │                                                       │
          ▼                                                       ▼
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │                           ROS2 Humble (DDS 通信)                                │
 └────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 控制回路详解

### 1️⃣ SAGE-MPPI 接收的信息

**话题**: `/joint_states`  
**消息类型**: `sensor_msgs/JointState`  
**频率**: ~500 Hz (UR 驱动发布频率)

```yaml
接收内容:
  - position: [q1, q2, q3, q4, q5, q6]    # 当前关节角度 (rad)
  - velocity: [v1, v2, v3, v4, v5, v6]    # 当前关节速度 (rad/s)
  - effort:   [τ1, τ2, τ3, τ4, τ5, τ6]    # 当前关节力矩 (Nm)
```

**处理流程**:
```
/joint_states → JointStateFilter(滤波) → 状态估计 → MPC 求解
```

### 2️⃣ SAGE-MPPI 发布的信息

**话题**: `/forward_position_controller/commands`  
**消息类型**: `std_msgs/Float64MultiArray`  
**频率**: 50 Hz (可配置)

```yaml
发送内容:
  data: [q1_cmd, q2_cmd, q3_cmd, q4_cmd, q5_cmd, q6_cmd]  # 期望关节角度 (rad)
```

---

## ⚙️ MPC 控制计算流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SAGE-MPPI 控制循环 (50 Hz)                            │
└─────────────────────────────────────────────────────────────────────────────┘

  1. 获取状态          2. MPPI 优化           3. 安全限制           4. 发送指令
 ┌───────────┐      ┌───────────────┐      ┌─────────────┐      ┌───────────┐
 │ 读取      │      │ 采样 SAGE     │      │ 速度/加速度 │      │ 发布      │
 │ /joint_   │ ──>  │ 轨迹并评估    │ ──>  │ 加速度限制  │ ──>  │ Float64   │
 │ states    │      │ 成本，加权    │      │ 平滑滤波    │      │ MultiArray│
 │           │      │ 平均得最优    │      │             │      │           │
 └───────────┘      └───────────────┘      └─────────────┘      └───────────┘
      │                    │                     │                    │
      ▼                    ▼                     ▼                    ▼
   当前状态            最优动作序列          安全的位置指令          机械臂运动
  [q, q̇, q̈]         [a_0, a_1, ..., a_H]    [q_cmd]
```

### MPPI 优化过程

```python
# 伪代码
for iteration in control_loop:
    # 1. 获取当前状态
    state = get_joint_state()  # [position, velocity, acceleration]
    
    # 2. MPPI 采样与评估
    trajectories = sample_trajectories(num_particles=500, horizon=30)
    costs = evaluate_costs(trajectories)  # 目标跟踪 + 碰撞避免 + 平滑性
    
    # 3. 加权平均计算最优动作
    weights = softmax(-costs / temperature)
    optimal_action = weighted_average(trajectories, weights)
    
    # 4. 积分得到位置指令
    q_cmd = state.position + optimal_action * dt
    
    # 5. 安全限制
    q_cmd = apply_velocity_limit(q_cmd, max_velocity=0.5)
    q_cmd = apply_smoothing_filter(q_cmd)
    
    # 6. 发送到机械臂
    publish(q_cmd)  # → /forward_position_controller/commands
```

---

## 🦾 机械臂如何响应位置指令

### forward_position_controller 工作原理

UR ROS2 Driver 中的 `forward_position_controller` 是一个 **JointGroupPositionController**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    forward_position_controller 内部结构                       │
└─────────────────────────────────────────────────────────────────────────────┘

   MPC 发送                     ROS2 控制器                    UR 控制器
   ───────                     ──────────                    ─────────
                                                             
  Float64MultiArray     ┌────────────────────┐     RTDE      ┌──────────────┐
  [q1_cmd, ..., q6_cmd] │ forward_position_  │   Protocol    │  UR 内部     │
  ─────────────────────>│ controller         │ ────────────> │  伺服控制    │
                        │                    │               │              │
                        │ • 接收位置指令     │               │ • PID 位置环 │
                        │ • 直接转发给硬件   │               │ • 电流控制   │
                        │ • 无轨迹插值       │               │ • 力矩输出   │
                        └────────────────────┘               └──────────────┘
```

### UR 机械臂内部控制结构

UR 机械臂内部使用**级联控制结构**：

```
                    UR 控制柜内部 (500 Hz 或 125 Hz)
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   位置指令          位置环 (P)        速度环 (PI)       电流环 (PI)          │
│   q_cmd ────────> ┌─────────┐ ───> ┌─────────┐ ───> ┌─────────┐ ───> 电机  │
│                   │    Kp   │      │  Kp,Ki  │      │  Kp,Ki  │            │
│                   └────┬────┘      └────┬────┘      └────┬────┘            │
│                        │ q             │ q̇             │ i                │
│                        │               │               │                  │
│   编码器 <─────────────┴───────────────┴───────────────┴──────────────────  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

控制层级:
  1. 位置环: 比较 q_cmd 和 q_actual，输出期望速度 v_cmd
  2. 速度环: 比较 v_cmd 和 v_actual，输出期望力矩 τ_cmd  
  3. 电流环: 比较 τ_cmd/Kt 和 i_actual，输出 PWM 驱动电机
```

### 控制器参数 (UR 内部，不可直接访问)

| 控制环 | 类型 | 典型参数 | 更新频率 |
|--------|------|----------|----------|
| 位置环 | P 控制 | Kp ≈ 10-50 | 500 Hz |
| 速度环 | PI 控制 | Kp ≈ 1-5, Ki ≈ 0.1-1 | 500 Hz |
| 电流环 | PI 控制 | 固件级别 | 8 kHz+ |

---

## 🔌 通信协议栈

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              通信协议栈                                       │
└─────────────────────────────────────────────────────────────────────────────┘

  SAGE-MPPI                    UR ROS2 Driver                  UR 控制柜
  ─────────                    ──────────────                  ─────────
      │                             │                              │
      │ DDS/ROS2                    │                              │
      │ (/forward_position_         │                              │
      │  controller/commands)       │                              │
      ├────────────────────────────>│                              │
      │                             │ RTDE (Real-Time              │
      │                             │ Data Exchange)               │
      │                             │ TCP/IP 125Hz                 │
      │                             ├─────────────────────────────>│
      │                             │                              │
      │                             │ 关节角度 (q_cmd)              │
      │                             │─────────────────────────────>│
      │                             │                              │
      │                             │ 状态反馈 (q, v, τ)            │
      │                             │<─────────────────────────────│
      │                             │                              │
      │ DDS/ROS2                    │                              │
      │ (/joint_states)             │                              │
      │<────────────────────────────│                              │
      │                             │                              │
```

### 延迟分析

| 环节 | 典型延迟 | 说明 |
|------|----------|------|
| MPC 计算 | 5-15 ms | GPU 加速后约 5ms |
| ROS2 DDS 传输 | 1-3 ms | 本地通信 |
| RTDE 传输 | 2-8 ms | TCP/IP 到控制柜 |
| UR 内部控制 | 2-8 ms | 500Hz 控制周期 |
| **总延迟** | **10-30 ms** | 端到端 |

---

## 📈 时序图

```
时间轴 (ms)
    0       20      40      60      80      100     120     140
    │       │       │       │       │       │       │       │
    ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼

MPC:  ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐
      │优化 │       │优化 │       │优化 │       │优化 │
      └──┬──┘       └──┬──┘       └──┬──┘       └──┬──┘
         │             │             │             │
         ▼ cmd         ▼ cmd         ▼ cmd         ▼ cmd

UR:       ├───────────┼───────────┼───────────┼───────────┤
          │  位置跟踪 │  位置跟踪 │  位置跟踪 │  位置跟踪 │
          │ (内部PID) │ (内部PID) │ (内部PID) │ (内部PID) │
          ├───────────┼───────────┼───────────┼───────────┤

状态:  ◄──┼───────────┼───────────┼───────────┼───────────►
       q₀    q₁          q₂          q₃          q₄
       (反馈到 MPC 作为下一次优化的初始状态)
```

---

## 🆚 控制器对比: forward_position_controller vs scaled_joint_trajectory_controller

| 特性 | forward_position_controller | scaled_joint_trajectory_controller |
|------|-----------------------------|------------------------------------|
| 消息类型 | `Float64MultiArray` | `JointTrajectory` |
| 控制模式 | 直接位置指令 | 轨迹跟踪 |
| 插值 | 无 (直接转发) | 有 (时间插值) |
| 延迟 | 低 (~2ms) | 高 (~20-50ms) |
| 适用场景 | **高频 MPC (>20Hz)** | 离线轨迹执行 |
| 速度缩放 | 无 | 有 (示教器速度滑块) |

**为什么 HIL 使用 forward_position_controller?**

1. **低延迟**: MPC 每 20ms 计算一次新指令，需要快速响应
2. **直接控制**: MPC 已经做了轨迹优化，不需要额外插值
3. **高频率**: 支持 50Hz+ 的控制频率
4. **简单性**: 直接发送位置数组，无需构建复杂的 JointTrajectory 消息

---

## ⚠️ 安全机制

### 1. 软件层安全 (SAGE-MPPI 侧)

```python
# 速度限制
max_delta = max_velocity * control_dt  # 0.5 * 0.02 = 0.01 rad/step
delta = np.clip(delta, -max_delta, max_delta)

# 平滑滤波
command_filter = JointStateFilter(filter_coeff=0.1, dt=0.02)
```

### 2. 硬件层安全 (UR 控制柜)

- 关节位置限制
- 关节速度限制 (默认 180°/s)
- 关节加速度限制
- 力矩限制
- 安全平面
- 急停按钮

---

## 📝 总结

| 组件 | 功能 |
|------|------|
| **SAGE-MPPI** | SAGE-MPPI 优化计算最优关节位置序列 |
| **ROS2 DDS** | 发布/订阅消息传输 |
| **UR ROS2 Driver** | ROS2 ↔ RTDE 协议转换 |
| **forward_position_controller** | 直接转发位置指令到硬件 |
| **UR 控制柜** | 级联 PID 控制 (位置环→速度环→电流环) |
| **电机驱动器** | PWM 控制电机转动 |

**关键点**:
- MPC 发送的是**期望关节位置** (不是速度或力矩)
- UR 内部使用**级联 PID** 跟踪位置指令
- 使用 `forward_position_controller` 实现**低延迟**控制
- 整个系统以 **50Hz** 闭环运行

---

## 📚 参考资料

- [UR ROS2 Driver 文档](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver)
- [ros2_control 架构](https://control.ros.org/)
- [RTDE 协议规范](https://www.universal-robots.com/articles/ur/interface-communication/real-time-data-exchange-rtde-guide/)
- [STORM MPC 论文](https://arxiv.org/abs/2104.13542)
- [STORM MPPI 算法原理](/home/wqj/storm/Algorithm.md) - 详细算法说明
