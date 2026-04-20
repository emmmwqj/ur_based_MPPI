# SAGE_MPPI 项目总览

这份文档总结当前项目：

- `/home/wqj/storm/examples/SAGE_MPPI`

到底在做什么、是怎么接进现有 STORM/whole_control 框架的、控制与部署的技术路线是什么、以及当前这套项目的边界在哪里。

这份文档的定位是**项目总览**。  
如果你要看更细的内容，可继续参考：

- [changes.md](/home/wqj/storm/examples/SAGE_MPPI/paper/changes.md)：当前 SAGE 实现与 STORM 原生控制器的差异
- [ablation_plan.md](/home/wqj/storm/examples/SAGE_MPPI/paper/ablation_plan.md)：后续建议做的消融实验清单
- [implementation_gaps.md](/home/wqj/storm/examples/SAGE_MPPI/paper/implementation_gaps.md)：当前实现与论文理想形式的差距
- [main.pdf](/home/wqj/storm/examples/SAGE_MPPI/paper/main.pdf)：SAGE_MPPI 对应的方法文档

---

## 1. 这个项目在做什么

`examples/SAGE_MPPI` 这个项目的目标很明确：

**在不修改原 baseline sim_gazebo tall 项目的前提下，独立搭一套基于 SAGE_MPPI 控制器的 UR7e Gazebo 高墙避障 reaching 实验入口。**

它做的不是一套新机器人平台，也不是一套新场景系统，而是：

1. 复用原有的 Gazebo 仿真环境
2. 复用原有的 tall primitive-world 场景
3. 复用原有的 ROS2 / RViz / Gazebo 控制主链
4. 仅把核心控制器从 baseline STORM MPPI 替换成当前实现的 `SAGE_MPPI`
5. 再在部署层补上近目标精修、到达保持、停滞恢复和更实时的可视化

所以这个项目本质上是：

**“SAGE 控制器在 UR7e 高墙场景 Gazebo 闭环中的独立对照实现与部署入口。”**

---

## 2. 这个项目不做什么

为了避免误解，也要先明确它不做什么。

当前这个项目：

- 不是通用多场景 benchmark runner
- 不是 round2 / round3 / round4 那类批量实验入口
- 不是 diffusion baseline 的入口
- 不是新的世界建图系统
- 不是点云 / voxel SDF 的在线建图项目

它的定位更接近：

1. 一个**单场景、可交互、可联机、可可视化**的 SAGE 部署入口
2. 一个把 SAGE 从离线 benchmark 推到 Gazebo 闭环中的工程版本
3. 一个用于观察真实控制行为、调试到达、回弹、局部最小值、可视化轨迹与碰撞球的项目

---

## 3. 项目目录里有哪些关键文件

当前 `examples/SAGE_MPPI` 目录下的关键文件非常少，符合“只在项目目录里保留主脚本和启动脚本”的原则。

### 3.1 启动器

- [run_all_reach_static_tall.sh](/home/wqj/storm/examples/SAGE_MPPI/run_all_reach_static_tall.sh)  
  一键启动器。负责：
  - 拉起 Gazebo
  - 指定 `forward_position_controller`
  - 等待 `/joint_states` 与 `/forward_position_controller/commands` 就绪
  - 再启动 SAGE 控制器

- [run_reach_static_tall.sh](/home/wqj/storm/examples/SAGE_MPPI/run_reach_static_tall.sh)  
  控制器启动器。负责：
  - source ROS2
  - 激活 `storm_py310` conda 环境
  - 可选启动 RViz
  - 启动 Python 主程序

### 3.2 主程序

- [reach_static_ur7e_tall.py](/home/wqj/storm/examples/SAGE_MPPI/reach_static_ur7e_tall.py)  
  这是项目核心。负责：
  - 连接 Gazebo / ROS2
  - 实例化 `SageReacherTask`
  - 进入控制循环
  - 发布关节位置命令
  - 发布目标、末端、预测轨迹、黄色碰撞球等 RViz marker
  - 处理动态目标更新
  - 处理近目标精修、Jacobian 精修、保持、停滞恢复

### 3.3 说明文档

- [paper](/home/wqj/storm/examples/SAGE_MPPI/paper)  
  当前用于整理方法、实现差异、消融计划和实现缺口的文档目录。

---

## 4. 这个项目依赖哪些外部部分

`examples/SAGE_MPPI` 本身文件不多，但它是建立在几个现有模块上的。

### 4.1 依赖 `storm_kit`

控制与任务装配依赖：

- [sage_mppi.py](/home/wqj/storm/storm_kit/mpc/control/sage_mppi.py)
- [sage_arm_task.py](/home/wqj/storm/storm_kit/mpc/task/sage_arm_task.py)
- [sage_reacher_task.py](/home/wqj/storm/storm_kit/mpc/task/sage_reacher_task.py)

rollout / dynamics / cost 仍然依赖原有 STORM 结构：

- `ArmReacher`
- `ArmBase`
- `urdf_kinematic_model`
- `differentiable_robot_model`
- `primitive_collision`
- `self_collision`
- `goal_pose` 等 cost

### 4.2 依赖 `examples/SAGE_MPPI/config` 与 `examples/sim_gazebo/config`

当前项目现在按角色拆分配置：

SAGE 自己的 task/controller 配置放在项目目录内：

- [ur7e_reacher_gazebo_tall_sage.yml](/home/wqj/storm/examples/SAGE_MPPI/config/ur7e_reacher_gazebo_tall_sage.yml)

Gazebo 共享场景/机器人/RViz 配置继续复用 `sim_gazebo`：

- [collision_world_gazebo_tall.yml](/home/wqj/storm/examples/sim_gazebo/config/collision_world_gazebo_tall.yml)
- [ur7e_robot_gazebo.yml](/home/wqj/storm/examples/sim_gazebo/config/ur7e_robot_gazebo.yml)
- [initial_positions.yaml](/home/wqj/storm/examples/sim_gazebo/config/initial_positions.yaml)
- [reach_static.rviz](/home/wqj/storm/examples/sim_gazebo/config/reach_static.rviz)

### 4.3 依赖 ROS2 / Gazebo / ros2_control

当前项目运行时要求：

- ROS2 Humble
- Gazebo 仿真
- `ur_simulation_gazebo`
- `ros2_control`
- `forward_position_controller`

该项目**明确要求** Gazebo 中的机械臂通过：

- `/forward_position_controller/commands`

来接收位置命令，而不是其他控制器话题。

---

## 5. 项目的核心运行链路

当前项目完整链路是：

`run_all_reach_static_tall.sh`
→ `run_reach_static_tall.sh`
→ `reach_static_ur7e_tall.py`
→ `SageReacherTask`
→ `ControlProcess`
→ `SAGE_MPPI.optimize()`
→ `rollout`
→ `forward_position_controller`
→ Gazebo UR7e

可以拆成下面几层。

### 5.1 启动层

`run_all_reach_static_tall.sh` 负责：

1. source ROS2 环境
2. 启动 Gazebo 仿真
3. 使用 `initial_joint_controller:=forward_position_controller`
4. 等待 `/joint_states` 和 `/forward_position_controller/commands` 出现
5. 再启动控制器侧脚本

这样做的目的，是保证：

- Gazebo 先起来
- 控制器只在话题 ready 后再接入
- 减少控制器先启动导致的初始化 race condition

### 5.2 控制器进程层

`run_reach_static_tall.sh` 负责：

1. source ROS2
2. 激活 conda 环境 `storm_py310`
3. 启动 RViz
4. 运行 `reach_static_ur7e_tall.py`

### 5.3 Python 主程序层

`reach_static_ur7e_tall.py` 的主逻辑是：

1. 初始化 ROS2 executor
2. 连接 Gazebo 机器人接口
3. 验证命令话题确实是 `forward_position_controller`
4. 加载 robot / task / world 配置
5. 实例化 `SageReacherTask`
6. 设置默认目标
7. 进入固定频率控制循环
8. 每轮读取机器人状态、求解控制命令、发布关节目标、更新可视化

---

## 6. 这套项目的技术实现路线

这部分是最核心的。  
“技术实现路线”可以理解为：为了把 SAGE 从方法文档变成一个可运行的 Gazebo 闭环项目，当前项目是按什么路径落地的。

### 6.1 第一步：不改 baseline，而是并列新建 SAGE 入口

这套项目的第一个技术原则是：

**不修改旧的 baseline tall 工程，而是并列新建 `examples/SAGE_MPPI`。**

这样做的好处是：

- baseline 仍然可直接跑
- SAGE 项目可以单独迭代
- 后续对比 baseline vs SAGE 不会互相污染

所以这条路线一开始就不是“在旧脚本上直接改”，而是：

- 新建独立启动器
- 新建独立主程序
- 新建独立 task/controller 接线

### 6.2 第二步：场景、机器人、世界保持与 baseline 一致

项目并没有重新设计场景，而是故意保持：

- 同一台 UR7e
- 同一 tall primitive world
- 同一套 reaching 任务
- 同一套 robot/world/cost 参数基础

高墙场景 world 定义在：

- [collision_world_gazebo_tall.yml](/home/wqj/storm/examples/sim_gazebo/config/collision_world_gazebo_tall.yml)

它本质上是 primitive world：

- 两个高墙立方体
- 一个球形障碍
- 地面

因此，项目的技术路线不是“换场景”，而是：

**在相同场景下，比较 baseline proposal 与 SAGE proposal 的差异。**

### 6.3 第三步：task 层不改旧 task，而是并列接一个 `SageReacherTask`

为了不改旧 task，当前项目在 `storm_kit` 下单独新建了：

- [sage_arm_task.py](/home/wqj/storm/storm_kit/mpc/task/sage_arm_task.py)
- [sage_reacher_task.py](/home/wqj/storm/storm_kit/mpc/task/sage_reacher_task.py)

这层的作用是：

1. 继续复用原有 `ArmReacher` rollout
2. 继续复用 `ControlProcess`
3. 继续复用 joint state filter / command filter
4. 仅把 controller 从 `MPPI` 换成 `SAGE_MPPI`

因此这一步的技术路线是：

**控制主链不变，只替换 controller 装配。**

### 6.4 第四步：controller 层用独立的 `SAGE_MPPI`

当前控制器文件是：

- [sage_mppi.py](/home/wqj/storm/storm_kit/mpc/control/sage_mppi.py)

它没有继承原来的 `MPPI`，而是：

- 独立实现
- 对外保持 STORM controller 风格兼容

它的核心实现路线是：

1. 保留 STORM 的 rollout / sample lib / hotstart / mean_action 风格
2. 把 proposal covariance 改成 `scale × shape`
3. 引入 stage-scaled proposal
4. 引入 safe-elite anisotropic covariance
5. 引入 stagnation-triggered amplification
6. 输出更完整的实验统计量

所以从技术上看，SAGE 项目的“算法核心路线”是：

**在 STORM 的控制链接口不变的前提下，重写 proposal 更新逻辑。**

### 6.5 第五步：配置层专门为 tall SAGE 单独建一份配置

项目没有改 baseline 的 tall config，而是单独新建：

- [ur7e_reacher_gazebo_tall_sage.yml](/home/wqj/storm/examples/SAGE_MPPI/config/ur7e_reacher_gazebo_tall_sage.yml)

这份配置的设计思路是：

1. 保持同一场景、同一成本结构
2. 切到 SAGE 所需的 proposal 参数
3. 再加上 Gazebo tall 场景运行时需要的 refinement / hold 参数

因此配置层实际上分成两块：

- `mppi:`  
  对应 SAGE core proposal 参数：
  - `sigma_0`
  - `sigma_1`
  - `sigma_2`
  - `eta`
  - `tau_p`
  - `stagnation_alpha`
  - `n_iters`
  - `num_particles`

- `sage:`  
  对应部署层策略参数：
  - `success_threshold`
  - `refine_*`
  - `cart_refine_*`
  - `hold_*`

### 6.6 第六步：Gazebo 闭环里再叠加部署层增强器

仅靠 `SAGE_MPPI` controller 本体，虽然已经能改善搜索，但在 Gazebo tall 闭环里，近目标最后几厘米和到达后回弹仍然是实际问题。

所以当前项目的技术路线并没有停在“只接控制器”这一层，而是继续在主脚本里叠加了四个部署层增强器：

1. `_NearGoalRefinementController`
2. `_CartesianGoalRefiner`
3. `_GoalHoldController`
4. `_StallMonitor`

这四层的作用分别是：

#### 6.6.1 Near-goal refinement

当末端已经进入较近区域时：

- 减小 `sigma_0`
- 关闭或减弱 stagnation amplification
- 提高目标位置权重
- 把 retract state 临时设为当前关节位姿

目的：

- 让控制器从“全局探索模式”切换到“局部精修模式”

#### 6.6.2 Cartesian goal refiner

当误差进一步进入更小阈值后：

- 通过当前 Jacobian
- 做阻尼最小二乘 Cartesian position correction

目的：

- 把最后几厘米的误差更快压下去

#### 6.6.3 Goal hold

当已经连续多步稳定进入目标区域且速度够小：

- 锁定当前关节位姿
- 持续发送 hold command

目的：

- 防止刚到目标附近又因持续重规划而回弹

#### 6.6.4 Stall monitor

当系统处于：

- 误差还大
- 但末端几乎不动
- 且关节速度也很小

这种停滞状态时：

- reset SAGE distribution
- reset control process timing

目的：

- 让控制器在局部最小附近重新获得探索能力

因此，这个项目的部署技术路线并不是“只把论文控制器塞进 Gazebo”，而是：

**SAGE core proposal + 近目标精修 + 稳定保持 + 停滞恢复。**

---

## 7. 当前项目中的控制与执行路径

### 7.1 状态获取

机器人状态来自 Gazebo/ROS2：

- `/joint_states`

主程序读取：

- joint position
- joint velocity
- joint acceleration

### 7.2 控制求解

状态传给 `SageReacherTask`，再通过：

- `ControlProcess`
- `SAGE_MPPI.optimize()`

得到一段 horizon 上的动作序列。

最终真正发给 Gazebo 的不是整段轨迹，而是当前控制时刻应执行的下一拍命令。

### 7.3 执行接口

执行接口被显式限定为：

- `/forward_position_controller/commands`

也就是说，当前项目使用的是：

**ros2_control 的 `forward_position_controller` 位置控制方式**

而不是 velocity controller、trajectory controller 或 effort controller。

### 7.4 目标更新

当前项目支持通过：

- `/target_pose`

动态更新目标位置。  
主程序在检测到目标变化后会：

1. 更新 `goal_ee_pos`
2. reset SAGE distribution
3. reset control process timing
4. 立即同步重规划

这使得项目不仅能跑默认目标，还能做交互式目标切换。

---

## 8. 当前项目中的可视化路线

这套项目非常强调在线可观察性。

### 8.1 RViz 显示的内容

当前会显示：

- 目标点
- 当前末端位置
- world 障碍物 marker
- 黄色机械臂包裹碰撞球
- top-5 预测轨迹

### 8.2 可视化的两个关键工程点

为了让可视化真正可用，当前项目做了两项重要修复：

#### 8.2.1 预测轨迹每个控制周期都刷新

不再低频刷新，而是跟随控制循环实时更新。

#### 8.2.2 预测轨迹起点显式从当前末端开始

因为 rollout 里的末端轨迹默认从未来第一个预测步开始，如果直接显示，红线会看起来“悬在机械臂前方”。  
当前项目在显示时会把当前末端位置 prepend 到预测轨迹前面，因此 RViz 看到的是一条从当前末端连续接出的未来轨迹。

---

## 9. 这个项目相对 baseline tall 项目的核心区别

如果只从项目层面概括，与原 tall baseline 相比，当前 `examples/SAGE_MPPI` 的区别可以总结为两层。

### 9.1 第一层：控制器替换

原 baseline 项目使用：

- baseline `MPPI`

当前项目使用：

- `SAGE_MPPI`

这对应 proposal 设计的根本变化：

- stage-scaled covariance
- safe-elite anisotropic covariance
- stagnation-triggered amplification

### 9.2 第二层：部署增强

当前项目为了让 Gazebo tall 场景运行得更稳，又额外加了：

- near-goal refinement
- Jacobian Cartesian refinement
- hold
- stall recovery

因此，当前项目不是“单纯把 baseline controller 名字换掉”，而是：

**SAGE controller + 面向 Gazebo tall 场景的工程稳定化增强版项目。**

---

## 10. 这个项目当前已经具备的能力

从实际运行效果来看，当前项目已经具备：

### 10.1 可独立启动

可以通过：

```bash
cd /home/wqj/storm/examples/SAGE_MPPI
./run_all_reach_static_tall.sh
```

一键拉起整套 Gazebo + SAGE controller 项目。

### 10.2 可在 Gazebo 闭环中到达默认目标

当前已经解决了：

- 默认目标到不了
- 接近目标后回弹
- RViz 轨迹起点不对齐
- marker 刷新过慢

等问题。

### 10.3 可以做到更稳定的最终收敛

通过：

- near-goal refinement
- Cartesian Jacobian refiner
- hold

当前项目已经不再停在几厘米级误差平台，而能把最终误差压到更小量级并保持稳定。

### 10.4 可以做在线调试与行为观察

通过 RViz 与终端日志，当前项目适合分析：

- 局部最小值
- 停滞
- 预测轨迹
- 碰撞球与障碍物关系
- 目标更新后的重规划行为

---

## 11. 当前项目的边界与局限

虽然这套项目已经可跑、可看、可分析，但它的边界也要明确。

### 11.1 它仍然是单场景 Gazebo 部署入口

当前项目聚焦：

- UR7e
- tall scene
- static reaching

它不是统一 benchmark 基础设施本体。  
round2/round3/round4 的统计实验是另一条线。

### 11.2 它的最终闭环表现不完全等于 SAGE core 本体表现

因为主脚本里额外有：

- refinement
- Jacobian refiner
- hold
- stall recovery

所以如果后续要做严格论文归因，必须区分：

1. `SAGE_MPPI` controller core
2. `examples/SAGE_MPPI` deployed system

### 11.3 当前场景仍是 primitive world

当前避障 world 不是来自在线点云建图，而是来自：

- `collision_world_gazebo_tall.yml`

中的 primitive obstacles，再转成 planner 需要的 primitive collision/SDF 查询形式。

### 11.4 当前成功定义主要还是位置 reaching 导向

项目核心关注的是：

- 末端位置误差

而不是更严格的：

- 位置 + 姿态 + 长时稳定 + 严格安全共同成功定义

---

## 12. 从技术演化角度，这个项目的路线图可以怎样理解

如果把整个项目按技术路线压缩成一句话，可以这么理解：

### 阶段 1：并列新建入口

不碰 baseline tall 项目，单独在 `examples/SAGE_MPPI` 下建立：

- 启动器
- 主脚本

### 阶段 2：接入 SAGE controller

在 `storm_kit` 下新建：

- `SageArmTask`
- `SageReacherTask`
- `SAGE_MPPI`

形成与 baseline 并列的 controller 链路。

### 阶段 3：复用原场景与 cost

场景、机器人、cost 结构不变，只改变 proposal 设计。

### 阶段 4：解决 Gazebo 实际闭环问题

为了让它在 tall 场景里真正稳定工作，又逐步加入：

- 停滞恢复
- 近目标精修
- Jacobian 末端精修
- 到达保持
- 高频 RViz 更新

### 阶段 5：沉淀分析文档

再在 `paper/` 目录下逐步整理：

- 方法文档
- 差异说明
- 消融计划
- 实现缺口
- 项目总览

所以这套项目不是一次性写出来的，而是沿着下面这条路线逐步长成的：

**baseline tall Gazebo 项目**
→ **并列接入 SAGE controller**
→ **在 Gazebo 中闭环跑通**
→ **修复到达、回弹、显示与恢复问题**
→ **沉淀成可分析、可改进的研究型工程项目**

---

## 13. 一句话总结

`/home/wqj/storm/examples/SAGE_MPPI` 这个项目本质上是：

**一个面向 UR7e Gazebo 高墙避障 reaching 场景的、基于 SAGE_MPPI 的独立闭环控制项目。**

它的技术路线是：

1. 复用 STORM/whole_control 的 rollout、cost、ControlProcess 与 Gazebo 基础设施
2. 并列接入独立实现的 `SAGE_MPPI` controller
3. 通过专用 tall SAGE config 打开 staged proposal / safe-elite covariance / stagnation amplification
4. 再在 Gazebo 主脚本中叠加近目标精修、Jacobian 末端精修、保持与停滞恢复
5. 最终形成一套既能真实运行、又方便观察与后续研究分析的 SAGE 部署项目

如果后续你要继续改进算法，这个项目就是当前最直接的“方法落地入口”和“Gazebo 行为观察入口”。
