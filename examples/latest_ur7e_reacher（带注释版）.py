#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
"""
UR7e 机械臂 MPC 控制程序 - Isaac Sim 5.1 版本

本程序实现了基于 STORM (Stochastic Tensor Optimization for Robot Motion) 的
模型预测控制 (MPC) 来控制 UR7e 机械臂在 Isaac Sim 仿真环境中运动。

=== 程序架构 ===

1. IsaacSimRobotWrapper: 机器人接口封装类
   - 封装 Isaac Sim 的机器人 API
   - 提供关节状态读取和控制指令发送功能

2. Transform: 坐标变换类
   - 处理位置和姿态的坐标变换
   - 支持变换的组合和逆变换

3. IsaacSimWorld: 世界/场景管理类
   - 管理障碍物的生成和更新
   - 管理目标标记点的可视化

4. mpc_robot_interactive: 主控制函数
   - 初始化仿真环境
   - 创建 MPC 控制器
   - 执行控制循环

=== 控制流程 ===

                    ┌─────────────────┐
                    │  Isaac Sim 仿真  │
                    │  (物理引擎+渲染) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ IsaacSimRobotWrapper │
                    │   读取关节状态      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   STORM MPC     │
                    │  (轨迹优化+避障)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   发送控制指令    │
                    │  (关节位置目标)   │
                    └─────────────────┘

=== 关键配置文件 ===

- ur7e_isaacsim.yml: 机器人基座位置、初始关节角度
- ur7e_reacher_isaacsim.yml: MPC 参数、代价函数权重、碰撞检测配置
- collision_primitives_3d.yml: 场景中的障碍物定义
- robot/ur7e.yml: 机器人碰撞球模型定义

Author: Auto-generated
Date: 2024
"""

import argparse
import copy
import time
import yaml
import numpy as np

# 全局变量：Isaac Sim 应用实例
# 必须在 __main__ 中初始化，以避免多进程问题
simulation_app = None

# 设置 numpy 打印精度为 2 位小数
np.set_printoptions(precision=2)


class IsaacSimRobotWrapper:
    """
    Isaac Sim 机器人接口封装类
    
    该类封装了 Isaac Sim 中机器人的底层 API，提供统一的接口用于：
    1. 读取机器人当前状态（关节位置、速度、加速度）
    2. 发送控制指令（关节位置目标）
    
    === 属性说明 ===
    - world: Isaac Sim 的 World 对象，管理整个仿真场景
    - robot_prim_path: 机器人在 USD 场景中的路径，如 "/World/UR7e"
    - robot_name: 机器人名称，用于从场景中获取机器人对象
    - n_dof: 机器人自由度数量，UR7e 为 6
    - _articulation_controller: 关节控制器，用于发送控制指令
    
    === 使用流程 ===
    1. 创建实例
    2. 调用 world.reset() 初始化场景
    3. 调用 initialize() 获取机器人对象
    4. 在控制循环中调用 get_state() 和 command_robot_position()
    """
    
    def __init__(self, world, robot_prim_path, robot_name, n_dof=6, device='cuda'):
        """
        初始化机器人封装器
        
        Args:
            world: Isaac Sim World 对象
            robot_prim_path: 机器人在场景中的 USD 路径
            robot_name: 机器人名称（用于场景查询）
            n_dof: 自由度数量，UR7e 默认为 6
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.world = world
        self.robot_prim_path = robot_prim_path
        self.robot_name = robot_name
        self.device = device
        self.robot = None
        self.n_dof = n_dof  # UR7e 有 6 个自由度
        
        # 用于估算加速度的历史数据
        self.prev_joint_velocities = None
        self.prev_time = None
        
        # 关节控制器
        self._articulation_controller = None
        
    def initialize(self):
        """
        初始化机器人关节控制
        
        必须在 world.reset() 之后调用，此时机器人的物理属性已经初始化完成。
        该方法获取机器人对象并设置关节控制器。
        """
        # 从场景中获取机器人对象
        self.robot = self.world.scene.get_object(self.robot_name)
        if self.robot is None:
            raise RuntimeError(f"机器人 '{self.robot_name}' 未在场景中找到")
        
        # 获取关节控制器，用于发送位置/速度/力矩指令
        self._articulation_controller = self.robot.get_articulation_controller()
        
        # 初始化加速度估算所需的历史数据
        self.prev_joint_velocities = np.zeros(self.n_dof)
        self.prev_time = time.time()
        
    def get_state(self):
        """
        获取机器人当前状态
        
        返回格式与原 Isaac Gym 版本兼容，包含：
        - name: 关节名称列表
        - position: 关节位置（弧度）
        - velocity: 关节速度（弧度/秒）
        - acceleration: 关节加速度（通过速度差分估算）
        
        Returns:
            dict: 包含 name, position, velocity, acceleration 的字典
        """
        # 读取关节位置和速度（只取前 n_dof 个，忽略可能存在的夹爪关节）
        joint_positions = self.robot.get_joint_positions()[:self.n_dof]
        joint_velocities = self.robot.get_joint_velocities()[:self.n_dof]
        
        # 通过速度差分估算加速度
        # 注意：这是一个简化的估算方法，实际加速度可能需要更精确的计算
        current_time = time.time()
        dt = current_time - self.prev_time if self.prev_time else 0.01
        dt = max(dt, 0.001)  # 防止除零
        
        joint_accelerations = (joint_velocities - self.prev_joint_velocities) / dt
        
        # 更新历史数据
        self.prev_joint_velocities = joint_velocities.copy()
        self.prev_time = current_time
        
        # UR7e 关节名称（与 URDF 中定义一致）
        joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        return {
            'name': joint_names[:self.n_dof],
            'position': joint_positions,
            'velocity': joint_velocities,
            'acceleration': joint_accelerations
        }
    
    def command_robot_position(self, q_des):
        """
        发送关节位置控制指令
        
        使用 Isaac Sim 的 ArticulationAction 来设置目标关节位置。
        机器人的 PD 控制器会自动跟踪这个目标位置。
        
        Args:
            q_des: 目标关节位置数组（弧度），长度应为 n_dof
        """
        from isaacsim.core.utils.types import ArticulationAction
        
        # 处理自由度数量不匹配的情况
        num_dof = self.robot.num_dof
        if len(q_des) < num_dof:
            # 如果目标位置少于实际自由度，用零填充（处理可能的夹爪关节）
            q_des_full = np.zeros(num_dof)
            q_des_full[:len(q_des)] = q_des
        else:
            q_des_full = q_des[:num_dof]
        
        # 创建并发送控制动作
        action = ArticulationAction(joint_positions=q_des_full)
        self._articulation_controller.apply_action(action)
        
    def set_robot_state(self, q_des, qd_des):
        """
        直接设置机器人关节状态（位置和速度）
        
        与 command_robot_position 不同，这个方法会直接设置关节状态，
        而不是通过控制器跟踪目标。主要用于初始化或重置机器人状态。
        
        Args:
            q_des: 目标关节位置数组（弧度）
            qd_des: 目标关节速度数组（弧度/秒）
        """
        num_dof = self.robot.num_dof
        
        # 处理自由度数量不匹配
        if len(q_des) < num_dof:
            q_des_full = np.zeros(num_dof)
            q_des_full[:len(q_des)] = q_des
        else:
            q_des_full = q_des[:num_dof]
            
        if len(qd_des) < num_dof:
            qd_des_full = np.zeros(num_dof)
            qd_des_full[:len(qd_des)] = qd_des
        else:
            qd_des_full = qd_des[:num_dof]
            
        # 直接设置关节状态
        self.robot.set_joint_positions(q_des_full)
        self.robot.set_joint_velocities(qd_des_full)


class Transform:
    """
    坐标变换类
    
    用于处理三维空间中的刚体变换，包括平移和旋转。
    模仿 Isaac Gym 中的 Transform 类接口，便于代码迁移。
    
    === 坐标系约定 ===
    - 位置 p: [x, y, z]
    - 姿态 r: 四元数 [qx, qy, qz, qw]（xyzw 格式，与 scipy 一致）
    
    === 主要用途 ===
    1. 将障碍物从机器人坐标系转换到世界坐标系
    2. 将目标位置从机器人坐标系转换到世界坐标系
    3. 计算变换的逆和组合
    
    === 变换数学 ===
    点 p_world 在世界坐标系中的位置 = R * p_local + t
    其中 R 是旋转矩阵，t 是平移向量
    """
    
    def __init__(self, p=None, r=None):
        """
        初始化变换
        
        Args:
            p: 位置 [x, y, z]，默认为原点 [0, 0, 0]
            r: 四元数 [qx, qy, qz, qw]（xyzw 格式），默认为单位四元数 [0, 0, 0, 1]
        """
        self.p = np.array(p) if p is not None else np.array([0.0, 0.0, 0.0])
        self.r = np.array(r) if r is not None else np.array([0.0, 0.0, 0.0, 1.0])
        
    def inverse(self):
        """
        计算逆变换
        
        如果 T 将点从 A 坐标系变换到 B 坐标系，
        则 T.inverse() 将点从 B 坐标系变换到 A 坐标系。
        
        Returns:
            Transform: 逆变换
        """
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(self.r)  # scipy 使用 xyzw 格式
        rot_inv = rot.inv()
        p_inv = -rot_inv.apply(self.p)
        return Transform(p=p_inv, r=rot_inv.as_quat())
    
    def __mul__(self, other):
        """
        组合两个变换（变换乘法）
        
        T1 * T2 表示先应用 T2，再应用 T1
        
        Args:
            other: 另一个 Transform 对象
            
        Returns:
            Transform: 组合后的变换
        """
        from scipy.spatial.transform import Rotation
        rot1 = Rotation.from_quat(self.r)
        rot2 = Rotation.from_quat(other.r)
        
        new_rot = rot1 * rot2
        new_p = rot1.apply(other.p) + self.p
        
        return Transform(p=new_p, r=new_rot.as_quat())
    
    def transform_point(self, point):
        """
        将点从局部坐标系变换到世界坐标系
        
        计算: p_world = R * p_local + t
        
        Args:
            point: 局部坐标系中的点 [x, y, z]
            
        Returns:
            numpy.ndarray: 世界坐标系中的点 [x, y, z]
        """
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(self.r)
        return rot.apply(point) + self.p


class IsaacSimWorld:
    """
    Isaac Sim 场景/世界管理类
    
    负责管理仿真场景中的障碍物和可视化标记，包括：
    1. 从配置文件生成障碍物（球体、立方体）
    2. 创建和更新目标位置标记（红球、绿球）
    3. 处理坐标变换（机器人坐标系 -> 世界坐标系）
    
    === 障碍物来源 ===
    障碍物定义在 collision_primitives_3d.yml 文件中，格式如下：
    
    world_model:
      coll_objs:
        sphere:
          sphere1:
            radius: 0.1
            position: [0.4, 0.4, 0.1]
        cube:
          cube1:
            dims: [0.3, 0.1, 0.4]
            pose: [0.4, 0.2, 0.2, 0, 0, 0, 1.0]
    
    === 重要说明 ===
    这些障碍物同时用于：
    1. Isaac Sim 中的可视化显示
    2. STORM MPC 中的碰撞检测（MPC 直接读取同一配置文件）
    
    因此，可视化障碍物和 MPC 避障计算使用的是完全相同的障碍物模型。
    """
    
    def __init__(self, world, world_params, w_T_r=None):
        """
        初始化世界管理器
        
        Args:
            world: Isaac Sim World 对象
            world_params: 从 collision_primitives_3d.yml 加载的世界参数
            w_T_r: 世界坐标系到机器人坐标系的变换（Transform 对象）
                   用于将障碍物从机器人坐标系转换到世界坐标系
        """
        self.world = world
        self.world_params = world_params
        self.w_T_r = w_T_r  # world_T_robot: 机器人在世界坐标系中的位姿
        self.objects = {}   # 存储所有生成的对象，用于后续更新
        
    def spawn_primitives(self):
        """
        从配置参数生成碰撞基元（障碍物）
        
        遍历 world_params 中定义的所有球体和立方体，
        在 Isaac Sim 场景中创建对应的可视化对象。
        
        注意：这些是 VisualSphere/VisualCuboid，仅用于可视化，
        不参与物理碰撞（MPC 通过自己的碰撞检测模块处理避障）
        """
        from isaacsim.core.api.objects import VisualCuboid, VisualSphere
        
        if self.world_params is None:
            return
            
        world_model = self.world_params.get('world_model', {})
        coll_objs = world_model.get('coll_objs', {})
        
        # === 生成球体障碍物 ===
        spheres = coll_objs.get('sphere', {})
        for name, params in spheres.items():
            radius = params.get('radius', 0.1)
            position = np.array(params.get('position', [0, 0, 0]))
            
            # 将位置从机器人坐标系转换到世界坐标系
            if self.w_T_r is not None:
                position = self.w_T_r.transform_point(position)
                
            # 创建红色可视化球体
            sphere = VisualSphere(
                prim_path=f"/World/obstacles/{name}",  # USD 场景路径
                name=name,
                position=position,
                radius=radius,
                color=np.array([0.8, 0.2, 0.2])  # 红色
            )
            self.world.scene.add(sphere)
            self.objects[name] = sphere
            
        # === 生成立方体障碍物 ===
        cubes = coll_objs.get('cube', {})
        for name, params in cubes.items():
            dims = np.array(params.get('dims', [0.1, 0.1, 0.1]))  # 尺寸 [x, y, z]
            pose = params.get('pose', [0, 0, 0, 0, 0, 0, 1])      # 位姿 [x, y, z, qx, qy, qz, qw]
            position = np.array(pose[:3])
            orientation_xyzw = np.array(pose[3:])
            
            # 将位置从机器人坐标系转换到世界坐标系
            if self.w_T_r is not None:
                position = self.w_T_r.transform_point(position)
                
            # 将四元数从 xyzw 格式转换为 Isaac Sim 使用的 wxyz 格式
            orientation_wxyz = np.array([orientation_xyzw[3], orientation_xyzw[0], 
                                         orientation_xyzw[1], orientation_xyzw[2]])
            
            # 创建蓝色可视化立方体
            cube = VisualCuboid(
                prim_path=f"/World/obstacles/{name}",
                name=name,
                position=position,
                orientation=orientation_wxyz,
                size=1.0,        # 基础尺寸
                scale=dims,      # 缩放到实际尺寸
                color=np.array([0.5, 0.5, 0.8])  # 蓝色
            )
            self.world.scene.add(cube)
            self.objects[name] = cube
            
    def spawn_target_marker(self, name, position, orientation=None, color=None):
        """
        生成目标位置标记（小球）
        
        用于可视化显示：
        - 目标末端位置（红色小球）
        - 当前末端位置（绿色小球）
        
        Args:
            name: 标记名称
            position: 标记位置 [x, y, z]
            orientation: 标记姿态（可选，标记为球体所以不需要）
            color: 标记颜色 [r, g, b]，默认红色
            
        Returns:
            VisualSphere: 创建的标记对象
        """
        from isaacsim.core.api.objects import VisualSphere
        
        if color is None:
            color = np.array([0.8, 0.1, 0.1])  # 默认红色
        
        position = np.array(position)
            
        marker = VisualSphere(
            prim_path=f"/World/markers/{name}",
            name=name,
            position=position,
            radius=0.03,  # 3cm 半径的小球
            color=color
        )
        self.world.scene.add(marker)
        self.objects[name] = marker
        return marker
        
    def update_marker_pose(self, name, position, orientation=None):
        """
        更新标记位置
        
        在控制循环中调用，用于实时更新当前末端位置标记
        
        Args:
            name: 标记名称
            position: 新位置 [x, y, z]
            orientation: 新姿态（可选）
        """
        if name in self.objects:
            marker = self.objects[name]
            marker.set_world_pose(position=np.array(position))
            
    def get_pose(self, name):
        """
        获取对象当前位姿
        
        Args:
            name: 对象名称
            
        Returns:
            Transform: 对象的位姿变换，如果对象不存在则返回 None
        """
        if name in self.objects:
            obj = self.objects[name]
            pos, rot_wxyz = obj.get_world_pose()
            # 将四元数从 wxyz 格式转换为 xyzw 格式
            rot_xyzw = np.array([rot_wxyz[1], rot_wxyz[2], rot_wxyz[3], rot_wxyz[0]])
            return Transform(p=pos, r=rot_xyzw)
        return None


def mpc_robot_interactive(args):
    """
    MPC 机器人交互控制主函数
    
    这是程序的核心函数，负责：
    1. 初始化 Isaac Sim 仿真环境
    2. 加载机器人和障碍物
    3. 创建 STORM MPC 控制器
    4. 运行控制循环
    
    === 控制循环流程 ===
    
    每个控制周期（约 20ms）执行以下步骤：
    
    1. world.step() - 推进物理仿真一步
    2. get_state() - 读取机器人当前关节状态
    3. mpc_control.get_command() - MPC 计算最优控制指令
       - MPPI 算法采样 500 条候选轨迹
       - 计算每条轨迹的代价（目标误差 + 碰撞代价 + 平滑度等）
       - 加权平均得到最优轨迹
       - 返回下一时刻的目标关节位置
    4. command_robot_position() - 发送控制指令到机器人
    
    === MPC 避障原理 ===
    
    STORM 使用基于模型的碰撞检测，而非传感器感知：
    
    1. 碰撞球模型：机器人各连杆用球体近似（定义在 robot/ur7e.yml）
    2. 世界模型：障碍物预定义为基元（定义在 collision_primitives_3d.yml）
    3. 碰撞代价：计算碰撞球与障碍物的距离，距离越近代价越高
    4. 轨迹优化：MPPI 选择总代价最低的轨迹，自动避开障碍物
    
    Args:
        args: 命令行参数，包含：
            - robot: 机器人类型（'ur7e'）
            - cuda: 是否使用 GPU
            - headless: 是否无头模式运行
    """
    
    # ========== 第一部分：导入模块 ==========
    # 导入全局变量和必要的模块
    global simulation_app
    import torch
    
    # Isaac Sim Core API 导入
    from isaacsim.core.api.world import World
    from isaacsim.core.api.objects import VisualCuboid, VisualSphere
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.nucleus import get_assets_root_path
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    import omni.timeline
    import carb
    
    # STORM 库导入
    from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
    from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
    from storm_kit.mpc.task.reacher_task import ReacherTask
    
    print("=" * 50)
    print("启动 UR7e MPC 机器人交互控制")
    print("=" * 50)
    
    # ========== 第二部分：配置文件设置 ==========
    vis_ee_target = True  # 是否可视化末端目标
    
    # 配置文件名称（使用 Isaac Sim 专用配置）
    robot_file = args.robot + '_isaacsim.yml'           # 机器人配置：ur7e_isaacsim.yml
    task_file = args.robot + '_reacher_isaacsim.yml'    # MPC 任务配置：ur7e_reacher_isaacsim.yml
    world_file = 'collision_primitives_3d.yml'          # 场景障碍物配置
    
    print(f"机器人配置文件: {robot_file}")
    print(f"MPC 任务配置文件: {task_file}")
    
    # ========== 第三部分：加载配置文件 ==========
    print("加载配置文件...")
    
    # 加载世界/障碍物配置
    world_yml = join_path(get_gym_configs_path(), world_file)
    with open(world_yml) as file:
        world_params = yaml.safe_load(file)
        
    # 加载机器人配置（包含基座位置、初始关节角度等）
    robot_yml = join_path(get_gym_configs_path(), robot_file)
    print(f"机器人配置路径: {robot_yml}")
    with open(robot_yml) as file:
        robot_params = yaml.safe_load(file)
        
    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    
    # 设置计算设备
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"计算设备: {device}")
    
    # ========== 第四部分：创建 Isaac Sim 世界 ==========
    print("创建 Isaac Sim 世界...")
    world = World(stage_units_in_meters=1.0)  # 使用米作为单位
    
    # 添加地面
    print("添加地面...")
    world.scene.add_default_ground_plane()
    
    # ========== 第五部分：设置机器人基座位姿 ==========
    # 从配置文件读取机器人基座位姿
    # 格式: [x, y, z, qx, qy, qz, qw]
    robot_pose_cfg = sim_params.get('robot_pose', [0, 0, 0, 0, 0, 0, 1])
    robot_position = np.array(robot_pose_cfg[:3])           # 位置 [x, y, z]
    robot_orientation_xyzw = np.array(robot_pose_cfg[3:])   # 四元数 [qx, qy, qz, qw]
    
    # Isaac Sim 使用 wxyz 格式的四元数，需要转换
    robot_orientation_wxyz = np.array([
        robot_orientation_xyzw[3],  # qw
        robot_orientation_xyzw[0],  # qx
        robot_orientation_xyzw[1],  # qy
        robot_orientation_xyzw[2]   # qz
    ])
    
    print(f"机器人位置: {robot_position}")
    print(f"机器人姿态 (wxyz): {robot_orientation_wxyz}")
    
    # 创建机器人基座变换（用于坐标转换）
    w_T_r = Transform(p=robot_position, r=robot_orientation_xyzw)
    
    # ========== 第六部分：加载 UR7e 机器人 ==========
    print("加载 UR7e 机器人...")
    
    # 使用本地 USD 文件（从 URDF 转换而来）
    ur7e_usd_path = get_assets_path() + "/urdf/ur7e/ur7e.usd"
    print(f"机器人 USD 路径: {ur7e_usd_path}")
    
    ur7e_prim_path = "/World/UR7e"  # 机器人在场景中的路径
    
    # 将机器人 USD 添加到场景
    print("添加机器人到场景...")
    add_reference_to_stage(usd_path=ur7e_usd_path, prim_path=ur7e_prim_path)
    
    # 创建 Robot 对象并添加到场景
    print("创建 Robot 对象...")
    ur7e_robot = world.scene.add(
        Robot(
            prim_path=ur7e_prim_path,
            name="ur7e_robot",
            position=robot_position,
            orientation=robot_orientation_wxyz
        )
    )
    
    # ========== 第七部分：初始化物理仿真 ==========
    # 重置世界以初始化机器人关节
    print("执行初始世界重置以配置机器人...")
    world.reset()
    
    # 重置后重新设置机器人位姿（确保位置正确）
    print("显式设置机器人位姿...")
    ur7e_robot.set_world_pose(position=robot_position, orientation=robot_orientation_wxyz)
    
    # 运行几步仿真以应用设置
    for _ in range(3):
        world.step(render=True)
    
    # ========== 第八部分：配置关节驱动参数 ==========
    print("配置关节驱动参数...")
    from pxr import UsdPhysics, PhysxSchema
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    
    # 设置 PD 控制器的刚度和阻尼
    # 这些参数影响机器人跟踪目标位置的响应速度和稳定性
    joint_drive_stiffness = 400.0  # 刚度：越大跟踪越快，但可能振荡
    joint_drive_damping = 40.0     # 阻尼：抑制振荡，但会减慢响应
    
    # 遍历所有关节，设置驱动参数
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.DriveAPI):
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                drive.GetStiffnessAttr().Set(joint_drive_stiffness)
                drive.GetDampingAttr().Set(joint_drive_damping)
                    
    print(f"关节驱动配置完成: 刚度={joint_drive_stiffness}, 阻尼={joint_drive_damping}")
    
    # ========== 第九部分：创建机器人封装器 ==========
    n_dof = 6  # UR7e 有 6 个自由度
    robot_sim = IsaacSimRobotWrapper(world, ur7e_prim_path, "ur7e_robot", n_dof=n_dof, device=device)
    
    # 设置 PyTorch 张量参数
    torch_device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    tensor_args = {'device': torch_device, 'dtype': torch.float32}
    
    # ========== 第十部分：创建障碍物 ==========
    print("创建障碍物...")
    world_instance = IsaacSimWorld(world, world_params, w_T_r=w_T_r)
    world_instance.spawn_primitives()
    
    # ========== 第十一部分：创建 MPC 控制器 ==========
    print("创建 MPC 控制器...")
    # ReacherTask 封装了 STORM 的 MPPI 控制器
    # 它会加载：
    # - 机器人运动学模型（从 URDF）
    # - 碰撞球模型（从 robot/ur7e.yml）
    # - 代价函数配置（从 ur7e_reacher_isaacsim.yml）
    # - 世界障碍物模型（从 collision_primitives_3d.yml）
    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)
    
    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
    print(f"机器人自由度: {n_dof}")
    
    # ========== 第十二部分：设置目标状态 ==========
    mpc_tensor_dtype = {'device': torch_device, 'dtype': torch.float32}
    
    # 目标状态说明：
    # 前 6 个值是目标关节角度（弧度），系统会通过正向运动学计算目标末端位置
    # 后 6 个值是目标关节速度（通常设为 0）
    #
    # UR7e 初始状态: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    # 下面的目标状态会让机械臂移动到一个可达的位置
    ur7e_goal_state = np.array([
        0.5,      # shoulder_pan: 稍微侧转
        -1.2,     # shoulder_lift: 抬高手臂（值越小臂越高）
        1.2,      # elbow: 弯曲肘部
        -1.57,    # wrist_1: 保持手腕姿态
        -1.57,    # wrist_2: 保持手腕姿态
        0.0,      # wrist_3: 保持手腕姿态
        # 目标速度（6 个零）
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])
    x_des_list = [ur7e_goal_state]
    x_des = x_des_list[0]
    
    # 更新 MPC 控制器的目标参数
    # 这会触发正向运动学计算，得到目标末端位置和姿态
    mpc_control.update_params(goal_state=x_des)
    
    # 获取计算得到的目标末端位置和姿态
    g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
    print(f"目标末端位置: {g_pos}")
    print(f"目标末端姿态: {g_q}")
    
    # ========== 第十三部分：设置坐标变换 ==========
    # 用于将轨迹从机器人坐标系转换到世界坐标系（可视化用）
    w_T_robot = torch.eye(4)
    w_T_robot[0, 3] = robot_position[0]
    w_T_robot[1, 3] = robot_position[1]
    w_T_robot[2, 3] = robot_position[2]
    
    # quaternion_to_matrix 期望 [w, x, y, z] 格式
    quat_wxyz = torch.tensor([robot_orientation_wxyz[0], robot_orientation_wxyz[1], 
                              robot_orientation_wxyz[2], robot_orientation_wxyz[3]]).unsqueeze(0)
    rot = quaternion_to_matrix(quat_wxyz)
    w_T_robot[:3, :3] = rot[0]
    
    w_robot_coord = CoordinateTransform(
        trans=w_T_robot[0:3, 3].unsqueeze(0),
        rot=w_T_robot[0:3, 0:3].unsqueeze(0)
    )
    
    # ========== 第十四部分：控制参数设置 ==========
    # 控制周期（从配置文件读取，通常为 0.02 秒 = 50Hz）
    sim_dt = mpc_control.exp_params['control_dt']
    print(f"控制周期: {sim_dt} 秒")
    
    # 初始化机器人封装器
    print("初始化机器人封装器...")
    robot_sim.initialize()
    
    # ========== 第十五部分：设置初始关节配置 ==========
    init_state = sim_params.get('init_state', [0.0] * n_dof)
    init_positions = np.array(init_state[:n_dof])
    print(f"初始关节角度: {init_positions}")
    
    # 获取实际机器人自由度数量
    num_robot_dof = ur7e_robot.num_dof
    print(f"机器人总自由度: {num_robot_dof}")
    
    if num_robot_dof is None:
        print("警告: num_dof 为 None，使用默认 6 自由度")
        num_robot_dof = 6
        
    # 处理自由度数量不匹配的情况
    if num_robot_dof > n_dof:
        init_positions_full = np.zeros(num_robot_dof)
        init_positions_full[:n_dof] = init_positions
    else:
        init_positions_full = init_positions[:num_robot_dof]
    
    # 设置初始关节状态
    ur7e_robot.set_joint_positions(init_positions_full)
    ur7e_robot.set_joint_velocities(np.zeros(num_robot_dof))
    
    # 再次确保机器人位姿正确
    print("关节初始化后重新设置机器人位姿...")
    ur7e_robot.set_world_pose(position=robot_position, orientation=robot_orientation_wxyz)
    
    # 调试：打印初始化后的实际位姿
    robot_actual_pos, robot_actual_ori = ur7e_robot.get_world_pose()
    print(f"期望位置: {robot_position}")
    print(f"期望姿态 (wxyz): {robot_orientation_wxyz}")
    print(f"实际位置: {robot_actual_pos}")
    print(f"实际姿态 (wxyz): {robot_actual_ori}")
    
    # 运行几步仿真让机器人稳定
    for _ in range(5):
        world.step(render=True)
    
    # 最终位姿检查
    robot_actual_pos, robot_actual_ori = ur7e_robot.get_world_pose()
    print(f"最终位置: {robot_actual_pos}")
    print(f"最终姿态 (wxyz): {robot_actual_ori}")
    
    # ========== 第十六部分：创建可视化标记 ==========
    if vis_ee_target:
        print("创建目标标记...")
        # 将目标位置转换到世界坐标系
        goal_pos_world = w_T_r.transform_point(g_pos)
        
        # 红色小球：目标末端位置
        world_instance.spawn_target_marker(
            "ee_target",
            position=goal_pos_world,
            color=np.array([0.8, 0.1, 0.1])  # 红色
        )
        # 绿色小球：当前末端位置
        world_instance.spawn_target_marker(
            "ee_current",
            position=[0, 0, 0],
            color=np.array([0.1, 0.8, 0.1])  # 绿色
        )
    
    # ========== 第十七部分：控制循环初始化 ==========
    t_step = 0.0  # 仿真时间
    i = 0         # 迭代计数器
    
    print("=" * 50)
    print("开始 MPC 控制循环...")
    print("按 Ctrl+C 退出")
    print("=" * 50)
    
    # 启动时间线（开始物理仿真）
    omni.timeline.get_timeline_interface().play()
    
    # 让仿真运行几步以稳定
    for _ in range(10):
        world.step(render=True)
    
    # ========== 第十八部分：MPC 预热 ==========
    # MPC 控制器使用多进程，需要预热以建立进程间通信
    current_robot_state = robot_sim.get_state()
    
    print("预热 MPC 控制器...")
    for warmup_i in range(5):
        try:
            # WAIT=False 表示不等待 MPC 计算完成
            _ = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=False)
            t_step += sim_dt
            world.step(render=True)
            current_robot_state = robot_sim.get_state()
        except Exception as e:
            print(f"预热迭代 {warmup_i}: {e}")
            t_step += sim_dt
            world.step(render=True)
    
    print("MPC 预热完成，开始主控制循环...")
    
    # ========== 第十九部分：主控制循环 ==========
    while simulation_app.is_running():
        try:
            # --- 步骤 1: 推进仿真 ---
            world.step(render=True)
            
            # 检查仿真状态
            if not world.is_playing():
                if world.is_stopped():
                    world.reset()
                    robot_sim.initialize()
                world.step(render=True)
                continue
                
            t_step += sim_dt
            
            # --- 步骤 2: 读取当前状态 ---
            current_robot_state = robot_sim.get_state()
            
            # --- 步骤 3: 获取 MPC 控制指令 ---
            try:
                # WAIT=True 表示等待 MPC 计算完成后再返回
                # MPC 会在后台进程中：
                # 1. 采样 500 条候选轨迹
                # 2. 通过正向运动学计算每条轨迹的末端位置
                # 3. 计算每条轨迹的代价（目标误差 + 碰撞 + 平滑度等）
                # 4. 按代价加权平均得到最优控制序列
                command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
            except IndexError as e:
                # MPC 时序问题，跳过本次迭代
                if i % 50 == 0:
                    print(f"MPC 时序跳过 at t={t_step:.3f}")
                i += 1
                continue
            
            # 构建状态张量（用于计算当前末端位置）
            filtered_state_mpc = current_robot_state
            curr_state = np.hstack((
                filtered_state_mpc['position'],      # 关节位置
                filtered_state_mpc['velocity'],      # 关节速度
                filtered_state_mpc['acceleration']   # 关节加速度
            ))
            
            curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
            
            # 提取控制指令
            q_des = copy.deepcopy(command['position'])      # 目标关节位置
            qd_des = copy.deepcopy(command['velocity'])     # 目标关节速度
            qdd_des = copy.deepcopy(command['acceleration']) # 目标关节加速度
            
            # --- 步骤 4: 计算误差 ---
            # ee_error 包含三个值：
            # [0] 总代价（包含碰撞、可操作度等）
            # [1] 姿态误差
            # [2] 位置误差（米）
            ee_error = mpc_control.get_current_error(filtered_state_mpc)
            
            # 获取当前末端位置（用于可视化）
            pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
            e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
            e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
            
            # --- 步骤 5: 更新可视化 ---
            if vis_ee_target:
                # 将当前末端位置转换到世界坐标系并更新绿色小球
                ee_pos_world = w_T_r.transform_point(e_pos)
                world_instance.update_marker_pose("ee_current", ee_pos_world)
            
            # 每 10 次迭代打印一次状态
            # Error 格式: [总代价, 姿态误差, 位置误差(米)]
            if i % 10 == 0:
                print(f"[{i}] Error: {['{:.3f}'.format(x) for x in ee_error]}, "
                      f"opt_dt: {mpc_control.opt_dt:.3f}, mpc_dt: {mpc_control.mpc_dt:.3f}")
            
            # --- 步骤 6: 发送控制指令 ---
            robot_sim.command_robot_position(q_des)
            
            i += 1
            
        except KeyboardInterrupt:
            print('正在关闭...')
            break
        except Exception as e:
            print(f"主循环错误: {e}")
            import traceback
            traceback.print_exc()
            # 不因单次错误而退出，尝试继续运行
            i += 1
            continue
    
    # ========== 第二十部分：清理资源 ==========
    print("清理资源...")
    mpc_control.close()  # 关闭 MPC 后台进程
    simulation_app.close()  # 关闭 Isaac Sim
    
    return 1


if __name__ == '__main__':
    """
    程序入口点
    
    === 启动顺序说明 ===
    
    1. 解析命令行参数
    2. 创建 SimulationApp（必须在 __main__ 中，避免多进程问题）
    3. 配置 PyTorch
    4. 导入 Isaac Sim 和 STORM 模块
    5. 调用主控制函数
    
    === 命令行参数 ===
    
    --robot: 机器人类型，默认 'ur7e'
    --cuda: 使用 GPU 加速（默认开启）
    --headless: 无头模式运行（不显示窗口）
    --control_space: 控制空间，默认 'acc'（加速度控制）
    
    === 运行示例 ===
    
    # 正常运行（带 GUI）
    python latest_ur7e_reacher.py --robot ur7e --cuda
    
    # 无头模式运行
    python latest_ur7e_reacher.py --robot ur7e --cuda --headless
    """
    
    # ===== 解析命令行参数 =====
    parser = argparse.ArgumentParser(description='UR7e Reacher MPC 控制 - Isaac Sim 5.1')
    parser.add_argument('--robot', type=str, default='ur7e', help='机器人类型')
    parser.add_argument('--cuda', action='store_true', default=True, help='使用 CUDA GPU 加速')
    parser.add_argument('--headless', action='store_true', default=False, help='无头模式（不显示窗口）')
    parser.add_argument('--control_space', type=str, default='acc', help='控制空间')
    args = parser.parse_args()
    
    # ===== 创建 Isaac Sim 应用 =====
    # 重要：SimulationApp 必须在 __main__ 中创建
    # 这是因为 STORM 的 MPC 使用多进程，如果在函数中创建会导致子进程重复创建
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})
    
    # ===== 配置 PyTorch =====
    import torch
    
    # 设置多进程启动方式为 'spawn'（CUDA 兼容）
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过了
    
    # 性能优化设置
    torch.set_num_threads(8)                        # CPU 线程数
    torch.backends.cudnn.benchmark = False          # 禁用自动优化（保证可重复性）
    torch.backends.cuda.matmul.allow_tf32 = True    # 允许 TF32 加速矩阵乘法
    torch.backends.cudnn.allow_tf32 = True          # 允许 TF32 加速卷积
    
    # ===== 导入 Isaac Sim 模块 =====
    # 必须在 SimulationApp 创建之后导入
    from isaacsim.core.api.world import World
    from isaacsim.core.api.objects import VisualCuboid, VisualSphere
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.nucleus import get_assets_root_path
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    import omni.timeline
    import carb
    
    # ===== 导入 STORM 模块 =====
    from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
    from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
    from storm_kit.mpc.task.reacher_task import ReacherTask
    
    # ===== 运行主控制函数 =====
    try:
        mpc_robot_interactive(args)
    except Exception as e:
        print(f"致命错误: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
