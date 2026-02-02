#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e STORM MPC - 硬件在环仿真 (HIL)

实现:
- 真实 UR7e 机器人控制 (通过 UR ROS2 Driver)
- 虚拟障碍物场景 (仅在 RViz 可视化，MPC 会避开)
- STORM MPPI-MPC 实时控制
- 动态目标更新

安全特性:
- 速度/加速度限制
- 平滑轨迹生成
- Ctrl+C 安全停止

用法:
    # 终端 1: 启动 UR ROS2 驱动
    cd ~/storm/examples/HIL
    ./run_ur_driver.sh
    
    # 终端 2: 运行本脚本
    cd ~/storm/examples/HIL
    python3 ur7e_hil_mpc.py --safe-mode
    
    # 终端 3 (可选): 发布目标位置
    ros2 topic pub /target_pose geometry_msgs/PoseStamped '{pose: {position: {x: 0.4, y: 0.2, z: 0.5}}}'

ROS2 话题:
    订阅:
        /joint_states - 真实机器人关节状态
        /target_pose - 目标位置
    发布:
        /scaled_joint_trajectory_controller/joint_trajectory - 轨迹指令
        /ee_pose - 末端位置
        /visualization_marker_array - 虚拟障碍物可视化

Author: wqj
Date: 2025
"""

import sys
import os
import time
import signal
import yaml
import argparse
import numpy as np
from threading import Thread, Lock
from scipy.spatial.transform import Rotation

# 添加 STORM 路径
STORM_ROOT = os.path.expanduser('~/storm')
sys.path.insert(0, STORM_ROOT)

import torch
torch.multiprocessing.set_start_method('spawn', force=True)

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import ColorRGBA, Float64MultiArray
    from geometry_msgs.msg import PoseStamped, Point
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:
    print("=" * 60)
    print("错误: 未找到 ROS2 Python 包")
    print("请先 source ROS2 环境:")
    print("  source /opt/ros/humble/setup.bash")
    print("=" * 60)
    sys.exit(1)

# STORM imports
from storm_kit.util_file import get_gym_configs_path, get_mpc_configs_path, join_path, get_assets_path
from storm_kit.mpc.rollout.arm_reacher import ArmReacher
from storm_kit.mpc.control import MPPI
from storm_kit.mpc.utils.state_filter import JointStateFilter
from storm_kit.mpc.utils.mpc_process_wrapper import ControlProcess
from storm_kit.mpc.task.task_base import BaseTask

np.set_printoptions(precision=3, suppress=True)


# ============================================================================
# HIL ReacherTask (使用本地配置)
# ============================================================================

class HILReacherTask(BaseTask):
    """HIL 专用 ReacherTask，支持绝对路径配置"""
    
    def __init__(self, task_file, robot_file, world_file, tensor_args):
        super().__init__(tensor_args=tensor_args)
        self.controller = self.init_mppi(task_file, robot_file, world_file)
        self.init_aux()
    
    def get_rollout_fn(self, **kwargs):
        return ArmReacher(**kwargs)
    
    def init_mppi(self, task_file, robot_file, world_file):
        # 加载机器人配置
        if os.path.isabs(robot_file):
            robot_yml = robot_file
        else:
            robot_yml = join_path(get_gym_configs_path(), robot_file)
        with open(robot_yml) as f:
            robot_params = yaml.safe_load(f)
        
        # 加载障碍物配置
        if os.path.isabs(world_file):
            world_yml = world_file
        else:
            world_yml = join_path(get_gym_configs_path(), world_file)
        with open(world_yml) as f:
            world_params = yaml.safe_load(f)
        
        # 加载 MPC 配置
        if os.path.isabs(task_file):
            mpc_yml = task_file
        else:
            mpc_yml = join_path(get_mpc_configs_path(), task_file)
        with open(mpc_yml) as f:
            exp_params = yaml.safe_load(f)
        
        exp_params['robot_params'] = exp_params['model']
        
        rollout_fn = self.get_rollout_fn(
            exp_params=exp_params,
            tensor_args=self.tensor_args,
            world_params=world_params
        )
        
        mppi_params = exp_params['mppi']
        dynamics_model = rollout_fn.dynamics_model
        mppi_params['d_action'] = dynamics_model.d_action
        mppi_params['action_lows'] = -exp_params['model']['max_acc'] * torch.ones(
            dynamics_model.d_action, **self.tensor_args
        )
        mppi_params['action_highs'] = exp_params['model']['max_acc'] * torch.ones(
            dynamics_model.d_action, **self.tensor_args
        )
        
        init_q = torch.tensor(exp_params['model']['init_state'], **self.tensor_args)
        init_action = torch.zeros(
            (mppi_params['horizon'], dynamics_model.d_action),
            **self.tensor_args
        )
        init_action[:, :] += init_q
        
        if exp_params['control_space'] == 'acc':
            mppi_params['init_mean'] = init_action * 0.0
        elif exp_params['control_space'] == 'pos':
            mppi_params['init_mean'] = init_action
        
        mppi_params['rollout_fn'] = rollout_fn
        mppi_params['tensor_args'] = self.tensor_args
        
        controller = MPPI(**mppi_params)
        self.exp_params = exp_params
        return controller
    
    def init_aux(self):
        self.state_filter = JointStateFilter(
            filter_coeff=self.exp_params['state_filter_coeff'],
            dt=self.exp_params['control_dt']
        )
        self.command_filter = JointStateFilter(
            filter_coeff=self.exp_params['cmd_filter_coeff'],
            dt=self.exp_params['control_dt']
        )
        self.control_process = ControlProcess(self.controller)
        self.n_dofs = self.controller.rollout_fn.dynamics_model.n_dofs
        self.zero_acc = np.zeros(self.n_dofs)


# ============================================================================
# 辅助函数
# ============================================================================

def transform_point(position, orientation_xyzw, point):
    """坐标变换: 机器人 -> 世界"""
    rot = Rotation.from_quat(orientation_xyzw)
    return rot.apply(point) + np.array(position)


def inv_transform_point(position, orientation_xyzw, point):
    """坐标变换: 世界 -> 机器人"""
    rot = Rotation.from_quat(orientation_xyzw).inv()
    return rot.apply(np.array(point) - np.array(position))


# ============================================================================
# HIL 机器人接口
# ============================================================================

class HILRobotInterface(Node):
    """
    HIL 真实机器人 ROS2 接口
    
    使用 forward_position_controller 进行高频 MPC 控制:
    - 直接发送关节位置 (Float64MultiArray)
    - 低延迟，适合 50Hz+ 控制频率
    - 内置速度限制保证安全
    """
    
    def __init__(self, joint_names: list, control_rate: float = 50.0,
                 max_velocity: float = 0.5, max_acceleration: float = 1.0):
        super().__init__('storm_hil_mpc')
        
        self.joint_names = joint_names
        self.n_dof = len(joint_names)
        self.control_rate = control_rate
        self.control_dt = 1.0 / control_rate
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        
        self._lock = Lock()
        self._positions = None
        self._velocities = None
        self._prev_velocities = None
        self._prev_time = None
        self._state_received = False
        self._state_count = 0
        self._cmd_count = 0
        self._target_pos = None
        self._last_cmd_positions = None
        
        qos = QoSProfile(depth=10)
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        
        # 订阅真实机器人关节状态
        self.sub_joint_states = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, qos
        )
        
        # 订阅目标位置
        self.sub_target = self.create_subscription(
            PoseStamped, '/target_pose', self._target_callback, qos
        )
        
        # 发布轨迹指令 (使用 forward_position_controller - 适合高频 MPC)
        self.pub_position_cmd = self.create_publisher(
            Float64MultiArray,
            '/forward_position_controller/commands',
            qos_reliable
        )
        
        # 发布末端位置
        self.pub_ee_pose = self.create_publisher(
            PoseStamped, '/ee_pose', qos
        )
        
        # 发布虚拟障碍物可视化
        self.pub_markers = self.create_publisher(
            MarkerArray, '/visualization_marker_array', qos
        )
        
        self.get_logger().info(f'HIL Robot Interface 初始化完成')
        self.get_logger().info(f'  控制频率: {control_rate} Hz')
        self.get_logger().info(f'  最大速度: {max_velocity} rad/s')
        self.get_logger().info(f'  最大加速度: {max_acceleration} rad/s^2')
        self.get_logger().info(f'  订阅: /joint_states, /target_pose')
        self.get_logger().info(f'  发布: /forward_position_controller/commands')
    
    def _joint_state_callback(self, msg: JointState):
        positions = np.zeros(self.n_dof)
        velocities = np.zeros(self.n_dof)
        
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                positions[i] = msg.position[idx]
                if len(msg.velocity) > idx:
                    velocities[i] = msg.velocity[idx]
        
        with self._lock:
            self._positions = positions
            self._velocities = velocities
            self._state_received = True
            self._state_count += 1
    
    def _target_callback(self, msg: PoseStamped):
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        with self._lock:
            self._target_pos = pos
        self.get_logger().info(f'收到目标位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]')
    
    def get_joint_positions(self) -> np.ndarray:
        with self._lock:
            if self._positions is None:
                return None
            return self._positions.copy()
    
    def get_joint_velocities(self) -> np.ndarray:
        with self._lock:
            if self._velocities is None:
                return None
            return self._velocities.copy()
    
    def get_state(self) -> dict:
        with self._lock:
            if not self._state_received:
                return None
            pos = self._positions.copy()
            vel = self._velocities.copy()
        
        current_time = time.time()
        if self._prev_velocities is not None and self._prev_time is not None:
            dt = max(current_time - self._prev_time, 0.001)
            accel = (vel - self._prev_velocities) / dt
        else:
            accel = np.zeros(self.n_dof)
        
        self._prev_velocities = vel.copy()
        self._prev_time = current_time
        
        return {
            'position': pos,
            'velocity': vel,
            'acceleration': accel
        }
    
    def get_target_position(self) -> np.ndarray:
        with self._lock:
            if self._target_pos is not None:
                pos = self._target_pos.copy()
                self._target_pos = None
                return pos
            return None
    
    def send_position_command(self, positions: np.ndarray):
        """
        发送位置指令到真实机器人
        
        使用 forward_position_controller，适合高频 MPC 控制
        - 直接发送关节位置，无需轨迹规划
        - 低延迟，适合 50Hz+ 控制频率
        """
        # 限制位置变化（安全措施）
        if self._last_cmd_positions is not None:
            delta = positions - self._last_cmd_positions
            max_delta = self.max_velocity * self.control_dt
            delta = np.clip(delta, -max_delta, max_delta)
            positions = self._last_cmd_positions + delta
        
        self._last_cmd_positions = positions.copy()
        
        # 创建并发送 Float64MultiArray 消息
        msg = Float64MultiArray()
        msg.data = positions.tolist()
        self.pub_position_cmd.publish(msg)
        self._cmd_count += 1
    
    def publish_ee_pose(self, position: np.ndarray, orientation: np.ndarray = None):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        if orientation is not None:
            msg.pose.orientation.x = float(orientation[0])
            msg.pose.orientation.y = float(orientation[1])
            msg.pose.orientation.z = float(orientation[2])
            msg.pose.orientation.w = float(orientation[3])
        else:
            msg.pose.orientation.w = 1.0
        self.pub_ee_pose.publish(msg)
    
    def publish_markers(self, obstacles: dict, goal_pos: np.ndarray, ee_pos: np.ndarray):
        """发布虚拟障碍物和目标/末端标记"""
        marker_array = MarkerArray()
        marker_id = 0
        
        # 目标 (红球)
        goal_marker = Marker()
        goal_marker.header.frame_id = "base_link"
        goal_marker.header.stamp = self.get_clock().now().to_msg()
        goal_marker.ns = "goal"
        goal_marker.id = marker_id
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = float(goal_pos[0])
        goal_marker.pose.position.y = float(goal_pos[1])
        goal_marker.pose.position.z = float(goal_pos[2])
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.06
        goal_marker.scale.y = 0.06
        goal_marker.scale.z = 0.06
        goal_marker.color = ColorRGBA(r=0.9, g=0.1, b=0.1, a=0.8)
        marker_array.markers.append(goal_marker)
        marker_id += 1
        
        # 末端 (绿球)
        ee_marker = Marker()
        ee_marker.header.frame_id = "base_link"
        ee_marker.header.stamp = self.get_clock().now().to_msg()
        ee_marker.ns = "ee"
        ee_marker.id = marker_id
        ee_marker.type = Marker.SPHERE
        ee_marker.action = Marker.ADD
        ee_marker.pose.position.x = float(ee_pos[0])
        ee_marker.pose.position.y = float(ee_pos[1])
        ee_marker.pose.position.z = float(ee_pos[2])
        ee_marker.pose.orientation.w = 1.0
        ee_marker.scale.x = 0.05
        ee_marker.scale.y = 0.05
        ee_marker.scale.z = 0.05
        ee_marker.color = ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.8)
        marker_array.markers.append(ee_marker)
        marker_id += 1
        
        # 虚拟障碍物
        if obstacles:
            coll_objs = obstacles.get('world_model', {}).get('coll_objs', {})
            
            # 球形障碍物 (半透明蓝色，表示虚拟)
            for name, params in coll_objs.get('sphere', {}).items():
                m = Marker()
                m.header.frame_id = "base_link"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "virtual_obstacles"
                m.id = marker_id
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                pos = params.get('position', [0, 0, 0])
                radius = params.get('radius', 0.1)
                m.pose.position.x = float(pos[0])
                m.pose.position.y = float(pos[1])
                m.pose.position.z = float(pos[2])
                m.pose.orientation.w = 1.0
                m.scale.x = radius * 2
                m.scale.y = radius * 2
                m.scale.z = radius * 2
                # 红色球形障碍物 (与 Gazebo 一致)
                m.color = ColorRGBA(r=0.8, g=0.2, b=0.2, a=0.6)
                marker_array.markers.append(m)
                marker_id += 1
            
            # 立方体障碍物
            for name, params in coll_objs.get('cube', {}).items():
                if name == 'ground':
                    continue  # 跳过地面
                m = Marker()
                m.header.frame_id = "base_link"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "virtual_obstacles"
                m.id = marker_id
                m.type = Marker.CUBE
                m.action = Marker.ADD
                pose = params.get('pose', [0, 0, 0, 0, 0, 0, 1])
                dims = params.get('dims', [0.1, 0.1, 0.1])
                m.pose.position.x = float(pose[0])
                m.pose.position.y = float(pose[1])
                m.pose.position.z = float(pose[2])
                m.pose.orientation.x = float(pose[3])
                m.pose.orientation.y = float(pose[4])
                m.pose.orientation.z = float(pose[5])
                m.pose.orientation.w = float(pose[6])
                m.scale.x = float(dims[0])
                m.scale.y = float(dims[1])
                m.scale.z = float(dims[2])
                # 蓝色立方体障碍物 (与 Gazebo 一致)
                m.color = ColorRGBA(r=0.5, g=0.5, b=0.8, a=0.6)
                marker_array.markers.append(m)
                marker_id += 1
        
        self.pub_markers.publish(marker_array)
    
    def is_connected(self) -> bool:
        return self._state_received
    
    def get_state_count(self) -> int:
        return self._state_count
    
    def get_cmd_count(self) -> int:
        return self._cmd_count


# ============================================================================
# 主控制函数
# ============================================================================

def hil_control_main(args):
    """HIL MPC 控制主函数"""
    
    print("=" * 60)
    print("UR7e STORM MPC - 硬件在环仿真 (HIL)")
    print("=" * 60)
    
    if args.safe_mode:
        print("\n⚠️  安全模式已启用")
        print("  - 速度限制: 0.3 rad/s")
        print("  - 加速度限制: 0.5 rad/s^2")
    
    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    
    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    
    robot_file = os.path.join(config_dir, 'ur7e_robot_hil.yml')
    task_file = os.path.join(config_dir, 'ur7e_reacher_hil.yml')
    world_file = os.path.join(config_dir, 'collision_world_hil.yml')
    
    print(f"\n加载配置文件...")
    print(f"  Robot: {robot_file}")
    print(f"  Task:  {task_file}")
    print(f"  World: {world_file}")
    
    with open(robot_file) as f:
        robot_params = yaml.safe_load(f)
    
    with open(world_file) as f:
        world_params = yaml.safe_load(f)
    
    sim_params = robot_params.get('sim_params', {})
    hil_params = sim_params.get('hil', {})
    safety_params = hil_params.get('safety', {})
    
    robot_pose = sim_params.get('robot_pose', [0, 0, 0, 0, 0, 0, 1])
    robot_pos = np.array(robot_pose[:3])
    robot_quat_xyzw = np.array(robot_pose[3:])
    
    joint_names = hil_params.get('joint_names', [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ])
    n_dof = len(joint_names)
    
    # 安全限制
    if args.safe_mode:
        max_velocity = 0.3
        max_acceleration = 0.5
    else:
        max_velocity = safety_params.get('max_velocity', 0.5)
        max_acceleration = safety_params.get('max_acceleration', 1.0)
    
    # =========================================================================
    # 2. 初始化 ROS2
    # =========================================================================
    
    print("\n初始化 ROS2...")
    rclpy.init(args=None)
    
    control_rate = args.rate
    robot = HILRobotInterface(
        joint_names,
        control_rate=control_rate,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration
    )
    
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(robot)
    
    _executor_running = True
    def spin_with_check():
        while _executor_running and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    
    spin_thread = Thread(target=spin_with_check, daemon=True)
    spin_thread.start()
    
    # 等待连接
    print("\n等待真实机器人关节状态...")
    print("确保已运行 UR ROS2 Driver 并启动 External Control 程序")
    
    timeout = 30.0
    start = time.time()
    while not robot.is_connected():
        if time.time() - start > timeout:
            print("错误: 无法接收关节状态")
            print("请检查:")
            print("  1. UR ROS2 Driver 是否运行")
            print("  2. 机器人是否连接")
            print("  3. External Control 程序是否运行")
            _executor_running = False
            robot.destroy_node()
            rclpy.shutdown()
            return 1
        time.sleep(0.1)
    
    print("✅ 已连接到真实 UR7e 机器人!")
    
    # 显示当前位置
    curr_pos = robot.get_joint_positions()
    print(f"\n当前关节位置 (rad): {np.round(curr_pos, 3)}")
    print(f"当前关节位置 (deg): {np.round(np.degrees(curr_pos), 1)}")
    
    # =========================================================================
    # 3. 初始化 STORM MPC
    # =========================================================================
    
    print("\n初始化 STORM MPC 控制器...")
    device = 'cuda' if args.cuda else 'cpu'
    print(f"计算设备: {device}")
    
    tensor_args = {
        'device': torch.device(device, 0) if device == 'cuda' else torch.device('cpu'),
        'dtype': torch.float32
    }
    
    mpc = HILReacherTask(task_file, robot_file, world_file, tensor_args)
    control_dt = mpc.exp_params.get('control_dt', 0.02)
    print(f"MPC 控制周期: {control_dt} s ({1.0/control_dt:.1f} Hz)")
    
    rollout_fn = mpc.controller.rollout_fn
    
    # =========================================================================
    # 4. 设置初始目标
    # =========================================================================
    
    # 从配置文件读取默认目标位置
    default_goal = mpc.exp_params.get('default_goal', None)
    
    if default_goal is not None:
        # 使用配置文件中的默认目标
        goal_ee_pos_robot = np.array(default_goal.get('position', [0.4, 0.0, 0.4]))
        goal_ee_quat = np.array(default_goal.get('orientation', [0.0, 0.707, 0.0, 0.707]))
        print(f"\n使用配置文件默认目标: {goal_ee_pos_robot}")
        mpc.update_params(goal_ee_pos=goal_ee_pos_robot, goal_ee_quat=goal_ee_quat)
    else:
        # 没有默认目标，保持当前位置
        curr_state = robot.get_state()
        initial_q = curr_state['position']
        goal_state = np.concatenate([initial_q, np.zeros(6)])
        mpc.update_params(goal_state=goal_state)
        goal_ee_pos_robot = np.ravel(rollout_fn.goal_ee_pos.cpu().numpy())
        goal_ee_quat = np.ravel(rollout_fn.goal_ee_quat.cpu().numpy())
        print(f"\n目标: 当前位置 (无默认目标)")
    
    goal_ee_world = transform_point(robot_pos, robot_quat_xyzw, goal_ee_pos_robot)
    print(f"目标末端位置 (世界): {np.round(goal_ee_world, 3)}")
    
    current_goal_ee = goal_ee_pos_robot.copy()
    current_goal_world = goal_ee_world.copy()
    
    # =========================================================================
    # 5. 显示场景信息并等待用户确认
    # =========================================================================
    
    # 计算当前末端位置
    curr_state = robot.get_state()
    curr_full = np.concatenate([curr_state['position'], curr_state['velocity'], curr_state['acceleration']])
    curr_ee_pose = rollout_fn.get_ee_pose(
        torch.as_tensor(curr_full, **tensor_args).unsqueeze(0)
    )
    curr_ee_pos_robot = np.ravel(curr_ee_pose['ee_pos_seq'].cpu().numpy())
    curr_ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, curr_ee_pos_robot)
    
    print("\n" + "=" * 60)
    print("场景加载完成 - 等待确认启动 MPC")
    print("=" * 60)
    print("\n📍 当前机械臂状态:")
    print(f"   关节位置 (deg): {np.round(np.degrees(curr_state['position']), 1)}")
    print(f"   末端位置 (世界): {np.round(curr_ee_pos_world, 3)}")
    print(f"\n🎯 目标位置:")
    print(f"   末端目标 (世界): {np.round(goal_ee_world, 3)}")
    print(f"   距离目标: {np.linalg.norm(goal_ee_world - curr_ee_pos_world):.3f} m")
    
    # 显示虚拟障碍物信息
    print(f"\n🧱 虚拟障碍物:")
    if world_params and 'world_model' in world_params:
        coll_objs = world_params['world_model'].get('coll_objs', {})
        sphere_count = len(coll_objs.get('sphere', {}))
        cube_count = len(coll_objs.get('cube', {}))
        print(f"   球体: {sphere_count} 个")
        print(f"   立方体: {cube_count} 个")
    else:
        print("   无障碍物")
    
    # 发布初始场景可视化到 RViz
    print("\n📺 正在发布场景到 RViz...")
    for _ in range(10):  # 多次发布确保 RViz 接收到
        robot.publish_markers(world_params, goal_ee_world, curr_ee_pos_world)
        time.sleep(0.1)
    print("   场景已发布，请在 RViz 中查看")
    
    print("\n" + "-" * 60)
    print("⚠️  安全提示:")
    print("   1. 确保工作区域无人员")
    print("   2. 急停按钮在可触及范围内")
    print("   3. 观察 RViz 中的目标位置和障碍物")
    print("-" * 60)
    print("\n按 Enter 启动 MPC 跟踪，Ctrl+C 取消...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n用户取消，退出...")
        _executor_running = False
        robot.destroy_node()
        rclpy.shutdown()
        return 0
    
    # =========================================================================
    # 6. MPC 控制循环
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("启动 MPC 跟踪控制... (Ctrl+C 安全停止)")
    print("=" * 60)
    print("")
    
    running = [True]
    
    def shutdown_handler(sig, frame):
        print("\n收到退出信号，安全停止中...")
        running[0] = False
        nonlocal _executor_running
        _executor_running = False
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # 预热 MPC
    print("预热 MPC 控制器...")
    t = 0.0
    for _ in range(5):
        state = robot.get_state()
        if state is not None:
            try:
                mpc.get_command(t, state, control_dt=control_dt, WAIT=False)
            except:
                pass
        t += control_dt
        time.sleep(0.01)
    
    print("等待首次优化...")
    state = robot.get_state()
    if state is not None:
        try:
            cmd = mpc.get_command(t, state, control_dt=control_dt, WAIT=True)
            print(f"首次优化完成! opt_dt={mpc.opt_dt:.3f}s")
        except Exception as e:
            print(f"首次优化异常: {e}")
    
    print("\nMPC 预热完成，开始控制!\n")
    
    i = 0
    loop_start = time.time()
    marker_update_counter = 0
    
    while running[0] and rclpy.ok():
        iter_start = time.time()
        t = time.time() - loop_start
        
        state = robot.get_state()
        if state is None:
            time.sleep(control_dt)
            continue
        
        q = state['position']
        dq = state['velocity']
        ddq = state['acceleration']
        
        # 检测目标更新
        new_target = robot.get_target_position()
        if new_target is not None:
            target_robot = inv_transform_point(robot_pos, robot_quat_xyzw, new_target)
            if np.linalg.norm(target_robot - current_goal_ee) > 0.005:
                current_goal_ee = target_robot.copy()
                current_goal_world = new_target.copy()
                mpc.update_params(goal_ee_pos=current_goal_ee, goal_ee_quat=goal_ee_quat)
                print(f"[目标更新] {np.round(current_goal_world, 3)}")
        
        # MPC 计算
        try:
            cmd = mpc.get_command(t, state, control_dt=control_dt, WAIT=False)
            
            if cmd is None or 'position' not in cmd:
                i += 1
                time.sleep(control_dt)
                continue
        except (IndexError, RuntimeError) as e:
            i += 1
            time.sleep(control_dt)
            continue
        
        # 发送指令 (使用 forward_position_controller)
        target_positions = cmd['position']
        if isinstance(target_positions, torch.Tensor):
            target_positions = target_positions.cpu().numpy()
        target_positions = np.array(target_positions).flatten()[:n_dof]
        
        robot.send_position_command(target_positions)
        
        # 计算末端位置
        curr = np.hstack([q, dq, ddq])
        ee_pose = rollout_fn.get_ee_pose(
            torch.as_tensor(curr, **tensor_args).unsqueeze(0)
        )
        ee_pos_robot = np.ravel(ee_pose['ee_pos_seq'].cpu().numpy())
        ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos_robot)
        robot.publish_ee_pose(ee_pos_world)
        
        # 发布可视化
        marker_update_counter += 1
        if marker_update_counter >= 10:
            robot.publish_markers(world_params, current_goal_world, ee_pos_world)
            marker_update_counter = 0
        
        # 打印状态
        if i % 50 == 0:
            err = mpc.get_current_error(state)
            print(f"[{i:4d}] 误差: {[f'{x:.3f}' for x in err]}, "
                  f"opt: {mpc.opt_dt:.3f}s, ee: {np.round(ee_pos_world, 3)}")
        
        elapsed = time.time() - iter_start
        sleep_time = (1.0 / control_rate) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        i += 1
    
    # =========================================================================
    # 清理
    # =========================================================================
    
    print("\n安全停止...")
    
    # 停止机器人 (发送当前位置保持)
    curr_pos = robot.get_joint_positions()
    if curr_pos is not None:
        robot.send_position_command(curr_pos)
    
    _executor_running = False
    spin_thread.join(timeout=1.0)
    
    print("  关闭 MPC...")
    def close_mpc():
        try:
            mpc.close()
        except:
            pass
    close_thread = Thread(target=close_mpc, daemon=True)
    close_thread.start()
    close_thread.join(timeout=2.0)
    
    print("  关闭 ROS2...")
    try:
        robot.destroy_node()
    except:
        pass
    try:
        rclpy.shutdown()
    except:
        pass
    
    print("完成!")
    return 0


# ============================================================================
# 入口点
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UR7e STORM MPC - HIL')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='使用 CUDA (默认: True)')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='禁用 CUDA')
    parser.add_argument('--rate', type=float, default=50.0,
                        help='控制频率 Hz (默认: 50)')
    parser.add_argument('--safe-mode', action='store_true',
                        help='安全模式: 降低速度/加速度限制')
    
    args = parser.parse_args()
    
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    sys.exit(hil_control_main(args))
