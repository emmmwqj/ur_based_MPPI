#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e STORM MPC Reach Static Target - Gazebo 仿真

完整的 Reach 任务实现：
- STORM MPPI-MPC 控制器
- 动态目标更新（通过 /target_pose 话题或 RViz InteractiveMarker）
- 末端位置实时计算和发布
- 障碍物场景可视化（RViz Markers）
- 与 SIL 方案功能一致，使用系统 ROS2 通信

用法:
    # 终端 1: 启动 Gazebo 仿真
    cd ~/storm/examples/sim_gazebo
    ./run_gazebo.sh
    
    # 终端 2: 运行本脚本
    cd ~/storm/examples/sim_gazebo
    python3 reach_static_ur7e.py
    
    # 终端 3 (可选): 发布目标位置
    ros2 topic pub /target_pose geometry_msgs/PoseStamped '{pose: {position: {x: 0.4, y: 0.2, z: 0.5}}}'

ROS2 话题:
    订阅:
        /joint_states (sensor_msgs/JointState) - 关节状态
        /target_pose (geometry_msgs/PoseStamped) - 目标位置
    发布:
        /forward_position_controller/commands (Float64MultiArray) - 关节指令
        /ee_pose (geometry_msgs/PoseStamped) - 末端位置
        /visualization_marker_array (MarkerArray) - 障碍物和目标可视化

Author: wqj
Date: 2025
"""

import sys
import os
import time
import signal
import queue
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
    from std_msgs.msg import Float64MultiArray, ColorRGBA
    from geometry_msgs.msg import PoseStamped, Point, Pose, Vector3
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

FORWARD_POSITION_CMD_TOPIC = '/forward_position_controller/commands'


def _drain_mp_queue(mp_queue) -> None:
    if mp_queue is None:
        return
    while True:
        try:
            mp_queue.get_nowait()
        except queue.Empty:
            break
        except Exception:
            break


def _reset_control_process_timing(control_process, t_step: float, control_dt: float) -> None:
    if control_process is None:
        return
    control_process.command = None
    control_process.command_tstep = control_process.traj_tstep + t_step
    control_process.prev_mpc_tstep = max(0.0, t_step - control_dt)
    control_process.mpc_dt = control_dt
    _drain_mp_queue(getattr(control_process, 'result_queue', None))
    _drain_mp_queue(getattr(control_process, 'opt_queue', None))


def _recover_command(mpc, t_step: float, state: dict, control_dt: float):
    _reset_control_process_timing(getattr(mpc, 'control_process', None), t_step, control_dt)
    return _get_sync_command(mpc, t_step, state, control_dt)


def _get_execution_mode(mpc) -> str:
    mppi_cfg = getattr(mpc, 'exp_params', {}).get('mppi', {})
    mode = str(mppi_cfg.get('execution_mode', 'best_sample')).strip().lower()
    if mode not in ('best_sample', 'mean'):
        return 'best_sample'
    return mode


def _get_sync_command(mpc, t_step: float, state: dict, control_dt: float):
    filt_state = mpc.state_filter.filter_joint_state(state)
    state_tensor = mpc._state_to_tensor(filt_state)
    next_command, _, _, _ = mpc.control_process.get_command_debug(
        t_step,
        state_tensor.numpy(),
        control_dt=control_dt,
    )

    qdd_des = np.asarray(next_command, dtype=np.float64)
    if _get_execution_mode(mpc) == 'best_sample' and mpc.exp_params.get('control_space', 'acc') == 'acc':
        best_traj = getattr(mpc.controller, 'best_traj', None)
        if best_traj is not None:
            if isinstance(best_traj, torch.Tensor):
                best_traj_np = best_traj.detach().cpu().numpy()
            else:
                best_traj_np = np.asarray(best_traj)
            if best_traj_np.ndim == 2 and best_traj_np.shape[0] > 0 and best_traj_np.shape[1] == mpc.n_dofs:
                qdd_des = np.asarray(best_traj_np[0], dtype=np.float64)
                if getattr(mpc.control_process, 'command', None) is not None:
                    mpc.control_process.command[0] = best_traj_np

    mpc.prev_qdd_des = qdd_des
    cmd_des = mpc.state_filter.integrate_acc(qdd_des)
    return cmd_des


# ============================================================================
# 自定义 Gazebo ReacherTask (使用本地配置文件)
# ============================================================================

class GazeboReacherTask(BaseTask):
    """
    Gazebo 专用 ReacherTask
    
    与原始 ReacherTask 相比，支持绝对路径配置文件加载
    """
    
    def __init__(self, task_file, robot_file, world_file, tensor_args):
        super().__init__(tensor_args=tensor_args)
        self.controller = self.init_mppi(task_file, robot_file, world_file)
        self.init_aux()
    
    def get_rollout_fn(self, **kwargs):
        return ArmReacher(**kwargs)
    
    def init_mppi(self, task_file, robot_file, world_file):
        """初始化 MPPI 控制器，支持绝对路径"""
        
        # 加载机器人配置 (支持绝对路径)
        if os.path.isabs(robot_file):
            robot_yml = robot_file
        else:
            robot_yml = join_path(get_gym_configs_path(), robot_file)
        
        with open(robot_yml) as f:
            robot_params = yaml.safe_load(f)
        
        # 加载世界/障碍物配置 (支持绝对路径)
        if os.path.isabs(world_file):
            world_yml = world_file
        else:
            world_yml = join_path(get_gym_configs_path(), world_file)
        
        with open(world_yml) as f:
            world_params = yaml.safe_load(f)
        
        # 加载 MPC 任务配置 (支持绝对路径)
        if os.path.isabs(task_file):
            mpc_yml = task_file
        else:
            mpc_yml = join_path(get_mpc_configs_path(), task_file)
        
        with open(mpc_yml) as f:
            exp_params = yaml.safe_load(f)
        
        exp_params['robot_params'] = exp_params['model']
        
        # 创建 rollout 函数
        rollout_fn = self.get_rollout_fn(
            exp_params=exp_params,
            tensor_args=self.tensor_args,
            world_params=world_params
        )
        
        # 配置 MPPI 参数
        mppi_params = dict(exp_params['mppi'])
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
        mppi_params.pop('execution_mode', None)
        
        controller = MPPI(**mppi_params)
        self.exp_params = exp_params
        
        return controller
    
    def init_aux(self):
        """初始化辅助组件（状态滤波器、控制进程）"""
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

    def set_position_only_goal_mode(self) -> None:
        """Disable orientation tracking for Cartesian reach targets."""
        rollout_fn = getattr(self.controller, 'rollout_fn', None)
        goal_cost = getattr(rollout_fn, 'goal_cost', None)
        if goal_cost is None:
            return
        if isinstance(goal_cost.weight, (list, tuple)):
            goal_cost.weight = [0.0, float(goal_cost.weight[1])]
        else:
            goal_cost.weight[0] = 0.0


# ============================================================================
# 辅助函数
# ============================================================================

def transform_point(position, orientation_xyzw, point):
    """将点从机器人坐标系变换到世界坐标系"""
    rot = Rotation.from_quat(orientation_xyzw)
    return rot.apply(point) + np.array(position)


def inv_transform_point(position, orientation_xyzw, point):
    """将点从世界坐标系变换到机器人坐标系"""
    rot = Rotation.from_quat(orientation_xyzw).inv()
    return rot.apply(np.array(point) - np.array(position))


class CollisionSphereVisualizer:
    def __init__(self, robot_collision_params: dict):
        sphere_config = os.path.expanduser(robot_collision_params["collision_spheres"])
        if not os.path.isabs(sphere_config):
            sphere_config = join_path(get_mpc_configs_path(), sphere_config)

        with open(sphere_config) as f:
            sphere_params = yaml.safe_load(f)

        self.link_names = list(robot_collision_params["link_objs"])
        raw_spheres_by_link = sphere_params["collision_spheres"]
        self.spheres_by_link = {}
        marker_id = 0
        for link_name in self.link_names:
            self.spheres_by_link[link_name] = []
            for sphere in raw_spheres_by_link.get(link_name, []):
                sphere_entry = dict(sphere)
                sphere_entry["marker_id"] = marker_id
                self.spheres_by_link[link_name].append(sphere_entry)
                marker_id += 1
        self.total_sphere_count = marker_id

    def get_world_spheres(
        self,
        link_pos_robot: np.ndarray,
        link_rot_robot: np.ndarray,
        robot_pos_world: np.ndarray,
        robot_quat_xyzw: np.ndarray,
    ):
        world_spheres = []
        for link_idx, link_name in enumerate(self.link_names):
            link_pos = link_pos_robot[link_idx]
            link_rot = link_rot_robot[link_idx]
            for sphere in self.spheres_by_link.get(link_name, []):
                center_local = np.asarray(sphere["center"], dtype=np.float64)
                center_robot = link_rot @ center_local + link_pos
                center_world = transform_point(robot_pos_world, robot_quat_xyzw, center_robot)
                world_spheres.append(
                    {
                        "marker_id": int(sphere["marker_id"]),
                        "link_name": link_name,
                        "center_world": center_world,
                        "radius": float(sphere["radius"]),
                    }
                )
        return world_spheres


def _compute_link_poses_robot_frame(rollout_fn, q: np.ndarray, dq: np.ndarray, tensor_args: dict):
    q_tensor = torch.as_tensor(q, **tensor_args).unsqueeze(0)
    dq_tensor = torch.as_tensor(dq, **tensor_args).unsqueeze(0)
    robot_model = rollout_fn.dynamics_model.robot_model
    robot_model.compute_fk_and_jacobian(
        q_tensor,
        dq_tensor,
        rollout_fn.exp_params["model"]["ee_link_name"],
    )

    link_pos_robot = []
    link_rot_robot = []
    for link_name in rollout_fn.dynamics_model.link_names:
        link_pos, link_rot = robot_model.get_link_pose(link_name)
        link_pos_robot.append(link_pos[0].detach().cpu().numpy())
        link_rot_robot.append(link_rot[0].detach().cpu().numpy())

    return np.stack(link_pos_robot, axis=0), np.stack(link_rot_robot, axis=0)


def _get_top_ee_trajs_world(
    mpc,
    robot_pos_world: np.ndarray,
    robot_quat_xyzw: np.ndarray,
    current_ee_pos_world: np.ndarray,
    max_trajs: int = 5,
):
    # In the default STORM async chain, top trajectories are produced by the
    # background ControlProcess and stored on the task wrapper (`mpc.top_trajs`),
    # not on `mpc.controller.top_trajs`. Reading the controller field first can
    # leave RViz stuck on stale warm-up data, which makes the displayed
    # trajectories look like they start from the initial EE pose instead of the
    # live EE pose.
    top_trajs = getattr(mpc, "top_trajs", None)
    if top_trajs is None:
        controller = getattr(mpc, "controller", None)
        top_trajs = getattr(controller, "top_trajs", None)
    if top_trajs is not None:
        if isinstance(top_trajs, torch.Tensor):
            top_trajs_np = top_trajs.detach().cpu().numpy()
        else:
            top_trajs_np = np.asarray(top_trajs)
        if top_trajs_np.ndim == 2:
            top_trajs_np = top_trajs_np[None, ...]
    else:
        trajectories = getattr(controller, "trajectories", None)
        total_costs = getattr(controller, "total_costs", None)
        if trajectories is None or total_costs is None:
            return None

        ee_pos_seq = trajectories.get("ee_pos_seq", None)
        if ee_pos_seq is None:
            return None

        if isinstance(ee_pos_seq, torch.Tensor):
            ee_pos_seq_np = ee_pos_seq.detach().cpu().numpy()
        else:
            ee_pos_seq_np = np.asarray(ee_pos_seq)

        if isinstance(total_costs, torch.Tensor):
            total_costs_np = total_costs.detach().cpu().numpy()
        else:
            total_costs_np = np.asarray(total_costs)

        if ee_pos_seq_np.ndim != 3 or ee_pos_seq_np.shape[-1] != 3 or total_costs_np.ndim != 1:
            return None

        top_count = min(max_trajs, ee_pos_seq_np.shape[0], total_costs_np.shape[0])
        if top_count <= 0:
            return None

        top_indices = np.argsort(total_costs_np)[:top_count]
        top_trajs_np = ee_pos_seq_np[top_indices]

    if top_trajs_np.ndim != 3 or top_trajs_np.shape[-1] != 3:
        return None

    top_count = min(max_trajs, top_trajs_np.shape[0])
    if top_count <= 0:
        return None
    top_trajs_np = top_trajs_np[:top_count]

    current_ee_pos_world = np.asarray(current_ee_pos_world, dtype=np.float64).reshape(1, 3)
    world_trajs = []
    for traj_points in top_trajs_np:
        traj_points_world = transform_point(robot_pos_world, robot_quat_xyzw, traj_points)
        if traj_points_world.ndim != 2 or traj_points_world.shape[-1] != 3:
            continue
        if len(traj_points_world) == 0:
            continue
        if np.linalg.norm(traj_points_world[0] - current_ee_pos_world[0]) < 1.0e-4:
            stitched = traj_points_world
        else:
            stitched = np.concatenate([current_ee_pos_world, traj_points_world], axis=0)
        world_trajs.append(stitched)

    if not world_trajs:
        return None
    return world_trajs


# ============================================================================
# Gazebo ROS2 机器人接口
# ============================================================================

class GazeboRobotInterface(Node):
    """
    Gazebo 机器人 ROS2 接口
    
    订阅:
        /joint_states - 关节状态
        /target_pose - 目标位置（用于动态更新目标）
    发布:
        /forward_position_controller/commands - 关节位置指令
        /ee_pose - 末端位置
        /visualization_marker_array - 可视化标记
    """
    
    def __init__(self, joint_names: list, control_rate: float = 50.0):
        super().__init__('storm_mpc_reach_static')
        
        self.joint_names = joint_names
        self.n_dof = len(joint_names)
        self.control_rate = control_rate
        self.control_dt = 1.0 / control_rate
        
        # 线程锁
        self._lock = Lock()
        
        # 当前状态
        self._positions = None
        self._velocities = None
        self._prev_velocities = None
        self._prev_time = None
        self._state_received = False
        self._state_count = 0
        self._cmd_count = 0
        self._runtime_control_topic = FORWARD_POSITION_CMD_TOPIC
        
        # 目标位置（从 ROS2 接收）
        self._target_pos = None
        
        # QoS 配置
        qos = QoSProfile(depth=10)
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        
        # 订阅关节状态
        self.sub_joint_states = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, qos
        )
        
        # 订阅目标位置（用于动态更新目标）
        self.sub_target = self.create_subscription(
            PoseStamped, '/target_pose', self._target_callback, qos
        )
        
        # 发布位置指令 (ForwardPositionController)
        self.pub_position_cmd = self.create_publisher(
            Float64MultiArray, self._runtime_control_topic, qos_reliable
        )
        
        # 发布末端位置
        self.pub_ee_pose = self.create_publisher(
            PoseStamped, '/ee_pose', qos
        )
        
        # 发布可视化标记
        self.pub_markers = self.create_publisher(
            MarkerArray, '/visualization_marker_array', qos
        )
        self.pub_top_traj_markers = self.create_publisher(
            MarkerArray, '/mppi_top_traj_markers', qos
        )
        self.pub_collision_sphere_markers = self.create_publisher(
            MarkerArray, '/collision_sphere_markers', qos
        )
        self._prev_collision_marker_count = 0
        
        self.get_logger().info(f'Gazebo Robot Interface 初始化完成')
        self.get_logger().info(f'  控制频率: {control_rate} Hz')
        self.get_logger().info(f'  关节数: {self.n_dof}')
        self.get_logger().info(f'  订阅: /joint_states, /target_pose')
        self.get_logger().info(f'  发布: {self._runtime_control_topic}, /ee_pose, /visualization_marker_array')
        self.get_logger().info('  额外发布: /mppi_top_traj_markers')
        self.get_logger().info('  额外发布: /collision_sphere_markers')
        self.get_logger().info(
            '  运行时关节控制约束: 仅通过 ros2_control/forward_position_controller 下发命令; '
            '除 Gazebo 初始姿态外, 不直接设置机械臂关节位置'
        )
    
    def _joint_state_callback(self, msg: JointState):
        """处理关节状态消息"""
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
        """处理目标位置消息"""
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        with self._lock:
            self._target_pos = pos
        self.get_logger().info(f'收到目标位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]')
    
    def get_joint_positions(self) -> np.ndarray:
        """获取当前关节位置"""
        with self._lock:
            if self._positions is None:
                return None
            return self._positions.copy()
    
    def get_joint_velocities(self) -> np.ndarray:
        """获取当前关节速度"""
        with self._lock:
            if self._velocities is None:
                return None
            return self._velocities.copy()
    
    def get_state(self) -> dict:
        """
        获取完整机器人状态
        
        Returns:
            dict: {'position': np.array, 'velocity': np.array, 'acceleration': np.array}
        """
        with self._lock:
            if not self._state_received:
                return None
            
            pos = self._positions.copy()
            vel = self._velocities.copy()
        
        # 估算加速度
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
        """获取目标位置（来自 ROS2 话题，读取后清除）"""
        with self._lock:
            if self._target_pos is not None:
                pos = self._target_pos.copy()
                self._target_pos = None
                return pos
            return None
    
    def send_position_command(self, positions: np.ndarray):
        """发送关节位置指令"""
        if self._runtime_control_topic != FORWARD_POSITION_CMD_TOPIC:
            raise RuntimeError(
                f'非法控制话题: {self._runtime_control_topic}, 仅允许 {FORWARD_POSITION_CMD_TOPIC}'
            )
        msg = Float64MultiArray()
        msg.data = positions.tolist()
        self.pub_position_cmd.publish(msg)
        self._cmd_count += 1
    
    def publish_ee_pose(self, position: np.ndarray, orientation: np.ndarray = None):
        """发布末端位置"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
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

    def publish_live_goal_ee_markers(self, goal_pos: np.ndarray, ee_pos: np.ndarray):
        """发布轻量目标/末端 marker，供高频 RViz 刷新使用。"""
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        goal_marker = Marker()
        goal_marker.header.frame_id = "world"
        goal_marker.header.stamp = stamp
        goal_marker.ns = "goal"
        goal_marker.id = 0
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

        ee_marker = Marker()
        ee_marker.header.frame_id = "world"
        ee_marker.header.stamp = stamp
        ee_marker.ns = "ee"
        ee_marker.id = 1
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

        self.pub_markers.publish(marker_array)
    
    def publish_markers(
        self,
        obstacles: dict,
        goal_pos: np.ndarray,
        ee_pos: np.ndarray,
        collision_spheres=None,
    ):
        """
        发布可视化标记到 RViz
        
        Args:
            obstacles: 障碍物配置
            goal_pos: 目标位置 (世界坐标系)
            ee_pos: 末端位置 (世界坐标系)
        """
        marker_array = MarkerArray()
        marker_id = 0
        
        # 1. 目标标记（红色球）
        goal_marker = Marker()
        goal_marker.header.frame_id = "world"
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
        
        # 2. 末端标记（绿色球）
        ee_marker = Marker()
        ee_marker.header.frame_id = "world"
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
        
        # 3. 障碍物标记
        if obstacles:
            coll_objs = obstacles.get('world_model', {}).get('coll_objs', {})
            
            # 球体障碍物
            for name, params in coll_objs.get('sphere', {}).items():
                sphere_marker = Marker()
                sphere_marker.header.frame_id = "world"
                sphere_marker.header.stamp = self.get_clock().now().to_msg()
                sphere_marker.ns = "obstacles"
                sphere_marker.id = marker_id
                sphere_marker.type = Marker.SPHERE
                sphere_marker.action = Marker.ADD
                
                pos = params.get('position', [0, 0, 0])
                radius = params.get('radius', 0.1)
                
                sphere_marker.pose.position.x = float(pos[0])
                sphere_marker.pose.position.y = float(pos[1])
                sphere_marker.pose.position.z = float(pos[2])
                sphere_marker.pose.orientation.w = 1.0
                sphere_marker.scale.x = radius * 2
                sphere_marker.scale.y = radius * 2
                sphere_marker.scale.z = radius * 2
                if name == 'dynamic_ball':
                    sphere_marker.color = ColorRGBA(r=0.2, g=0.35, b=0.9, a=0.75)
                else:
                    sphere_marker.color = ColorRGBA(r=0.8, g=0.2, b=0.2, a=0.6)
                marker_array.markers.append(sphere_marker)
                marker_id += 1
            
            # 立方体障碍物
            for name, params in coll_objs.get('cube', {}).items():
                cube_marker = Marker()
                cube_marker.header.frame_id = "world"
                cube_marker.header.stamp = self.get_clock().now().to_msg()
                cube_marker.ns = "obstacles"
                cube_marker.id = marker_id
                cube_marker.type = Marker.CUBE
                cube_marker.action = Marker.ADD
                
                pose = params.get('pose', [0, 0, 0, 0, 0, 0, 1])
                dims = params.get('dims', [0.1, 0.1, 0.1])
                
                cube_marker.pose.position.x = float(pose[0])
                cube_marker.pose.position.y = float(pose[1])
                cube_marker.pose.position.z = float(pose[2])
                cube_marker.pose.orientation.x = float(pose[3])
                cube_marker.pose.orientation.y = float(pose[4])
                cube_marker.pose.orientation.z = float(pose[5])
                cube_marker.pose.orientation.w = float(pose[6])
                cube_marker.scale.x = float(dims[0])
                cube_marker.scale.y = float(dims[1])
                cube_marker.scale.z = float(dims[2])
                cube_marker.color = ColorRGBA(r=0.5, g=0.5, b=0.8, a=0.6)
                marker_array.markers.append(cube_marker)
                marker_id += 1
        
        self.pub_markers.publish(marker_array)

        collision_marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        current_count = 0

        if collision_spheres:
            for marker_id, sphere in enumerate(collision_spheres):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = stamp
                marker.ns = "collision_spheres"
                marker.id = int(sphere.get("marker_id", marker_id))
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.frame_locked = True
                marker.pose.position.x = float(sphere["center_world"][0])
                marker.pose.position.y = float(sphere["center_world"][1])
                marker.pose.position.z = float(sphere["center_world"][2])
                marker.pose.orientation.w = 1.0
                marker.scale.x = 2.0 * sphere["radius"]
                marker.scale.y = 2.0 * sphere["radius"]
                marker.scale.z = 2.0 * sphere["radius"]
                marker.color = ColorRGBA(r=1.0, g=0.78, b=0.12, a=0.45)
                collision_marker_array.markers.append(marker)
            current_count = len(collision_spheres)

        for marker_id in range(current_count, self._prev_collision_marker_count):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = stamp
            marker.ns = "collision_spheres"
            marker.id = marker_id
            marker.action = Marker.DELETE
            collision_marker_array.markers.append(marker)

        self._prev_collision_marker_count = current_count
        self.pub_collision_sphere_markers.publish(collision_marker_array)

    def publish_top_trajectories(self, top_trajs_world):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        clear_marker = Marker()
        clear_marker.header.frame_id = "world"
        clear_marker.header.stamp = stamp
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        if top_trajs_world is None or len(top_trajs_world) == 0:
            self.pub_top_traj_markers.publish(marker_array)
            return

        for traj_id, traj_points in enumerate(top_trajs_world[:5]):
            line_marker = Marker()
            line_marker.header.frame_id = "world"
            line_marker.header.stamp = stamp
            line_marker.ns = "mppi_top_trajs"
            line_marker.id = traj_id
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.pose.orientation.w = 1.0
            line_marker.scale.x = 0.002
            line_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.95)
            line_marker.points = []

            for point in np.asarray(traj_points, dtype=np.float64):
                if not np.all(np.isfinite(point)):
                    continue
                point_msg = Point()
                point_msg.x = float(point[0])
                point_msg.y = float(point[1])
                point_msg.z = float(point[2])
                line_marker.points.append(point_msg)

            if len(line_marker.points) >= 2:
                marker_array.markers.append(line_marker)

        self.pub_top_traj_markers.publish(marker_array)
    
    def is_connected(self) -> bool:
        """检查是否已连接到机器人"""
        return self._state_received
    
    def get_state_count(self) -> int:
        """获取接收的状态数量"""
        return self._state_count
    
    def get_cmd_count(self) -> int:
        """获取发送的指令数量"""
        return self._cmd_count


# ============================================================================
# 主控制函数
# ============================================================================

def mpc_control_main(args):
    """STORM MPC Reach Static 控制主函数"""
    
    print("=" * 60)
    print("UR7e STORM MPC Reach Static - Gazebo 仿真")
    print("=" * 60)
    
    # =========================================================================
    # 1. 加载配置 (使用 Gazebo 专用配置文件)
    # =========================================================================
    
    # 配置文件路径 (sim_gazebo/config 目录)
    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    
    robot_file = os.path.join(config_dir, 'ur7e_robot_gazebo.yml')
    task_file = 'ur7e_reacher_gazebo.yml'  # MPC 任务配置
    world_file = os.path.join(config_dir, 'collision_world_gazebo.yml')
    
    print(f"\n加载配置文件...")
    print(f"  Robot: {robot_file}")
    print(f"  Task:  {task_file}")
    print(f"  World: {world_file}")
    
    # 加载机器人配置
    with open(robot_file) as f:
        robot_params = yaml.safe_load(f)
    
    # 加载障碍物配置
    with open(world_file) as f:
        world_params = yaml.safe_load(f)
    
    sim_params = robot_params.get('sim_params', {})
    
    # 机器人位姿
    robot_pose = sim_params.get('robot_pose', [0, 0, 0, 0, 0, 0, 1])
    robot_pos = np.array(robot_pose[:3])
    robot_quat_xyzw = np.array(robot_pose[3:])
    
    # 关节名称
    joint_names = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]
    n_dof = len(joint_names)
    
    # =========================================================================
    # 2. 初始化 ROS2
    # =========================================================================
    
    print("\n初始化 ROS2...")
    rclpy.init(args=None)
    
    control_rate = args.rate
    robot = GazeboRobotInterface(joint_names, control_rate=control_rate)
    
    # 创建多线程执行器
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(robot)
    
    # 后台 spin
    _executor_running = True
    def spin_with_check():
        while _executor_running and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    
    spin_thread = Thread(target=spin_with_check, daemon=True)
    spin_thread.start()
    
    # 等待连接
    print("\n等待 Gazebo 关节状态...")
    timeout = 10.0
    start = time.time()
    while not robot.is_connected():
        if time.time() - start > timeout:
            print("错误: 无法接收关节状态，请确保 Gazebo 仿真正在运行")
            _executor_running = False
            robot.destroy_node()
            rclpy.shutdown()
            return 1
        time.sleep(0.1)
    
    print("已连接到 Gazebo 机器人!")
    
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
    
    # 创建 MPC 控制器 (使用 Gazebo 专用配置)
    task_file_abs = os.path.join(config_dir, 'ur7e_reacher_gazebo.yml')
    mpc = GazeboReacherTask(task_file_abs, robot_file, world_file, tensor_args)
    mpc.set_position_only_goal_mode()
    control_dt = mpc.exp_params.get('control_dt', 0.02)
    print(f"MPC 控制周期: {control_dt} s ({1.0/control_dt:.1f} Hz)")
    print("目标模式: position-only (/target_pose 只约束 xyz, 不约束末端姿态)")
    
    # =========================================================================
    # 4. 设置初始目标
    # =========================================================================
    
    # 目标关节状态
    goal_state = np.array([0.5, -1.2, 1.2, -1.57, -1.57, 0.0, 0, 0, 0, 0, 0, 0])
    mpc.update_params(goal_state=goal_state)
    
    # 获取目标末端位置
    goal_ee_pos_robot = np.ravel(mpc.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    goal_ee_quat = np.ravel(mpc.controller.rollout_fn.goal_ee_quat.cpu().numpy())
    goal_ee_world = transform_point(robot_pos, robot_quat_xyzw, goal_ee_pos_robot)
    
    print(f"\n目标末端位置 (机器人坐标系): {goal_ee_pos_robot}")
    print(f"目标末端位置 (世界坐标系): {goal_ee_world}")
    
    # 保存 rollout_fn 用于计算末端位置
    rollout_fn = mpc.controller.rollout_fn
    collision_sphere_visualizer = CollisionSphereVisualizer(
        mpc.exp_params['model']['robot_collision_params']
    )
    n_collision_spheres = sum(
        len(collision_sphere_visualizer.spheres_by_link.get(link_name, []))
        for link_name in collision_sphere_visualizer.link_names
    )
    
    # 当前目标
    current_goal_ee = goal_ee_pos_robot.copy()
    current_goal_world = goal_ee_world.copy()
    
    # =========================================================================
    # 5. MPC 控制循环
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("开始 MPC 控制循环... (Ctrl+C 退出)")
    print("=" * 60)
    print("\n提示:")
    print("  - 发布 PoseStamped 到 /target_pose 可动态更新目标")
    print("  - /target_pose 的 orientation 不参与目标更新")
    print("  - 在 RViz 中查看 /visualization_marker_array")
    print("  - 在 RViz 中查看 /mppi_top_traj_markers (MPPI 前若干条预测轨迹)")
    print("  - 红球=目标, 绿球=末端")
    print(f"  - 黄球=机械臂碰撞球模型 ({n_collision_spheres} 个)")
    print("  - 红线=MPPI 前若干条末端预测轨迹")
    print(f"  - 控制器使用同步求解，执行模式={_get_execution_mode(mpc)}")
    print("")
    
    # 信号处理
    running = [True]
    
    def shutdown_handler(sig, frame):
        print("\n收到退出信号，正在退出...")
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
                _get_sync_command(mpc, t, state, control_dt)
            except:
                pass
        t += control_dt
        time.sleep(0.01)
    
    # 等待第一次优化完成
    print("等待首次优化完成...")
    state = robot.get_state()
    if state is not None:
        try:
            cmd = _get_sync_command(mpc, t, state, control_dt)
            print(f"首次优化完成! opt_dt={mpc.opt_dt:.3f}s")
        except Exception as e:
            print(f"首次优化异常: {e}")
    
    print("\nMPC 预热完成，开始控制!\n")
    
    # 控制变量
    i = 0
    loop_start = time.time()
    prev_vel = np.zeros(n_dof)
    while running[0] and rclpy.ok():
        iter_start = time.time()
        t = time.time() - loop_start
        cmd = None
        
        # --- 1. 获取状态 ---
        state = robot.get_state()
        if state is None:
            time.sleep(control_dt)
            continue
        
        q = state['position']
        dq = state['velocity']
        ddq = state['acceleration']
        
        # --- 2. 检测目标更新 ---
        new_target = robot.get_target_position()
        if new_target is not None:
            # 目标是世界坐标系，转换到机器人坐标系
            target_robot = inv_transform_point(robot_pos, robot_quat_xyzw, new_target)
            if np.linalg.norm(target_robot - current_goal_ee) > 0.005:
                current_goal_ee = target_robot.copy()
                current_goal_world = new_target.copy()
                mpc.update_params(goal_ee_pos=current_goal_ee)
                print(f"[目标更新] 世界: {np.round(current_goal_world, 3)}, 机器人: {np.round(current_goal_ee, 3)}")
                try:
                    cmd = _recover_command(mpc, t, state, control_dt)
                    print("[目标更新] 已同步重规划并重置 MPC 时间基准")
                except Exception as sync_exc:
                    print(f"[MPC异常] 目标更新后的同步重规划失败: {sync_exc}")
                    i += 1
                    time.sleep(control_dt)
                    continue
        
        # --- 3. MPC 计算 ---
        if cmd is None:
            try:
                cmd = _get_sync_command(mpc, t, state, control_dt)
            except (IndexError, RuntimeError, ValueError) as exc:
                print(f"[MPC恢复] 同步取命令失败 ({exc})，重置控制进程时间基准后重规划")
                try:
                    cmd = _recover_command(mpc, t, state, control_dt)
                except Exception as recover_exc:
                    print(f"[MPC异常] 同步重规划失败: {recover_exc}")
                    i += 1
                    time.sleep(control_dt)
                    continue

        if cmd is None or 'position' not in cmd:
            i += 1
            time.sleep(control_dt)
            continue
        
        # --- 4. 发送指令 ---
        target_positions = cmd['position']
        if isinstance(target_positions, torch.Tensor):
            target_positions = target_positions.cpu().numpy()
        target_positions = np.array(target_positions).flatten()[:n_dof]
        robot.send_position_command(target_positions)
        
        # --- 5. 计算并发布末端位置 ---
        curr = np.hstack([q, dq, ddq])
        ee_pose = rollout_fn.get_ee_pose(
            torch.as_tensor(curr, **tensor_args).unsqueeze(0)
        )
        ee_pos_robot = np.ravel(ee_pose['ee_pos_seq'].cpu().numpy())
        ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos_robot)
        robot.publish_ee_pose(ee_pos_world)
        
        # --- 6. 发布可视化标记（与控制频率一致）---
        link_pos_robot, link_rot_robot = _compute_link_poses_robot_frame(
            rollout_fn,
            q,
            dq,
            tensor_args,
        )
        collision_spheres_world = collision_sphere_visualizer.get_world_spheres(
            link_pos_robot,
            link_rot_robot,
            robot_pos,
            robot_quat_xyzw,
        )
        robot.publish_markers(
            world_params,
            current_goal_world,
            ee_pos_world,
            collision_spheres=collision_spheres_world,
        )

        top_trajs_world = _get_top_ee_trajs_world(
            mpc,
            robot_pos,
            robot_quat_xyzw,
            ee_pos_world,
            max_trajs=5,
        )
        robot.publish_top_trajectories(top_trajs_world)
        
        # --- 7. 打印状态 ---
        if i % 50 == 0:
            err = mpc.get_current_error(state)
            rx_count = robot.get_state_count()
            tx_count = robot.get_cmd_count()
            print(f"[{i:4d}] 误差: {[f'{x:.3f}' for x in err]}, "
                  f"opt: {mpc.opt_dt:.3f}s, ee: {np.round(ee_pos_world, 3)}, "
                  f"rx/tx: {rx_count}/{tx_count}")
        
        # --- 8. 保持控制频率 ---
        elapsed = time.time() - iter_start
        sleep_time = (1.0 / control_rate) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        i += 1
    
    # =========================================================================
    # 清理
    # =========================================================================
    
    print("\n清理资源...")
    
    _executor_running = False
    spin_thread.join(timeout=1.0)
    
    # 关闭 MPC
    print("  关闭 MPC...")
    def close_mpc():
        try:
            mpc.close()
        except:
            pass
    
    close_thread = Thread(target=close_mpc, daemon=True)
    close_thread.start()
    close_thread.join(timeout=2.0)
    
    # 关闭 ROS2
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
    parser = argparse.ArgumentParser(description='UR7e STORM MPC Reach Static - Gazebo')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='使用 CUDA 加速 (默认: True)')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='禁用 CUDA')
    parser.add_argument('--rate', type=float, default=50.0,
                        help='控制频率 Hz (默认: 50)')
    
    args = parser.parse_args()
    
    # Torch 配置
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    sys.exit(mpc_control_main(args))
