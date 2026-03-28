#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e STORM MPC Control - Whole Gazebo Tall ESDF 驱动示例
"""

import argparse
import os
import queue
import signal
import sys
import threading
import time

import numpy as np
import torch
import yaml

STORM_ROOT = os.path.expanduser('~/storm')
if STORM_ROOT not in sys.path:
    sys.path.insert(0, STORM_ROOT)

torch.multiprocessing.set_start_method('spawn', force=True)

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    from std_msgs.msg import ColorRGBA, Float64MultiArray
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:
    print('=' * 60)
    print('错误: 未找到 ROS2 Python 包')
    print('请先 source ROS2 环境:')
    print('  source /opt/ros/humble/setup.bash')
    print('=' * 60)
    sys.exit(1)

from storm_kit.mpc.control import MPPI
from storm_kit.mpc.task.task_base import BaseTask
from storm_kit.mpc.utils.mpc_process_wrapper import ControlProcess
from storm_kit.mpc.utils.state_filter import JointStateFilter

from examples.whole_sim_gazebo.arm_reacher_esdf import ArmReacherESDF
from examples.whole_sim_gazebo.gazebo_obstacle_utils import (
    count_primitive_obstacles,
    iter_primitive_obstacles,
    load_primitive_world,
    spawn_gazebo_obstacles,
)

np.set_printoptions(precision=3, suppress=True)


def _log(message: str) -> None:
    print(message, flush=True)


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


def _close_mp_queue(mp_queue) -> None:
    for method_name in ('close', 'cancel_join_thread'):
        method = getattr(mp_queue, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass


def _shutdown_control_process(control_process, join_timeout: float = 2.0) -> None:
    if control_process is None:
        return

    control_process.done = True
    _drain_mp_queue(getattr(control_process, 'result_queue', None))

    done_message = {'state': None, 'dt': None, 'done': True, 'params': None}
    opt_queue = getattr(control_process, 'opt_queue', None)
    if opt_queue is not None:
        try:
            opt_queue.put_nowait(done_message)
        except queue.Full:
            _drain_mp_queue(opt_queue)
            try:
                opt_queue.put_nowait(done_message)
            except Exception:
                pass
        except Exception:
            pass

    opt_process = getattr(control_process, 'opt_process', None)
    if opt_process is not None:
        opt_process.join(timeout=join_timeout)
        if opt_process.is_alive():
            _log('后台 MPC 进程未在超时内退出，强制终止...')
            opt_process.terminate()
            opt_process.join(timeout=join_timeout)

    if opt_queue is not None:
        _close_mp_queue(opt_queue)
    result_queue = getattr(control_process, 'result_queue', None)
    if result_queue is not None:
        _close_mp_queue(result_queue)


def _restart_control_process(mpc, join_timeout: float = 0.2) -> None:
    old_control_process = getattr(mpc, 'control_process', None)
    _shutdown_control_process(old_control_process, join_timeout=join_timeout)
    mpc.control_process = ControlProcess(mpc.controller)


def _reset_control_process_timing(control_process, t_step: float, control_dt: float) -> None:
    control_process.command = None
    control_process.command_tstep = control_process.traj_tstep + t_step
    control_process.prev_mpc_tstep = max(0.0, t_step - control_dt)
    control_process.mpc_dt = control_dt
    _drain_mp_queue(getattr(control_process, 'result_queue', None))
    _drain_mp_queue(getattr(control_process, 'opt_queue', None))


def _recover_command(mpc, t_step: float, state: dict, control_dt: float):
    control_process = mpc.control_process
    _reset_control_process_timing(control_process, t_step, control_dt)
    return mpc.get_command(t_step, state, control_dt=control_dt, WAIT=True)


def _quat_xyzw_to_rot_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_point(translation_world: np.ndarray, quat_xyzw: np.ndarray, point_local: np.ndarray) -> np.ndarray:
    rot = _quat_xyzw_to_rot_matrix(quat_xyzw)
    return np.asarray(translation_world, dtype=np.float64) + rot @ np.asarray(point_local, dtype=np.float64)


def inv_transform_point(translation_world: np.ndarray, quat_xyzw: np.ndarray, point_world: np.ndarray) -> np.ndarray:
    rot = _quat_xyzw_to_rot_matrix(quat_xyzw)
    return rot.T @ (np.asarray(point_world, dtype=np.float64) - np.asarray(translation_world, dtype=np.float64))


def build_esdf_surface_points(snapshot, max_points: int = 5000, surface_band_scale: float = 1.5) -> np.ndarray:
    esdf = snapshot.esdf.detach().cpu().numpy()
    valid_mask = snapshot.valid_mask.detach().cpu().numpy()
    voxel_size = float(snapshot.voxel_size)
    surface_band = max(voxel_size * surface_band_scale, voxel_size)

    surface_mask = np.logical_and(valid_mask, np.abs(esdf) <= surface_band)
    voxel_idx = np.argwhere(surface_mask)
    if voxel_idx.shape[0] == 0:
        surface_mask = np.logical_and(valid_mask, esdf <= surface_band)
        voxel_idx = np.argwhere(surface_mask)
    if voxel_idx.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    if voxel_idx.shape[0] > max_points:
        step = int(np.ceil(float(voxel_idx.shape[0]) / float(max_points)))
        voxel_idx = voxel_idx[::step]

    origin_world = snapshot.origin_world.detach().cpu().numpy().astype(np.float32)
    return origin_world.reshape(1, 3) + voxel_idx.astype(np.float32) * voxel_size


class WholeGazeboReacherTask(BaseTask):
    def __init__(self, task_file, robot_file, world_file, tensor_args):
        super().__init__(tensor_args=tensor_args)
        self.controller = self.init_mppi(task_file, robot_file, world_file)
        self.init_aux()

    def get_rollout_fn(self, **kwargs):
        return ArmReacherESDF(**kwargs)

    def init_mppi(self, task_file, robot_file, world_file):
        with open(robot_file) as f:
            robot_params = yaml.safe_load(f)
        with open(world_file) as f:
            world_params = yaml.safe_load(f)
        with open(task_file) as f:
            exp_params = yaml.safe_load(f)

        exp_params['robot_params'] = exp_params['model']
        rollout_fn = self.get_rollout_fn(
            exp_params=exp_params,
            tensor_args=self.tensor_args,
            world_params=world_params,
        )

        mppi_params = exp_params['mppi']
        dynamics_model = rollout_fn.dynamics_model
        mppi_params['d_action'] = dynamics_model.d_action
        mppi_params['action_lows'] = -exp_params['model']['max_acc'] * torch.ones(
            dynamics_model.d_action,
            **self.tensor_args,
        )
        mppi_params['action_highs'] = exp_params['model']['max_acc'] * torch.ones(
            dynamics_model.d_action,
            **self.tensor_args,
        )

        init_q = torch.tensor(exp_params['model']['init_state'], **self.tensor_args)
        init_action = torch.zeros(
            (mppi_params['horizon'], dynamics_model.d_action),
            **self.tensor_args,
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

        _log('[WholeGazeboReacherTask] Cost source summary:')
        _log(
            '  primitive_collision.weight = %.1f'
            % float(exp_params['cost']['primitive_collision']['weight'])
        )
        _log(
            '  voxel_collision.weight     = %.1f'
            % float(exp_params['cost']['voxel_collision']['weight'])
        )
        _log(
            '  esdf_collision.weight      = %.1f'
            % float(exp_params['cost']['esdf_collision']['weight'])
        )
        _log(
            '  esdf_snapshot_path         = %s'
            % world_params['world_model']['esdf_snapshot_path']
        )
        _log('  environment_collision      = ESDF snapshot')
        return controller

    def init_aux(self):
        self.state_filter = JointStateFilter(
            filter_coeff=self.exp_params['state_filter_coeff'],
            dt=self.exp_params['control_dt'],
        )
        self.command_filter = JointStateFilter(
            filter_coeff=self.exp_params['cmd_filter_coeff'],
            dt=self.exp_params['control_dt'],
        )
        self.control_process = ControlProcess(self.controller)
        self.n_dofs = self.controller.rollout_fn.dynamics_model.n_dofs
        self.zero_acc = np.zeros(self.n_dofs)

    def close(self):
        control_process = getattr(self, 'control_process', None)
        _shutdown_control_process(control_process)


class GazeboRobotInterface(Node):
    def __init__(self, joint_names, control_rate=50.0):
        super().__init__('storm_mpc_whole_gazebo')
        self.joint_names = joint_names
        self.n_dof = len(joint_names)
        self.control_rate = control_rate
        self.control_dt = 1.0 / control_rate
        self._lock = threading.Lock()
        self.current_positions = None
        self.current_velocities = None
        self.prev_velocities = None
        self.prev_time = None
        self.state_received = False
        self.obstacle_world = None
        self.target_position_world = None

        qos = QoSProfile(depth=10)
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.sub_joint_states = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            qos,
        )
        self.sub_target_pose = self.create_subscription(
            PoseStamped,
            '/target_pose',
            self._target_pose_callback,
            qos,
        )
        self.pub_position_cmd = self.create_publisher(
            Float64MultiArray,
            '/forward_position_controller/commands',
            qos_reliable,
        )
        self.pub_ee_pose = self.create_publisher(
            PoseStamped,
            '/ee_pose',
            qos,
        )
        self.pub_markers = self.create_publisher(
            MarkerArray,
            '/visualization_marker_array',
            qos,
        )

        self.get_logger().info('Gazebo Robot Interface 初始化完成')
        self.get_logger().info(f'控制频率: {control_rate} Hz')
        self.get_logger().info(f'关节数: {self.n_dof}')
        self.get_logger().info('订阅: /joint_states, /target_pose')
        self.get_logger().info('发布: /forward_position_controller/commands, /ee_pose, /visualization_marker_array')

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
            self.current_positions = positions
            self.current_velocities = velocities
            self.state_received = True

    def _target_pose_callback(self, msg: PoseStamped):
        target = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        with self._lock:
            self.target_position_world = target
        self.get_logger().info(
            '收到目标位置: [%.3f, %.3f, %.3f]'
            % (target[0], target[1], target[2])
        )

    def get_state(self):
        with self._lock:
            if not self.state_received:
                return None
            positions = self.current_positions.copy()
            velocities = self.current_velocities.copy()

        current_time = time.time()
        if self.prev_velocities is not None and self.prev_time is not None:
            dt = max(current_time - self.prev_time, 0.001)
            acceleration = (velocities - self.prev_velocities) / dt
        else:
            acceleration = np.zeros(self.n_dof)

        self.prev_velocities = velocities.copy()
        self.prev_time = current_time
        return {
            'position': positions,
            'velocity': velocities,
            'acceleration': acceleration,
        }

    def send_position_command(self, positions):
        msg = Float64MultiArray()
        msg.data = positions.tolist()
        self.pub_position_cmd.publish(msg)

    def get_target_position(self):
        with self._lock:
            if self.target_position_world is None:
                return None
            target_position = self.target_position_world.copy()
            self.target_position_world = None
        return target_position

    def set_obstacles(self, obstacle_world: dict):
        self.obstacle_world = obstacle_world
        n_spheres, n_cubes = count_primitive_obstacles(obstacle_world, include_ground=False)
        self.get_logger().info(
            '已加载 primitive 障碍物配置: spheres=%d cubes=%d'
            % (n_spheres, n_cubes)
        )

    def publish_ee_pose(self, position_world: np.ndarray):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(position_world[0])
        msg.pose.position.y = float(position_world[1])
        msg.pose.position.z = float(position_world[2])
        msg.pose.orientation.w = 1.0
        self.pub_ee_pose.publish(msg)

    def publish_markers(self, goal_pos_world: np.ndarray, ee_pos_world: np.ndarray):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker_id = 0

        if self.obstacle_world is not None:
            for obstacle in iter_primitive_obstacles(self.obstacle_world, include_ground=False):
                obstacle_marker = Marker()
                obstacle_marker.header.frame_id = 'world'
                obstacle_marker.header.stamp = stamp
                obstacle_marker.ns = 'obstacles'
                obstacle_marker.id = marker_id
                obstacle_marker.action = Marker.ADD

                if obstacle['kind'] == 'sphere':
                    obstacle_marker.type = Marker.SPHERE
                    obstacle_marker.pose.position.x = obstacle['position'][0]
                    obstacle_marker.pose.position.y = obstacle['position'][1]
                    obstacle_marker.pose.position.z = obstacle['position'][2]
                    obstacle_marker.pose.orientation.w = 1.0
                    obstacle_marker.scale.x = obstacle['radius'] * 2.0
                    obstacle_marker.scale.y = obstacle['radius'] * 2.0
                    obstacle_marker.scale.z = obstacle['radius'] * 2.0
                    obstacle_marker.color = ColorRGBA(r=0.8, g=0.2, b=0.2, a=0.75)
                else:
                    obstacle_marker.type = Marker.CUBE
                    pose = obstacle['pose']
                    dims = obstacle['dims']
                    obstacle_marker.pose.position.x = pose[0]
                    obstacle_marker.pose.position.y = pose[1]
                    obstacle_marker.pose.position.z = pose[2]
                    obstacle_marker.pose.orientation.x = pose[3]
                    obstacle_marker.pose.orientation.y = pose[4]
                    obstacle_marker.pose.orientation.z = pose[5]
                    obstacle_marker.pose.orientation.w = pose[6]
                    obstacle_marker.scale.x = dims[0]
                    obstacle_marker.scale.y = dims[1]
                    obstacle_marker.scale.z = dims[2]
                    obstacle_marker.color = ColorRGBA(r=0.5, g=0.5, b=0.8, a=0.75)

                marker_array.markers.append(obstacle_marker)
                marker_id += 1

        goal_marker = Marker()
        goal_marker.header.frame_id = 'world'
        goal_marker.header.stamp = stamp
        goal_marker.ns = 'goal'
        goal_marker.id = marker_id
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = float(goal_pos_world[0])
        goal_marker.pose.position.y = float(goal_pos_world[1])
        goal_marker.pose.position.z = float(goal_pos_world[2])
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.06
        goal_marker.scale.y = 0.06
        goal_marker.scale.z = 0.06
        goal_marker.color = ColorRGBA(r=0.9, g=0.1, b=0.1, a=0.85)
        marker_array.markers.append(goal_marker)
        marker_id += 1

        ee_marker = Marker()
        ee_marker.header.frame_id = 'world'
        ee_marker.header.stamp = stamp
        ee_marker.ns = 'ee'
        ee_marker.id = marker_id
        ee_marker.type = Marker.SPHERE
        ee_marker.action = Marker.ADD
        ee_marker.pose.position.x = float(ee_pos_world[0])
        ee_marker.pose.position.y = float(ee_pos_world[1])
        ee_marker.pose.position.z = float(ee_pos_world[2])
        ee_marker.pose.orientation.w = 1.0
        ee_marker.scale.x = 0.05
        ee_marker.scale.y = 0.05
        ee_marker.scale.z = 0.05
        ee_marker.color = ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.85)
        marker_array.markers.append(ee_marker)

        self.pub_markers.publish(marker_array)

    def wait_for_state(self, timeout=5.0, stop_event=None):
        start_time = time.time()
        while True:
            with self._lock:
                state_received = self.state_received
            if state_received:
                break
            if stop_event is not None and stop_event.is_set():
                return False
            time.sleep(0.05)
            if time.time() - start_time > timeout:
                return False
        return True


def mpc_control_main(args):
    shutdown_event = threading.Event()
    executor = None
    spin_thread = None
    robot = None
    mpc = None
    exit_code = 0

    def request_shutdown(signum=None, _frame=None):
        if shutdown_event.is_set():
            return
        if signum is not None:
            signal_name = signal.Signals(signum).name
            _log(f'\n收到 {signal_name}，准备退出...')
        shutdown_event.set()

    try:
        _log('=' * 60)
        _log('UR7e STORM MPC Control - Whole Gazebo Tall ESDF')
        _log('=' * 60)

        config_dir = os.path.join(os.path.dirname(__file__), 'config')
        robot_file = os.path.join(config_dir, 'ur7e_robot_gazebo.yml')
        task_file = os.path.join(config_dir, 'ur7e_reacher_whole_gazebo_tall.yml')
        world_file = os.path.join(config_dir, 'esdf_world_gazebo_tall.yml')
        obstacle_file = os.path.join(config_dir, 'collision_world_gazebo_tall.yml')

        _log('\n加载配置文件...')
        _log(f'  Robot: {robot_file}')
        _log(f'  Task:  {task_file}')
        _log(f'  World: {world_file}')
        _log(f'  Obstacles: {obstacle_file}')

        with open(robot_file) as f:
            robot_params = yaml.safe_load(f)
        obstacle_world = load_primitive_world(obstacle_file)
        sim_params = robot_params.get('sim_params', {})
        robot_pose = sim_params.get('robot_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        robot_pos_world = np.array(robot_pose[:3], dtype=np.float64)
        robot_quat_xyzw = np.array(robot_pose[3:7], dtype=np.float64)

        joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]
        n_dof = len(joint_names)

        _log('\n初始化 ROS2...')
        rclpy.init(args=None)
        signal.signal(signal.SIGINT, request_shutdown)
        signal.signal(signal.SIGTERM, request_shutdown)

        robot = GazeboRobotInterface(joint_names, control_rate=args.rate)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(robot)

        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        _log('等待 Gazebo 关节状态...')
        if not robot.wait_for_state(timeout=10.0, stop_event=shutdown_event):
            if shutdown_event.is_set():
                _log('初始化阶段收到退出请求，停止启动流程')
                exit_code = 130
            else:
                _log('错误: 无法接收关节状态，请确保 Gazebo 仿真正在运行')
                exit_code = 1
            return exit_code

        _log('已连接到 Gazebo 机器人!')
        robot.set_obstacles(obstacle_world)
        if spawn_gazebo_obstacles(robot, obstacle_world, model_prefix='whole_tall', include_ground=False):
            _log('Gazebo 真实障碍物已生成: 使用 primitive 高墙/球体模型')
        else:
            _log('警告: Gazebo 真实障碍物未完整生成，请检查 /spawn_entity 与 /delete_entity 服务')

        device = 'cuda' if args.cuda else 'cpu'
        _log('\n初始化 STORM MPC 控制器...')
        _log(f'计算设备: {device}')
        tensor_args = {
            'device': torch.device(device, 0) if device == 'cuda' else torch.device('cpu'),
            'dtype': torch.float32,
        }

        mpc = WholeGazeboReacherTask(task_file, robot_file, world_file, tensor_args)
        mpc_control_dt = mpc.exp_params.get('control_dt', 0.02)
        _log(f'MPC 控制周期: {mpc_control_dt} s ({1.0 / mpc_control_dt:.1f} Hz)')

        init_state = sim_params.get('init_state', [0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        if args.goal is not None:
            goal_joint_positions = np.array(args.goal, dtype=np.float64)
        else:
            goal_joint_positions = np.array([0.5, -1.2, 1.2, -1.57, -1.57, 0.0], dtype=np.float64)
        goal_state = np.concatenate([goal_joint_positions, np.zeros(n_dof)])

        _log(f'\n初始关节角: {np.array(init_state)}')
        _log(f'目标关节角: {goal_joint_positions}')
        mpc.update_params(goal_state=goal_state)

        rollout_fn = mpc.controller.rollout_fn
        goal_ee_pos_robot = np.ravel(rollout_fn.goal_ee_pos.detach().cpu().numpy())
        goal_ee_quat = np.ravel(rollout_fn.goal_ee_quat.detach().cpu().numpy())
        current_goal_world = transform_point(robot_pos_world, robot_quat_xyzw, goal_ee_pos_robot)
        current_goal_robot = goal_ee_pos_robot.copy()
        n_spheres, n_cubes = count_primitive_obstacles(obstacle_world, include_ground=False)
        _log(
            'RViz scene markers prepared: primitive spheres=%d cubes=%d'
            % (n_spheres, n_cubes)
        )
        _log(
            '初始目标末端位置(世界坐标系): [%.3f, %.3f, %.3f]'
            % (current_goal_world[0], current_goal_world[1], current_goal_world[2])
        )

        _log('\n' + '=' * 60)
        _log('开始 MPC 控制循环... (Ctrl+C 退出)')
        _log('=' * 60 + '\n')

        t = 0.0
        loop_count = 0
        marker_update_counter = 0
        control_dt = 1.0 / args.rate

        _log('预热 MPC 控制器...')
        current_state = robot.get_state()
        for _ in range(5):
            if shutdown_event.is_set():
                break
            try:
                mpc.get_command(t, current_state, control_dt=mpc_control_dt, WAIT=True)
            except Exception as exc:
                _log(f'预热异常 (可忽略): {exc}')
            t += mpc_control_dt
            time.sleep(0.01)

        _log('开始控制!\n')

        try:
            while rclpy.ok() and not shutdown_event.is_set():
                loop_start = time.time()
                current_state = robot.get_state()
                if current_state is None:
                    time.sleep(0.01)
                    continue

                new_target_world = robot.get_target_position()
                if new_target_world is not None:
                    target_robot = inv_transform_point(robot_pos_world, robot_quat_xyzw, new_target_world)
                    if np.linalg.norm(target_robot - current_goal_robot) > 0.005:
                        current_goal_robot = target_robot.copy()
                        current_goal_world = new_target_world.copy()
                        mpc.update_params(goal_ee_pos=current_goal_robot, goal_ee_quat=goal_ee_quat)
                        _log(
                            '[目标更新] world=[%.3f, %.3f, %.3f] robot=[%.3f, %.3f, %.3f]'
                            % (
                                current_goal_world[0],
                                current_goal_world[1],
                                current_goal_world[2],
                                current_goal_robot[0],
                                current_goal_robot[1],
                                current_goal_robot[2],
                            )
                        )
                        try:
                            command = _recover_command(mpc, t, current_state, mpc_control_dt)
                            _log('[目标更新] 已同步重规划并重置 MPC 时间基准')
                        except Exception as sync_exc:
                            _log(f'[MPC异常] 目标更新后的同步重规划失败: {sync_exc}')
                            time.sleep(control_dt)
                            continue
                    else:
                        command = None
                else:
                    command = None

                if command is None:
                    try:
                        command = mpc.get_command(t, current_state, control_dt=mpc_control_dt, WAIT=True)
                    except (IndexError, RuntimeError, ValueError) as exc:
                        _log('[MPC恢复] 同步取命令失败 (%s)，重置控制进程时间基准后重规划' % exc)
                        try:
                            command = _recover_command(mpc, t, current_state, mpc_control_dt)
                        except Exception as recover_exc:
                            _log(f'[MPC异常] 同步重规划失败: {recover_exc}')
                            time.sleep(control_dt)
                            continue

                if command is not None and 'position' in command:
                    target_positions = command['position']
                    if isinstance(target_positions, torch.Tensor):
                        target_positions = target_positions.detach().cpu().numpy()
                    target_positions = np.array(target_positions).flatten()[:n_dof]
                else:
                    target_positions = current_state['position']

                robot.send_position_command(target_positions)

                q = current_state['position']
                dq = current_state['velocity']
                ddq = current_state['acceleration']
                ee_pose = rollout_fn.get_ee_pose(
                    torch.as_tensor(np.hstack([q, dq, ddq]), **tensor_args).unsqueeze(0)
                )
                ee_pos_robot = np.ravel(ee_pose['ee_pos_seq'].detach().cpu().numpy())
                ee_pos_world = transform_point(robot_pos_world, robot_quat_xyzw, ee_pos_robot)
                robot.publish_ee_pose(ee_pos_world)

                marker_update_counter += 1
                if marker_update_counter >= 10:
                    robot.publish_markers(current_goal_world, ee_pos_world)
                    marker_update_counter = 0

                loop_count += 1
                if loop_count % 50 == 0:
                    current_pos = current_state['position']
                    ee_pos_error = np.linalg.norm(ee_pos_world - current_goal_world)
                    valid_ratio = getattr(
                        mpc.controller.rollout_fn.esdf_collision_cost,
                        'last_valid_ratio',
                        0.0,
                    )
                    _log(
                        f'[{loop_count:5d}] t={t:.2f}s | '
                        f'q=[{current_pos[0]:+.2f}, {current_pos[1]:+.2f}, {current_pos[2]:+.2f}, '
                        f'{current_pos[3]:+.2f}, {current_pos[4]:+.2f}, {current_pos[5]:+.2f}] | '
                        f'ee_pos_error={ee_pos_error:.4f}m | esdf_valid_ratio={100.0 * valid_ratio:.1f}%'
                    )

                t += control_dt
                elapsed = time.time() - loop_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            _log('\n用户中断，正在退出...')
            shutdown_event.set()
            exit_code = 130

        if shutdown_event.is_set() and exit_code == 0:
            exit_code = 130

        return exit_code
    finally:
        _log('清理资源...')
        shutdown_event.set()

        if mpc is not None:
            try:
                mpc.close()
            except Exception as exc:
                _log(f'关闭 MPC 资源时出现异常: {exc}')

        if robot is not None:
            try:
                robot.destroy_node()
            except Exception as exc:
                _log(f'销毁 ROS2 节点时出现异常: {exc}')

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as exc:
                _log(f'关闭 ROS2 时出现异常: {exc}')

        if executor is not None:
            try:
                executor.shutdown(timeout_sec=0.0)
            except TypeError:
                pass
            except Exception as exc:
                _log(f'关闭 ROS2 executor 时出现异常: {exc}')

        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=1.0)
            if spin_thread.is_alive():
                _log('ROS2 spin 线程未在超时内退出')
        _log('程序结束')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UR7e STORM MPC Whole Gazebo Tall ESDF Control')
    parser.add_argument('--cuda', action='store_true', default=True, help='使用 CUDA 加速 (默认: True)')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='禁用 CUDA')
    parser.add_argument('--rate', type=float, default=50.0, help='控制频率 Hz (默认: 50)')
    parser.add_argument('--goal', type=float, nargs=6, default=None, help='目标关节角度 (6个值，单位: rad)')
    args = parser.parse_args()
    sys.exit(mpc_control_main(args))
