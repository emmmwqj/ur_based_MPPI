#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predictive dynamic-ball reaching demo with horizon ball forecast."""

from __future__ import annotations

import argparse
import os
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
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped, TwistStamped
except ImportError:
    print('=' * 60)
    print('错误: 未找到 ROS2 Python 包')
    print('请先 source ROS2 环境: source /opt/ros/humble/setup.bash')
    print('=' * 60)
    sys.exit(1)

from examples.sim_dynamic.scripts.dynamic_reacher_task_predict import DynamicGazeboReacherTaskPredict
from examples.sim_dynamic.scripts.gazebo_dynamic_utils import (
    build_dynamic_sphere_sdf,
    count_primitive_obstacles,
    spawn_obstacle_model,
    spawn_static_obstacles,
)
from examples.sim_gazebo.reach_static_ur7e import (
    GazeboRobotInterface,
    CollisionSphereVisualizer,
    transform_point,
    inv_transform_point,
    _compute_link_poses_robot_frame,
    _get_top_ee_trajs_world,
    _get_sync_command,
    _recover_command,
)
from examples.sim_gazebo.reach_static_ur7e_tall import _shutdown_control_process

np.set_printoptions(precision=3, suppress=True)


class DynamicGazeboRobotInterfacePredict(GazeboRobotInterface):
    def __init__(
        self,
        joint_names: list[str],
        dynamic_ball_topic: str,
        dynamic_ball_velocity_topic: str,
        control_rate: float = 50.0,
    ):
        super().__init__(joint_names, control_rate=control_rate)
        self._dynamic_ball_pos = None
        self._dynamic_ball_vel_y = None
        self._dynamic_ball_pose_stamp_sec = None
        self._dynamic_ball_last_sign = 1.0
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.sub_dynamic_ball = self.create_subscription(
            PoseStamped,
            dynamic_ball_topic,
            self._dynamic_ball_callback,
            qos,
        )
        self.sub_dynamic_ball_velocity = self.create_subscription(
            TwistStamped,
            dynamic_ball_velocity_topic,
            self._dynamic_ball_velocity_callback,
            qos,
        )
        self.get_logger().info(f'  订阅动态障碍物(预测): {dynamic_ball_topic}')
        self.get_logger().info(f'  订阅动态障碍物速度(预测): {dynamic_ball_velocity_topic}')

    def _dynamic_ball_callback(self, msg: PoseStamped):
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=np.float64)
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1.0e-9
        if stamp_sec <= 0.0:
            stamp_sec = time.time()
        with self._lock:
            if self._dynamic_ball_vel_y is not None and abs(self._dynamic_ball_vel_y) > 1.0e-4:
                self._dynamic_ball_last_sign = 1.0 if self._dynamic_ball_vel_y >= 0.0 else -1.0
            self._dynamic_ball_pos = pos
            self._dynamic_ball_pose_stamp_sec = stamp_sec

    def _dynamic_ball_velocity_callback(self, msg: TwistStamped):
        vel_y = float(msg.twist.linear.y)
        with self._lock:
            self._dynamic_ball_vel_y = vel_y
            if abs(vel_y) > 1.0e-4:
                self._dynamic_ball_last_sign = 1.0 if vel_y >= 0.0 else -1.0

    def get_dynamic_ball_state(self):
        with self._lock:
            if self._dynamic_ball_pos is None:
                return None, None, None
            vel_y = None if self._dynamic_ball_vel_y is None else float(self._dynamic_ball_vel_y)
            stamp = self._dynamic_ball_pose_stamp_sec
            return self._dynamic_ball_pos.copy(), vel_y, stamp

    @staticmethod
    def _reflect_y(y_value: float, y_min: float, y_max: float) -> float:
        span = y_max - y_min
        if span <= 0.0:
            return min(max(y_value, y_min), y_max)
        period = 2.0 * span
        shifted = y_value - y_min
        wrapped = shifted % period
        if wrapped <= span:
            return y_min + wrapped
        return y_max - (wrapped - span)

    def get_dynamic_ball_state_estimate(
        self,
        nominal_speed: float,
        y_limits,
        current_time_sec: float,
    ):
        with self._lock:
            if self._dynamic_ball_pos is None:
                return None, None
            pos = self._dynamic_ball_pos.copy()
            stamp = self._dynamic_ball_pose_stamp_sec
            vel_y = self._dynamic_ball_vel_y
            sign = float(self._dynamic_ball_last_sign)
        if vel_y is None or abs(vel_y) < 1.0e-3:
            vel_y = sign * nominal_speed
        else:
            vel_y = (1.0 if vel_y >= 0.0 else -1.0) * nominal_speed

        if stamp is None:
            return pos, vel_y

        lag = max(current_time_sec - stamp, 0.0)
        pos[1] = self._reflect_y(pos[1] + vel_y * lag, float(y_limits[0]), float(y_limits[1]))
        return pos, vel_y


def _log(message: str) -> None:
    print(message, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='UR7e predictive dynamic-ball reaching demo')
    parser.add_argument('--task-file', default=os.path.join(STORM_ROOT, 'examples/sim_dynamic/config/ur7e_reacher_dynamic_ball_predict.yml'))
    parser.add_argument('--robot-file', default=os.path.join(STORM_ROOT, 'examples/sim_dynamic/config/ur7e_robot_dynamic_ball.yml'))
    parser.add_argument('--world-file', default=os.path.join(STORM_ROOT, 'examples/sim_dynamic/config/collision_world_dynamic_ball_predict.yml'))
    parser.add_argument('--rate', type=float, default=50.0)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--max-steps', type=int, default=0)
    args = parser.parse_args(argv)

    with open(args.robot_file) as f:
        robot_params = yaml.safe_load(f)
    with open(args.world_file) as f:
        world_params = yaml.safe_load(f)
    with open(args.task_file) as f:
        task_params = yaml.safe_load(f)

    sim_params = robot_params.get('sim_params', {})
    robot_pose = sim_params.get('robot_pose', [0, 0, 0, 0, 0, 0, 1])
    robot_pos = np.array(robot_pose[:3], dtype=np.float64)
    robot_quat_xyzw = np.array(robot_pose[3:7], dtype=np.float64)
    joint_names = sim_params.get('gazebo', {}).get('joint_names', [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ])

    dynamic_cfg = world_params['world_model']['dynamic_obstacles']['dynamic_ball']
    dynamic_ball_name = 'dynamic_ball'
    dynamic_ball_pos_world = np.array(dynamic_cfg['initial_position'], dtype=np.float64)
    dynamic_ball_topic = str(dynamic_cfg.get('topic', '/dynamic_ball/pose'))
    dynamic_ball_velocity_topic = str(dynamic_cfg.get('velocity_topic', '/dynamic_ball/velocity'))
    dynamic_ball_nominal_speed = float(dynamic_cfg.get('speed', 0.1))
    dynamic_ball_y_limits = [float(v) for v in dynamic_cfg.get('y_limits', [-0.18, 0.08])]
    log_every = int(task_params.get('task', {}).get('log_dynamic_ball_every', 25))
    default_goal_world = np.array(task_params.get('task', {}).get('default_goal_world', [0.5, -0.45, 0.4]), dtype=np.float64)

    rclpy.init(args=None)
    robot = DynamicGazeboRobotInterfacePredict(
        joint_names,
        dynamic_ball_topic=dynamic_ball_topic,
        dynamic_ball_velocity_topic=dynamic_ball_velocity_topic,
        control_rate=args.rate,
    )
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(robot)

    running = [True]
    def shutdown_handler(sig, frame):
        _log('\n收到退出信号，准备退出...')
        running[0] = False
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    spin_thread = None
    mpc = None
    try:
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        _log('=' * 60)
        _log('UR7e Predictive Dynamic Primitive Ball Reaching Demo')
        _log('=' * 60)
        _log(f'Robot: {args.robot_file}')
        _log(f'Task:  {args.task_file}')
        _log(f'World: {args.world_file}')
        _log(f'Dynamic ball topic: {dynamic_ball_topic}')
        _log('predictive_dynamic_obstacle_enabled=True')

        start_wait = time.time()
        while not robot.is_connected():
            if time.time() - start_wait > 10.0:
                raise RuntimeError('无法接收 /joint_states, 请确认 Gazebo 已启动')
            time.sleep(0.1)
        _log('已连接到 Gazebo 机器人!')

        n_spheres, n_cubes = count_primitive_obstacles(world_params, skip_names=[dynamic_ball_name])
        spawn_ok = spawn_static_obstacles(robot, world_params, model_prefix='sim_dynamic_predict', skip_names=[dynamic_ball_name])
        _log(f'Gazebo 静态障碍物生成: spheres={n_spheres} cubes={n_cubes} success={spawn_ok}')
        dynamic_spawn_ok = spawn_obstacle_model(
            robot,
            dynamic_ball_name,
            build_dynamic_sphere_sdf(dynamic_ball_name, float(dynamic_cfg['radius'])),
            dynamic_ball_pos_world.tolist(),
            service_timeout_sec=8.0,
        )
        _log(f'Gazebo 动态球生成: success={dynamic_spawn_ok} position={np.round(dynamic_ball_pos_world, 3)}')

        device = 'cuda' if args.cuda else 'cpu'
        tensor_args = {
            'device': torch.device(device, 0) if device == 'cuda' else torch.device('cpu'),
            'dtype': torch.float32,
        }
        mpc = DynamicGazeboReacherTaskPredict(args.task_file, args.robot_file, args.world_file, tensor_args)
        mpc.set_position_only_goal_mode()
        control_dt = mpc.exp_params.get('control_dt', 0.02)
        rollout_fn = mpc.controller.rollout_fn
        collision_sphere_visualizer = CollisionSphereVisualizer(mpc.exp_params['model']['robot_collision_params'])

        current_state = robot.get_state()
        if current_state is None:
            raise RuntimeError('已连接 Gazebo 但无法读取当前关节状态')
        current_q = current_state['position']
        current_dq = current_state['velocity']
        current_ddq = current_state['acceleration']
        current_pose = rollout_fn.get_ee_pose(
            torch.as_tensor(np.hstack([current_q, current_dq, current_ddq]), **tensor_args).unsqueeze(0)
        )
        current_goal_quat = np.ravel(current_pose['ee_quat_seq'].detach().cpu().numpy())

        current_goal_world = default_goal_world.copy()
        current_goal_ee = inv_transform_point(robot_pos, robot_quat_xyzw, current_goal_world)
        mpc.update_params(goal_ee_pos=current_goal_ee, goal_ee_quat=current_goal_quat)
        _log(f'默认目标末端位置 (世界坐标系): {np.round(current_goal_world, 3)}')
        _log(f'动态球初始位置 (世界坐标系): {np.round(dynamic_ball_pos_world, 3)}')

        for _ in range(5):
            state = robot.get_state()
            if state is not None:
                mpc.set_dynamic_sphere_state_world(dynamic_ball_name, dynamic_ball_pos_world, dynamic_ball_nominal_speed)
                _get_sync_command(mpc, 0.0, state, control_dt)
            time.sleep(0.01)

        loop_index = 0
        loop_start = time.time()
        while running[0] and rclpy.ok():
            iter_start = time.time()
            t = time.time() - loop_start
            state = robot.get_state()
            if state is None:
                time.sleep(control_dt)
                continue

            new_target = robot.get_target_position()
            if new_target is not None:
                target_robot = inv_transform_point(robot_pos, robot_quat_xyzw, new_target)
                if np.linalg.norm(target_robot - current_goal_ee) > 0.005:
                    current_goal_ee = target_robot.copy()
                    current_goal_world = new_target.copy()
                    mpc.update_params(goal_ee_pos=current_goal_ee, goal_ee_quat=current_goal_quat)
                    _log(f'[目标更新] world={np.round(current_goal_world, 3)} robot={np.round(current_goal_ee, 3)}')

            current_time_sec = robot.get_clock().now().nanoseconds * 1.0e-9
            estimated_ball_pos, dynamic_ball_vel_y = robot.get_dynamic_ball_state_estimate(
                dynamic_ball_nominal_speed,
                dynamic_ball_y_limits,
                current_time_sec,
            )
            if estimated_ball_pos is not None:
                dynamic_ball_pos_world = estimated_ball_pos

            mpc.set_dynamic_sphere_state_world(dynamic_ball_name, dynamic_ball_pos_world, dynamic_ball_vel_y)
            world_params['world_model']['coll_objs']['sphere'][dynamic_ball_name]['position'] = dynamic_ball_pos_world.tolist()

            try:
                cmd = _get_sync_command(mpc, t, state, control_dt)
            except (IndexError, RuntimeError, ValueError) as exc:
                _log(f'[MPC恢复] 同步取命令失败 ({exc})，重置时间基准后重规划')
                cmd = _recover_command(mpc, t, state, control_dt)

            if cmd is None or 'position' not in cmd:
                time.sleep(control_dt)
                continue

            target_positions = cmd['position']
            if isinstance(target_positions, torch.Tensor):
                target_positions = target_positions.detach().cpu().numpy()
            target_positions = np.asarray(target_positions).flatten()[: len(joint_names)]
            robot.send_position_command(target_positions)

            q = state['position']
            dq = state['velocity']
            ddq = state['acceleration']
            curr = np.hstack([q, dq, ddq])
            ee_pose = rollout_fn.get_ee_pose(torch.as_tensor(curr, **tensor_args).unsqueeze(0))
            ee_pos_robot = np.ravel(ee_pose['ee_pos_seq'].detach().cpu().numpy())
            ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos_robot)
            robot.publish_ee_pose(ee_pos_world)

            link_pos_robot, link_rot_robot = _compute_link_poses_robot_frame(rollout_fn, q, dq, tensor_args)
            collision_spheres_world = collision_sphere_visualizer.get_world_spheres(
                link_pos_robot,
                link_rot_robot,
                robot_pos,
                robot_quat_xyzw,
            )
            robot.publish_markers(world_params, current_goal_world, ee_pos_world, collision_spheres=collision_spheres_world)
            top_trajs_world = _get_top_ee_trajs_world(mpc, robot_pos, robot_quat_xyzw, ee_pos_world, max_trajs=5)
            robot.publish_top_trajectories(top_trajs_world)

            if loop_index % log_every == 0:
                ee_error = float(np.linalg.norm(ee_pos_world - current_goal_world))
                metrics = mpc.get_dynamic_ball_metrics()
                _log(
                    f'[{loop_index:5d}] t={t:.2f}s | ee_error={ee_error:.4f}m | '
                    f'predictive_dynamic_obstacle_enabled={metrics["predictive_dynamic_obstacle_enabled"]} | '
                    f'dynamic_ball_pos={np.round(dynamic_ball_pos_world, 3).tolist()} | '
                    f'dynamic_ball_vel_y={float(dynamic_ball_vel_y):.3f} | '
                    f'min_dynamic_ball_distance={metrics["min_dynamic_ball_distance"]:.4f} | '
                    f'min_dynamic_ball_margin={metrics["min_dynamic_ball_margin"]:.4f} | '
                    f'dynamic_collision_violation_count={metrics["dynamic_collision_violation_count"]} | '
                    f'opt_dt={mpc.opt_dt:.3f}s'
                )

            loop_index += 1
            if args.max_steps > 0 and loop_index >= args.max_steps:
                _log(f'达到 max_steps={args.max_steps}, 退出 smoke run')
                break

            sleep_time = (1.0 / args.rate) - (time.time() - iter_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        running[0] = False
        try:
            _shutdown_control_process(getattr(mpc, 'control_process', None))
        except Exception:
            pass
        try:
            executor.shutdown(timeout_sec=0.0)
        except Exception:
            pass
        try:
            robot.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
