#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple SAGE-MPPI predictive dynamic-ball reaching demo without deployment extras."""

from __future__ import annotations

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
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped, TwistStamped
except ImportError:
    print('=' * 60)
    print('错误: 未找到 ROS2 Python 包')
    print('请先 source ROS2 环境: source /opt/ros/humble/setup.bash')
    print('=' * 60)
    sys.exit(1)

from examples.sage_sim_dynamic.scripts.gazebo_dynamic_utils import (
    build_dynamic_sphere_sdf,
    count_primitive_obstacles,
    spawn_obstacle_model,
    spawn_static_obstacles,
)
from examples.sage_sim_dynamic.scripts.sage_dynamic_reacher_task_predict_simple import (
    SageDynamicReacherTaskPredictSimple,
)
from examples.sim_gazebo.reach_static_ur7e import (
    GazeboRobotInterface,
    CollisionSphereVisualizer,
    transform_point,
    inv_transform_point,
    _get_top_ee_trajs_world,
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

    def get_dynamic_ball_state_estimate(self, nominal_speed: float, y_limits, current_time_sec: float):
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


def _reset_control_process_timing_strict(control_process, t_step: float, control_dt: float) -> None:
    control_process.command = None
    control_process.command_tstep = control_process.traj_tstep + t_step + control_dt
    control_process.prev_mpc_tstep = max(0.0, t_step - control_dt)
    control_process.mpc_dt = control_dt
    control_process.params = None
    _drain_mp_queue(getattr(control_process, 'result_queue', None))
    _drain_mp_queue(getattr(control_process, 'opt_queue', None))


def _recover_command_strict(mpc, t_step: float, state: dict, control_dt: float):
    _reset_control_process_timing_strict(mpc.control_process, t_step, control_dt)
    return mpc.get_command(t_step, state, control_dt=control_dt, WAIT=True)


def _configure_default_goal(mpc, robot_pos_world, robot_quat_xyzw):
    default_goal_world = np.array(
        mpc.exp_params.get('task', {}).get('default_goal_world', [0.5, -0.45, 0.4]),
        dtype=np.float64,
    )
    default_goal_seed_state = np.array([0.5, -1.2, 1.2, -1.57, -1.57, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    mpc.update_params(goal_state=default_goal_seed_state)
    goal_ee_quat = np.ravel(mpc.controller.rollout_fn.goal_ee_quat.detach().cpu().numpy())
    goal_ee_pos_robot = inv_transform_point(robot_pos_world, robot_quat_xyzw, default_goal_world)
    mpc.update_params(goal_ee_pos=goal_ee_pos_robot, goal_ee_quat=goal_ee_quat)
    return goal_ee_pos_robot, goal_ee_quat, default_goal_world.copy()


def _get_execution_mode(mpc) -> str:
    mppi_cfg = getattr(mpc, "exp_params", {}).get("mppi", {})
    mode = str(mppi_cfg.get("execution_mode", "mean")).strip().lower()
    if mode not in ("best_sample", "mean"):
        return "mean"
    return mode


def _apply_execution_mode(mpc) -> str:
    mode = _get_execution_mode(mpc)
    controller = getattr(mpc, "controller", None)
    if controller is None:
        return mode
    use_best = mode == "best_sample"
    if hasattr(controller, "execute_best"):
        controller.execute_best = use_best
    return mode


def executor_spin_loop(executor, spin_running, shutdown_event):
    while spin_running[0] and rclpy.ok() and not shutdown_event.is_set():
        try:
            executor.spin_once(timeout_sec=0.1)
        except RuntimeError as exc:
            if 'Destroyable' in str(exc):
                break
            raise


def _compute_ee_and_link_poses_robot_frame(rollout_fn, q, dq, tensor_args):
    q_t = torch.as_tensor(np.asarray(q, dtype=np.float64), **tensor_args).unsqueeze(0)
    dq_t = torch.as_tensor(np.asarray(dq, dtype=np.float64), **tensor_args).unsqueeze(0)
    robot_model = rollout_fn.dynamics_model.robot_model
    ee_link_name = rollout_fn.exp_params['model']['ee_link_name']
    robot_model.compute_fk_and_jacobian(q_t, dq_t, ee_link_name)

    ee_pos, _ = robot_model.get_link_pose(ee_link_name)
    ee_pos_robot = ee_pos[0].detach().cpu().numpy()

    link_pos = []
    link_rot = []
    for link_name in rollout_fn.dynamics_model.link_names:
        pos, rot = robot_model.get_link_pose(link_name)
        link_pos.append(pos[0].detach().cpu().numpy())
        link_rot.append(rot[0].detach().cpu().numpy())

    return ee_pos_robot, np.stack(link_pos, axis=0), np.stack(link_rot, axis=0)


def _evaluate_position_command_dynamic_margin(mpc, q_current, q_target, eval_horizon_steps):
    rollout_fn = mpc.controller.rollout_fn
    dyn_model = rollout_fn.dynamics_model
    robot_model = dyn_model.robot_model
    ee_link_name = rollout_fn.exp_params['model']['ee_link_name']
    q_current = np.asarray(q_current, dtype=np.float64).reshape(-1)
    q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)
    dq_zero = np.zeros_like(q_current)
    alphas = np.linspace(1.0 / eval_horizon_steps, 1.0, eval_horizon_steps, dtype=np.float64)
    q_seq = q_current.reshape(1, -1) + alphas.reshape(-1, 1) * (q_target - q_current).reshape(1, -1)
    link_pos_seq = []
    link_rot_seq = []
    for q_step in q_seq:
        q_t = torch.as_tensor(q_step, **mpc.tensor_args).unsqueeze(0)
        dq_t = torch.as_tensor(dq_zero, **mpc.tensor_args).unsqueeze(0)
        robot_model.compute_fk_and_jacobian(q_t, dq_t, ee_link_name)
        link_pos_step = []
        link_rot_step = []
        for link_name in dyn_model.link_names:
            link_pos, link_rot = robot_model.get_link_pose(link_name)
            link_pos_step.append(link_pos[0])
            link_rot_step.append(link_rot[0])
        link_pos_seq.append(torch.stack(link_pos_step, dim=0))
        link_rot_seq.append(torch.stack(link_rot_step, dim=0))
    link_pos_seq = torch.stack(link_pos_seq, dim=0).unsqueeze(0)
    link_rot_seq = torch.stack(link_rot_seq, dim=0).unsqueeze(0)
    return rollout_fn.primitive_collision_cost.evaluate_link_pose_sequence(link_pos_seq, link_rot_seq)


def main():
    parser = argparse.ArgumentParser(description='UR7e SAGE predictive dynamic-ball simple baseline')
    example_root = os.path.dirname(os.path.dirname(__file__))
    parser.add_argument('--task-file', default=os.path.join(example_root, 'config', 'ur7e_reacher_sage_dynamic_ball_predict_simple.yml'))
    parser.add_argument('--robot-file', default=os.path.join(example_root, 'config', 'ur7e_robot_sage_dynamic_ball.yml'))
    parser.add_argument('--world-file', default=os.path.join(example_root, 'config', 'collision_world_sage_dynamic_ball.yml'))
    parser.add_argument('--cuda', dest='cuda', action='store_true', default=True)
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--max-steps', type=int, default=0)
    args = parser.parse_args()

    with open(args.world_file) as f:
        world_params = yaml.safe_load(f)
    with open(args.robot_file) as f:
        robot_params = yaml.safe_load(f)
    with open(args.task_file) as f:
        task_params = yaml.safe_load(f)

    tensor_args = {
        'device': torch.device('cuda', 0) if args.cuda else torch.device('cpu'),
        'dtype': torch.float32,
    }

    sim_params = robot_params.get('sim_params', {})
    robot_pose = sim_params.get('robot_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    robot_pos = np.array(robot_pose[:3], dtype=np.float64)
    robot_quat_xyzw = np.array(robot_pose[3:7], dtype=np.float64)

    dynamic_ball_cfg = world_params['world_model']['dynamic_obstacles']['dynamic_ball']
    dynamic_ball_name = 'dynamic_ball'
    dynamic_ball_pos_world = np.array(dynamic_ball_cfg.get('initial_position', [0.4, -0.6, 0.4]), dtype=np.float64)
    dynamic_ball_nominal_speed = float(dynamic_ball_cfg.get('speed', 0.1))
    dynamic_ball_y_limits = dynamic_ball_cfg.get('y_limits', [-0.6, 0.6])

    control_dt = float(task_params.get('control_dt', 0.05))
    log_every = int(task_params.get('task', {}).get('log_dynamic_ball_every', 25))
    success_threshold = float(task_params.get('task_metrics', {}).get('success_threshold', 0.05))
    eval_horizon_steps = int(task_params.get('task', {}).get('selected_command_eval_horizon_steps', 20))
    dynamic_diagnostic_eval_every = int(task_params.get('task', {}).get('dynamic_diagnostic_eval_every', 10))
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else None

    joint_names = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint',
    ]
    n_dof = len(joint_names)

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
            _log(f'收到 {signal.Signals(signum).name}，准备退出...')
        shutdown_event.set()

    try:
        signal.signal(signal.SIGINT, request_shutdown)
        signal.signal(signal.SIGTERM, request_shutdown)
        rclpy.init(args=None)
        robot = DynamicGazeboRobotInterfacePredict(
            joint_names,
            dynamic_ball_topic=task_params['task']['dynamic_obstacle_topic'],
            dynamic_ball_velocity_topic=task_params['task']['dynamic_obstacle_velocity_topic'],
            control_rate=50.0,
        )
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(robot)
        spin_running = [True]
        spin_thread = threading.Thread(target=executor_spin_loop, args=(executor, spin_running, shutdown_event), daemon=True)
        spin_thread.start()

        _log('=' * 60)
        _log('UR7e SAGE Predictive Dynamic Primitive Ball Simple Baseline')
        _log('=' * 60)
        _log(f'Robot: {args.robot_file}')
        _log(f'Task:  {args.task_file}')
        _log(f'World: {args.world_file}')
        _log('predictive_dynamic_obstacle_enabled=True')

        wait_deadline = time.time() + 10.0
        while rclpy.ok() and not shutdown_event.is_set():
            if robot.is_connected():
                break
            if time.time() >= wait_deadline:
                _log('错误: 无法接收关节状态')
                return 1
            time.sleep(0.05)
        _log('已连接到 Gazebo 机器人!')

        n_spheres, n_cubes = count_primitive_obstacles(world_params, skip_names=[dynamic_ball_name])
        spawn_ok = spawn_static_obstacles(
            robot,
            world_params,
            model_prefix='sage_dynamic_safe',
            skip_names=[dynamic_ball_name],
        )
        _log(f'Gazebo 静态障碍物生成: spheres={n_spheres} cubes={n_cubes} success={spawn_ok}')
        dynamic_sdf = build_dynamic_sphere_sdf(dynamic_ball_name, float(dynamic_ball_cfg['radius']))
        dynamic_spawn_ok = spawn_obstacle_model(robot, dynamic_ball_name, dynamic_sdf, tuple(dynamic_ball_pos_world.tolist()))
        _log(f'Gazebo 动态球生成: success={dynamic_spawn_ok} position={np.round(dynamic_ball_pos_world, 3)}')

        mpc = SageDynamicReacherTaskPredictSimple(args.task_file, args.robot_file, args.world_file, tensor_args)
        mpc.set_position_only_goal_mode()
        applied_execution_mode = _apply_execution_mode(mpc)
        rollout_fn = mpc.controller.rollout_fn
        world_params['world_model']['coll_objs']['sphere'][dynamic_ball_name]['position'] = dynamic_ball_pos_world.tolist()
        collision_sphere_visualizer = CollisionSphereVisualizer(mpc.exp_params['model']['robot_collision_params'])

        current_goal_ee, goal_ee_quat, current_goal_world = _configure_default_goal(
            mpc,
            robot_pos,
            robot_quat_xyzw,
        )

        _log(f'默认目标末端位置 (世界坐标系): {np.round(current_goal_world, 3)}')
        _log(f'动态球初始位置 (世界坐标系): {np.round(dynamic_ball_pos_world, 3)}')
        _log(f'execution_mode={applied_execution_mode}')
        _log('simple_baseline: deployment refinement / local refinement / hold / stall recovery / dynamic guard 已禁用')

        _log('预热 SAGE dynamic MPC 控制器...')
        current_state = robot.get_state()
        for warm_idx in range(5):
            if current_state is None:
                break
            mpc.get_command(warm_idx * control_dt, current_state, control_dt=control_dt, WAIT=True)
        _log('开始控制!')

        loop_start = time.time()
        last_wall_time = None
        loop_count = 0
        final_ee_error = float('nan')
        min_ee_error = float('inf')
        current_state_dynamic_margin_min = float('inf')
        selected_traj_dynamic_margin_min = float('inf')
        executed_dynamic_margin_min = float('inf')
        overall_dynamic_margin_min = float('inf')
        rollout_sample_violation_count_max = 0
        dynamic_collision_risk = False
        last_selected_seq_metrics = None
        last_current_state_metrics = {'min_dynamic_ball_margin': float('nan')}
        last_executed_metrics = {'min_dynamic_ball_margin': float('nan')}
        success_reported = False

        while rclpy.ok() and not shutdown_event.is_set():
            iter_start = time.time()
            t_step = time.time() - loop_start
            actual_loop_dt_wall = None if last_wall_time is None else (iter_start - last_wall_time)
            last_wall_time = iter_start

            state = robot.get_state()
            if state is None:
                time.sleep(control_dt)
                continue

            current_time_sec = robot.get_clock().now().nanoseconds * 1.0e-9
            estimated_ball_pos, dynamic_ball_vel_y = robot.get_dynamic_ball_state_estimate(
                dynamic_ball_nominal_speed,
                dynamic_ball_y_limits,
                current_time_sec,
            )
            if estimated_ball_pos is not None:
                dynamic_ball_pos_world = estimated_ball_pos
            if dynamic_ball_vel_y is None:
                dynamic_ball_vel_y = dynamic_ball_nominal_speed
            mpc.set_dynamic_sphere_state_world(dynamic_ball_name, dynamic_ball_pos_world, dynamic_ball_vel_y)
            world_params['world_model']['coll_objs']['sphere'][dynamic_ball_name]['position'] = dynamic_ball_pos_world.tolist()

            q = state['position']
            dq = state['velocity']
            ddq = state['acceleration']
            curr = np.hstack([q, dq, ddq])

            new_target = robot.get_target_position()
            if new_target is not None:
                target_robot = inv_transform_point(robot_pos, robot_quat_xyzw, new_target)
                if np.linalg.norm(target_robot - current_goal_ee) > 0.005:
                    current_goal_ee = target_robot.copy()
                    current_goal_world = new_target.copy()
                    mpc.update_params(goal_ee_pos=current_goal_ee, goal_ee_quat=goal_ee_quat)
                    _log('[目标更新] 世界: %s, 机器人: %s' % (np.round(current_goal_world, 3), np.round(current_goal_ee, 3)))

            ee_pos_robot, link_pos_robot, link_rot_robot = _compute_ee_and_link_poses_robot_frame(
                rollout_fn,
                q,
                dq,
                tensor_args,
            )
            ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos_robot)
            ee_error = float(np.linalg.norm(ee_pos_world - current_goal_world))
            final_ee_error = ee_error
            min_ee_error = min(min_ee_error, ee_error)

            try:
                cmd = mpc.get_command(t_step, state, control_dt=control_dt, WAIT=True)
            except Exception as exc:
                _log(f'[SAGEDynamicSimple] 同步取命令失败，执行恢复重规划: {exc}')
                try:
                    cmd = _recover_command_strict(mpc, t_step, state, control_dt)
                except Exception as recover_exc:
                    _log(f'[SAGEDynamicSimple] 恢复失败: {recover_exc}')
                    time.sleep(control_dt)
                    continue

            nominal_position_cmd = None
            if cmd is not None and 'position' in cmd:
                nominal_position_cmd = cmd['position']
                if isinstance(nominal_position_cmd, torch.Tensor):
                    nominal_position_cmd = nominal_position_cmd.detach().cpu().numpy()
            if nominal_position_cmd is None:
                time.sleep(control_dt)
                continue
            target_positions = np.asarray(nominal_position_cmd, dtype=np.float64).flatten()[:n_dof]

            selected_seq_metrics = last_selected_seq_metrics
            current_state_metrics = last_current_state_metrics
            executed_metrics = last_executed_metrics
            run_dynamic_diagnostics = (
                loop_count == 0
                or (dynamic_diagnostic_eval_every > 0 and loop_count % dynamic_diagnostic_eval_every == 0)
            )
            if run_dynamic_diagnostics:
                selected_action_seq = mpc.get_selected_action_sequence()
                selected_seq_metrics = None
                if selected_action_seq is not None:
                    selected_seq_metrics = mpc.evaluate_action_sequence_dynamic_margin(
                        curr,
                        selected_action_seq,
                        t_step=t_step,
                        pred_mpc_dt=mpc.mpc_dt,
                    )
                current_state_metrics = mpc.evaluate_current_state_dynamic_margin(q, dq)
                executed_metrics = _evaluate_position_command_dynamic_margin(mpc, q, target_positions, eval_horizon_steps)
                last_selected_seq_metrics = selected_seq_metrics
                last_current_state_metrics = current_state_metrics
                last_executed_metrics = executed_metrics

            robot.send_position_command(target_positions)
            robot.publish_ee_pose(ee_pos_world)

            metrics = mpc.get_dynamic_ball_metrics()
            rollout_sample_violation_count_max = max(
                rollout_sample_violation_count_max,
                int(metrics['rollout_sample_violation_count']),
            )
            if run_dynamic_diagnostics:
                current_state_dynamic_margin_min = min(current_state_dynamic_margin_min, float(current_state_metrics['min_dynamic_ball_margin']))
                if selected_seq_metrics is not None:
                    selected_traj_dynamic_margin_min = min(selected_traj_dynamic_margin_min, float(selected_seq_metrics['min_dynamic_ball_margin']))
                executed_dynamic_margin_min = min(executed_dynamic_margin_min, float(executed_metrics['min_dynamic_ball_margin']))
                current_min_margin = min(
                    float(current_state_metrics['min_dynamic_ball_margin']),
                    float(executed_metrics['min_dynamic_ball_margin']),
                    float('inf') if selected_seq_metrics is None else float(selected_seq_metrics['min_dynamic_ball_margin']),
                )
                overall_dynamic_margin_min = min(overall_dynamic_margin_min, current_min_margin)
                dynamic_collision_risk = dynamic_collision_risk or any([
                    float(current_state_metrics['min_dynamic_ball_margin']) < 0.0,
                    (selected_seq_metrics is not None and float(selected_seq_metrics['min_dynamic_ball_margin']) < 0.0),
                    float(executed_metrics['min_dynamic_ball_margin']) < 0.0,
                ])

            loop_count += 1
            collision_spheres_world = collision_sphere_visualizer.get_world_spheres(
                link_pos_robot,
                link_rot_robot,
                robot_pos,
                robot_quat_xyzw,
            )
            robot.publish_markers(world_params, current_goal_world, ee_pos_world, collision_spheres=collision_spheres_world)
            top_trajs_world = _get_top_ee_trajs_world(mpc, robot_pos, robot_quat_xyzw, ee_pos_world, max_trajs=5)
            robot.publish_top_trajectories(top_trajs_world)

            if loop_count % log_every == 0 or loop_count == 0:
                _log(
                    f'[{loop_count:5d}] t={t_step:.2f}s | '
                    f'ee_error={ee_error:.4f}m | '
                    f'predictive_dynamic_obstacle_enabled={metrics["predictive_dynamic_obstacle_enabled"]} | '
                    f'dynamic_ball_pos={np.round(dynamic_ball_pos_world, 3).tolist()} | '
                    f'dynamic_ball_vel_y={float(dynamic_ball_vel_y):.3f} | '
                    f'rollout_min_dynamic_ball_margin={metrics["rollout_min_dynamic_ball_margin"]:.4f} | '
                    f'rollout_sample_violation_count={metrics["rollout_sample_violation_count"]} | '
                    f'selected_traj_min_dynamic_margin={float("nan") if selected_seq_metrics is None else selected_seq_metrics["min_dynamic_ball_margin"]:.4f} | '
                    f'executed_min_dynamic_margin={executed_metrics["min_dynamic_ball_margin"]:.4f} | '
                    f'current_state_dynamic_margin={current_state_metrics["min_dynamic_ball_margin"]:.4f} | '
                    f'actual_loop_dt_wall={float("nan") if actual_loop_dt_wall is None else actual_loop_dt_wall:.3f}s | '
                    f'opt_dt={mpc.opt_dt:.3f}s'
                )

            if max_steps is not None and loop_count >= max_steps:
                _log(f'达到 max_steps={max_steps}, 退出 SAGE dynamic simple 控制循环')
                break
            if (not success_reported) and min_ee_error <= success_threshold:
                _log(f'达到 success_threshold={success_threshold:.3f}, 保持运行并继续接受新的目标点')
                success_reported = True

            elapsed = time.time() - iter_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        reached_goal = bool(min_ee_error <= success_threshold)
        _log(
            'episode_summary '
            f'final_ee_error={final_ee_error:.4f} '
            f'min_ee_error={min_ee_error:.4f} '
            f'min_dynamic_ball_margin={overall_dynamic_margin_min:.4f} '
            f'current_state_dynamic_margin_min={current_state_dynamic_margin_min:.4f} '
            f'selected_traj_min_dynamic_margin_min={selected_traj_dynamic_margin_min:.4f} '
            f'executed_min_dynamic_margin_min={executed_dynamic_margin_min:.4f} '
            f'dynamic_collision_violation_count={rollout_sample_violation_count_max} '
            f'reached_goal={reached_goal} '
            f'dynamic_collision_risk={dynamic_collision_risk}'
        )
        return exit_code
    finally:
        shutdown_event.set()
        if mpc is not None:
            try:
                _shutdown_control_process(getattr(mpc, 'control_process', None))
            except Exception:
                pass
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=0.0)
            except TypeError:
                pass
            except Exception:
                pass
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=1.0)
        if robot is not None:
            try:
                robot.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    sys.exit(main())
