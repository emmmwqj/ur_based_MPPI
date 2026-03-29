#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug entry for UR7e STORM MPC tall-scene control.

This keeps the same control path as reach_static_ur7e_tall.py, but when the
end-effector is still far from the goal and motion nearly stalls, it captures
controller sampling diagnostics to debug local minima:
- covariance / scale_tril / mean action
- sampled action sequences
- sampled ee trajectories
- sampled costs / total costs
- current robot state and goal state
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

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
except ImportError:
    print('=' * 60)
    print('错误: 未找到 ROS2 Python 包')
    print('请先 source ROS2 环境:')
    print('  source /opt/ros/humble/setup.bash')
    print('=' * 60)
    sys.exit(1)

from examples.sim_gazebo.reach_static_ur7e import GazeboReacherTask, inv_transform_point, transform_point
from examples.sim_gazebo.reach_static_ur7e_tall import (
    _compute_link_poses_robot_frame,
    _get_top_ee_trajs_world,
    _log,
    _recover_command,
    _reset_control_process_timing,
    _shutdown_control_process,
    CollisionSphereVisualizer,
    TallGazeboRobotInterface,
)
from examples.sim_gazebo.gazebo_obstacle_utils import count_primitive_obstacles, spawn_gazebo_obstacles

np.set_printoptions(precision=3, suppress=True)


def _get_base_init_cov(controller) -> float:
    base_init_cov = getattr(controller, "_debug_base_init_cov", None)
    if base_init_cov is None:
        base_init_cov = float(controller.init_cov)
        controller._debug_base_init_cov = base_init_cov
    return float(base_init_cov)


def _get_base_step_size_mean(controller) -> float:
    base_step_size_mean = getattr(controller, "_debug_base_step_size_mean", None)
    if base_step_size_mean is None:
        base_step_size_mean = float(controller.step_size_mean)
        controller._debug_base_step_size_mean = base_step_size_mean
    return float(base_step_size_mean)


def _set_controller_distribution(mpc, cov_scale: float, step_size_mean: float | None = None) -> None:
    controller = mpc.controller
    base_init_cov = _get_base_init_cov(controller)
    base_step = _get_base_step_size_mean(controller)
    controller.init_cov = float(base_init_cov * cov_scale)
    controller.step_size_mean = float(base_step if step_size_mean is None else step_size_mean)
    controller.reset()


def _restore_controller_distribution(mpc) -> None:
    controller = mpc.controller
    controller.init_cov = _get_base_init_cov(controller)
    controller.step_size_mean = _get_base_step_size_mean(controller)
    controller.reset()


class StallDebugRecorder:
    def __init__(
        self,
        output_dir: Path,
        history_len: int = 50,
        min_runtime: float = 8.0,
        error_threshold: float = 0.12,
        motion_threshold: float = 0.01,
        velocity_threshold: float = 0.08,
        capture_cooldown: float = 8.0,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = deque(maxlen=history_len)
        self.min_runtime = float(min_runtime)
        self.error_threshold = float(error_threshold)
        self.motion_threshold = float(motion_threshold)
        self.velocity_threshold = float(velocity_threshold)
        self.capture_cooldown = float(capture_cooldown)
        self.last_capture_t = -1e9
        self.capture_count = 0

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _history_motion(self) -> float:
        if len(self.history) < 2:
            return np.inf
        points = np.asarray([item['ee_pos_world'] for item in self.history], dtype=np.float64)
        ref = points[0]
        disp = np.linalg.norm(points - ref.reshape(1, 3), axis=1)
        return float(np.max(disp))

    def update(self, t_step: float, ee_pos_world: np.ndarray) -> None:
        self.history.append({
            't_step': float(t_step),
            'ee_pos_world': np.asarray(ee_pos_world, dtype=np.float64).copy(),
        })

    def maybe_capture(
        self,
        mpc,
        t_step: float,
        state: dict,
        ee_pos_world: np.ndarray,
        goal_world: np.ndarray,
    ) -> str | None:
        self.update(t_step, ee_pos_world)
        if t_step < self.min_runtime:
            return None
        if len(self.history) < self.history.maxlen:
            return None
        if t_step - self.last_capture_t < self.capture_cooldown:
            return None

        ee_error = float(np.linalg.norm(ee_pos_world - goal_world))
        history_motion = self._history_motion()
        velocity_norm = float(np.linalg.norm(state['velocity']))
        if ee_error <= self.error_threshold:
            return None
        if history_motion >= self.motion_threshold:
            return None
        if velocity_norm >= self.velocity_threshold:
            return None

        controller = getattr(mpc, 'controller', None)
        trajectories = getattr(controller, 'trajectories', None)
        total_costs = self._to_numpy(getattr(controller, 'total_costs', None))
        mean_action = self._to_numpy(getattr(controller, 'mean_action', None))
        best_traj = self._to_numpy(getattr(controller, 'best_traj', None))
        cov_action = self._to_numpy(getattr(controller, 'cov_action', None))
        scale_tril = self._to_numpy(getattr(controller, 'scale_tril', None))
        full_scale_tril = self._to_numpy(getattr(controller, 'full_scale_tril', None))
        best_idx = getattr(controller, 'best_idx', None)

        if trajectories is None or total_costs is None:
            return None

        actions = self._to_numpy(trajectories.get('actions'))
        costs = self._to_numpy(trajectories.get('costs'))
        ee_pos_seq = self._to_numpy(trajectories.get('ee_pos_seq'))
        state_seq = self._to_numpy(trajectories.get('state_seq'))
        if actions is None or costs is None or ee_pos_seq is None:
            return None

        ee_world_seq = np.asarray(
            [transform_point([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], traj) for traj in ee_pos_seq],
            dtype=np.float64,
        )

        top_indices = np.argsort(total_costs)[:5]
        top_total_costs = total_costs[top_indices]
        action_std = np.std(actions.reshape(actions.shape[0], -1), axis=0)
        action_std_per_joint = np.std(actions, axis=(0, 1))

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = f'stall_capture_{timestamp}_{self.capture_count:02d}'
        npz_path = self.output_dir / f'{stem}.npz'
        json_path = self.output_dir / f'{stem}.json'

        np.savez_compressed(
            npz_path,
            t_step=np.array(t_step, dtype=np.float64),
            ee_error=np.array(ee_error, dtype=np.float64),
            history_motion=np.array(history_motion, dtype=np.float64),
            velocity_norm=np.array(velocity_norm, dtype=np.float64),
            goal_world=np.asarray(goal_world, dtype=np.float64),
            ee_pos_world=np.asarray(ee_pos_world, dtype=np.float64),
            q=np.asarray(state['position'], dtype=np.float64),
            dq=np.asarray(state['velocity'], dtype=np.float64),
            ddq=np.asarray(state['acceleration'], dtype=np.float64),
            total_costs=total_costs,
            top_indices=top_indices,
            top_total_costs=top_total_costs,
            mean_action=mean_action,
            best_traj=best_traj,
            best_idx=np.array(-1 if best_idx is None else int(best_idx), dtype=np.int64),
            cov_action=np.array([]) if cov_action is None else cov_action,
            scale_tril=np.array([]) if scale_tril is None else scale_tril,
            full_scale_tril=np.array([]) if full_scale_tril is None else full_scale_tril,
            sample_actions=actions,
            sample_cost_seq=costs,
            sample_ee_pos_seq=ee_pos_seq,
            sample_ee_world_seq=ee_world_seq,
            sample_state_seq=np.array([]) if state_seq is None else state_seq,
            action_std_per_joint=action_std_per_joint,
            action_std_flat=action_std,
        )

        summary = {
            't_step': float(t_step),
            'ee_error_m': ee_error,
            'history_motion_m': history_motion,
            'velocity_norm': velocity_norm,
            'goal_world': np.asarray(goal_world, dtype=np.float64).round(6).tolist(),
            'ee_pos_world': np.asarray(ee_pos_world, dtype=np.float64).round(6).tolist(),
            'top_indices': top_indices.tolist(),
            'top_total_costs': np.asarray(top_total_costs, dtype=np.float64).round(6).tolist(),
            'total_cost_min': float(np.min(total_costs)),
            'total_cost_mean': float(np.mean(total_costs)),
            'total_cost_max': float(np.max(total_costs)),
            'action_std_per_joint': np.asarray(action_std_per_joint, dtype=np.float64).round(6).tolist(),
            'cov_action_shape': [] if cov_action is None else list(np.shape(cov_action)),
            'cov_action': [] if cov_action is None else np.asarray(cov_action).round(8).tolist(),
            'scale_tril_shape': [] if scale_tril is None else list(np.shape(scale_tril)),
            'scale_tril': [] if scale_tril is None else np.asarray(scale_tril).round(8).tolist(),
            'npz_path': str(npz_path),
        }
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

        self.last_capture_t = t_step
        self.capture_count += 1

        _log(
            '[DEBUG捕获] 末端停滞且离目标较远: '
            f'ee_error={ee_error:.4f}m, history_motion={history_motion:.4f}m, '
            f'vel_norm={velocity_norm:.4f}, saved={npz_path.name}'
        )
        _log(
            '[DEBUG采样] total_cost min/mean/max=%.4f / %.4f / %.4f, top5=%s'
            % (
                float(np.min(total_costs)),
                float(np.mean(total_costs)),
                float(np.max(total_costs)),
                np.asarray(top_total_costs, dtype=np.float64).round(4).tolist(),
            )
        )
        _log(
            '[DEBUG协方差] cov_action=%s scale_tril=%s action_std_per_joint=%s'
            % (
                'None' if cov_action is None else np.asarray(cov_action).round(6).tolist(),
                'None' if scale_tril is None else np.asarray(scale_tril).round(6).tolist(),
                np.asarray(action_std_per_joint, dtype=np.float64).round(6).tolist(),
            )
        )
        return str(npz_path)


def mpc_control_main(args):
    shutdown_event = threading.Event()
    executor = None
    spin_thread = None
    robot = None
    mpc = None
    exit_code = 0

    script_dir = Path(__file__).resolve().parent
    captures_dir = script_dir / 'captures'
    captures_dir.mkdir(parents=True, exist_ok=True)
    recorder = StallDebugRecorder(captures_dir)

    def request_shutdown(signum=None, _frame=None):
        if shutdown_event.is_set():
            return
        if signum is not None:
            _log(f'\n收到 {signal.Signals(signum).name}，准备退出...')
        shutdown_event.set()

    try:
        _log('=' * 60)
        _log('UR7e STORM MPC Reach Static - Gazebo 高墙场景 DEBUG')
        _log('=' * 60)

        config_dir = Path(__file__).resolve().parents[1] / 'config'
        robot_file = str(config_dir / 'ur7e_robot_gazebo.yml')
        task_file = str(config_dir / 'ur7e_reacher_gazebo_tall.yml')
        world_file = str(config_dir / 'collision_world_gazebo_tall.yml')

        _log('\n加载配置文件...')
        _log(f'  Robot: {robot_file}')
        _log(f'  Task:  {task_file}')
        _log(f'  World: {world_file}')
        _log(f'  Debug captures: {captures_dir}')

        with open(robot_file) as f:
            robot_params = yaml.safe_load(f)
        with open(world_file) as f:
            world_params = yaml.safe_load(f)

        sim_params = robot_params.get('sim_params', {})
        robot_pose = sim_params.get('robot_pose', [0, 0, 0, 0, 0, 0, 1])
        robot_pos = np.array(robot_pose[:3], dtype=np.float64)
        robot_quat_xyzw = np.array(robot_pose[3:], dtype=np.float64)

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

        control_rate = args.rate
        robot = TallGazeboRobotInterface(joint_names, control_rate=control_rate)

        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(robot)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        _log('\n等待 Gazebo 关节状态...')
        timeout = 10.0
        start = time.time()
        while not robot.is_connected():
            if shutdown_event.is_set():
                return 130
            if time.time() - start > timeout:
                _log('错误: 无法接收关节状态，请确保 Gazebo 仿真正在运行')
                return 1
            time.sleep(0.1)

        _log('已连接到 Gazebo 机器人!')
        n_world_spheres, n_world_cubes = count_primitive_obstacles(world_params, include_ground=False)
        if spawn_gazebo_obstacles(robot, world_params, model_prefix='sim_tall_debug', include_ground=False):
            _log('Gazebo 真实障碍物已生成: spheres=%d cubes=%d' % (n_world_spheres, n_world_cubes))
        else:
            _log('警告: Gazebo 真实障碍物未完整生成，请检查 /spawn_entity 与 /delete_entity 服务')

        _log('\n初始化 STORM MPC 控制器...')
        device = 'cuda' if args.cuda else 'cpu'
        _log(f'计算设备: {device}')
        tensor_args = {
            'device': torch.device(device, 0) if device == 'cuda' else torch.device('cpu'),
            'dtype': torch.float32,
        }

        mpc = GazeboReacherTask(task_file, robot_file, world_file, tensor_args)
        control_dt = mpc.exp_params.get('control_dt', 0.02)
        _log(f'MPC 控制周期: {control_dt} s ({1.0 / control_dt:.1f} Hz)')

        default_goal_seed_state = np.array([0.5, -1.2, 1.2, -1.57, -1.57, 0.0, 0, 0, 0, 0, 0, 0])
        mpc.update_params(goal_state=default_goal_seed_state)

        goal_ee_quat = np.ravel(mpc.controller.rollout_fn.goal_ee_quat.cpu().numpy())
        goal_ee_world = np.array([0.5, -0.45, 0.4], dtype=np.float64)
        goal_ee_pos_robot = inv_transform_point(robot_pos, robot_quat_xyzw, goal_ee_world)
        mpc.update_params(goal_ee_pos=goal_ee_pos_robot, goal_ee_quat=goal_ee_quat)

        _log(f'\n默认目标末端位置 (机器人坐标系): {goal_ee_pos_robot}')
        _log(f'默认目标末端位置 (世界坐标系): {goal_ee_world}')

        rollout_fn = mpc.controller.rollout_fn
        collision_sphere_visualizer = CollisionSphereVisualizer(mpc.exp_params['model']['robot_collision_params'])
        n_collision_spheres = sum(
            len(collision_sphere_visualizer.spheres_by_link.get(link_name, []))
            for link_name in collision_sphere_visualizer.link_names
        )
        current_goal_ee = goal_ee_pos_robot.copy()
        current_goal_world = goal_ee_world.copy()

        _log('\n' + '=' * 60)
        _log('开始 MPC DEBUG 控制循环... (Ctrl+C 退出)')
        _log('=' * 60)
        _log('\n提示:')
        _log('  - 当前使用高墙场景 primitive world 避障')
        _log('  - 发布 PoseStamped 到 /target_pose 可动态更新目标')
        _log('  - 末端停滞且远离目标时，会自动保存采样调试数据')
        _log('  - 在 RViz 中查看 /visualization_marker_array')
        _log('  - 在 RViz 中查看 /mppi_top_traj_markers (MPPI top-5 预测轨迹)')
        _log('  - 红线=MPPI top-5 末端预测轨迹')
        _log(f'  - 黄球=机械臂碰撞球模型 ({n_collision_spheres} 个)')
        _log('')

        _log('预热 MPC 控制器...')
        current_state = robot.get_state()
        for warm_idx in range(5):
            if shutdown_event.is_set():
                return 130
            if current_state is None:
                time.sleep(0.01)
                current_state = robot.get_state()
                continue
            try:
                mpc.get_command(warm_idx * control_dt, current_state, control_dt=control_dt, WAIT=True)
            except Exception as exc:
                _log(f'预热异常 (可忽略): {exc}')
            time.sleep(0.01)

        _log('MPC 预热完成，开始控制!\n')

        loop_count = 0
        marker_update_counter = 0
        loop_start = time.time()

        while rclpy.ok() and not shutdown_event.is_set():
            iter_start = time.time()
            t_step = time.time() - loop_start
            cmd = None

            state = robot.get_state()
            if state is None:
                time.sleep(control_dt)
                continue

            q = state['position']
            dq = state['velocity']
            ddq = state['acceleration']

            new_target = robot.get_target_position()
            if new_target is not None:
                target_robot = inv_transform_point(robot_pos, robot_quat_xyzw, new_target)
                if np.linalg.norm(target_robot - current_goal_ee) > 0.005:
                    current_goal_ee = target_robot.copy()
                    current_goal_world = new_target.copy()
                    mpc.update_params(goal_ee_pos=current_goal_ee, goal_ee_quat=goal_ee_quat)
                    _log('[目标更新] 世界: %s, 机器人: %s' % (np.round(current_goal_world, 3), np.round(current_goal_ee, 3)))
                    try:
                        _set_controller_distribution(mpc, cov_scale=9.0, step_size_mean=0.45)
                        _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                        cmd = mpc.get_command(t_step, state, control_dt=control_dt, WAIT=True)
                        _log('[目标更新] 已重置时间基准与采样分布，并放大探索协方差')
                    except Exception as sync_exc:
                        _log(f'[MPC异常] 目标更新后的同步重规划失败: {sync_exc}')
                        time.sleep(control_dt)
                        continue

            if cmd is None:
                try:
                    cmd = mpc.get_command(t_step, state, control_dt=control_dt, WAIT=True)
                except (IndexError, RuntimeError, ValueError) as exc:
                    _log('[MPC恢复] 同步取命令失败 (%s)，重置控制进程时间基准后重规划' % exc)
                    try:
                        cmd = _recover_command(mpc, t_step, state, control_dt)
                    except Exception as recover_exc:
                        _log(f'[MPC异常] 同步重规划失败: {recover_exc}')
                        time.sleep(control_dt)
                        continue

            if cmd is None or 'position' not in cmd:
                time.sleep(control_dt)
                continue

            target_positions = cmd['position']
            if isinstance(target_positions, torch.Tensor):
                target_positions = target_positions.cpu().numpy()
            target_positions = np.array(target_positions).flatten()[:n_dof]
            robot.send_position_command(target_positions)

            curr = np.hstack([q, dq, ddq])
            ee_pose = rollout_fn.get_ee_pose(torch.as_tensor(curr, **tensor_args).unsqueeze(0))
            ee_pos_robot = np.ravel(ee_pose['ee_pos_seq'].cpu().numpy())
            ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos_robot)
            robot.publish_ee_pose(ee_pos_world)

            capture_path = recorder.maybe_capture(mpc, t_step, state, ee_pos_world, current_goal_world)
            if capture_path is not None:
                _set_controller_distribution(mpc, cov_scale=16.0, step_size_mean=0.35)
                _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                _log('[DEBUG恢复] 已放大采样协方差并重置时间基准，尝试跳出当前局部 basin')

            if np.linalg.norm(ee_pos_world - current_goal_world) < 0.12:
                controller = mpc.controller
                if (
                    abs(float(controller.init_cov) - _get_base_init_cov(controller)) > 1e-9
                    or abs(float(controller.step_size_mean) - _get_base_step_size_mean(controller)) > 1e-9
                ):
                    _restore_controller_distribution(mpc)
                    _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                    _log('[DEBUG恢复] 已恢复默认采样分布')

            marker_update_counter += 1
            if marker_update_counter >= 10:
                link_pos_robot, link_rot_robot = _compute_link_poses_robot_frame(rollout_fn, q, dq, tensor_args)
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
                top_trajs_world = _get_top_ee_trajs_world(mpc, robot_pos, robot_quat_xyzw, max_trajs=5)
                robot.publish_top_trajectories(top_trajs_world)
                marker_update_counter = 0

            loop_count += 1
            if loop_count % 50 == 0:
                error = np.linalg.norm(ee_pos_world - current_goal_world)
                _log(
                    f'[{loop_count:5d}] t={t_step:.2f}s | '
                    f'q=[{q[0]:+.2f}, {q[1]:+.2f}, {q[2]:+.2f}, {q[3]:+.2f}, {q[4]:+.2f}, {q[5]:+.2f}] | '
                    f'ee_error={error:.4f} | opt_dt={mpc.opt_dt:.3f}s'
                )

            elapsed = time.time() - iter_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if shutdown_event.is_set():
            exit_code = 130
        return exit_code
    finally:
        _log('\n清理资源...')
        shutdown_event.set()

        if mpc is not None:
            try:
                _shutdown_control_process(getattr(mpc, 'control_process', None))
            except Exception as exc:
                _log(f'关闭 MPC 资源时出现异常: {exc}')

        if robot is not None:
            try:
                robot.destroy_node()
            except Exception as exc:
                _log(f'销毁 ROS2 节点时出现异常: {exc}')

        if executor is not None:
            try:
                executor.shutdown(timeout_sec=0.0)
            except TypeError:
                pass
            except Exception as exc:
                _log(f'关闭 ROS2 executor 时出现异常: {exc}')

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as exc:
                _log(f'关闭 ROS2 时出现异常: {exc}')

        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=1.0)
            if spin_thread.is_alive():
                _log('ROS2 spin 线程未在超时内退出')

        _log('程序结束')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UR7e STORM MPC Reach Static Gazebo Tall Scene DEBUG')
    parser.add_argument('--cuda', action='store_true', default=True, help='使用 CUDA 加速 (默认: True)')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='禁用 CUDA')
    parser.add_argument('--rate', type=float, default=50.0, help='控制频率 Hz (默认: 50)')
    args = parser.parse_args()
    sys.exit(mpc_control_main(args))
