#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e Diffusion MPPI Reach Static - Gazebo high-wall scene.

Uses the SAME tall task/world config files as run_reach_static_tall.sh,
but swaps the controller from MPPI to DiffusionMPPI.
Environment collision remains primitive-world based.
"""

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
except ImportError:
    print('=' * 60)
    print('错误: 未找到 ROS2 Python 包')
    print('请先 source ROS2 环境:')
    print('  source /opt/ros/humble/setup.bash')
    print('=' * 60)
    sys.exit(1)

from examples.sim_gazebo.diffusion_gazebo_reacher_task import DiffusionGazeboReacherTask
from examples.sim_gazebo.gazebo_obstacle_utils import (
    count_primitive_obstacles,
    spawn_gazebo_obstacles,
)
from examples.sim_gazebo.reach_static_ur7e import inv_transform_point, transform_point
from examples.sim_gazebo.reach_static_ur7e_tall import (
    _compute_link_poses_robot_frame,
    _log,
    _shutdown_control_process,
    CollisionSphereVisualizer,
    TallGazeboRobotInterface,
)

np.set_printoptions(precision=3, suppress=True)


def _reset_diffusion_solver_state(mpc, t_step: float, control_dt: float) -> None:
    mpc.controller.reset()
    control_process = mpc.control_process
    control_process.command = None
    control_process.command_tstep = control_process.traj_tstep + t_step
    control_process.prev_mpc_tstep = max(0.0, t_step - control_dt)
    control_process.mpc_dt = control_dt
    control_process.params = None


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
            _log(f"\n收到 {signal.Signals(signum).name}，准备退出...")
        shutdown_event.set()

    try:
        _log('=' * 60)
        _log('UR7e Diffusion MPPI Reach Static - Gazebo 高墙场景')
        _log('=' * 60)

        config_dir = os.path.join(os.path.dirname(__file__), 'config')
        robot_file = os.path.join(config_dir, 'ur7e_robot_gazebo.yml')
        task_file = os.path.join(config_dir, 'ur7e_reacher_gazebo_tall.yml')
        world_file = os.path.join(config_dir, 'collision_world_gazebo_tall.yml')

        _log('\n加载配置文件...')
        _log(f'  Robot: {robot_file}')
        _log(f'  Task:  {task_file}')
        _log(f'  World: {world_file}')

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
        if spawn_gazebo_obstacles(robot, world_params, model_prefix='sim_tall_diff', include_ground=False):
            _log(
                'Gazebo 真实障碍物已生成: spheres=%d cubes=%d'
                % (n_world_spheres, n_world_cubes)
            )
        else:
            _log('警告: Gazebo 真实障碍物未完整生成，请检查 /spawn_entity 与 /delete_entity 服务')

        _log('\n初始化 Diffusion MPC 控制器...')
        device = 'cuda' if args.cuda else 'cpu'
        _log(f'计算设备: {device}')

        tensor_args = {
            'device': torch.device(device, 0) if device == 'cuda' else torch.device('cpu'),
            'dtype': torch.float32,
        }

        mpc = DiffusionGazeboReacherTask(task_file, robot_file, world_file, tensor_args)
        control_dt = mpc.exp_params.get('control_dt', 0.02)
        _log(f'Diffusion MPC 控制周期: {control_dt} s ({1.0 / control_dt:.1f} Hz)')

        default_goal_seed_state = np.array([0.5, -1.2, 1.2, -1.57, -1.57, 0.0, 0, 0, 0, 0, 0, 0])
        mpc.update_params(goal_state=default_goal_seed_state)

        goal_ee_quat = np.ravel(mpc.controller.rollout_fn.goal_ee_quat.cpu().numpy())
        goal_ee_world = np.array([0.5, -0.45, 0.4], dtype=np.float64)
        goal_ee_pos_robot = inv_transform_point(robot_pos, robot_quat_xyzw, goal_ee_world)
        mpc.update_params(goal_ee_pos=goal_ee_pos_robot, goal_ee_quat=goal_ee_quat)

        _log(f'\n默认目标末端位置 (机器人坐标系): {goal_ee_pos_robot}')
        _log(f'默认目标末端位置 (世界坐标系): {goal_ee_world}')

        rollout_fn = mpc.controller.rollout_fn
        collision_sphere_visualizer = CollisionSphereVisualizer(
            mpc.exp_params['model']['robot_collision_params']
        )
        n_collision_spheres = sum(
            len(collision_sphere_visualizer.spheres_by_link.get(link_name, []))
            for link_name in collision_sphere_visualizer.link_names
        )
        current_goal_ee = goal_ee_pos_robot.copy()
        current_goal_world = goal_ee_world.copy()

        _log('\n' + '=' * 60)
        _log('开始 Diffusion MPC 控制循环... (Ctrl+C 退出)')
        _log('=' * 60)
        _log('\n提示:')
        _log('  - 当前使用高墙场景 primitive world 避障')
        _log('  - 发布 PoseStamped 到 /target_pose 可动态更新目标')
        _log('  - 在 RViz 中查看 /visualization_marker_array')
        _log('  - Gazebo 中已真实生成高墙/球体障碍物，可直接观察物理碰撞')
        _log('  - 红球=目标, 绿球=末端, 蓝色障碍物=高墙场景')
        _log(f'  - 黄球=机械臂碰撞球模型 ({n_collision_spheres} 个)')
        _log('  - 控制器=DiffusionMPPI (同步求解)')
        _log('')

        _log('预热 Diffusion MPC 控制器...')
        current_state = robot.get_state()
        for warm_idx in range(3):
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

        _log('Diffusion MPC 预热完成，开始控制!\n')

        loop_count = 0
        marker_update_counter = 0
        loop_start = time.time()
        best_goal_error = float('inf')
        stagnation_steps = 0

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
                    _reset_diffusion_solver_state(mpc, t_step, control_dt)
                    best_goal_error = float('inf')
                    stagnation_steps = 0
                    _log(
                        '[目标更新] 世界: %s, 机器人: %s'
                        % (np.round(current_goal_world, 3), np.round(current_goal_ee, 3))
                    )
                    try:
                        cmd = mpc.get_command(t_step, state, control_dt=control_dt, WAIT=True)
                        _log('[目标更新] 已重置 diffusion 分布并按新目标同步重规划')
                    except Exception as exc:
                        _log(f'[Diffusion MPC异常] 目标更新后的同步重规划失败: {exc}')
                        time.sleep(control_dt)
                        continue

            if cmd is None:
                try:
                    cmd = mpc.get_command(t_step, state, control_dt=control_dt, WAIT=True)
                except Exception as exc:
                    _log(f'[Diffusion MPC异常] {exc}')
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

            marker_update_counter += 1
            if marker_update_counter >= 10:
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
                marker_update_counter = 0

            loop_count += 1
            current_error = np.linalg.norm(ee_pos_world - current_goal_world)
            if current_error < (best_goal_error - 0.003):
                best_goal_error = current_error
                stagnation_steps = 0
            elif current_error > 0.10:
                stagnation_steps += 1
            else:
                stagnation_steps = 0

            if stagnation_steps >= 60:
                _reset_diffusion_solver_state(mpc, t_step, control_dt)
                stagnation_steps = 0
                best_goal_error = current_error
                _log(
                    '[Diffusion MPC恢复] 误差在 %.3f 附近停滞，重置采样分布后继续搜索'
                    % current_error
                )

            if loop_count % 20 == 0:
                diff_info = mpc.controller.get_diffusion_info()
                _log(
                    f'[{loop_count:5d}] t={t_step:.2f}s | '
                    f'q=[{q[0]:+.2f}, {q[1]:+.2f}, {q[2]:+.2f}, {q[3]:+.2f}, {q[4]:+.2f}, {q[5]:+.2f}] | '
                    f'ee_error={current_error:.4f} | opt_dt={mpc.opt_dt:.3f}s | '
                    f'n_diffuse={diff_info["n_diffuse"]}'
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
                _log(f'关闭 Diffusion MPC 资源时出现异常: {exc}')

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
    parser = argparse.ArgumentParser(description='UR7e Diffusion MPC Reach Static Gazebo Tall Scene')
    parser.add_argument('--cuda', action='store_true', default=True, help='使用 CUDA 加速 (默认: True)')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='禁用 CUDA')
    parser.add_argument('--rate', type=float, default=50.0, help='控制频率 Hz (默认: 50)')
    args = parser.parse_args()
    sys.exit(mpc_control_main(args))
