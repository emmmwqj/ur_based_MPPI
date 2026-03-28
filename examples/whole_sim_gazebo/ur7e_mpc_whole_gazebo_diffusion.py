#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e Diffusion MPPI Control - Whole Gazebo Tall ESDF 示例
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

from examples.whole_sim_gazebo.ur7e_mpc_whole_gazebo import (
    _log,
    inv_transform_point,
    transform_point,
)
from examples.whole_sim_gazebo.ur7e_mpc_whole_gazebo_tall import (
    GazeboRobotInterface,
    _recover_command,
)
from examples.whole_sim_gazebo.gazebo_obstacle_utils import (
    count_primitive_obstacles,
    load_primitive_world,
    spawn_gazebo_obstacles,
)
from examples.whole_sim_gazebo.whole_gazebo_diffusion_task import WholeGazeboDiffusionReacherTask

np.set_printoptions(precision=3, suppress=True)


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
        _log('UR7e Diffusion MPPI Control - Whole Gazebo Tall ESDF')
        _log('=' * 60)

        config_dir = os.path.join(os.path.dirname(__file__), 'config')
        robot_file = os.path.join(config_dir, 'ur7e_robot_gazebo.yml')
        task_file = os.path.join(config_dir, 'ur7e_reacher_whole_gazebo_diffusion_tall.yml')
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
        n_world_spheres, n_world_cubes = count_primitive_obstacles(obstacle_world, include_ground=False)
        if spawn_gazebo_obstacles(robot, obstacle_world, model_prefix='whole_tall_diff', include_ground=False):
            _log(
                'Gazebo 真实障碍物已生成: spheres=%d cubes=%d'
                % (n_world_spheres, n_world_cubes)
            )
        else:
            _log('警告: Gazebo 真实障碍物未完整生成，请检查 /spawn_entity 与 /delete_entity 服务')

        device = 'cuda' if args.cuda else 'cpu'
        _log('\n初始化 Diffusion MPC 控制器...')
        _log(f'计算设备: {device}')
        tensor_args = {
            'device': torch.device(device, 0) if device == 'cuda' else torch.device('cpu'),
            'dtype': torch.float32,
        }

        mpc = WholeGazeboDiffusionReacherTask(task_file, robot_file, world_file, tensor_args)
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
        _log('开始 Diffusion MPC 控制循环... (Ctrl+C 退出)')
        _log('=' * 60 + '\n')

        t = 0.0
        loop_count = 0
        marker_update_counter = 0
        control_dt = 1.0 / args.rate

        _log('预热 Diffusion MPC 控制器...')
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
                            _log('[目标更新] 已同步重规划并重置 Diffusion MPC 时间基准')
                        except Exception as sync_exc:
                            _log(f'[Diffusion MPC异常] 目标更新后的同步重规划失败: {sync_exc}')
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
                        _log('[Diffusion MPC恢复] 同步取命令失败 (%s)，重置控制进程时间基准后重规划' % exc)
                        try:
                            command = _recover_command(mpc, t, current_state, mpc_control_dt)
                        except Exception as recover_exc:
                            _log(f'[Diffusion MPC异常] 同步重规划失败: {recover_exc}')
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
                    opt_info = getattr(mpc, '_last_opt_info', {}) or {}
                    iteration_costs = opt_info.get('iteration_costs', [])
                    variance_schedule = opt_info.get('variance_schedule', [])
                    best_cost = iteration_costs[-1] if iteration_costs else float('nan')
                    mean_sigma = float(np.mean(variance_schedule)) if variance_schedule else 0.0
                    _log(
                        f'[{loop_count:5d}] t={t:.2f}s | '
                        f'q=[{current_pos[0]:+.2f}, {current_pos[1]:+.2f}, {current_pos[2]:+.2f}]'
                        f' | q4-6=[{current_pos[3]:+.2f}, {current_pos[4]:+.2f}, {current_pos[5]:+.2f}]'
                        f' | ee_pos_error={ee_pos_error:.4f}m'
                        f' | esdf_valid_ratio={100.0 * valid_ratio:.1f}%'
                        f' | diff_sigma={mean_sigma:.4f}'
                        f' | diff_best_cost={best_cost:.4f}'
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
    parser = argparse.ArgumentParser(description='UR7e Diffusion MPPI Whole Gazebo ESDF Control')
    parser.add_argument('--cuda', action='store_true', default=True, help='使用 CUDA 加速 (默认: True)')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='禁用 CUDA')
    parser.add_argument('--rate', type=float, default=50.0, help='控制频率 Hz (默认: 50)')
    parser.add_argument('--goal', type=float, nargs=6, default=None, help='目标关节角度 (6个值，单位: rad)')
    args = parser.parse_args()
    sys.exit(mpc_control_main(args))
