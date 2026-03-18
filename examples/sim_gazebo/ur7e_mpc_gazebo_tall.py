#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e STORM MPC Control - Gazebo 高墙场景

与 ur7e_mpc_gazebo.py 保持相同的 primitive-world 避障链路，
仅切换到高墙场景的 world/task 配置。
"""

import sys
import os
import time
import yaml
import argparse
import numpy as np

STORM_ROOT = os.path.expanduser('~/storm')
sys.path.insert(0, STORM_ROOT)

import torch
torch.multiprocessing.set_start_method('spawn', force=True)

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
except ImportError:
    print("=" * 60)
    print("错误: 未找到 ROS2 Python 包")
    print("请先 source ROS2 环境:")
    print("  source /opt/ros/humble/setup.bash")
    print("=" * 60)
    sys.exit(1)

from examples.sim_gazebo.ur7e_mpc_gazebo import GazeboReacherTask, GazeboRobotInterface

np.set_printoptions(precision=3, suppress=True)


def mpc_control_main(args):
    print("=" * 60)
    print("UR7e STORM MPC Control - Gazebo 高墙场景")
    print("=" * 60)

    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    robot_file = os.path.join(config_dir, 'ur7e_robot_gazebo.yml')
    task_file = os.path.join(config_dir, 'ur7e_reacher_gazebo_tall.yml')
    world_file = os.path.join(config_dir, 'collision_world_gazebo_tall.yml')

    print(f"\n加载配置文件...")
    print(f"  Robot: {robot_file}")
    print(f"  Task:  {task_file}")
    print(f"  World: {world_file}")

    with open(robot_file) as f:
        robot_params = yaml.safe_load(f)

    sim_params = robot_params.get('sim_params', {})

    joint_names = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]
    n_dof = len(joint_names)

    print("\n初始化 ROS2...")
    rclpy.init(args=None)

    control_rate = args.rate
    robot = GazeboRobotInterface(joint_names, control_rate=control_rate)

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(robot)

    import threading
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("等待 Gazebo 关节状态...")
    if not robot.wait_for_state(timeout=10.0):
        print("错误: 无法接收关节状态，请确保 Gazebo 仿真正在运行")
        robot.destroy_node()
        rclpy.shutdown()
        return 1

    print("已连接到 Gazebo 机器人!")

    print("\n初始化 STORM MPC 控制器...")
    device = 'cuda' if args.cuda else 'cpu'
    print(f"计算设备: {device}")

    tensor_args = {
        'device': torch.device(device, 0) if device == 'cuda' else torch.device('cpu'),
        'dtype': torch.float32,
    }

    mpc = GazeboReacherTask(task_file, robot_file, world_file, tensor_args)
    mpc_control_dt = mpc.exp_params.get('control_dt', 0.02)
    print(f"MPC 控制周期: {mpc_control_dt} s ({1.0/mpc_control_dt:.1f} Hz)")

    init_state = sim_params.get('init_state', [0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    goal_joint_positions = np.array([
        0.5,
        -1.2,
        1.2,
        -1.57,
        -1.57,
        0.0,
    ])
    goal_state = np.concatenate([goal_joint_positions, np.zeros(n_dof)])

    print(f"\n初始关节角度: {init_state}")
    print(f"目标关节角度: {goal_joint_positions}")
    print("避障场景: 高墙 primitive world")

    mpc.update_params(goal_state=goal_state)

    print("\n" + "=" * 60)
    print("开始 MPC 控制循环... (Ctrl+C 退出)")
    print("=" * 60 + "\n")

    t = 0.0
    loop_count = 0
    control_dt = 1.0 / control_rate

    print("预热 MPC 控制器...")
    current_state = robot.get_state()
    for _ in range(5):
        try:
            mpc.get_command(t, current_state, control_dt=mpc_control_dt, WAIT=False)
        except Exception as e:
            print(f"预热异常 (可忽略): {e}")
        t += mpc_control_dt
        time.sleep(0.01)

    print("开始控制!\n")

    try:
        while rclpy.ok():
            loop_start = time.time()

            current_state = robot.get_state()
            if current_state is None:
                time.sleep(0.01)
                continue

            try:
                command = mpc.get_command(t, current_state, control_dt=mpc_control_dt, WAIT=True)
            except Exception as e:
                print(f"MPC 异常: {e}")
                continue

            if command is not None and 'position' in command:
                target_positions = command['position']
                if isinstance(target_positions, torch.Tensor):
                    target_positions = target_positions.cpu().numpy()
                target_positions = np.array(target_positions).flatten()[:n_dof]
            else:
                target_positions = current_state['position']

            robot.send_position_command(target_positions)

            loop_count += 1
            if loop_count % 50 == 0:
                current_pos = current_state['position']
                error = np.linalg.norm(current_pos - goal_joint_positions)
                print(
                    f"[{loop_count:5d}] t={t:.2f}s | "
                    f"q=[{current_pos[0]:+.2f}, {current_pos[1]:+.2f}, {current_pos[2]:+.2f}, "
                    f"{current_pos[3]:+.2f}, {current_pos[4]:+.2f}, {current_pos[5]:+.2f}] | "
                    f"error={error:.4f}"
                )

            t += control_dt
            elapsed = time.time() - loop_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n用户中断，正在退出...")

    finally:
        print("清理资源...")
        mpc.close()
        robot.destroy_node()
        rclpy.shutdown()

    print("程序结束")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UR7e STORM MPC Gazebo Tall Scene Control')
    parser.add_argument('--cuda', action='store_true', default=True, help='使用 CUDA 加速 (默认: True)')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='禁用 CUDA')
    parser.add_argument('--rate', type=float, default=50.0, help='控制频率 Hz (默认: 50)')
    parser.add_argument('--goal', type=float, nargs=6, default=None, help='目标关节角度 (6个值，单位: rad)')
    args = parser.parse_args()
    sys.exit(mpc_control_main(args))
