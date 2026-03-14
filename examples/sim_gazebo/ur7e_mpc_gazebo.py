#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e STORM MPC Control - Gazebo 仿真验证

使用 ForwardPositionController 实现高频实时 MPC 控制
发布 Float64MultiArray 到 /forward_position_controller/commands

Author: wqj
Date: 2025
"""

import sys
import os
import time
import copy
import yaml
import argparse
import numpy as np

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
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
except ImportError:
    print("=" * 60)
    print("错误: 未找到 ROS2 Python 包")
    print("请先 source ROS2 环境:")
    print("  source /opt/ros/humble/setup.bash")
    print("=" * 60)
    sys.exit(1)

# STORM imports
from storm_kit.util_file import get_gym_configs_path, get_mpc_configs_path, join_path
from storm_kit.mpc.rollout.arm_reacher import ArmReacher
from storm_kit.mpc.control import MPPI
from storm_kit.mpc.utils.state_filter import JointStateFilter
from storm_kit.mpc.utils.mpc_process_wrapper import ControlProcess
from storm_kit.mpc.task.task_base import BaseTask

np.set_printoptions(precision=3, suppress=True)


class GazeboReacherTask(BaseTask):
    """Gazebo 专用 ReacherTask，支持加载 examples/sim_gazebo/config 下的绝对路径配置。"""

    def __init__(self, task_file, robot_file, world_file, tensor_args):
        super().__init__(tensor_args=tensor_args)
        self.controller = self.init_mppi(task_file, robot_file, world_file)
        self.init_aux()

    def get_rollout_fn(self, **kwargs):
        return ArmReacher(**kwargs)

    def init_mppi(self, task_file, robot_file, world_file):
        if os.path.isabs(robot_file):
            robot_yml = robot_file
        else:
            robot_yml = join_path(get_gym_configs_path(), robot_file)

        with open(robot_yml) as f:
            robot_params = yaml.safe_load(f)

        if os.path.isabs(world_file):
            world_yml = world_file
        else:
            world_yml = join_path(get_gym_configs_path(), world_file)

        with open(world_yml) as f:
            world_params = yaml.safe_load(f)

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


class GazeboRobotInterface(Node):
    """
    Gazebo 机器人 ROS2 接口
    
    订阅: /joint_states (sensor_msgs/JointState)
    发布: /forward_position_controller/commands (std_msgs/Float64MultiArray)
    """
    
    def __init__(self, joint_names: list, control_rate: float = 50.0):
        super().__init__('storm_mpc_gazebo')
        
        self.joint_names = joint_names
        self.n_dof = len(joint_names)
        self.control_rate = control_rate
        self.control_dt = 1.0 / control_rate
        
        # 当前状态
        self.current_positions = None
        self.current_velocities = None
        self.prev_velocities = None
        self.prev_time = None
        self.state_received = False
        
        # 订阅关节状态
        self.sub_joint_states = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )
        
        # 发布位置指令 (ForwardPositionController)
        self.pub_position_cmd = self.create_publisher(
            Float64MultiArray,
            '/forward_position_controller/commands',
            10
        )
        
        self.get_logger().info(f'Gazebo Robot Interface 初始化完成')
        self.get_logger().info(f'控制频率: {control_rate} Hz')
        self.get_logger().info(f'关节数: {self.n_dof}')
    
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
        
        self.current_positions = positions
        self.current_velocities = velocities
        self.state_received = True
    
    def get_state(self) -> dict:
        """
        获取当前机器人状态
        
        Returns:
            dict: {'position': np.array, 'velocity': np.array, 'acceleration': np.array}
        """
        if not self.state_received:
            return None
        
        # 估算加速度
        current_time = time.time()
        if self.prev_velocities is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            dt = max(dt, 0.001)  # 防止除零
            acceleration = (self.current_velocities - self.prev_velocities) / dt
        else:
            acceleration = np.zeros(self.n_dof)
        
        self.prev_velocities = self.current_velocities.copy()
        self.prev_time = current_time
        
        return {
            'position': self.current_positions.copy(),
            'velocity': self.current_velocities.copy(),
            'acceleration': acceleration
        }
    
    def send_position_command(self, positions: np.ndarray):
        """
        发送位置指令
        
        Args:
            positions: 目标关节位置 [q1, q2, q3, q4, q5, q6] (rad)
        """
        msg = Float64MultiArray()
        msg.data = positions.tolist()
        self.pub_position_cmd.publish(msg)
    
    def wait_for_state(self, timeout: float = 5.0) -> bool:
        """
        等待接收到第一个关节状态
        
        Args:
            timeout: 超时时间 (秒)
        
        Returns:
            bool: 是否成功接收到状态
        """
        start_time = time.time()
        while not self.state_received:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                return False
        return True


def mpc_control_main(args):
    """
    STORM MPC Gazebo 控制主函数
    """
    print("=" * 60)
    print("UR7e STORM MPC Control - Gazebo 仿真")
    print("=" * 60)
    
    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    
    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    robot_file = os.path.join(config_dir, 'ur7e_robot_gazebo.yml')
    task_file = os.path.join(config_dir, 'ur7e_reacher_gazebo.yml')
    world_file = os.path.join(config_dir, 'collision_world_gazebo.yml')
    
    print(f"\n加载配置文件...")
    print(f"  Robot: {robot_file}")
    print(f"  Task:  {task_file}")
    print(f"  World: {world_file}")
    
    # 加载机器人配置
    with open(robot_file) as f:
        robot_params = yaml.safe_load(f)
    
    sim_params = robot_params.get('sim_params', {})
    
    # 关节名称
    joint_names = [
        'shoulder_pan_joint',
        'shoulder_lift_joint', 
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]
    n_dof = len(joint_names)
    
    # =========================================================================
    # 2. 初始化 ROS2
    # =========================================================================
    
    print("\n初始化 ROS2...")
    rclpy.init(args=None)
    
    control_rate = args.rate
    robot = GazeboRobotInterface(joint_names, control_rate=control_rate)
    
    # 创建多线程执行器（用于后台 spin）
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(robot)
    
    # 后台 spin
    import threading
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    # 等待连接
    print("等待 Gazebo 关节状态...")
    if not robot.wait_for_state(timeout=10.0):
        print("错误: 无法接收关节状态，请确保 Gazebo 仿真正在运行")
        robot.destroy_node()
        rclpy.shutdown()
        return 1
    
    print("已连接到 Gazebo 机器人!")
    
    # =========================================================================
    # 3. 初始化 STORM MPC
    # =========================================================================
    
    print("\n初始化 STORM MPC 控制器...")
    device = 'cuda' if args.cuda else 'cpu'
    print(f"计算设备: {device}")
    
    tensor_args = {'device': torch.device(device, 0) if device == 'cuda' else torch.device('cpu'), 
                   'dtype': torch.float32}
    
    # 创建 MPC 控制器
    mpc = GazeboReacherTask(task_file, robot_file, world_file, tensor_args)
    
    # 获取 MPC 控制周期
    mpc_control_dt = mpc.exp_params.get('control_dt', 0.02)
    print(f"MPC 控制周期: {mpc_control_dt} s ({1.0/mpc_control_dt:.1f} Hz)")
    
    # =========================================================================
    # 4. 设置目标状态
    # =========================================================================
    
    # 目标关节角度 (从初始位置移动到目标位置)
    init_state = sim_params.get('init_state', [0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    
    # 目标状态：移动到一个不同的位置
    goal_joint_positions = np.array([
        0.5,      # shoulder_pan: 略微旋转
        -1.2,     # shoulder_lift: 抬高一点
        1.2,      # elbow: 调整肘关节
        -1.57,    # wrist_1: 保持
        -1.57,    # wrist_2: 保持
        0.0       # wrist_3: 保持
    ])
    
    # 目标状态格式: [position (6), velocity (6)]
    goal_state = np.concatenate([goal_joint_positions, np.zeros(n_dof)])
    
    print(f"\n目标关节角度: {goal_joint_positions}")
    
    # 更新 MPC 目标
    mpc.update_params(goal_state=goal_state)
    
    # =========================================================================
    # 5. MPC 控制循环
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("开始 MPC 控制循环... (Ctrl+C 退出)")
    print("=" * 60 + "\n")
    
    t = 0.0
    loop_count = 0
    control_dt = 1.0 / control_rate
    
    # 预热 MPC
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
            
            # 1. 获取当前状态
            current_state = robot.get_state()
            if current_state is None:
                time.sleep(0.01)
                continue
            
            # 2. MPC 计算
            try:
                command = mpc.get_command(t, current_state, control_dt=mpc_control_dt, WAIT=True)
            except Exception as e:
                print(f"MPC 异常: {e}")
                continue
            
            # 3. 提取位置指令
            if command is not None and 'position' in command:
                target_positions = command['position']
                if isinstance(target_positions, torch.Tensor):
                    target_positions = target_positions.cpu().numpy()
                target_positions = np.array(target_positions).flatten()[:n_dof]
            else:
                # 如果没有有效指令，保持当前位置
                target_positions = current_state['position']
            
            # 4. 发送指令
            robot.send_position_command(target_positions)
            
            # 5. 日志输出
            loop_count += 1
            if loop_count % 50 == 0:  # 每 50 次循环输出一次
                current_pos = current_state['position']
                error = np.linalg.norm(current_pos - goal_joint_positions)
                print(f"[{loop_count:5d}] t={t:.2f}s | "
                      f"q=[{current_pos[0]:+.2f}, {current_pos[1]:+.2f}, {current_pos[2]:+.2f}, "
                      f"{current_pos[3]:+.2f}, {current_pos[4]:+.2f}, {current_pos[5]:+.2f}] | "
                      f"error={error:.4f}")
            
            # 6. 控制频率
            t += control_dt
            elapsed = time.time() - loop_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n\n用户中断，正在退出...")
    
    finally:
        # 清理
        print("清理资源...")
        mpc.close()
        robot.destroy_node()
        rclpy.shutdown()
    
    print("程序结束")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UR7e STORM MPC Gazebo Control')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='使用 CUDA 加速 (默认: True)')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='禁用 CUDA')
    parser.add_argument('--rate', type=float, default=50.0,
                        help='控制频率 Hz (默认: 50)')
    parser.add_argument('--goal', type=float, nargs=6, default=None,
                        help='目标关节角度 (6个值，单位: rad)')
    
    args = parser.parse_args()
    
    sys.exit(mpc_control_main(args))
