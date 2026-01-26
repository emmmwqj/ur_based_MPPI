#!/usr/bin/env python3
"""
UR7e MPC Control - ROS2 软件在环 (SIL) 版本

本脚本与 ur7e_mpc_main.py 结构相同，但使用 ROS2 进行通信。
用于 Step 2/3：与 Isaac Sim ROS2 桥接或真实机器人通信。

架构说明:
    - Isaac Sim 运行仿真并发布 /joint_states
    - 本脚本订阅状态、运行 MPC、发布控制指令到 /joint_command
    - 形成软件在环 (SIL) 闭环

用法:
    # 终端 1: 启动 Isaac Sim + ROS2 桥接
    cd /home/wqj/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64
    ./python.sh ~/storm/examples/ur7e_ros2_sim.py
    
    # 终端 2: 运行本脚本 (在 ROS2 环境中)
    source /opt/ros/humble/setup.bash
    python3 ~/storm/examples/ur7e_mpc_ros2.py

依赖:
    - ROS2 Humble+
    - sensor_msgs, geometry_msgs
"""

import argparse
import signal
import sys
import time
import numpy as np
import torch
import yaml
from threading import Thread, Lock
from scipy.spatial.transform import Rotation

# ============================================================================
# ROS2 检查与导入
# ============================================================================

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from builtin_interfaces.msg import Duration
except ImportError:
    print("=" * 60)
    print("错误: 需要 ROS2！")
    print("请先 source ROS2 环境:")
    print("  source /opt/ros/humble/setup.bash")
    print("=" * 60)
    sys.exit(1)

np.set_printoptions(precision=2)


# ============================================================================
# 辅助函数
# ============================================================================

def transform_point(position, orientation_xyzw, point):
    """将点从机器人坐标系变换到世界坐标系"""
    rot = Rotation.from_quat(orientation_xyzw)
    return rot.apply(point) + position

def inv_transform_point(position, orientation_xyzw, point):
    """将点从世界坐标系变换到机器人坐标系"""
    rot = Rotation.from_quat(orientation_xyzw).inv()
    return rot.apply(point - position)


# ============================================================================
# ROS2 机器人接口
# ============================================================================

class ROS2RobotInterface(Node):
    """
    ROS2 机器人接口 - 与 Isaac Sim 或真实机器人通信
    
    接口与 ur7e_mpc_main.py 中的 robot 对象保持一致:
        - get_joint_positions() -> np.ndarray
        - get_joint_velocities() -> np.ndarray
        - apply_action(position) -> None
    """
    
    def __init__(self, n_dof=6, 
                 state_topic='/joint_states',
                 cmd_topic='/joint_command',
                 target_topic='/mpc_target'):
        super().__init__('ur7e_mpc_controller')
        
        self.n_dof = n_dof
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ][:n_dof]
        
        # 状态缓存
        self._lock = Lock()
        self._q = None
        self._dq = None
        self._target_pos = None  # 从 ROS2 接收的目标位置
        
        # 加速度估计
        self._prev_dq = np.zeros(n_dof)
        self._prev_time = time.time()
        
        # QoS 配置
        qos = QoSProfile(depth=10)
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        
        # 订阅关节状态
        self._state_sub = self.create_subscription(
            JointState, state_topic, self._state_callback, qos
        )
        
        # 订阅目标位置 (可选 - 用于动态目标)
        self._target_sub = self.create_subscription(
            PoseStamped, target_topic, self._target_callback, qos
        )
        
        # 发布关节指令 (使用 JointState 消息，与 OmniGraph 兼容)
        self._cmd_pub = self.create_publisher(
            JointState, cmd_topic, qos_reliable
        )
        
        self.get_logger().info(f'订阅关节状态: {state_topic}')
        self.get_logger().info(f'订阅目标位置: {target_topic}')
        self.get_logger().info(f'发布关节指令: {cmd_topic}')
    
    def _state_callback(self, msg: JointState):
        """关节状态回调"""
        q = np.zeros(self.n_dof)
        dq = np.zeros(self.n_dof)
        
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                if idx < len(msg.position):
                    q[i] = msg.position[idx]
                if idx < len(msg.velocity):
                    dq[i] = msg.velocity[idx]
        
        with self._lock:
            self._q = q
            self._dq = dq
            
        # 计数器
        if not hasattr(self, '_state_count'):
            self._state_count = 0
        self._state_count += 1
    
    def _target_callback(self, msg: PoseStamped):
        """目标位置回调 (可选)"""
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        with self._lock:
            self._target_pos = pos
    
    def get_joint_positions(self) -> np.ndarray:
        """获取关节位置 - 与仿真版接口一致"""
        with self._lock:
            if self._q is None:
                return None
            return self._q.copy()
    
    def get_joint_velocities(self) -> np.ndarray:
        """获取关节速度 - 与仿真版接口一致"""
        with self._lock:
            if self._dq is None:
                return None
            return self._dq.copy()
    
    def get_acceleration_estimate(self) -> np.ndarray:
        """估计关节加速度"""
        dq = self.get_joint_velocities()
        if dq is None:
            return np.zeros(self.n_dof)
        
        now = time.time()
        dt = max(now - self._prev_time, 0.001)
        ddq = (dq - self._prev_dq) / dt
        
        self._prev_dq = dq.copy()
        self._prev_time = now
        
        return ddq
    
    def get_target_position(self) -> np.ndarray:
        """获取目标位置 (如果通过 ROS2 发布)"""
        with self._lock:
            return self._target_pos.copy() if self._target_pos is not None else None
    
    def apply_action(self, q_des: np.ndarray, duration_sec: float = 0.02):
        """
        发送关节位置指令 - 与仿真版接口一致
        使用 JointState 消息格式，与 Isaac Sim OmniGraph 兼容
        
        Args:
            q_des: 目标关节位置
            duration_sec: 到达时间 (未使用，保留接口兼容)
        """
        from std_msgs.msg import Header
        
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(x) for x in q_des[:self.n_dof]]
        msg.velocity = []  # 空，只发送位置
        msg.effort = []
        
        self._cmd_pub.publish(msg)
        
        # 计数器
        if not hasattr(self, '_cmd_count'):
            self._cmd_count = 0
        self._cmd_count += 1
    
    def get_cmd_count(self):
        """获取发送的指令数量"""
        return getattr(self, '_cmd_count', 0)
    
    def get_state_count(self):
        """获取接收的状态数量"""
        return getattr(self, '_state_count', 0)
    
    def is_connected(self) -> bool:
        """检查是否已连接到机器人"""
        return self._q is not None


# ============================================================================
# 主控制函数
# ============================================================================

def mpc_control_main(args):
    """主控制函数"""
    
    # STORM 导入
    sys.path.insert(0, '/home/wqj/storm')
    from storm_kit.util_file import get_gym_configs_path, join_path, get_assets_path
    from storm_kit.mpc.task.reacher_task import ReacherTask
    from storm_kit.differentiable_robot_model.coordinate_transform import (
        quaternion_to_matrix, CoordinateTransform
    )
    
    print("=" * 60)
    print("UR7e MPC Control - ROS2 软件在环 (SIL)")
    print("=" * 60)
    
    # ========================================================================
    # 加载配置
    # ========================================================================
    
    robot_file = 'ur7e_isaacsim.yml'
    task_file = 'ur7e_reacher_isaacsim.yml'
    world_file = 'collision_primitives_3d.yml'
    
    with open(join_path(get_gym_configs_path(), robot_file)) as f:
        robot_params = yaml.safe_load(f)
    
    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    
    # 机器人位姿 (需要与仿真中一致)
    robot_pose = sim_params.get('robot_pose', [0, 0, 0, 0, 0, 0, 1])
    robot_pos = np.array(robot_pose[:3])
    robot_quat_xyzw = np.array(robot_pose[3:])
    robot_quat_wxyz = np.array([robot_quat_xyzw[3], *robot_quat_xyzw[:3]])
    
    n_dof = 6
    
    # ========================================================================
    # 初始化 ROS2 节点
    # ========================================================================
    
    print("\n初始化 ROS2...")
    rclpy.init()
    
    robot = ROS2RobotInterface(
        n_dof=n_dof,
        state_topic=args.joint_state_topic,
        cmd_topic=args.joint_cmd_topic,
        target_topic=args.target_topic
    )
    
    # 使用 executor 进行后台 spin（可以安全停止）
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(robot)
    
    # 定义可中断的 spin 函数
    _executor_running = True
    def spin_with_check():
        while _executor_running:
            executor.spin_once(timeout_sec=0.1)
    
    spin_thread = Thread(target=spin_with_check, daemon=True)
    spin_thread.start()
    
    # ========================================================================
    # 等待机器人连接
    # ========================================================================
    
    print("\n等待机器人状态...")
    timeout = 30.0
    start = time.time()
    while not robot.is_connected():
        if time.time() - start > timeout:
            print("错误: 超时等待机器人状态!")
            print(f"请确保 Isaac Sim 正在运行并发布 {args.joint_state_topic}")
            try:
                executor.shutdown()
                robot.destroy_node()
                rclpy.shutdown()
            except:
                pass
            return 1
        time.sleep(0.1)
    print("已连接到机器人!\n")
    
    # ========================================================================
    # MPC 控制器
    # ========================================================================
    
    print("设置 MPC 控制器...")
    device = 'cuda' if args.cuda else 'cpu'
    tensor_args = {'device': torch.device(device, 0), 'dtype': torch.float32}
    
    mpc = ReacherTask(task_file, robot_file, world_file, tensor_args)
    control_dt = mpc.exp_params['control_dt']
    
    # 目标状态
    goal_state = np.array([0.5, -1.2, 1.2, -1.57, -1.57, 0.0, 0, 0, 0, 0, 0, 0])
    mpc.update_params(goal_state=goal_state)
    
    goal_ee_pos = np.ravel(mpc.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    goal_ee_quat = np.ravel(mpc.controller.rollout_fn.goal_ee_quat.cpu().numpy())
    print(f"目标末端位置: {goal_ee_pos}")
    print(f"控制周期: {control_dt}s")
    
    # ========================================================================
    # 预热 MPC (完整预热，包括等待第一次优化完成)
    # ========================================================================
    
    print("\n预热 MPC (首次优化可能需要几秒钟)...")
    t = 0.0
    
    # 先发送几次状态启动优化线程
    for _ in range(3):
        q = robot.get_joint_positions()
        dq = robot.get_joint_velocities()
        if q is not None and dq is not None:
            state = {'position': q, 'velocity': dq, 'acceleration': np.zeros(n_dof)}
            try:
                mpc.get_command(t, state, control_dt=control_dt, WAIT=False)
            except:
                pass
        t += control_dt
        time.sleep(control_dt)
    
    # 等待第一次优化完成
    print("  等待首次优化完成...")
    q = robot.get_joint_positions()
    dq = robot.get_joint_velocities()
    if q is not None and dq is not None:
        state = {'position': q, 'velocity': dq, 'acceleration': np.zeros(n_dof)}
        try:
            cmd = mpc.get_command(t, state, control_dt=control_dt, WAIT=True)
            print(f"  首次优化完成! opt_dt={mpc.opt_dt:.3f}s")
        except Exception as e:
            print(f"  首次优化异常: {e}")
    
    print("MPC 预热完成\n")
    
    # ========================================================================
    # 主循环
    # ========================================================================
    
    print("=" * 60)
    print("开始 MPC 控制循环... (Ctrl+C 退出)")
    print("=" * 60)
    print("\n提示: 可以通过发布 PoseStamped 到 /mpc_target 来动态更新目标\n")
    
    # 信号处理 - 使用全局变量确保能正确传递
    running = [True]  # 使用列表避免 nonlocal 问题
    
    def shutdown_handler(sig, frame):
        print("\n收到退出信号，正在退出...")
        running[0] = False
        # 立即停止 executor
        nonlocal _executor_running
        _executor_running = False
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # 控制变量
    i = 0
    loop_start = time.time()
    current_goal_ee = goal_ee_pos.copy()
    prev_vel = np.zeros(n_dof)
    
    while running[0]:
        iter_start = time.time()
        t = time.time() - loop_start
        
        # --- 获取状态 ---
        q = robot.get_joint_positions()
        dq = robot.get_joint_velocities()
        
        if q is None or dq is None:
            time.sleep(control_dt)
            continue
        
        # 估计加速度
        ddq = (dq - prev_vel) / max(control_dt, 0.001)
        prev_vel = dq.copy()
        
        state = {'position': q, 'velocity': dq, 'acceleration': ddq}
        
        # --- 检测目标更新 (通过 ROS2) ---
        new_target = robot.get_target_position()
        if new_target is not None:
            # 目标位置是世界坐标系，需要转换到机器人坐标系
            target_robot = inv_transform_point(robot_pos, robot_quat_xyzw, new_target)
            if np.linalg.norm(target_robot - current_goal_ee) > 0.005:
                current_goal_ee = target_robot.copy()
                mpc.update_params(goal_ee_pos=current_goal_ee, goal_ee_quat=goal_ee_quat)
                print(f"[目标更新] 新位置: {np.round(current_goal_ee, 3)}")
        
        # --- MPC 计算 ---
        try:
            # 使用 WAIT=False 避免阻塞，然后检查是否有结果
            cmd = mpc.get_command(t, state, control_dt=control_dt, WAIT=False)
            
            # 如果没有命令（优化还没完成），使用上一次的命令或当前位置
            if cmd is None or 'position' not in cmd:
                i += 1
                time.sleep(control_dt)
                continue
                
        except (IndexError, RuntimeError) as e:
            i += 1
            time.sleep(control_dt)
            continue
        
        # --- 发送指令 ---
        robot.apply_action(cmd['position'], duration_sec=control_dt)
        
        # --- 打印状态 ---
        if i % 50 == 0:
            err = mpc.get_current_error(state)
            rx_count = robot.get_state_count()
            tx_count = robot.get_cmd_count()
            print(f"[{i:4d}] 误差: {[f'{x:.3f}' for x in err]}, "
                  f"opt: {mpc.opt_dt:.3f}s, mpc: {mpc.mpc_dt:.3f}s, "
                  f"rx/tx: {rx_count}/{tx_count}")
        
        # --- 保持控制频率 ---
        elapsed = time.time() - iter_start
        if elapsed < control_dt:
            time.sleep(control_dt - elapsed)
        
        i += 1
    
    # ========================================================================
    # 清理 (使用超时避免卡住)
    # ========================================================================
    
    print("\n清理资源...")
    
    # 停止 executor 线程
    _executor_running = False
    spin_thread.join(timeout=1.0)
    
    # 关闭 MPC (在单独线程中执行，带超时)
    print("  关闭 MPC...")
    def close_mpc():
        try:
            mpc.close()
        except:
            pass
    
    close_thread = Thread(target=close_mpc, daemon=True)
    close_thread.start()
    close_thread.join(timeout=2.0)  # 最多等待 2 秒
    
    if close_thread.is_alive():
        print("  MPC 关闭超时，强制退出")
    
    # 停止 ROS2
    print("  关闭 ROS2...")
    try:
        executor.shutdown()
    except:
        pass
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
    # 解析参数
    parser = argparse.ArgumentParser(description='UR7e MPC Control - ROS2 SIL')
    parser.add_argument('--cuda', action='store_true', default=True, help='使用CUDA')
    parser.add_argument('--joint_state_topic', default='/joint_states', 
                        help='关节状态话题')
    parser.add_argument('--joint_cmd_topic', 
                        default='/joint_command',
                        help='关节指令话题 (JointState)')
    parser.add_argument('--target_topic', default='/mpc_target',
                        help='目标位置话题 (PoseStamped)')
    args = parser.parse_args()
    
    # Torch 配置
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 运行主函数
    sys.exit(mpc_control_main(args))
