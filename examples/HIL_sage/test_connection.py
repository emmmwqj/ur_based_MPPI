#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e 连接测试脚本

测试与真实 UR7e 机器人的 ROS2 通信：
- 接收关节状态 (/joint_states)
- 验证控制器状态

用法:
    source /opt/ros/humble/setup.bash
    python3 test_connection.py
"""

import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import JointState
except ImportError:
    print("错误: 需要 ROS2 环境")
    print("请运行: source /opt/ros/humble/setup.bash")
    sys.exit(1)

import numpy as np


class ConnectionTester(Node):
    def __init__(self):
        super().__init__('ur7e_connection_tester')
        
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        self.received_count = 0
        self.last_positions = None
        self.last_velocities = None
        self.last_time = None
        
        qos = QoSProfile(depth=10)
        self.sub = self.create_subscription(
            JointState, '/joint_states', self.callback, qos
        )
        
        self.get_logger().info('等待 /joint_states 话题...')
    
    def callback(self, msg: JointState):
        self.received_count += 1
        
        positions = []
        velocities = []
        
        for name in self.joint_names:
            if name in msg.name:
                idx = msg.name.index(name)
                positions.append(msg.position[idx])
                if len(msg.velocity) > idx:
                    velocities.append(msg.velocity[idx])
                else:
                    velocities.append(0.0)
            else:
                positions.append(0.0)
                velocities.append(0.0)
        
        self.last_positions = np.array(positions)
        self.last_velocities = np.array(velocities)
        self.last_time = time.time()


def main():
    print("=" * 60)
    print("UR7e ROS2 连接测试")
    print("=" * 60)
    
    rclpy.init()
    node = ConnectionTester()
    
    print("\n等待机器人关节状态 (10秒超时)...\n")
    
    start = time.time()
    timeout = 10.0
    
    while rclpy.ok() and (time.time() - start) < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        
        if node.received_count > 0:
            break
    
    if node.received_count == 0:
        print("❌ 超时: 未收到关节状态")
        print("\n请检查:")
        print("  1. UR ROS2 Driver 是否正在运行")
        print("  2. 机器人是否已连接")
        print("  3. 运行 'ros2 topic list' 检查话题")
        node.destroy_node()
        rclpy.shutdown()
        return 1
    
    print("✅ 成功接收关节状态!")
    print(f"\n收到消息数: {node.received_count}")
    
    # 继续接收几秒钟统计频率
    print("\n统计话题频率 (3秒)...")
    count_start = node.received_count
    time_start = time.time()
    
    while rclpy.ok() and (time.time() - time_start) < 3.0:
        rclpy.spin_once(node, timeout_sec=0.01)
    
    count_end = node.received_count
    time_end = time.time()
    
    hz = (count_end - count_start) / (time_end - time_start)
    
    print(f"\n话题频率: {hz:.1f} Hz")
    print(f"\n当前关节位置 (rad):")
    for i, name in enumerate(node.joint_names):
        print(f"  {name}: {node.last_positions[i]:+.4f}")
    
    print(f"\n当前关节位置 (deg):")
    for i, name in enumerate(node.joint_names):
        print(f"  {name}: {np.degrees(node.last_positions[i]):+.2f}°")
    
    print(f"\n当前关节速度 (rad/s):")
    for i, name in enumerate(node.joint_names):
        print(f"  {name}: {node.last_velocities[i]:+.4f}")
    
    print("\n" + "=" * 60)
    print("连接测试完成!")
    print("=" * 60)
    
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
