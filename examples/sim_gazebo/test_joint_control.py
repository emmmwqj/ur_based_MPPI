#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e 关节控制测试脚本

用于快速测试 ForwardPositionController 是否正常工作
"""

import sys
import time
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
except ImportError:
    print("错误: 请先 source ROS2 环境")
    print("  source /opt/ros/humble/setup.bash")
    sys.exit(1)


class JointControlTest(Node):
    def __init__(self):
        super().__init__('joint_control_test')
        
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        self.current_positions = None
        
        self.sub = self.create_subscription(
            JointState, '/joint_states', self.joint_cb, 10)
        
        self.pub = self.create_publisher(
            Float64MultiArray, '/forward_position_controller/commands', 10)
        
        self.get_logger().info('Joint Control Test 初始化完成')
    
    def joint_cb(self, msg):
        positions = []
        for name in self.joint_names:
            if name in msg.name:
                idx = msg.name.index(name)
                positions.append(msg.position[idx])
        if len(positions) == 6:
            self.current_positions = np.array(positions)
    
    def send_position(self, positions):
        msg = Float64MultiArray()
        msg.data = positions.tolist()
        self.pub.publish(msg)
        self.get_logger().info(f'发送位置: {positions}')


def main():
    rclpy.init()
    node = JointControlTest()
    
    print("\n" + "=" * 60)
    print("UR7e 关节控制测试")
    print("=" * 60)
    
    # 等待关节状态
    print("\n等待关节状态...")
    for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.current_positions is not None:
            break
    
    if node.current_positions is None:
        print("错误: 无法接收关节状态")
        node.destroy_node()
        rclpy.shutdown()
        return 1
    
    print(f"当前关节位置: {node.current_positions}")
    
    # 定义测试位置
    home_position = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    test_position = np.array([0.5, -1.2, 1.2, -1.57, -1.57, 0.0])
    
    print("\n测试序列:")
    print("  1. 回到 Home 位置")
    print("  2. 等待 3 秒")
    print("  3. 移动到测试位置")
    print("  4. 等待 3 秒")
    print("  5. 回到 Home 位置")
    
    input("\n按 Enter 开始测试...")
    
    try:
        # 1. 回到 Home
        print("\n[1/5] 移动到 Home 位置...")
        node.send_position(home_position)
        for _ in range(30):
            rclpy.spin_once(node, timeout_sec=0.1)
        
        # 2. 等待
        print("[2/5] 等待 3 秒...")
        time.sleep(3)
        
        # 3. 移动到测试位置
        print("[3/5] 移动到测试位置...")
        node.send_position(test_position)
        for _ in range(30):
            rclpy.spin_once(node, timeout_sec=0.1)
        
        # 4. 等待
        print("[4/5] 等待 3 秒...")
        time.sleep(3)
        
        # 5. 回到 Home
        print("[5/5] 回到 Home 位置...")
        node.send_position(home_position)
        for _ in range(30):
            rclpy.spin_once(node, timeout_sec=0.1)
        
        print("\n测试完成!")
        
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
