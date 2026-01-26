#!/usr/bin/env python3
"""
测试 ROS2 话题通信
用于诊断 Isaac Sim 和 MPC 控制器之间的通信问题
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time


class TopicMonitor(Node):
    """监控 ROS2 话题"""
    
    def __init__(self):
        super().__init__('topic_monitor')
        
        self.joint_state_count = 0
        self.joint_command_count = 0
        self.last_joint_state = None
        self.last_joint_command = None
        
        # 订阅关节状态
        self.sub_state = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_cb,
            10
        )
        
        # 订阅关节指令
        self.sub_cmd = self.create_subscription(
            JointState,
            '/joint_command',
            self._joint_command_cb,
            10
        )
        
        # 定时打印统计
        self.timer = self.create_timer(1.0, self._print_stats)
        
        self.get_logger().info("开始监控 ROS2 话题...")
        self.get_logger().info("  - /joint_states (Isaac Sim 发布)")
        self.get_logger().info("  - /joint_command (MPC 发布)")
        
    def _joint_state_cb(self, msg):
        self.joint_state_count += 1
        self.last_joint_state = msg
        
    def _joint_command_cb(self, msg):
        self.joint_command_count += 1
        self.last_joint_command = msg
        
    def _print_stats(self):
        print(f"\n{'='*60}")
        print(f"[统计] joint_states: {self.joint_state_count} 条, "
              f"joint_command: {self.joint_command_count} 条")
        
        if self.last_joint_state:
            pos = [f"{x:.3f}" for x in self.last_joint_state.position[:6]]
            print(f"[joint_states] 位置: {pos}")
            print(f"  关节名称: {self.last_joint_state.name[:6]}")
            
        if self.last_joint_command:
            pos = [f"{x:.3f}" for x in self.last_joint_command.position[:6]]
            print(f"[joint_command] 目标: {pos}")
        
        print(f"{'='*60}")


def main():
    print("=" * 60)
    print("ROS2 话题监控工具")
    print("=" * 60)
    
    rclpy.init()
    
    # 检查可用话题
    node = rclpy.create_node('topic_checker')
    
    print("\n等待话题发现...")
    time.sleep(2.0)
    
    topics = node.get_topic_names_and_types()
    print("\n发现的话题:")
    for topic, types in topics:
        if 'joint' in topic.lower():
            print(f"  {topic}: {types}")
    
    node.destroy_node()
    
    # 启动监控
    print("\n")
    monitor = TopicMonitor()
    
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
