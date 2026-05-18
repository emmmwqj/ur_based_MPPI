#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic moving-ball node for the sim_dynamic demo."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import yaml

STORM_ROOT = os.path.expanduser('~/storm')
if STORM_ROOT not in sys.path:
    sys.path.insert(0, STORM_ROOT)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped

class DynamicBallNode(Node):
    def __init__(self, world_file: str):
        super().__init__('sim_dynamic_ball')
        with open(world_file) as f:
            world_params = yaml.safe_load(f)
        self.world_params = world_params
        ball_cfg = world_params['world_model']['dynamic_obstacles']['dynamic_ball']

        self.model_name = str(ball_cfg.get('model_name', 'dynamic_ball'))
        self.topic = str(ball_cfg.get('topic', '/dynamic_ball/pose'))
        self.velocity_topic = str(ball_cfg.get('velocity_topic', '/dynamic_ball/velocity'))
        self.radius = float(ball_cfg['radius'])
        self.position = [float(v) for v in ball_cfg['initial_position']]
        self.y_min, self.y_max = [float(v) for v in ball_cfg['y_limits']]
        self.speed = float(ball_cfg.get('speed', 0.1))
        self.update_hz = float(ball_cfg.get('update_hz', 20.0))
        self.reference_frame = str(ball_cfg.get('reference_frame', 'world'))
        self.direction = 1.0
        self.last_time = time.time()
        self.last_log_time = 0.0

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pose_pub = self.create_publisher(PoseStamped, self.topic, qos)
        self.velocity_pub = self.create_publisher(TwistStamped, self.velocity_topic, qos)

        self.get_logger().info(
            f'dynamic ball config: pos={self.position}, y_limits={[self.y_min, self.y_max]}, '
            f'speed={self.speed:.3f}m/s, update_hz={self.update_hz:.1f}, mover=gz model, '
            f'velocity_topic={self.velocity_topic}'
        )

        self._publish_pose()
        self._publish_velocity()
        self.timer = self.create_timer(1.0 / self.update_hz, self._tick)

    def _publish_pose(self):
        if not rclpy.ok():
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.reference_frame
        msg.pose.position.x = float(self.position[0])
        msg.pose.position.y = float(self.position[1])
        msg.pose.position.z = float(self.position[2])
        msg.pose.orientation.w = 1.0
        try:
            self.pose_pub.publish(msg)
        except Exception:
            pass

    def _publish_velocity(self):
        if not rclpy.ok():
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.reference_frame
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = float(self.direction * self.speed)
        msg.twist.linear.z = 0.0
        try:
            self.velocity_pub.publish(msg)
        except Exception:
            pass

    def _send_state(self):
        cmd = [
            'gz', 'model',
            '-m', self.model_name,
            '-x', f'{self.position[0]:.6f}',
            '-y', f'{self.position[1]:.6f}',
            '-z', f'{self.position[2]:.6f}',
            '-R', '0.0',
            '-P', '0.0',
            '-Y', '0.0',
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    def _tick(self):
        if not rclpy.ok():
            return
        now = time.time()
        dt = max(now - self.last_time, 1.0 / self.update_hz)
        self.last_time = now
        self.position[1] += self.direction * self.speed * dt
        if self.position[1] >= self.y_max:
            self.position[1] = self.y_max
            self.direction = -1.0
        elif self.position[1] <= self.y_min:
            self.position[1] = self.y_min
            self.direction = 1.0

        self._send_state()
        self._publish_pose()
        self._publish_velocity()

        if now - self.last_log_time >= 1.0:
            self.get_logger().info(
                'dynamic ball world position: [%.3f, %.3f, %.3f]'
                % (self.position[0], self.position[1], self.position[2])
            )
            self.last_log_time = now


def main(argv=None):
    parser = argparse.ArgumentParser(description='Dynamic moving ball node')
    parser.add_argument('--world-file', required=True)
    args = parser.parse_args(argv)

    rclpy.init(args=None)
    node = DynamicBallNode(args.world_file)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
