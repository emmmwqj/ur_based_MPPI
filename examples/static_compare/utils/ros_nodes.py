from __future__ import annotations

import time
from typing import Iterable

import numpy as np

try:
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
except Exception:  # pragma: no cover - imported only in ROS-enabled runs
    PoseStamped = None
    Node = object


class TargetPosePublisher(Node):
    """Publish one static world-frame target through the tuned /target_pose path."""

    def __init__(
        self,
        goal_ee_position: Iterable[float],
        node_name: str,
        publish_period_sec: float = 0.25,
        publish_duration_sec: float = 2.0,
    ) -> None:
        super().__init__(node_name)
        self.goal_ee_position = np.asarray(goal_ee_position, dtype=float)
        self.publish_period_sec = float(publish_period_sec)
        self.publish_duration_sec = float(publish_duration_sec)
        self.publish_count = 0
        self.started_at: float | None = None
        self._publisher = self.create_publisher(PoseStamped, "/target_pose", 10)
        self._timer = self.create_timer(self.publish_period_sec, self._timer_cb)

    def start(self) -> None:
        self.started_at = time.time()
        self.publish_once()

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return min(time.time() - self.started_at, self.publish_duration_sec)

    def publish_once(self) -> None:
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(self.goal_ee_position[0])
        msg.pose.position.y = float(self.goal_ee_position[1])
        msg.pose.position.z = float(self.goal_ee_position[2])
        msg.pose.orientation.w = 1.0
        self._publisher.publish(msg)
        self.publish_count += 1

    def _timer_cb(self) -> None:
        if self.started_at is None:
            return
        if time.time() - self.started_at > self.publish_duration_sec:
            return
        self.publish_once()
