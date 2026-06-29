#!/usr/bin/env python3
"""Joint-space smoothing executor for SAGE-MPPI command outputs.

This module intentionally stays independent from the planner. It accepts
planner-produced joint targets, generates bounded-rate joint commands at a
higher frequency, and sends those commands through a backend.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray


def parse_joint_limit(value: str | float, n_dof: int, name: str) -> np.ndarray:
    """Parse a scalar or comma-separated joint limit into a vector."""
    if isinstance(value, (float, int)):
        values = [float(value)]
    else:
        values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(values) == 1:
        result = np.full(n_dof, values[0], dtype=np.float64)
    elif len(values) == n_dof:
        result = np.asarray(values, dtype=np.float64)
    else:
        raise ValueError(f"{name} must be scalar or contain exactly {n_dof} comma-separated values")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return result


class GazeboBackend:
    """Publish joint position commands to the existing Gazebo ros2_control topic."""

    def __init__(self, node: Node, controller_topic: str, joint_names: list[str]):
        self.node = node
        self.controller_topic = controller_topic
        self.joint_names = list(joint_names)
        self.n_dof = len(self.joint_names)
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._publisher = node.create_publisher(Float64MultiArray, controller_topic, qos_reliable)
        self.publish_count = 0

    def publish(self, q_cmd: np.ndarray) -> None:
        q_cmd = np.asarray(q_cmd, dtype=np.float64).reshape(-1)[: self.n_dof]
        if q_cmd.shape[0] != self.n_dof:
            raise ValueError(f"q_cmd has {q_cmd.shape[0]} elements, expected {self.n_dof}")
        msg = Float64MultiArray()
        msg.data = q_cmd.tolist()
        self._publisher.publish(msg)
        self.publish_count += 1


class ServoExecutionLogger:
    """CSV logger for planner target, smoothed command, and actual joint state."""

    def __init__(self, log_dir: str | os.PathLike, joint_names: list[str]):
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.joint_names = list(joint_names)
        self._lock = threading.Lock()
        self._file = (self.log_dir / "servo_execution_log.csv").open("w", newline="", buffering=1)
        self._writer = csv.writer(self._file)

        header = [
            "event",
            "wall_time",
            "elapsed_s",
            "sequence",
            "source",
            "target_age_s",
            "max_abs_target_error",
            "max_abs_cmd_step",
        ]
        for prefix in ("q_next", "q_cmd", "q_actual", "qd_cmd"):
            header.extend(f"{prefix}_{name}" for name in self.joint_names)
        header.extend(f"dq_actual_{name}" for name in self.joint_names)
        self._writer.writerow(header)
        self._start_wall_time = time.time()

    @property
    def path(self) -> Path:
        return self.log_dir / "servo_execution_log.csv"

    def log(
        self,
        event: str,
        sequence: int,
        source: str,
        q_next: Optional[np.ndarray],
        q_cmd: Optional[np.ndarray],
        q_actual: Optional[np.ndarray],
        qd_cmd: Optional[np.ndarray],
        dq_actual: Optional[np.ndarray] = None,
        target_age_s: Optional[float] = None,
        max_abs_cmd_step: Optional[float] = None,
    ) -> None:
        wall_time = time.time()
        q_next = self._vector_or_nan(q_next)
        q_cmd = self._vector_or_nan(q_cmd)
        q_actual = self._vector_or_nan(q_actual)
        qd_cmd = self._vector_or_nan(qd_cmd)
        dq_actual = self._vector_or_nan(dq_actual)

        if np.all(np.isfinite(q_next)) and np.all(np.isfinite(q_actual)):
            target_error = float(np.max(np.abs(q_next - q_actual)))
        else:
            target_error = np.nan

        row = [
            event,
            f"{wall_time:.9f}",
            f"{wall_time - self._start_wall_time:.9f}",
            sequence,
            source,
            self._format_float(target_age_s),
            self._format_float(target_error),
            self._format_float(max_abs_cmd_step),
        ]
        for vector in (q_next, q_cmd, q_actual, qd_cmd):
            row.extend(self._format_float(value) for value in vector)
        row.extend(self._format_float(value) for value in dq_actual)

        with self._lock:
            self._writer.writerow(row)

    def close(self) -> None:
        with self._lock:
            self._file.flush()
            self._file.close()

    def _vector_or_nan(self, value: Optional[np.ndarray]) -> np.ndarray:
        if value is None:
            return np.full(len(self.joint_names), np.nan, dtype=np.float64)
        result = np.asarray(value, dtype=np.float64).reshape(-1)
        if result.shape[0] < len(self.joint_names):
            padded = np.full(len(self.joint_names), np.nan, dtype=np.float64)
            padded[: result.shape[0]] = result
            return padded
        return result[: len(self.joint_names)]

    @staticmethod
    def _format_float(value: Optional[float]) -> str:
        if value is None:
            return ""
        value = float(value)
        if not np.isfinite(value):
            return ""
        return f"{value:.9f}"


class JointServoExecutor:
    """High-frequency bounded-rate joint command executor."""

    def __init__(
        self,
        backend: GazeboBackend,
        joint_names: list[str],
        executor_frequency: float,
        max_joint_speed: np.ndarray,
        actual_position_fn: Callable[[], Optional[np.ndarray]],
        actual_velocity_fn: Optional[Callable[[], Optional[np.ndarray]]] = None,
        logger: Optional[ServoExecutionLogger] = None,
        max_joint_acceleration: Optional[np.ndarray] = None,
        log_every_n_ticks: int = 1,
    ):
        if executor_frequency <= 0.0:
            raise ValueError("executor_frequency must be positive")
        self.backend = backend
        self.joint_names = list(joint_names)
        self.n_dof = len(self.joint_names)
        self.executor_frequency = float(executor_frequency)
        self.executor_dt = 1.0 / self.executor_frequency
        self.max_joint_speed = np.asarray(max_joint_speed, dtype=np.float64).reshape(self.n_dof)
        self.max_joint_acceleration = (
            None
            if max_joint_acceleration is None
            else np.asarray(max_joint_acceleration, dtype=np.float64).reshape(self.n_dof)
        )
        self.actual_position_fn = actual_position_fn
        self.actual_velocity_fn = actual_velocity_fn
        self.logger = logger
        self.log_every_n_ticks = max(1, int(log_every_n_ticks))

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._target: Optional[np.ndarray] = None
        self._target_source = "unset"
        self._target_wall_time: Optional[float] = None
        self._q_cmd: Optional[np.ndarray] = None
        self._qd_cmd = np.zeros(self.n_dof, dtype=np.float64)
        self._sequence = 0
        self._tick_count = 0

    def start(self, initial_position: Optional[np.ndarray] = None) -> None:
        if initial_position is None:
            initial_position = self.actual_position_fn()
        if initial_position is None:
            raise RuntimeError("JointServoExecutor requires an initial joint position")
        initial_position = np.asarray(initial_position, dtype=np.float64).reshape(-1)[: self.n_dof]
        with self._lock:
            self._q_cmd = initial_position.copy()
            self._target = initial_position.copy()
            self._target_source = "initial"
            self._target_wall_time = time.time()
            self._qd_cmd[:] = 0.0
        self.backend.publish(initial_position)
        self._thread = threading.Thread(target=self._run_loop, name="JointServoExecutor", daemon=True)
        self._thread.start()

    def set_target(
        self,
        q_next: np.ndarray,
        source: str = "mppi",
        q_actual: Optional[np.ndarray] = None,
        dq_actual: Optional[np.ndarray] = None,
    ) -> None:
        q_next = np.asarray(q_next, dtype=np.float64).reshape(-1)[: self.n_dof]
        if q_next.shape[0] != self.n_dof:
            raise ValueError(f"q_next has {q_next.shape[0]} elements, expected {self.n_dof}")
        with self._lock:
            self._target = q_next.copy()
            self._target_source = source
            self._target_wall_time = time.time()
            q_cmd = None if self._q_cmd is None else self._q_cmd.copy()
            qd_cmd = self._qd_cmd.copy()
            self._sequence += 1
            sequence = self._sequence
        if self.logger is not None:
            self.logger.log(
                event="target_update",
                sequence=sequence,
                source=source,
                q_next=q_next,
                q_cmd=q_cmd,
                q_actual=q_actual,
                qd_cmd=qd_cmd,
                dq_actual=dq_actual,
                target_age_s=0.0,
            )

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)

    def get_current_command(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._q_cmd is None else self._q_cmd.copy()

    def _run_loop(self) -> None:
        next_time = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_time:
                time.sleep(min(next_time - now, self.executor_dt))
                continue

            tick_start_wall = time.time()
            q_actual = self.actual_position_fn()
            dq_actual = self.actual_velocity_fn() if self.actual_velocity_fn is not None else None
            with self._lock:
                q_cmd = None if self._q_cmd is None else self._q_cmd.copy()
                q_target = None if self._target is None else self._target.copy()
                qd_prev = self._qd_cmd.copy()
                source = self._target_source
                target_wall_time = self._target_wall_time
                sequence = self._sequence

            if q_cmd is not None and q_target is not None:
                new_q_cmd, new_qd_cmd, max_step = self._step_toward(q_cmd, q_target, qd_prev, self.executor_dt)
                self.backend.publish(new_q_cmd)
                with self._lock:
                    self._q_cmd = new_q_cmd.copy()
                    self._qd_cmd = new_qd_cmd.copy()
                self._tick_count += 1

                if self.logger is not None and self._tick_count % self.log_every_n_ticks == 0:
                    target_age = None if target_wall_time is None else tick_start_wall - target_wall_time
                    self.logger.log(
                        event="servo_tick",
                        sequence=sequence,
                        source=source,
                        q_next=q_target,
                        q_cmd=new_q_cmd,
                        q_actual=q_actual,
                        qd_cmd=new_qd_cmd,
                        dq_actual=dq_actual,
                        target_age_s=target_age,
                        max_abs_cmd_step=max_step,
                    )

            next_time += self.executor_dt
            if next_time < time.monotonic() - self.executor_dt:
                next_time = time.monotonic() + self.executor_dt

    def _step_toward(
        self,
        q_cmd: np.ndarray,
        q_target: np.ndarray,
        qd_prev: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        delta = q_target - q_cmd
        max_delta = self.max_joint_speed * dt
        step = np.clip(delta, -max_delta, max_delta)
        qd_cmd = step / max(dt, 1.0e-9)

        if self.max_joint_acceleration is not None and np.any(self.max_joint_acceleration > 0.0):
            max_dv = self.max_joint_acceleration * dt
            qd_cmd = qd_prev + np.clip(qd_cmd - qd_prev, -max_dv, max_dv)
            qd_cmd = np.clip(qd_cmd, -self.max_joint_speed, self.max_joint_speed)
            step = qd_cmd * dt
            overshoot = np.sign(step) != np.sign(delta)
            step[overshoot] = 0.0
            too_far = np.abs(step) > np.abs(delta)
            step[too_far] = delta[too_far]
            qd_cmd = step / max(dt, 1.0e-9)

        q_new = q_cmd + step
        return q_new, qd_cmd, float(np.max(np.abs(step)))
