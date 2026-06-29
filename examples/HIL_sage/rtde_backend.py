#!/usr/bin/env python3
"""RTDE backend for UR servoJ command streaming."""

from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np


class RTDEBackend:
    """Connect to UR RTDE and send joint targets with servoJ."""

    def __init__(
        self,
        robot_ip: str,
        lookahead_time: float,
        gain: int,
        n_dof: int = 6,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        self.robot_ip = str(robot_ip)
        self.lookahead_time = float(lookahead_time)
        self.gain = int(gain)
        self.n_dof = int(n_dof)
        self.log_fn = log_fn or (lambda message: None)
        self._lock = threading.Lock()
        self._stopped = False

        try:
            import rtde_control
            import rtde_receive
        except ImportError as exc:
            raise RuntimeError(
                "ur_rtde Python package is required for RTDEBackend "
                "(install/import rtde_control and rtde_receive in the active environment)"
            ) from exc

        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.log_fn(f"[RTDEBackend] connected to {self.robot_ip}")

    def get_actual_q(self) -> Optional[np.ndarray]:
        try:
            actual_q = np.asarray(self.rtde_r.getActualQ(), dtype=np.float64).reshape(-1)[: self.n_dof]
        except Exception as exc:
            self.log_fn(f"[RTDEBackend] getActualQ failed: {exc}")
            return None
        if actual_q.shape[0] != self.n_dof or np.any(~np.isfinite(actual_q)):
            self.log_fn(f"[RTDEBackend] invalid actual_q: {actual_q}")
            return None
        return actual_q

    def publish(self, q_cmd: np.ndarray, dt: Optional[float] = None) -> None:
        q_cmd = np.asarray(q_cmd, dtype=np.float64).reshape(-1)[: self.n_dof]
        if q_cmd.shape[0] != self.n_dof:
            raise ValueError(f"q_cmd has {q_cmd.shape[0]} elements, expected {self.n_dof}")
        if np.any(~np.isfinite(q_cmd)):
            raise ValueError("q_cmd contains non-finite values")
        servo_time = 0.002 if dt is None else float(dt)
        with self._lock:
            if self._stopped:
                return
            self.rtde_c.servoJ(
                q_cmd.tolist(),
                0.0,
                0.0,
                servo_time,
                self.lookahead_time,
                self.gain,
            )

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True

        for method_name in ("servoStop", "stopScript"):
            method = getattr(self.rtde_c, method_name, None)
            if callable(method):
                try:
                    method()
                    self.log_fn(f"[RTDEBackend] {method_name} ok")
                except Exception as exc:
                    self.log_fn(f"[RTDEBackend] {method_name} failed: {exc}")

        for obj_name, obj in (("rtde_c", self.rtde_c), ("rtde_r", self.rtde_r)):
            disconnect = getattr(obj, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception as exc:
                    self.log_fn(f"[RTDEBackend] {obj_name}.disconnect failed: {exc}")


class DryRunBackend:
    """Non-actuating backend used when --no-use-rtde-backend is selected."""

    def __init__(self, n_dof: int = 6, log_fn: Optional[Callable[[str], None]] = None):
        self.n_dof = int(n_dof)
        self.log_fn = log_fn or (lambda message: None)
        self.last_q_cmd: Optional[np.ndarray] = None
        self.publish_count = 0
        self.log_fn("[DryRunBackend] active; q_cmd will be logged but not sent to ROS or RTDE")

    def publish(self, q_cmd: np.ndarray, dt: Optional[float] = None) -> None:
        del dt
        q_cmd = np.asarray(q_cmd, dtype=np.float64).reshape(-1)[: self.n_dof]
        if q_cmd.shape[0] != self.n_dof:
            raise ValueError(f"q_cmd has {q_cmd.shape[0]} elements, expected {self.n_dof}")
        self.last_q_cmd = q_cmd.copy()
        self.publish_count += 1

    def get_actual_q(self) -> Optional[np.ndarray]:
        return None

    def stop(self) -> None:
        self.log_fn("[DryRunBackend] stop")
