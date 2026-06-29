#!/usr/bin/env python3
"""RTDE receive/control backend for pure UR HIL execution."""

from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np


class RTDEBackend:
    """Read state from RTDEReceiveInterface and send commands with servoJ."""

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
        self._control_lock = threading.Lock()
        self._receive_lock = threading.Lock()
        self._stopped = False
        self._missing_methods_logged: set[str] = set()

        try:
            import rtde_control
            import rtde_receive
        except ImportError as exc:
            raise RuntimeError(
                "ur_rtde Python package is required: import rtde_control and rtde_receive failed"
            ) from exc

        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.log_fn(f"[RTDEBackend] connected to {self.robot_ip}")

    def get_actual_q(self) -> Optional[np.ndarray]:
        return self._read_vector("getActualQ", "actual_q")

    def get_actual_qd(self) -> Optional[np.ndarray]:
        return self._read_vector("getActualQd", "actual_qd")

    def get_target_q(self) -> Optional[np.ndarray]:
        return self._read_vector("getTargetQ", "target_q")

    def publish(self, q_cmd: np.ndarray, dt: Optional[float] = None) -> None:
        q_cmd = np.asarray(q_cmd, dtype=np.float64).reshape(-1)[: self.n_dof]
        if q_cmd.shape[0] != self.n_dof:
            raise ValueError(f"q_cmd has {q_cmd.shape[0]} elements, expected {self.n_dof}")
        if np.any(~np.isfinite(q_cmd)):
            raise ValueError("q_cmd contains non-finite values")
        servo_time = 0.002 if dt is None else float(dt)
        with self._control_lock:
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
        with self._control_lock:
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

        for label, obj in (("rtde_c", self.rtde_c), ("rtde_r", self.rtde_r)):
            disconnect = getattr(obj, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception as exc:
                    self.log_fn(f"[RTDEBackend] {label}.disconnect failed: {exc}")

    def _read_vector(self, method_name: str, label: str) -> Optional[np.ndarray]:
        with self._receive_lock:
            method = getattr(self.rtde_r, method_name, None)
            if not callable(method):
                if method_name not in self._missing_methods_logged:
                    self.log_fn(f"[RTDEBackend] RTDEReceiveInterface has no {method_name}")
                    self._missing_methods_logged.add(method_name)
                return None
            try:
                value = np.asarray(method(), dtype=np.float64).reshape(-1)[: self.n_dof]
            except Exception as exc:
                self.log_fn(f"[RTDEBackend] {method_name} failed: {exc}")
                return None
        if value.shape[0] != self.n_dof or np.any(~np.isfinite(value)):
            self.log_fn(f"[RTDEBackend] invalid {label}: {value}")
            return None
        return value
