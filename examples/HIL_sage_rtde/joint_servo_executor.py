#!/usr/bin/env python3
"""Joint-space servo executor for pure RTDE HIL control."""

from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np


class JointCommandBackend(Protocol):
    def publish(self, q_cmd: np.ndarray, dt: Optional[float] = None) -> None:
        ...


def parse_joint_limit(value: str | float | int, n_dof: int, name: str) -> np.ndarray:
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
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return result


class ServoExecutionLogger:
    """CSV logger for planner targets, servo commands, RTDE state, and timing."""

    def __init__(self, log_dir: str | os.PathLike, joint_names: list[str]):
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.joint_names = list(joint_names)
        self.n_dof = len(self.joint_names)
        self._lock = threading.Lock()
        self._closed = False
        self._path = self.log_dir / "pure_rtde_servo_log.csv"
        self._file = self._path.open("w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._start_wall_time = time.time()

        header = [
            "event",
            "wall_time",
            "elapsed_s",
            "sequence",
            "source",
            "loop_dt_s",
            "mpc_dt_s",
            "opt_dt_s",
            "target_age_s",
            "max_abs_cmd_step",
        ]
        for prefix in ("q_next", "q_cmd", "actual_q", "actual_qd", "target_q", "qd_cmd"):
            header.extend(f"{prefix}_{name}" for name in self.joint_names)
        self._writer.writerow(header)

    @property
    def path(self) -> Path:
        return self._path

    def log(
        self,
        event: str,
        sequence: int,
        source: str,
        q_next: Optional[np.ndarray],
        q_cmd: Optional[np.ndarray],
        actual_q: Optional[np.ndarray],
        actual_qd: Optional[np.ndarray],
        target_q: Optional[np.ndarray],
        qd_cmd: Optional[np.ndarray],
        loop_dt_s: Optional[float],
        mpc_dt_s: Optional[float],
        opt_dt_s: Optional[float],
        target_age_s: Optional[float] = None,
        max_abs_cmd_step: Optional[float] = None,
    ) -> None:
        wall_time = time.time()
        row = [
            event,
            f"{wall_time:.9f}",
            f"{wall_time - self._start_wall_time:.9f}",
            int(sequence),
            source,
            self._format_float(loop_dt_s),
            self._format_float(mpc_dt_s),
            self._format_float(opt_dt_s),
            self._format_float(target_age_s),
            self._format_float(max_abs_cmd_step),
        ]
        for vector in (
            self._vector_or_nan(q_next),
            self._vector_or_nan(q_cmd),
            self._vector_or_nan(actual_q),
            self._vector_or_nan(actual_qd),
            self._vector_or_nan(target_q),
            self._vector_or_nan(qd_cmd),
        ):
            row.extend(self._format_float(value) for value in vector)

        with self._lock:
            if not self._closed:
                self._writer.writerow(row)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._file.flush()
            self._file.close()
            self._closed = True

    def _vector_or_nan(self, value: Optional[np.ndarray]) -> np.ndarray:
        if value is None:
            return np.full(self.n_dof, np.nan, dtype=np.float64)
        result = np.asarray(value, dtype=np.float64).reshape(-1)
        if result.shape[0] < self.n_dof:
            padded = np.full(self.n_dof, np.nan, dtype=np.float64)
            padded[: result.shape[0]] = result
            return padded
        return result[: self.n_dof]

    @staticmethod
    def _format_float(value: Optional[float]) -> str:
        if value is None:
            return ""
        value = float(value)
        if not np.isfinite(value):
            return ""
        return f"{value:.9f}"


class JointServoExecutor:
    """Generate high-frequency q_cmd from low-frequency MPPI q_next targets."""

    def __init__(
        self,
        backend: JointCommandBackend,
        joint_names: list[str],
        servo_frequency: float,
        max_joint_speed: np.ndarray,
        actual_position_fn: Callable[[], Optional[np.ndarray]],
        actual_velocity_fn: Callable[[], Optional[np.ndarray]],
        target_position_fn: Optional[Callable[[], Optional[np.ndarray]]] = None,
        max_joint_acceleration: Optional[np.ndarray] = None,
        logger: Optional[ServoExecutionLogger] = None,
        log_every_n_ticks: int = 1,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        if servo_frequency <= 0.0:
            raise ValueError("servo_frequency must be positive")
        self.backend = backend
        self.joint_names = list(joint_names)
        self.n_dof = len(self.joint_names)
        self.servo_frequency = float(servo_frequency)
        self.servo_dt = 1.0 / self.servo_frequency
        self.max_joint_speed = np.asarray(max_joint_speed, dtype=np.float64).reshape(self.n_dof)
        self.max_joint_acceleration = (
            None
            if max_joint_acceleration is None
            else np.asarray(max_joint_acceleration, dtype=np.float64).reshape(self.n_dof)
        )
        self.actual_position_fn = actual_position_fn
        self.actual_velocity_fn = actual_velocity_fn
        self.target_position_fn = target_position_fn
        self.logger = logger
        self.log_every_n_ticks = max(1, int(log_every_n_ticks))
        self.log_fn = log_fn or (lambda message: None)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._q_cmd: Optional[np.ndarray] = None
        self._qd_cmd = np.zeros(self.n_dof, dtype=np.float64)
        self._q_next: Optional[np.ndarray] = None
        self._target_wall_time: Optional[float] = None
        self._source = "unset"
        self._sequence = 0
        self._tick_count = 0
        self._last_mppi_loop_dt_s: Optional[float] = None
        self._last_mpc_dt_s: Optional[float] = None
        self._last_opt_dt_s: Optional[float] = None

    def start(self, initial_position: Optional[np.ndarray] = None) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if initial_position is None:
            initial_position = self.actual_position_fn()
        if initial_position is None:
            raise RuntimeError("JointServoExecutor requires an initial RTDE actual_q")
        initial_position = self._validate_joint_vector(initial_position, "initial_position")
        with self._lock:
            self._q_cmd = initial_position.copy()
            self._q_next = initial_position.copy()
            self._target_wall_time = time.time()
            self._source = "initial_actual_q"
            self._qd_cmd[:] = 0.0
        self.backend.publish(initial_position, dt=self.servo_dt)
        self._thread = threading.Thread(target=self._run_loop, name="PureRTDEJointServoExecutor", daemon=True)
        self._thread.start()

    def set_target(
        self,
        q_next: np.ndarray,
        source: str = "mppi",
        actual_q: Optional[np.ndarray] = None,
        actual_qd: Optional[np.ndarray] = None,
        target_q: Optional[np.ndarray] = None,
        loop_dt_s: Optional[float] = None,
        mpc_dt_s: Optional[float] = None,
        opt_dt_s: Optional[float] = None,
    ) -> None:
        q_next = self._validate_joint_vector(q_next, "q_next")
        with self._lock:
            self._q_next = q_next.copy()
            self._target_wall_time = time.time()
            self._source = source
            q_cmd = None if self._q_cmd is None else self._q_cmd.copy()
            qd_cmd = self._qd_cmd.copy()
            self._sequence += 1
            sequence = self._sequence
            self._last_mppi_loop_dt_s = loop_dt_s
            self._last_mpc_dt_s = mpc_dt_s
            self._last_opt_dt_s = opt_dt_s
        if self.logger is not None:
            self.logger.log(
                event="target_update",
                sequence=sequence,
                source=source,
                q_next=q_next,
                q_cmd=q_cmd,
                actual_q=actual_q,
                actual_qd=actual_qd,
                target_q=target_q,
                qd_cmd=qd_cmd,
                loop_dt_s=loop_dt_s,
                mpc_dt_s=mpc_dt_s,
                opt_dt_s=opt_dt_s,
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
        next_time = time.perf_counter()
        last_tick_wall: Optional[float] = None
        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_time:
                time.sleep(min(next_time - now, self.servo_dt))
                continue

            tick_wall = time.time()
            servo_loop_dt_s = None if last_tick_wall is None else tick_wall - last_tick_wall
            last_tick_wall = tick_wall
            actual_q = self.actual_position_fn()
            actual_qd = self.actual_velocity_fn()
            rtde_target_q = self.target_position_fn() if self.target_position_fn is not None else None

            with self._lock:
                q_cmd = None if self._q_cmd is None else self._q_cmd.copy()
                q_next = None if self._q_next is None else self._q_next.copy()
                qd_prev = self._qd_cmd.copy()
                source = self._source
                target_wall_time = self._target_wall_time
                sequence = self._sequence
                mpc_dt_s = self._last_mpc_dt_s
                opt_dt_s = self._last_opt_dt_s

            if q_cmd is not None and q_next is not None:
                new_q_cmd, new_qd_cmd, max_step = self._step_toward(q_cmd, q_next, qd_prev, self.servo_dt)
                try:
                    self.backend.publish(new_q_cmd, dt=self.servo_dt)
                except Exception as exc:
                    self.log_fn(f"[JointServoExecutor] servoJ publish failed: {exc}")
                    self._stop_event.set()
                    break

                with self._lock:
                    self._q_cmd = new_q_cmd.copy()
                    self._qd_cmd = new_qd_cmd.copy()
                self._tick_count += 1

                if self.logger is not None and self._tick_count % self.log_every_n_ticks == 0:
                    target_age = None if target_wall_time is None else tick_wall - target_wall_time
                    self.logger.log(
                        event="servo_tick",
                        sequence=sequence,
                        source=source,
                        q_next=q_next,
                        q_cmd=new_q_cmd,
                        actual_q=actual_q,
                        actual_qd=actual_qd,
                        target_q=rtde_target_q,
                        qd_cmd=new_qd_cmd,
                        loop_dt_s=servo_loop_dt_s,
                        mpc_dt_s=mpc_dt_s,
                        opt_dt_s=opt_dt_s,
                        target_age_s=target_age,
                        max_abs_cmd_step=max_step,
                    )

            next_time += self.servo_dt
            if next_time < time.perf_counter() - self.servo_dt:
                next_time = time.perf_counter() + self.servo_dt

    def _step_toward(
        self,
        q_cmd: np.ndarray,
        q_next: np.ndarray,
        qd_prev: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        delta = q_next - q_cmd
        max_delta = self.max_joint_speed * dt
        step = np.clip(delta, -max_delta, max_delta)
        qd_cmd = step / max(dt, 1.0e-9)

        if self.max_joint_acceleration is not None and np.any(self.max_joint_acceleration > 0.0):
            max_dv = self.max_joint_acceleration * dt
            qd_cmd = qd_prev + np.clip(qd_cmd - qd_prev, -max_dv, max_dv)
            qd_cmd = np.clip(qd_cmd, -self.max_joint_speed, self.max_joint_speed)
            step = qd_cmd * dt
            overshoot = np.abs(step) > np.abs(delta)
            step[overshoot] = delta[overshoot]
            qd_cmd = step / max(dt, 1.0e-9)

        q_cmd_next = q_cmd + step
        return q_cmd_next, qd_cmd, float(np.max(np.abs(step)))

    def _validate_joint_vector(self, value: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64).reshape(-1)[: self.n_dof]
        if result.shape[0] != self.n_dof:
            raise ValueError(f"{name} has {result.shape[0]} elements, expected {self.n_dof}")
        if np.any(~np.isfinite(result)):
            raise ValueError(f"{name} contains non-finite values")
        return result
