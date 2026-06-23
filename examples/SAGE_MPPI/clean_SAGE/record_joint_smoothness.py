#!/usr/bin/env python3
"""Record UR7e joint smoothness data from the clean SAGE tall Gazebo run.

The recorder subscribes to:
  - /joint_states: executed joint position/velocity from Gazebo/ros2_control
  - /forward_position_controller/commands: controller target joint positions

At shutdown it writes CSV files, summary metrics, and optional PNG plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
except ImportError as exc:
    print("ROS2 Python packages are unavailable. Source ROS2 before running this recorder.")
    raise exc


DEFAULT_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _rms(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(finite))))


def _max_abs(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.max(np.abs(finite)))


def _integral_square(values: np.ndarray, t: np.ndarray) -> float | None:
    finite = np.isfinite(values) & np.isfinite(t)
    if np.count_nonzero(finite) < 2:
        return None
    return float(np.trapz(np.square(values[finite]), t[finite]))


def _derivative(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    deriv = np.full_like(values, np.nan, dtype=np.float64)
    if values.ndim != 2 or len(t) < 2:
        return deriv

    for joint_idx in range(values.shape[1]):
        y = values[:, joint_idx]
        finite = np.isfinite(y) & np.isfinite(t)
        if np.count_nonzero(finite) < 2:
            continue
        tt = t[finite]
        yy = y[finite]
        unique = np.concatenate(([True], np.diff(tt) > 1.0e-9))
        if np.count_nonzero(unique) < 2:
            continue
        tt = tt[unique]
        yy = yy[unique]
        grad = np.gradient(yy, tt, edge_order=1)
        indices = np.flatnonzero(finite)[unique]
        deriv[indices, joint_idx] = grad
    return deriv


def _choose_time(rows: list[dict], min_dt: float) -> tuple[np.ndarray, str, np.ndarray]:
    wall_time = np.asarray([row["wall_time"] for row in rows], dtype=np.float64)
    ros_time = np.asarray([row.get("ros_time", np.nan) for row in rows], dtype=np.float64)

    use_ros = (
        len(ros_time) >= 2
        and np.all(np.isfinite(ros_time))
        and float(ros_time[-1] - ros_time[0]) > 0.0
        and np.all(np.diff(ros_time) >= 0.0)
    )
    time_values = ros_time if use_ros else wall_time
    time_source = "ros_header" if use_ros else "wall_clock"

    keep = np.ones(len(time_values), dtype=bool)
    if len(time_values) > 1:
        keep[1:] = np.diff(time_values) > min_dt
    time_values = time_values[keep]
    return time_values - time_values[0], time_source, keep


def _rows_to_array(rows: list[dict], key: str, keep: np.ndarray, n_joints: int) -> np.ndarray:
    if not rows:
        return np.empty((0, n_joints), dtype=np.float64)
    return np.asarray([row[key] for row in rows], dtype=np.float64)[keep]


def _compute_metrics(
    label: str,
    t: np.ndarray,
    position: np.ndarray,
    joint_names: list[str],
    velocity_msg: np.ndarray | None = None,
) -> dict:
    velocity_fd = _derivative(position, t)
    acceleration_fd = _derivative(velocity_fd, t)
    jerk_fd = _derivative(acceleration_fd, t)

    per_joint = {}
    for idx, joint_name in enumerate(joint_names):
        position_values = position[:, idx] if position.size else np.asarray([], dtype=np.float64)
        finite_position = position_values[np.isfinite(position_values)]
        position_range = (
            float(np.max(finite_position) - np.min(finite_position))
            if finite_position.size
            else None
        )
        joint_metrics = {
            "position_range_rad": _finite_or_none(position_range),
            "velocity_fd_rms_rad_s": _finite_or_none(_rms(velocity_fd[:, idx])),
            "velocity_fd_max_abs_rad_s": _finite_or_none(_max_abs(velocity_fd[:, idx])),
            "acceleration_fd_rms_rad_s2": _finite_or_none(_rms(acceleration_fd[:, idx])),
            "acceleration_fd_max_abs_rad_s2": _finite_or_none(_max_abs(acceleration_fd[:, idx])),
            "jerk_fd_rms_rad_s3": _finite_or_none(_rms(jerk_fd[:, idx])),
            "jerk_fd_max_abs_rad_s3": _finite_or_none(_max_abs(jerk_fd[:, idx])),
            "integrated_squared_acceleration": _finite_or_none(
                _integral_square(acceleration_fd[:, idx], t)
            ),
            "integrated_squared_jerk": _finite_or_none(_integral_square(jerk_fd[:, idx], t)),
        }
        if velocity_msg is not None:
            joint_metrics.update(
                {
                    "velocity_msg_rms_rad_s": _finite_or_none(_rms(velocity_msg[:, idx])),
                    "velocity_msg_max_abs_rad_s": _finite_or_none(_max_abs(velocity_msg[:, idx])),
                }
            )
        per_joint[joint_name] = joint_metrics

    jerk_rms_values = [
        metrics["jerk_fd_rms_rad_s3"]
        for metrics in per_joint.values()
        if metrics["jerk_fd_rms_rad_s3"] is not None
    ]
    jerk_max_values = [
        metrics["jerk_fd_max_abs_rad_s3"]
        for metrics in per_joint.values()
        if metrics["jerk_fd_max_abs_rad_s3"] is not None
    ]
    isj_values = [
        metrics["integrated_squared_jerk"]
        for metrics in per_joint.values()
        if metrics["integrated_squared_jerk"] is not None
    ]

    median_dt = float(np.median(np.diff(t))) if len(t) > 1 else None
    max_dt = float(np.max(np.diff(t))) if len(t) > 1 else None
    return {
        "label": label,
        "sample_count": int(len(t)),
        "duration_s": _finite_or_none(float(t[-1] - t[0]) if len(t) > 1 else 0.0),
        "median_dt_s": _finite_or_none(median_dt),
        "max_dt_s": _finite_or_none(max_dt),
        "mean_joint_jerk_rms_rad_s3": _finite_or_none(np.mean(jerk_rms_values) if jerk_rms_values else None),
        "max_joint_jerk_abs_rad_s3": _finite_or_none(np.max(jerk_max_values) if jerk_max_values else None),
        "mean_integrated_squared_jerk": _finite_or_none(np.mean(isj_values) if isj_values else None),
        "per_joint": per_joint,
    }


def _write_csv(path: Path, header: list[str], rows: Iterable[Iterable]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _format_csv_value(value) -> str | float | int:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return float(value)
    return value


def _write_joint_csv(
    output_dir: Path,
    joint_names: list[str],
    t: np.ndarray,
    wall_time: np.ndarray,
    ros_time: np.ndarray,
    position: np.ndarray,
    velocity_msg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity_fd = _derivative(position, t)
    acceleration_fd = _derivative(velocity_fd, t)
    jerk_fd = _derivative(acceleration_fd, t)

    header = ["sample_index", "time_s", "wall_time", "ros_time"]
    for prefix in ("q", "dq_msg", "dq_fd", "ddq_fd", "jerk_fd"):
        header.extend(f"{prefix}_{_safe_name(name)}" for name in joint_names)

    rows = []
    for i in range(len(t)):
        row = [i, t[i], wall_time[i], ros_time[i]]
        for values in (position, velocity_msg, velocity_fd, acceleration_fd, jerk_fd):
            row.extend(values[i])
        rows.append([_format_csv_value(v) for v in row])

    _write_csv(output_dir / "joint_states.csv", header, rows)
    return velocity_fd, acceleration_fd, jerk_fd


def _write_command_csv(
    output_dir: Path,
    joint_names: list[str],
    t: np.ndarray,
    wall_time: np.ndarray,
    command_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity_fd = _derivative(command_position, t)
    acceleration_fd = _derivative(velocity_fd, t)
    jerk_fd = _derivative(acceleration_fd, t)

    header = ["sample_index", "time_s", "wall_time"]
    for prefix in ("cmd_q", "cmd_dq_fd", "cmd_ddq_fd", "cmd_jerk_fd"):
        header.extend(f"{prefix}_{_safe_name(name)}" for name in joint_names)

    rows = []
    for i in range(len(t)):
        row = [i, t[i], wall_time[i]]
        for values in (command_position, velocity_fd, acceleration_fd, jerk_fd):
            row.extend(values[i])
        rows.append([_format_csv_value(v) for v in row])

    _write_csv(output_dir / "commands.csv", header, rows)
    return velocity_fd, acceleration_fd, jerk_fd


def _plot_series(
    output_path: Path,
    title: str,
    ylabel: str,
    t: np.ndarray,
    values: np.ndarray,
    joint_names: list[str],
    x_min: float | None = None,
    x_max: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(joint_names), 1, figsize=(12, 10), sharex=True)
    if len(joint_names) == 1:
        axes = [axes]
    for idx, (axis, joint_name) in enumerate(zip(axes, joint_names)):
        axis.plot(t, values[:, idx], linewidth=1.2)
        axis.set_ylabel(joint_name, rotation=0, ha="right", va="center")
        axis.grid(True, alpha=0.25)
        if x_min is not None or x_max is not None:
            axis.set_xlim(left=x_min, right=x_max)
    axes[0].set_title(title)
    axes[-1].set_xlabel("time [s]")
    fig.text(0.02, 0.5, ylabel, rotation=90, va="center")
    fig.tight_layout(rect=(0.04, 0.02, 1.0, 0.98))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Joint Smoothness Report",
        "",
        "Lower acceleration, jerk, and integrated squared jerk indicate smoother motion.",
        "`joint_state` is the executed Gazebo trajectory. `command` is the target position stream sent to `/forward_position_controller/commands`.",
        "",
        "## Global Metrics",
        "",
        "| stream | samples | duration_s | median_dt_s | mean_joint_jerk_rms_rad_s3 | max_joint_jerk_abs_rad_s3 | mean_integrated_squared_jerk |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for stream_name in ("joint_state", "command"):
        metrics = summary.get(stream_name)
        if not metrics:
            continue
        lines.append(
            "| {name} | {samples} | {duration} | {dt} | {jerk_rms} | {jerk_max} | {isj} |".format(
                name=stream_name,
                samples=metrics.get("sample_count"),
                duration=_fmt(metrics.get("duration_s")),
                dt=_fmt(metrics.get("median_dt_s")),
                jerk_rms=_fmt(metrics.get("mean_joint_jerk_rms_rad_s3")),
                jerk_max=_fmt(metrics.get("max_joint_jerk_abs_rad_s3")),
                isj=_fmt(metrics.get("mean_integrated_squared_jerk")),
            )
        )

    lines.extend(["", "## Per-Joint Executed Trajectory", ""])
    joint_state = summary.get("joint_state", {})
    lines.extend(_per_joint_table(joint_state.get("per_joint", {})))

    if summary.get("command"):
        lines.extend(["", "## Per-Joint Command Trajectory", ""])
        lines.extend(_per_joint_table(summary["command"].get("per_joint", {})))

    path.write_text("\n".join(lines) + "\n")


def _fmt(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def _per_joint_table(per_joint: dict) -> list[str]:
    lines = [
        "| joint | vel_rms | acc_rms | jerk_rms | max_abs_jerk | integrated_squared_jerk |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for joint_name, metrics in per_joint.items():
        vel_rms = metrics.get("velocity_msg_rms_rad_s", metrics.get("velocity_fd_rms_rad_s"))
        lines.append(
            "| {joint} | {vel} | {acc} | {jerk} | {jerk_max} | {isj} |".format(
                joint=joint_name,
                vel=_fmt(vel_rms),
                acc=_fmt(metrics.get("acceleration_fd_rms_rad_s2")),
                jerk=_fmt(metrics.get("jerk_fd_rms_rad_s3")),
                jerk_max=_fmt(metrics.get("jerk_fd_max_abs_rad_s3")),
                isj=_fmt(metrics.get("integrated_squared_jerk")),
            )
        )
    return lines


class JointSmoothnessRecorder(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sage_joint_smoothness_recorder")
        self.joint_names = args.joint_names
        self.n_joints = len(self.joint_names)
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_dt = float(args.min_dt)
        self.no_plots = bool(args.no_plots)
        self.joint_state_topic = args.joint_state_topic
        self.command_topic = args.command_topic
        self.joint_rows: list[dict] = []
        self.command_rows: list[dict] = []

        self.create_subscription(JointState, args.joint_state_topic, self._joint_state_callback, 100)
        self.create_subscription(Float64MultiArray, args.command_topic, self._command_callback, 100)
        self.get_logger().info(f"Recording joint smoothness to {self.output_dir}")

    def _joint_state_callback(self, msg: JointState) -> None:
        name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
        position = np.full(self.n_joints, np.nan, dtype=np.float64)
        velocity = np.full(self.n_joints, np.nan, dtype=np.float64)
        for joint_idx, joint_name in enumerate(self.joint_names):
            msg_idx = name_to_idx.get(joint_name)
            if msg_idx is None:
                continue
            if msg_idx < len(msg.position):
                position[joint_idx] = msg.position[msg_idx]
            if msg_idx < len(msg.velocity):
                velocity[joint_idx] = msg.velocity[msg_idx]

        stamp = msg.header.stamp
        ros_time = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        self.joint_rows.append(
            {
                "wall_time": time.time(),
                "ros_time": ros_time,
                "position": position,
                "velocity": velocity,
            }
        )

    def _command_callback(self, msg: Float64MultiArray) -> None:
        values = np.full(self.n_joints, np.nan, dtype=np.float64)
        data = np.asarray(msg.data, dtype=np.float64).flatten()
        count = min(len(data), self.n_joints)
        if count:
            values[:count] = data[:count]
        self.command_rows.append({"wall_time": time.time(), "position": values})

    def save(self) -> None:
        summary = {
            "output_dir": str(self.output_dir),
            "joint_names": self.joint_names,
            "joint_state_topic": self.joint_state_topic,
            "command_topic": self.command_topic,
            "created_wall_time": time.time(),
        }

        if self.joint_rows:
            t, time_source, keep = _choose_time(self.joint_rows, self.min_dt)
            wall_time = np.asarray([row["wall_time"] for row in self.joint_rows], dtype=np.float64)[keep]
            ros_time = np.asarray([row["ros_time"] for row in self.joint_rows], dtype=np.float64)[keep]
            position = _rows_to_array(self.joint_rows, "position", keep, self.n_joints)
            velocity_msg = _rows_to_array(self.joint_rows, "velocity", keep, self.n_joints)
            velocity_fd, acceleration_fd, jerk_fd = _write_joint_csv(
                self.output_dir,
                self.joint_names,
                t,
                wall_time,
                ros_time,
                position,
                velocity_msg,
            )
            summary["joint_state_time_source"] = time_source
            summary["joint_state"] = _compute_metrics(
                "joint_state",
                t,
                position,
                self.joint_names,
                velocity_msg=velocity_msg,
            )
            if not self.no_plots:
                self._write_plots(
                    t,
                    position,
                    velocity_msg,
                    velocity_fd,
                    acceleration_fd,
                    jerk_fd,
                    prefix="joint_state",
                )
        else:
            summary["joint_state"] = None

        if self.command_rows:
            t_cmd, _, keep_cmd = _choose_time(self.command_rows, self.min_dt)
            cmd_wall_time = np.asarray([row["wall_time"] for row in self.command_rows], dtype=np.float64)[keep_cmd]
            record_start_wall_time = (
                float(self.joint_rows[0]["wall_time"])
                if self.joint_rows
                else float(cmd_wall_time[0])
            )
            t_cmd = cmd_wall_time - record_start_wall_time
            command_position = _rows_to_array(self.command_rows, "position", keep_cmd, self.n_joints)
            cmd_velocity_fd, cmd_acceleration_fd, cmd_jerk_fd = _write_command_csv(
                self.output_dir,
                self.joint_names,
                t_cmd,
                cmd_wall_time,
                command_position,
            )
            summary["command"] = _compute_metrics(
                    "command",
                    t_cmd - t_cmd[0],
                    command_position,
                    self.joint_names,
                )
            if not self.no_plots:
                x_max = None
                if self.joint_rows:
                    x_max = float(self.joint_rows[-1]["wall_time"] - record_start_wall_time)
                self._write_command_plots(
                    t_cmd,
                    command_position,
                    cmd_velocity_fd,
                    cmd_acceleration_fd,
                    cmd_jerk_fd,
                    x_max=x_max,
                )
        else:
            summary["command"] = None

        with (self.output_dir / "smoothness_summary.json").open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        _write_report(self.output_dir / "smoothness_report.md", summary)
        print(f"Saved smoothness outputs to {self.output_dir}", flush=True)

    def _write_plots(
        self,
        t: np.ndarray,
        position: np.ndarray,
        velocity_msg: np.ndarray,
        velocity_fd: np.ndarray,
        acceleration_fd: np.ndarray,
        jerk_fd: np.ndarray,
        prefix: str,
    ) -> None:
        try:
            velocity_for_plot = velocity_msg
            if not np.isfinite(velocity_for_plot).any():
                velocity_for_plot = velocity_fd
            _plot_series(
                self.output_dir / f"{prefix}_positions.png",
                "Executed joint positions",
                "position [rad]",
                t,
                position,
                self.joint_names,
            )
            _plot_series(
                self.output_dir / f"{prefix}_velocities.png",
                "Executed joint velocities",
                "velocity [rad/s]",
                t,
                velocity_for_plot,
                self.joint_names,
            )
            _plot_series(
                self.output_dir / f"{prefix}_accelerations_fd.png",
                "Executed finite-difference accelerations",
                "acceleration [rad/s^2]",
                t,
                acceleration_fd,
                self.joint_names,
            )
            _plot_series(
                self.output_dir / f"{prefix}_jerk_fd.png",
                "Executed finite-difference jerk",
                "jerk [rad/s^3]",
                t,
                jerk_fd,
                self.joint_names,
            )
        except Exception as exc:
            self.get_logger().warning(f"Plot generation failed: {exc}")

    def _write_command_plots(
        self,
        t: np.ndarray,
        command_position: np.ndarray,
        velocity_fd: np.ndarray,
        acceleration_fd: np.ndarray,
        jerk_fd: np.ndarray,
        x_max: float | None = None,
    ) -> None:
        try:
            _plot_series(
                self.output_dir / "command_positions.png",
                "Controller command joint positions",
                "position [rad]",
                t,
                command_position,
                self.joint_names,
                x_min=0.0,
                x_max=x_max,
            )
            _plot_series(
                self.output_dir / "command_velocities_fd.png",
                "Controller command finite-difference velocities",
                "velocity [rad/s]",
                t,
                velocity_fd,
                self.joint_names,
                x_min=0.0,
                x_max=x_max,
            )
            _plot_series(
                self.output_dir / "command_accelerations_fd.png",
                "Controller command finite-difference accelerations",
                "acceleration [rad/s^2]",
                t,
                acceleration_fd,
                self.joint_names,
                x_min=0.0,
                x_max=x_max,
            )
            _plot_series(
                self.output_dir / "command_jerk_fd.png",
                "Controller command finite-difference jerk",
                "jerk [rad/s^3]",
                t,
                jerk_fd,
                self.joint_names,
                x_min=0.0,
                x_max=x_max,
            )
        except Exception as exc:
            self.get_logger().warning(f"Command plot generation failed: {exc}")


def _parse_joint_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("joint name list cannot be empty")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Record joint smoothness for clean SAGE tall Gazebo")
    parser.add_argument("--output-dir", required=True, help="directory for CSV, plots, and summaries")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--command-topic", default="/forward_position_controller/commands")
    parser.add_argument("--joint-names", type=_parse_joint_names, default=DEFAULT_JOINT_NAMES)
    parser.add_argument("--min-dt", type=float, default=1.0e-5, help="minimum accepted sample interval")
    parser.add_argument("--no-plots", action="store_true", help="write CSV/JSON/Markdown only")
    args = parser.parse_args()

    stop_requested = False

    def _request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    rclpy.init(args=None)
    node = JointSmoothnessRecorder(args)
    try:
        while rclpy.ok() and not stop_requested:
            try:
                rclpy.spin_once(node, timeout_sec=0.1)
            except Exception as exc:
                if stop_requested or not rclpy.ok() or "context is not valid" in str(exc):
                    break
                raise
    finally:
        try:
            node.save()
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
