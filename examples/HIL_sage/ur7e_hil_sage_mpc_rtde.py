#!/usr/bin/env python3
"""UR7e HIL SAGE-MPPI entry with RTDE + servoJ execution.

The MPPI planner and refinement stack are reused from ur7e_hil_sage_mpc.py.
Only the post-planner execution interface is replaced:

    q_next -> JointServoExecutor -> RTDEBackend -> rtde_c.servoJ(q_cmd)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

STORM_ROOT = os.path.expanduser("~/storm")
if STORM_ROOT not in sys.path:
    sys.path.insert(0, STORM_ROOT)

from examples.HIL_sage import ur7e_hil_sage_mpc as base  # noqa: E402
from examples.HIL_sage.joint_servo_executor import (  # noqa: E402
    JointServoExecutor,
    ServoExecutionLogger,
    parse_joint_limit,
)
from examples.HIL_sage.rtde_backend import DryRunBackend, RTDEBackend  # noqa: E402

np.set_printoptions(precision=3, suppress=True)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RTDE_SERVO_FILE = Path(
    os.environ.get("SAGE_HIL_RTDE_SERVO_FILE", SCRIPT_DIR / "config" / "rtde_servo_hil_sage.yml")
)
ALLOWED_SERVO_FREQUENCIES = (125, 250, 500)


def _log(message: str) -> None:
    print(message, flush=True)


def _default_log_dir() -> str:
    return str(SCRIPT_DIR / "rtde_servo_logs" / time.strftime("%Y%m%d_%H%M%S"))


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _load_robot_ip_default() -> str:
    robot_params = _load_yaml(Path(base.ROBOT_FILE))
    return str(
        robot_params.get("sim_params", {})
        .get("hil", {})
        .get("robot_ip", "192.168.56.100")
    )


def _load_servo_defaults(path: Path) -> dict:
    defaults = {
        "robot_ip": _load_robot_ip_default(),
        "use_rtde_backend": True,
        "servo_frequency": 500,
        "lookahead_time": 0.10,
        "gain": 300,
        "max_joint_speed": "0.5",
        "max_joint_acceleration": "0.0",
        "log_every_n_ticks": 1,
    }
    defaults.update(_load_yaml(path))
    return defaults


def _add_original_hil_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cuda", action="store_true", default=True, help="use CUDA when available")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="disable CUDA")
    parser.add_argument("--rate", type=float, default=50.0, help="MPPI/control loop rate in Hz")
    parser.add_argument("--safe-mode", action="store_true", help="lower velocity and acceleration limits")
    parser.add_argument("--skip-confirm", action="store_true", help="skip interactive safety confirmation")
    parser.add_argument("--connection-timeout", type=float, default=30.0, help="seconds to wait for /joint_states")
    parser.add_argument("--max-steps", type=int, default=0, help="maximum control-loop steps; <=0 means unlimited")
    parser.add_argument("--viz-update-every", type=int, default=5, help="visualization update interval in loop steps")
    parser.add_argument("--enable-deployment-refinement", action="store_true", help="force enable deployment refinement")
    parser.add_argument("--disable-deployment-refinement", action="store_true", help="force disable deployment refinement")
    parser.add_argument("--enable-cartesian-refinement", action="store_true", help="force enable local Cartesian refinement")
    parser.add_argument("--disable-cartesian-refinement", action="store_true", help="force disable local Cartesian refinement")


def _build_parser() -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--rtde-servo-file", default=str(DEFAULT_RTDE_SERVO_FILE))
    pre_args, _ = pre_parser.parse_known_args()
    rtde_servo_file = Path(pre_args.rtde_servo_file).expanduser().resolve()
    defaults = _load_servo_defaults(rtde_servo_file)

    parser = argparse.ArgumentParser(description="UR7e SAGE-MPPI HIL controller with RTDE servoJ execution")
    _add_original_hil_args(parser)
    parser.add_argument("--rtde-servo-file", default=str(rtde_servo_file), help="RTDE servo parameter YAML")
    parser.add_argument("--robot-ip", default=str(defaults["robot_ip"]), help="URSim or real UR7e controller IP")
    parser.add_argument(
        "--servo-frequency",
        type=int,
        choices=ALLOWED_SERVO_FREQUENCIES,
        default=int(defaults["servo_frequency"]),
        help="JointServoExecutor frequency in Hz",
    )
    parser.add_argument("--lookahead-time", type=float, default=float(defaults["lookahead_time"]))
    parser.add_argument("--gain", type=int, default=int(defaults["gain"]))
    parser.add_argument(
        "--max-joint-speed",
        default=str(defaults["max_joint_speed"]),
        help="scalar or comma-separated rad/s joint speed limit",
    )
    parser.add_argument(
        "--max-joint-acceleration",
        default=str(defaults["max_joint_acceleration"]),
        help="0 disables acceleration limiting; otherwise scalar or comma-separated rad/s^2",
    )
    use_rtde_default = bool(defaults["use_rtde_backend"])
    rtde_group = parser.add_mutually_exclusive_group()
    rtde_group.add_argument(
        "--use-rtde-backend",
        dest="use_rtde_backend",
        action="store_true",
        default=use_rtde_default,
        help="send q_cmd to UR through RTDE servoJ",
    )
    rtde_group.add_argument(
        "--no-use-rtde-backend",
        dest="use_rtde_backend",
        action="store_false",
        help="dry-run q_cmd generation without ROS command publish or RTDE send",
    )
    parser.add_argument("--servo-log-dir", default=_default_log_dir(), help="directory for rtde_servo_execution_log.csv")
    parser.add_argument(
        "--servo-log-every",
        type=int,
        default=int(defaults["log_every_n_ticks"]),
        help="log every N servo ticks",
    )
    return parser


def _install_rtde_robot_interface(args: argparse.Namespace) -> None:
    class RTDESageRobotInterface(base.HILSageRobotInterface):
        """Robot interface that replaces forward_position_controller with RTDE servoJ."""

        def __init__(
            self,
            joint_names: list,
            control_rate: float = 50.0,
            max_velocity: float = 0.5,
            max_acceleration: float = 1.0,
        ):
            base.Node.__init__(self, "storm_hil_sage_mpc_rtde")
            self.joint_names = joint_names
            self.n_dof = len(joint_names)
            self.control_rate = float(control_rate)
            self.control_dt = 1.0 / self.control_rate
            self.max_velocity = float(max_velocity)
            self.max_acceleration = float(max_acceleration)

            self._lock = base.Lock()
            self._positions = None
            self._velocities = None
            self._prev_velocities = None
            self._prev_time = None
            self._state_received = False
            self._state_count = 0
            self._cmd_count = 0
            self._target_pos = None
            self._last_cmd_positions = None
            self._last_cmd_velocities = None
            self._prev_collision_marker_count = 0

            self._servo_started = False
            self._servo_close_lock = threading.Lock()
            self._servo_closed = False
            self._last_q_next_wall_time: Optional[float] = None

            qos = base.QoSProfile(depth=10)
            self.sub_joint_states = self.create_subscription(
                base.JointState,
                "/joint_states",
                self._joint_state_callback,
                qos,
            )
            self.sub_target = self.create_subscription(base.PoseStamped, "/target_pose", self._target_callback, qos)
            self.pub_ee_pose = self.create_publisher(base.PoseStamped, "/ee_pose", qos)
            self.pub_markers = self.create_publisher(base.MarkerArray, "/visualization_marker_array", qos)
            self.pub_collision_sphere_markers = self.create_publisher(
                base.MarkerArray,
                "/collision_sphere_marker_array",
                qos,
            )
            self.pub_top_traj_markers = self.create_publisher(
                base.MarkerArray,
                "/mppi_top_trajectories",
                qos,
            )

            configured_speed = parse_joint_limit(args.max_joint_speed, self.n_dof, "max_joint_speed")
            if getattr(args, "safe_mode", False):
                speed_cap = np.full(self.n_dof, self.max_velocity, dtype=np.float64)
                self._max_joint_speed = np.minimum(configured_speed, speed_cap)
            else:
                self._max_joint_speed = configured_speed
            parsed_acceleration = parse_joint_limit(
                args.max_joint_acceleration,
                self.n_dof,
                "max_joint_acceleration",
            )
            self._max_joint_acceleration = (
                parsed_acceleration if np.any(parsed_acceleration > 0.0) else None
            )

            if args.use_rtde_backend:
                self._backend = RTDEBackend(
                    robot_ip=args.robot_ip,
                    lookahead_time=args.lookahead_time,
                    gain=args.gain,
                    n_dof=self.n_dof,
                    log_fn=lambda message: self.get_logger().info(message),
                )
            else:
                self._backend = DryRunBackend(
                    n_dof=self.n_dof,
                    log_fn=lambda message: self.get_logger().info(message),
                )

            self._servo_logger = ServoExecutionLogger(args.servo_log_dir, self.joint_names)
            self._servo_executor = JointServoExecutor(
                backend=self._backend,
                joint_names=self.joint_names,
                servo_frequency=float(args.servo_frequency),
                max_joint_speed=self._max_joint_speed,
                max_joint_acceleration=self._max_joint_acceleration,
                actual_position_fn=self._read_actual_q,
                actual_velocity_fn=self.get_joint_velocities,
                logger=self._servo_logger,
                log_every_n_ticks=max(1, int(args.servo_log_every)),
                log_fn=lambda message: self.get_logger().error(message),
            )

            self.get_logger().info("HIL SAGE RTDE servo robot interface ready")
            self.get_logger().info(f"  robot_ip: {args.robot_ip}")
            self.get_logger().info(f"  use_rtde_backend: {bool(args.use_rtde_backend)}")
            self.get_logger().info(f"  servo_frequency: {float(args.servo_frequency):.1f} Hz")
            self.get_logger().info(f"  lookahead_time: {float(args.lookahead_time):.3f}")
            self.get_logger().info(f"  gain: {int(args.gain)}")
            self.get_logger().info(f"  max_joint_speed: {np.round(self._max_joint_speed, 4).tolist()} rad/s")
            self.get_logger().info(
                "  max_joint_acceleration: "
                + (
                    "disabled"
                    if self._max_joint_acceleration is None
                    else f"{np.round(self._max_joint_acceleration, 4).tolist()} rad/s^2"
                )
            )
            self.get_logger().info(f"  servo_log: {self._servo_logger.path}")
            self.get_logger().info("  forward_position_controller publish: disabled")

        def get_joint_velocities(self) -> Optional[np.ndarray]:
            with self._lock:
                if self._velocities is None:
                    return None
                return self._velocities.copy()

        def send_position_command(self, positions: np.ndarray):
            q_next = np.asarray(positions, dtype=np.float64).reshape(-1)[: self.n_dof]
            if q_next.shape[0] != self.n_dof:
                raise ValueError(f"q_next has {q_next.shape[0]} elements, expected {self.n_dof}")
            if np.any(~np.isfinite(q_next)):
                raise ValueError("q_next contains non-finite values")

            if not self._servo_started:
                self._start_servo_executor()

            now = time.time()
            loop_dt_s = None if self._last_q_next_wall_time is None else now - self._last_q_next_wall_time
            self._last_q_next_wall_time = now
            actual_q = self._read_actual_q()
            actual_dq = self.get_joint_velocities()
            self._servo_executor.set_target(
                q_next,
                source="mppi",
                actual_q=actual_q,
                actual_dq=actual_dq,
                loop_dt_s=loop_dt_s,
            )
            self._cmd_count += 1

            if self._cmd_count % 25 == 0:
                q_cmd = self._servo_executor.get_current_command()
                target_q = self._servo_executor.get_current_target()
                self.get_logger().info(
                    "rtde_servo "
                    f"q_next={np.round(q_next, 4).tolist()} "
                    f"q_cmd={np.round(q_cmd, 4).tolist() if q_cmd is not None else None} "
                    f"actual_q={np.round(actual_q, 4).tolist() if actual_q is not None else None} "
                    f"target_q={np.round(target_q, 4).tolist() if target_q is not None else None} "
                    f"loop_dt={float('nan') if loop_dt_s is None else loop_dt_s:.4f}s"
                )

        def destroy_node(self):
            self.close_servo()
            base.Node.destroy_node(self)

        def close_servo(self) -> None:
            with self._servo_close_lock:
                if self._servo_closed:
                    return
                self._servo_closed = True

            try:
                self._servo_executor.stop()
            except Exception as exc:
                self.get_logger().warn(f"JointServoExecutor stop failed: {exc}")
            try:
                self._backend.stop()
            except Exception as exc:
                self.get_logger().warn(f"RTDE backend stop failed: {exc}")
            try:
                self._servo_logger.close()
                self.get_logger().info(f"servo execution log: {self._servo_logger.path}")
            except Exception as exc:
                self.get_logger().warn(f"servo logger close failed: {exc}")

        def _start_servo_executor(self) -> None:
            initial_actual_q = self._read_actual_q()
            if initial_actual_q is None:
                initial_actual_q = self.get_joint_positions()
            if initial_actual_q is None:
                raise RuntimeError("cannot initialize q_cmd: no RTDE actual_q or ROS /joint_states position")
            self._servo_executor.start(initial_actual_q)
            self._servo_started = True
            self.get_logger().info(
                "JointServoExecutor started from actual_q="
                f"{np.round(initial_actual_q, 4).tolist()}"
            )

        def _read_actual_q(self) -> Optional[np.ndarray]:
            getter = getattr(self._backend, "get_actual_q", None)
            if callable(getter):
                actual_q = getter()
                if actual_q is not None:
                    return actual_q
            return self.get_joint_positions()

    base.HILSageRobotInterface = RTDESageRobotInterface


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    _log("=" * 60)
    _log("UR7e SAGE-MPPI HIL + RTDE servoJ")
    _log("=" * 60)
    _log("Execution chain: q_next -> JointServoExecutor -> RTDEBackend -> servoJ(q_cmd)")
    _log(f"RTDE servo config: {args.rtde_servo_file}")

    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    _install_rtde_robot_interface(args)
    return base.hil_sage_control_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
