#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e SAGE-MPPI hardware-in-loop entry.

This project keeps the HIL ROS2/UR-driver boundary from examples/HIL and swaps
the controller path to the clean SAGE stack:
- SAGE_MPPI
- SageReacherTask
- DeploymentRefinementStack
"""

from __future__ import annotations

import argparse
import os
import queue
import signal
import sys
import threading
import time
from threading import Lock, Thread
from typing import Optional

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

STORM_ROOT = os.path.expanduser("~/storm")
if STORM_ROOT not in sys.path:
    sys.path.insert(0, STORM_ROOT)

torch.multiprocessing.set_start_method("spawn", force=True)

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import Point, PoseStamped
    from sensor_msgs.msg import JointState
    from std_msgs.msg import ColorRGBA, Float64MultiArray
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:
    print("=" * 60)
    print("error: ROS2 Python packages were not found")
    print("source ROS2 first, for example:")
    print("  source /opt/ros/humble/setup.bash")
    print("=" * 60)
    sys.exit(1)

from examples.SAGE_MPPI.deployment_refinement import DeploymentRefinementStack
from storm_kit.mpc.control.sage_mppi import SAGE_MPPI
from storm_kit.mpc.task.sage_reacher_task import SageReacherTask

np.set_printoptions(precision=3, suppress=True)

EXAMPLE_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(EXAMPLE_DIR, "config")
TASK_FILE = os.environ.get(
    "SAGE_HIL_TASK_FILE",
    os.path.join(CONFIG_DIR, "ur7e_reacher_hil_sage.yml"),
)
ROBOT_FILE = os.environ.get(
    "SAGE_HIL_ROBOT_FILE",
    os.path.join(CONFIG_DIR, "ur7e_robot_hil_sage.yml"),
)
WORLD_FILE = os.environ.get(
    "SAGE_HIL_WORLD_FILE",
    os.path.join(CONFIG_DIR, "collision_world_hil.yml"),
)


def _log(message: str) -> None:
    print(message, flush=True)


def _build_tensor_args(use_cuda: bool) -> dict:
    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    return {
        "device": torch.device(device, 0) if device == "cuda" else torch.device("cpu"),
        "dtype": torch.float32,
    }


def transform_point(position, orientation_xyzw, point):
    rot = Rotation.from_quat(orientation_xyzw)
    return rot.apply(point) + np.asarray(position)


def inv_transform_point(position, orientation_xyzw, point):
    rot = Rotation.from_quat(orientation_xyzw).inv()
    return rot.apply(np.asarray(point) - np.asarray(position))


def _drain_mp_queue(mp_queue) -> None:
    if mp_queue is None:
        return
    while True:
        try:
            mp_queue.get_nowait()
        except queue.Empty:
            break
        except Exception:
            break


def _close_mp_queue(mp_queue) -> None:
    for method_name in ("close", "cancel_join_thread"):
        method = getattr(mp_queue, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass


def _shutdown_control_process(control_process, join_timeout: float = 2.0) -> None:
    if control_process is None:
        return

    control_process.done = True
    _drain_mp_queue(getattr(control_process, "result_queue", None))

    done_message = {"state": None, "dt": None, "done": True, "params": None}
    opt_queue = getattr(control_process, "opt_queue", None)
    if opt_queue is not None:
        try:
            opt_queue.put_nowait(done_message)
        except queue.Full:
            _drain_mp_queue(opt_queue)
            try:
                opt_queue.put_nowait(done_message)
            except Exception:
                pass
        except Exception:
            pass

    opt_process = getattr(control_process, "opt_process", None)
    if opt_process is not None:
        opt_process.join(timeout=join_timeout)
        if opt_process.is_alive():
            _log("background MPC process did not exit in time; terminating")
            opt_process.terminate()
            opt_process.join(timeout=join_timeout)

    if opt_queue is not None:
        _close_mp_queue(opt_queue)
    result_queue = getattr(control_process, "result_queue", None)
    if result_queue is not None:
        _close_mp_queue(result_queue)


def _reset_control_process_timing_strict(control_process, t_step: float, control_dt: float) -> None:
    control_process.command = None
    control_process.command_tstep = control_process.traj_tstep + t_step + control_dt
    control_process.prev_mpc_tstep = max(0.0, t_step - control_dt)
    control_process.mpc_dt = control_dt
    control_process.params = None
    _drain_mp_queue(getattr(control_process, "result_queue", None))
    _drain_mp_queue(getattr(control_process, "opt_queue", None))


def _recover_command_strict(mpc, t_step: float, state: dict, control_dt: float):
    _reset_control_process_timing_strict(mpc.control_process, t_step, control_dt)
    return mpc.get_command(t_step, state, control_dt=control_dt, WAIT=True)


def _get_robot_pose_world(robot_params: dict):
    sim_params = robot_params.get("sim_params", {})
    robot_pose = sim_params.get("robot_pose", [0, 0, 0, 0, 0, 0, 1])
    robot_pos = np.asarray(robot_pose[:3], dtype=np.float64)
    robot_quat_xyzw = np.asarray(robot_pose[3:], dtype=np.float64)
    return robot_pos, robot_quat_xyzw


def _get_execution_mode(mpc) -> str:
    mppi_cfg = getattr(mpc, "exp_params", {}).get("mppi", {})
    mode = str(mppi_cfg.get("execution_mode", "mean")).strip().lower()
    if mode not in ("best_sample", "mean"):
        return "mean"
    return mode


def _apply_execution_mode(mpc) -> str:
    mode = _get_execution_mode(mpc)
    controller = getattr(mpc, "controller", None)
    if controller is not None and hasattr(controller, "execute_best"):
        controller.execute_best = mode == "best_sample"
    return mode


def _make_sage_task(tensor_args: dict) -> SageReacherTask:
    task = SageReacherTask(
        task_file=TASK_FILE,
        robot_file=ROBOT_FILE,
        world_file=WORLD_FILE,
        tensor_args=tensor_args,
    )
    _apply_execution_mode(task)
    return task


def _apply_refinement_overrides(refinement_cfg: dict, args) -> dict:
    refinement_cfg = dict(refinement_cfg or {})
    if "cartesian_refinement" in refinement_cfg:
        refinement_cfg["cartesian_refinement"] = dict(refinement_cfg["cartesian_refinement"])
    if "local_refinement" in refinement_cfg:
        refinement_cfg["local_refinement"] = dict(refinement_cfg["local_refinement"])

    if args.disable_deployment_refinement:
        refinement_cfg["enabled"] = False
        return refinement_cfg
    if args.enable_deployment_refinement:
        refinement_cfg["enabled"] = True

    local_cfg = dict(refinement_cfg.get("local_refinement", refinement_cfg.get("cartesian_refinement", {})))
    if args.enable_cartesian_refinement:
        refinement_cfg["enabled"] = True
        local_cfg["enabled"] = True
    elif args.disable_cartesian_refinement:
        local_cfg["enabled"] = False
    if local_cfg:
        refinement_cfg["local_refinement"] = local_cfg
    return refinement_cfg


def _configure_initial_goal(
    mpc,
    robot,
    rollout_fn,
    tensor_args: dict,
):
    default_goal = mpc.exp_params.get("default_goal")
    if default_goal is not None:
        goal_ee_pos_robot = np.array(default_goal.get("position", [0.4, 0.0, 0.4]), dtype=np.float64)
        goal_ee_quat = np.array(default_goal.get("orientation", [0.0, 0.707, 0.0, 0.707]), dtype=np.float64)
        mpc.update_params(goal_ee_pos=goal_ee_pos_robot, goal_ee_quat=goal_ee_quat)
        return goal_ee_pos_robot, goal_ee_quat, "config default_goal"

    curr_state = robot.get_state()
    if curr_state is None:
        raise RuntimeError("robot state is unavailable while configuring the initial goal")
    q = curr_state["position"]
    zero = np.zeros_like(q)
    full_state = np.hstack([q, zero, zero])
    ee_pose = rollout_fn.get_ee_pose(torch.as_tensor(full_state, **tensor_args).unsqueeze(0))
    goal_ee_pos_robot = np.ravel(ee_pose["ee_pos_seq"].detach().cpu().numpy())
    goal_ee_quat = np.ravel(rollout_fn.goal_ee_quat.detach().cpu().numpy())
    mpc.update_params(goal_ee_pos=goal_ee_pos_robot, goal_ee_quat=goal_ee_quat)
    return goal_ee_pos_robot, goal_ee_quat, "current ee pose"


def _compute_link_poses_robot_frame(rollout_fn, q: np.ndarray, dq: np.ndarray, tensor_args: dict):
    q_tensor = torch.as_tensor(q, **tensor_args).unsqueeze(0)
    dq_tensor = torch.as_tensor(dq, **tensor_args).unsqueeze(0)
    robot_model = rollout_fn.dynamics_model.robot_model
    robot_model.compute_fk_and_jacobian(
        q_tensor,
        dq_tensor,
        rollout_fn.exp_params["model"]["ee_link_name"],
    )

    link_pos_robot = []
    link_rot_robot = []
    for link_name in rollout_fn.dynamics_model.link_names:
        link_pos, link_rot = robot_model.get_link_pose(link_name)
        link_pos_robot.append(link_pos[0].detach().cpu().numpy())
        link_rot_robot.append(link_rot[0].detach().cpu().numpy())
    return np.stack(link_pos_robot, axis=0), np.stack(link_rot_robot, axis=0)


def _get_clean_top_ee_trajs(
    mpc,
    current_ee_pos: np.ndarray,
    max_trajs: int = 5,
):
    controller = getattr(mpc, "controller", None)
    top_trajs = getattr(controller, "top_trajs", None)
    if top_trajs is None:
        return None
    if isinstance(top_trajs, torch.Tensor):
        top_trajs_np = top_trajs.detach().cpu().numpy()
    else:
        top_trajs_np = np.asarray(top_trajs)
    if top_trajs_np.ndim == 2:
        top_trajs_np = top_trajs_np[None, ...]
    if top_trajs_np.ndim != 3 or top_trajs_np.shape[-1] != 3:
        return None

    current_ee_pos = np.asarray(current_ee_pos, dtype=np.float64).reshape(1, 3)
    stitched = []
    for traj_points in top_trajs_np[:max_trajs]:
        if len(traj_points) == 0:
            continue
        if np.linalg.norm(traj_points[0] - current_ee_pos[0]) < 1.0e-4:
            stitched.append(traj_points)
        else:
            stitched.append(np.concatenate([current_ee_pos, traj_points], axis=0))
    if not stitched:
        return None
    return np.asarray(stitched, dtype=np.float64)


class CollisionSphereVisualizer:
    def __init__(self, robot_collision_params: dict):
        sphere_config = os.path.expanduser(robot_collision_params["collision_spheres"])
        with open(sphere_config) as f:
            sphere_params = yaml.safe_load(f)

        self.link_names = list(robot_collision_params["link_objs"])
        raw_spheres_by_link = sphere_params["collision_spheres"]
        self.spheres_by_link = {}
        marker_id = 0
        for link_name in self.link_names:
            self.spheres_by_link[link_name] = []
            for sphere in raw_spheres_by_link.get(link_name, []):
                sphere_entry = dict(sphere)
                sphere_entry["marker_id"] = marker_id
                self.spheres_by_link[link_name].append(sphere_entry)
                marker_id += 1

    def get_robot_frame_spheres(self, link_pos_robot: np.ndarray, link_rot_robot: np.ndarray):
        spheres = []
        for link_idx, link_name in enumerate(self.link_names):
            link_pos = link_pos_robot[link_idx]
            link_rot = link_rot_robot[link_idx]
            for sphere in self.spheres_by_link.get(link_name, []):
                center_local = np.asarray(sphere["center"], dtype=np.float64)
                center_robot = link_rot @ center_local + link_pos
                spheres.append(
                    {
                        "marker_id": int(sphere["marker_id"]),
                        "center": center_robot,
                        "radius": float(sphere["radius"]),
                    }
                )
        return spheres


class HILSageRobotInterface(Node):
    def __init__(
        self,
        joint_names: list,
        control_rate: float = 50.0,
        max_velocity: float = 0.5,
        max_acceleration: float = 1.0,
    ):
        super().__init__("storm_hil_sage_mpc")
        self.joint_names = joint_names
        self.n_dof = len(joint_names)
        self.control_rate = float(control_rate)
        self.control_dt = 1.0 / self.control_rate
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)

        self._lock = Lock()
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

        qos = QoSProfile(depth=10)
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.sub_joint_states = self.create_subscription(JointState, "/joint_states", self._joint_state_callback, qos)
        self.sub_target = self.create_subscription(PoseStamped, "/target_pose", self._target_callback, qos)
        self.pub_position_cmd = self.create_publisher(
            Float64MultiArray,
            "/forward_position_controller/commands",
            qos_reliable,
        )
        self.pub_ee_pose = self.create_publisher(PoseStamped, "/ee_pose", qos)
        self.pub_markers = self.create_publisher(MarkerArray, "/visualization_marker_array", qos)
        self.pub_collision_sphere_markers = self.create_publisher(
            MarkerArray,
            "/collision_sphere_marker_array",
            qos,
        )
        self.pub_top_traj_markers = self.create_publisher(
            MarkerArray,
            "/mppi_top_trajectories",
            qos,
        )

        self.get_logger().info("HIL SAGE robot interface ready")
        self.get_logger().info(f"  control_rate: {self.control_rate:.1f} Hz")
        self.get_logger().info(f"  max_velocity: {self.max_velocity:.3f} rad/s")
        self.get_logger().info(f"  max_acceleration: {self.max_acceleration:.3f} rad/s^2")

    def _joint_state_callback(self, msg: JointState):
        positions = np.zeros(self.n_dof)
        velocities = np.zeros(self.n_dof)
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                positions[i] = msg.position[idx]
                if len(msg.velocity) > idx:
                    velocities[i] = msg.velocity[idx]
        with self._lock:
            self._positions = positions
            self._velocities = velocities
            if self._last_cmd_positions is None:
                self._last_cmd_positions = positions.copy()
                self._last_cmd_velocities = np.zeros(self.n_dof)
            self._state_received = True
            self._state_count += 1

    def _target_callback(self, msg: PoseStamped):
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
        with self._lock:
            self._target_pos = pos
        self.get_logger().info(f"target update: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    def get_joint_positions(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._positions is None:
                return None
            return self._positions.copy()

    def get_state(self) -> Optional[dict]:
        with self._lock:
            if not self._state_received:
                return None
            pos = self._positions.copy()
            vel = self._velocities.copy()
        current_time = time.time()
        if self._prev_velocities is not None and self._prev_time is not None:
            dt = max(current_time - self._prev_time, 0.001)
            accel = (vel - self._prev_velocities) / dt
        else:
            accel = np.zeros(self.n_dof)
        self._prev_velocities = vel.copy()
        self._prev_time = current_time
        return {"position": pos, "velocity": vel, "acceleration": accel}

    def get_target_position(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._target_pos is None:
                return None
            pos = self._target_pos.copy()
            self._target_pos = None
            return pos

    def send_position_command(self, positions: np.ndarray):
        positions = np.asarray(positions, dtype=np.float64).flatten()[: self.n_dof]
        if self._last_cmd_positions is not None:
            desired_vel = (positions - self._last_cmd_positions) / self.control_dt
            desired_vel = np.clip(desired_vel, -self.max_velocity, self.max_velocity)
            if self._last_cmd_velocities is not None:
                max_dv = self.max_acceleration * self.control_dt
                desired_vel = self._last_cmd_velocities + np.clip(
                    desired_vel - self._last_cmd_velocities,
                    -max_dv,
                    max_dv,
                )
            positions = self._last_cmd_positions + desired_vel * self.control_dt
            self._last_cmd_velocities = desired_vel.copy()
        else:
            self._last_cmd_velocities = np.zeros(self.n_dof)

        self._last_cmd_positions = positions.copy()
        msg = Float64MultiArray()
        msg.data = positions.tolist()
        self.pub_position_cmd.publish(msg)
        self._cmd_count += 1

    def publish_ee_pose(self, position: np.ndarray, orientation: Optional[np.ndarray] = None):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        if orientation is not None:
            msg.pose.orientation.x = float(orientation[0])
            msg.pose.orientation.y = float(orientation[1])
            msg.pose.orientation.z = float(orientation[2])
            msg.pose.orientation.w = float(orientation[3])
        else:
            msg.pose.orientation.w = 1.0
        self.pub_ee_pose.publish(msg)

    def publish_live_goal_ee_markers(self, goal_pos: np.ndarray, ee_pos: np.ndarray):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for marker_id, (ns, pos, color, scale) in enumerate(
            (
                ("goal", goal_pos, ColorRGBA(r=0.9, g=0.1, b=0.1, a=0.8), 0.06),
                ("ee", ee_pos, ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.8), 0.05),
            )
        ):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = stamp
            marker.ns = ns
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            marker.color = color
            marker_array.markers.append(marker)
        self.pub_markers.publish(marker_array)

    def publish_markers(
        self,
        obstacles: dict,
        goal_pos: np.ndarray,
        ee_pos: np.ndarray,
        collision_spheres=None,
    ):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker_id = 0

        for ns, pos, color, scale in (
            ("goal", goal_pos, ColorRGBA(r=0.9, g=0.1, b=0.1, a=0.8), 0.06),
            ("ee", ee_pos, ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.8), 0.05),
        ):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = stamp
            marker.ns = ns
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            marker.color = color
            marker_array.markers.append(marker)
            marker_id += 1

        coll_objs = (obstacles or {}).get("world_model", {}).get("coll_objs", {})
        for params in coll_objs.get("sphere", {}).values():
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = stamp
            marker.ns = "virtual_obstacles"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            pos = params.get("position", [0, 0, 0])
            radius = float(params.get("radius", 0.1))
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = 2.0 * radius
            marker.scale.y = 2.0 * radius
            marker.scale.z = 2.0 * radius
            marker.color = ColorRGBA(r=0.8, g=0.2, b=0.2, a=0.6)
            marker_array.markers.append(marker)
            marker_id += 1

        for name, params in coll_objs.get("cube", {}).items():
            if name == "ground":
                continue
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = stamp
            marker.ns = "virtual_obstacles"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            pose = params.get("pose", [0, 0, 0, 0, 0, 0, 1])
            dims = params.get("dims", [0.1, 0.1, 0.1])
            marker.pose.position.x = float(pose[0])
            marker.pose.position.y = float(pose[1])
            marker.pose.position.z = float(pose[2])
            marker.pose.orientation.x = float(pose[3])
            marker.pose.orientation.y = float(pose[4])
            marker.pose.orientation.z = float(pose[5])
            marker.pose.orientation.w = float(pose[6])
            marker.scale.x = float(dims[0])
            marker.scale.y = float(dims[1])
            marker.scale.z = float(dims[2])
            marker.color = ColorRGBA(r=0.5, g=0.5, b=0.8, a=0.6)
            marker_array.markers.append(marker)
            marker_id += 1
        self.pub_markers.publish(marker_array)

        collision_marker_array = MarkerArray()
        current_count = 0
        if collision_spheres:
            for sphere in collision_spheres:
                marker = Marker()
                marker.header.frame_id = "base_link"
                marker.header.stamp = stamp
                marker.ns = "collision_spheres"
                marker.id = int(sphere.get("marker_id", current_count))
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(sphere["center"][0])
                marker.pose.position.y = float(sphere["center"][1])
                marker.pose.position.z = float(sphere["center"][2])
                marker.pose.orientation.w = 1.0
                marker.scale.x = 2.0 * sphere["radius"]
                marker.scale.y = 2.0 * sphere["radius"]
                marker.scale.z = 2.0 * sphere["radius"]
                marker.color = ColorRGBA(r=1.0, g=0.78, b=0.12, a=0.45)
                collision_marker_array.markers.append(marker)
                current_count += 1
        for marker_id in range(current_count, self._prev_collision_marker_count):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = stamp
            marker.ns = "collision_spheres"
            marker.id = marker_id
            marker.action = Marker.DELETE
            collision_marker_array.markers.append(marker)
        self._prev_collision_marker_count = current_count
        self.pub_collision_sphere_markers.publish(collision_marker_array)

    def publish_top_trajectories(self, top_trajs):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        clear_marker = Marker()
        clear_marker.header.frame_id = "base_link"
        clear_marker.header.stamp = stamp
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        if top_trajs is None or len(top_trajs) == 0:
            self.pub_top_traj_markers.publish(marker_array)
            return

        for traj_id, traj_points in enumerate(top_trajs[:5]):
            line_marker = Marker()
            line_marker.header.frame_id = "base_link"
            line_marker.header.stamp = stamp
            line_marker.ns = "sage_top_trajs"
            line_marker.id = traj_id
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.pose.orientation.w = 1.0
            line_marker.scale.x = 0.002
            line_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.95)
            for point in np.asarray(traj_points, dtype=np.float64):
                if not np.all(np.isfinite(point)):
                    continue
                point_msg = Point()
                point_msg.x = float(point[0])
                point_msg.y = float(point[1])
                point_msg.z = float(point[2])
                line_marker.points.append(point_msg)
            if len(line_marker.points) >= 2:
                marker_array.markers.append(line_marker)
        self.pub_top_traj_markers.publish(marker_array)

    def is_connected(self) -> bool:
        return self._state_received


def hil_sage_control_main(args) -> int:
    _log("=" * 60)
    _log("UR7e SAGE-MPPI HIL")
    _log("=" * 60)
    _log(f"Task:  {TASK_FILE}")
    _log(f"Robot: {ROBOT_FILE}")
    _log(f"World: {WORLD_FILE}")

    with open(ROBOT_FILE) as f:
        robot_params = yaml.safe_load(f)
    with open(WORLD_FILE) as f:
        world_params = yaml.safe_load(f)

    sim_params = robot_params.get("sim_params", {})
    hil_params = sim_params.get("hil", {})
    safety_params = hil_params.get("safety", {})
    robot_pos, robot_quat_xyzw = _get_robot_pose_world(robot_params)
    joint_names = hil_params.get(
        "joint_names",
        [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
    )
    n_dof = len(joint_names)
    if args.safe_mode:
        max_velocity = 0.3
        max_acceleration = 0.5
    else:
        max_velocity = float(safety_params.get("max_velocity", 0.5))
        max_acceleration = float(safety_params.get("max_acceleration", 1.0))

    tensor_args = _build_tensor_args(use_cuda=args.cuda)
    _log(f"Compute device: {tensor_args['device']}")

    robot = None
    executor = None
    spin_thread = None
    mpc = None
    spin_running = [True]
    shutdown_event = threading.Event()

    try:
        rclpy.init(args=None)
        robot = HILSageRobotInterface(
            joint_names,
            control_rate=args.rate,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
        )
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(robot)

        def spin_with_check():
            while spin_running[0] and rclpy.ok() and not shutdown_event.is_set():
                try:
                    executor.spin_once(timeout_sec=0.1)
                except RuntimeError as exc:
                    if "Destroyable" in str(exc):
                        break
                    raise

        spin_thread = Thread(target=spin_with_check, daemon=True)
        spin_thread.start()

        _log("\nWaiting for real robot /joint_states...")
        start = time.time()
        while not robot.is_connected():
            if time.time() - start > float(args.connection_timeout):
                _log("error: did not receive /joint_states")
                _log("check UR ROS2 Driver, External Control, and network connection")
                return 1
            time.sleep(0.1)
        curr_pos = robot.get_joint_positions()
        _log("Connected to UR7e")
        _log(f"Current q rad: {np.round(curr_pos, 3)}")
        _log(f"Current q deg: {np.round(np.degrees(curr_pos), 1)}")

        _log("\nInitializing SAGE-MPPI...")
        mpc = _make_sage_task(tensor_args)
        if not isinstance(mpc.controller, SAGE_MPPI):
            raise RuntimeError("HIL SAGE task did not instantiate SAGE_MPPI")
        control_dt = float(mpc.exp_params.get("control_dt", 0.02))
        rollout_fn = mpc.controller.rollout_fn
        _log(f"control_dt={control_dt:.4f}s ({1.0 / control_dt:.1f} Hz)")
        _log(f"execution_mode={_get_execution_mode(mpc)}")

        refinement_cfg = _apply_refinement_overrides(mpc.deployment_refinement_config, args)
        refinement = DeploymentRefinementStack(
            mpc=mpc,
            tensor_args=tensor_args,
            refinement_cfg=refinement_cfg,
            reset_timing_fn=_reset_control_process_timing_strict,
            log_fn=_log,
        )
        _log(f"deployment_refinement_enabled={refinement.enabled}")
        _log(f"local_refinement_enabled={refinement.local_refinement is not None}")
        _log(f"cartesian_refinement_enabled={refinement.cartesian is not None}")
        _log(f"stall_recovery_enabled={refinement.stall_monitor is not None}")

        goal_ee_pos_robot, goal_ee_quat, goal_source = _configure_initial_goal(
            mpc,
            robot,
            rollout_fn,
            tensor_args,
        )
        current_goal_ee = goal_ee_pos_robot.copy()
        current_goal_world = transform_point(robot_pos, robot_quat_xyzw, current_goal_ee)
        _log(f"Initial goal source: {goal_source}")
        _log(f"Initial goal robot/base_link: {np.round(current_goal_ee, 3)}")
        _log(f"Initial goal display frame: {np.round(current_goal_world, 3)}")
        _log("Note: /target_pose uses position x/y/z only; orientation is ignored for retargeting")

        state = robot.get_state()
        curr = np.hstack([state["position"], state["velocity"], state["acceleration"]])
        ee_pose = rollout_fn.get_ee_pose(torch.as_tensor(curr, **tensor_args).unsqueeze(0))
        ee_pos_robot = np.ravel(ee_pose["ee_pos_seq"].detach().cpu().numpy())
        ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos_robot)
        _log("\nScene summary")
        _log(f"  current ee display frame: {np.round(ee_pos_world, 3)}")
        _log(f"  target distance: {np.linalg.norm(ee_pos_world - current_goal_world):.3f} m")
        coll_objs = world_params.get("world_model", {}).get("coll_objs", {})
        _log(f"  virtual spheres: {len(coll_objs.get('sphere', {}))}")
        _log(f"  virtual cubes: {len(coll_objs.get('cube', {}))}")
        for _ in range(10):
            robot.publish_markers(world_params, current_goal_world, ee_pos_world)
            time.sleep(0.1)

        if not args.skip_confirm:
            _log("\nSafety check before motion:")
            _log("  1. real workspace is clear")
            _log("  2. emergency stop is reachable")
            _log("  3. RViz target and virtual obstacles look correct")
            input("Press Enter to start SAGE-MPPI control, or Ctrl+C to cancel...")

        running = [True]

        def shutdown_handler(sig, frame):
            running[0] = False
            shutdown_event.set()

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        _log("\nWarming SAGE-MPPI...")
        current_state = robot.get_state()
        for warm_idx in range(3):
            if current_state is None:
                time.sleep(0.01)
                current_state = robot.get_state()
                continue
            try:
                mpc.get_command_and_stats(
                    warm_idx * control_dt,
                    current_state,
                    control_dt=control_dt,
                    WAIT=True,
                )
            except Exception as exc:
                _log(f"warmup warning: {exc}")
            time.sleep(0.01)

        collision_sphere_visualizer = CollisionSphereVisualizer(
            mpc.exp_params["model"]["robot_collision_params"]
        )

        _log("\nStarting control loop")
        loop_count = 0
        loop_start = time.time()
        last_wall_time = None
        max_steps = None if args.max_steps <= 0 else int(args.max_steps)
        viz_update_every = max(1, int(args.viz_update_every))

        while running[0] and rclpy.ok() and not shutdown_event.is_set():
            iter_start = time.time()
            t_step = time.time() - loop_start
            actual_loop_dt_wall = None if last_wall_time is None else iter_start - last_wall_time
            last_wall_time = iter_start

            state = robot.get_state()
            if state is None:
                time.sleep(control_dt)
                continue
            q = state["position"]
            dq = state["velocity"]
            ddq = state["acceleration"]
            curr = np.hstack([q, dq, ddq])

            new_target = robot.get_target_position()
            if new_target is not None:
                target_robot = inv_transform_point(robot_pos, robot_quat_xyzw, new_target)
                if np.linalg.norm(target_robot - current_goal_ee) > 0.005:
                    current_goal_ee = target_robot.copy()
                    current_goal_world = new_target.copy()
                    mpc.update_params(goal_ee_pos=current_goal_ee, goal_ee_quat=goal_ee_quat)
                    refinement.on_goal_changed(t_step, control_dt)
                    _log(
                        "[target update] display=%s robot=%s"
                        % (np.round(current_goal_world, 3), np.round(current_goal_ee, 3))
                    )

            ee_pose = rollout_fn.get_ee_pose(torch.as_tensor(curr, **tensor_args).unsqueeze(0))
            ee_pos_robot = np.ravel(ee_pose["ee_pos_seq"].detach().cpu().numpy())
            ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos_robot)
            ee_error = float(np.linalg.norm(ee_pos_world - current_goal_world))

            refinement.update_modes(
                error=ee_error,
                q=q,
                dq=dq,
                t_step=t_step,
                control_dt=control_dt,
            )

            stats = {}
            try:
                cmd, stats = mpc.get_command_and_stats(
                    t_step,
                    state,
                    control_dt=control_dt,
                    WAIT=True,
                )
            except Exception as exc:
                _log(f"[SAGE-MPPI] synchronous command failed, recovering: {exc}")
                try:
                    cmd = _recover_command_strict(mpc, t_step, state, control_dt)
                    stats = mpc.get_latest_stats()
                except Exception as recover_exc:
                    _log(f"[SAGE-MPPI] recovery failed: {recover_exc}")
                    time.sleep(control_dt)
                    continue

            nominal_position_cmd = None
            if cmd is not None and "position" in cmd:
                nominal_position_cmd = cmd["position"]
                if isinstance(nominal_position_cmd, torch.Tensor):
                    nominal_position_cmd = nominal_position_cmd.detach().cpu().numpy()

            override_cmd = refinement.maybe_get_override_command(
                error=ee_error,
                q=q,
                dq=dq,
                goal_ee_pos_robot=current_goal_ee,
                t_step=t_step,
                control_dt=control_dt,
                nominal_position_cmd=nominal_position_cmd,
            )
            if override_cmd is not None:
                cmd = override_cmd
            if cmd is None or "position" not in cmd:
                time.sleep(control_dt)
                continue

            target_positions = cmd["position"]
            if isinstance(target_positions, torch.Tensor):
                target_positions = target_positions.detach().cpu().numpy()
            target_positions = np.asarray(target_positions, dtype=np.float64).flatten()[:n_dof]
            robot.send_position_command(target_positions)
            robot.publish_ee_pose(ee_pos_world)
            robot.publish_live_goal_ee_markers(current_goal_world, ee_pos_world)

            if refinement.enabled:
                refinement.maybe_trigger_recovery(
                    t_step=t_step,
                    ee_pos_world=ee_pos_world,
                    goal_world=current_goal_world,
                    joint_velocity=dq,
                    control_dt=control_dt,
                )

            loop_count += 1
            if loop_count % viz_update_every == 0:
                link_pos_robot, link_rot_robot = _compute_link_poses_robot_frame(
                    rollout_fn,
                    q,
                    dq,
                    tensor_args,
                )
                collision_spheres = collision_sphere_visualizer.get_robot_frame_spheres(
                    link_pos_robot,
                    link_rot_robot,
                )
                robot.publish_markers(
                    world_params,
                    current_goal_world,
                    ee_pos_world,
                    collision_spheres=collision_spheres,
                )
                robot.publish_top_trajectories(_get_clean_top_ee_trajs(mpc, ee_pos_world))

            if loop_count % 25 == 0:
                local_stats = dict(getattr(refinement, "latest_local_refinement_stats", {}) or {})
                _log(
                    f"[{loop_count:5d}] t={t_step:.2f}s | "
                    f"ee_error={ee_error:.4f} | "
                    f"shape_skip={bool(stats.get('shape_update_skipped', False))} | "
                    f"reason={stats.get('shape_skip_reason', '') or '-'} | "
                    f"near_goal={bool(stats.get('near_goal_active', False))} | "
                    f"lr_active={bool(local_stats.get('local_refinement_active', False))} | "
                    f"lr_mode={local_stats.get('local_refinement_mode', 'off')} | "
                    f"lr_step={float(local_stats.get('local_refinement_step_norm', 0.0)):.4f} | "
                    f"actual_loop_dt_wall={float('nan') if actual_loop_dt_wall is None else actual_loop_dt_wall:.3f}s | "
                    f"opt_dt={mpc.opt_dt:.3f}s"
                )

            if max_steps is not None and loop_count >= max_steps:
                _log(f"Reached max_steps={max_steps}; exiting")
                break

            elapsed = time.time() - iter_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        shutdown_event.set()
        spin_running[0] = False
        if robot is not None:
            curr_pos = robot.get_joint_positions()
            if curr_pos is not None:
                try:
                    robot.send_position_command(curr_pos)
                except Exception:
                    pass
        if mpc is not None:
            try:
                _shutdown_control_process(getattr(mpc, "control_process", None))
            except Exception:
                pass
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=0.0)
            except TypeError:
                pass
            except Exception:
                pass
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=1.0)
        if robot is not None:
            try:
                robot.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="UR7e SAGE-MPPI HIL controller")
    parser.add_argument("--cuda", action="store_true", default=True, help="use CUDA when available")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="disable CUDA")
    parser.add_argument("--rate", type=float, default=50.0, help="command publish rate in Hz")
    parser.add_argument("--safe-mode", action="store_true", help="lower velocity and acceleration limits")
    parser.add_argument("--skip-confirm", action="store_true", help="skip interactive safety confirmation")
    parser.add_argument("--connection-timeout", type=float, default=30.0, help="seconds to wait for /joint_states")
    parser.add_argument("--max-steps", type=int, default=0, help="maximum control-loop steps; <=0 means unlimited")
    parser.add_argument("--viz-update-every", type=int, default=5, help="visualization update interval in loop steps")
    parser.add_argument("--enable-deployment-refinement", action="store_true", help="force enable deployment refinement")
    parser.add_argument("--disable-deployment-refinement", action="store_true", help="force disable deployment refinement")
    parser.add_argument("--enable-cartesian-refinement", action="store_true", help="force enable local Cartesian refinement")
    parser.add_argument("--disable-cartesian-refinement", action="store_true", help="force disable local Cartesian refinement")
    args = parser.parse_args()

    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return hil_sage_control_main(args)


if __name__ == "__main__":
    sys.exit(main())
