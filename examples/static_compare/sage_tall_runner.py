#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.SAGE_MPPI.clean_SAGE.reach_static_ur7e_tall import (
    DeploymentRefinementStack,
    SAGE_MPPI,
    _apply_refinement_overrides,
    _build_tensor_args,
    _configure_default_goal,
    _get_clean_top_ee_trajs_world,
    _get_execution_mode,
    _get_robot_pose_world,
    _load_robot_and_world_params,
    _make_clean_task,
    _recover_command_strict,
    _reset_control_process_timing_strict,
)
from examples.sim_gazebo.gazebo_obstacle_utils import count_primitive_obstacles, spawn_gazebo_obstacles
from examples.sim_gazebo.reach_static_ur7e import GazeboRobotInterface, inv_transform_point, transform_point
from examples.sim_gazebo.reach_static_ur7e_tall import (
    CollisionSphereVisualizer,
    TallGazeboRobotInterface,
    _compute_link_poses_robot_frame,
    _shutdown_control_process,
)
from examples.static_compare.utils.metrics import (
    nan,
    smoothness_jerk,
    trajectory_length_ee,
    trajectory_length_joint,
)
from examples.static_compare.utils.ros_nodes import TargetPosePublisher
from examples.static_compare.utils.static_collision_checker import StaticTallCollisionChecker


SAGE_ENTRYPOINT = "examples/SAGE_MPPI/clean_SAGE/reach_static_ur7e_tall.py"
SAGE_REFERENCE_SCRIPT = "examples/SAGE_MPPI/clean_SAGE/run_all_reach_static_tall.sh"
SAGE_CONFIG = "examples/SAGE_MPPI/clean_SAGE/config/ur7e_reacher_gazebo_tall_sage_clean.yml"
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class CleanTallBenchmarkRobotInterface(TallGazeboRobotInterface):
    """Clean-SAGE tall Gazebo interface copied into the benchmark runner."""

    def __init__(self, joint_names: list[str], control_rate: float = 50.0):
        super().__init__(joint_names, control_rate=control_rate)
        self._latest_sim_time = None

    def _joint_state_callback(self, msg):
        self._latest_sim_time = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1.0e-9
        super()._joint_state_callback(msg)

    def get_latest_sim_time(self):
        return self._latest_sim_time

    def publish_markers(self, obstacles: dict, goal_pos: np.ndarray, ee_pos: np.ndarray, collision_spheres=None):
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker, MarkerArray

        GazeboRobotInterface.publish_markers(self, obstacles, goal_pos, ee_pos)
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        current_count = 0
        if collision_spheres:
            for marker_id, sphere in enumerate(collision_spheres):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = stamp
                marker.ns = "collision_spheres"
                marker.id = int(sphere.get("marker_id", marker_id))
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.frame_locked = True
                marker.pose.position.x = float(sphere["center_world"][0])
                marker.pose.position.y = float(sphere["center_world"][1])
                marker.pose.position.z = float(sphere["center_world"][2])
                marker.pose.orientation.w = 1.0
                marker.scale.x = 2.0 * sphere["radius"]
                marker.scale.y = 2.0 * sphere["radius"]
                marker.scale.z = 2.0 * sphere["radius"]
                marker.color = ColorRGBA(r=1.0, g=0.78, b=0.12, a=0.45)
                marker_array.markers.append(marker)
            current_count = len(collision_spheres)

        for marker_id in range(current_count, self._prev_collision_marker_count):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = stamp
            marker.ns = "collision_spheres"
            marker.id = marker_id
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)

        self._prev_collision_marker_count = current_count
        self.pub_collision_sphere_markers.publish(marker_array)


def _base_episode_row(episode_id: int, target_id: str) -> dict[str, Any]:
    return {
        "method_name": "sage_mppi_tuned",
        "episode_id": episode_id,
        "target_id": target_id,
        "scene": "tall",
        "success": False,
        "failure": True,
        "collision": False,
        "timeout": False,
        "final_ee_error": nan(),
        "final_joint_error": nan(),
        "minimum_safety_margin": nan(),
        "steps_to_goal": 0,
        "wall_time": nan(),
        "planning_time": nan(),
        "control_time_mean": nan(),
        "trajectory_length_joint": nan(),
        "trajectory_length_ee": nan(),
        "smoothness_jerk": nan(),
        "rrtstar_available": nan(),
        "rrtstar_exact_solution": nan(),
        "rrtstar_approximate_solution": nan(),
        "controller_class": "SAGE_MPPI",
        "controller_entrypoint": SAGE_ENTRYPOINT,
        "config_path": SAGE_CONFIG,
        "tuned_reference_script": SAGE_REFERENCE_SCRIPT,
        "target_publish_count": 0,
        "target_publish_duration": 0.0,
        "uses_clean_controller": True,
        "uses_native_margin": True,
        "deployment_refinement_enabled": True,
        "local_refinement_enabled": True,
        "margin_fallback": True,
        "target_not_available": False,
        "backend": "python_internal_tuned_logic",
        "skipped_reason": "",
    }


def _refinement_args() -> SimpleNamespace:
    return SimpleNamespace(
        disable_deployment_refinement=False,
        enable_deployment_refinement=False,
        enable_cartesian_refinement=False,
        disable_cartesian_refinement=False,
    )


def _wait_for_target(robot: CleanTallBenchmarkRobotInterface, goal_world: np.ndarray, timeout_sec: float = 5.0) -> np.ndarray | None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        target = robot.get_target_position()
        if target is not None and np.linalg.norm(np.asarray(target, dtype=float) - goal_world) < 1.0e-4:
            return np.asarray(target, dtype=float)
        time.sleep(0.05)
    return None


class SageTallRunner:
    """Direct Python runner mirroring clean_SAGE/reach_static_ur7e_tall.py."""

    def __init__(
        self,
        checker: StaticTallCollisionChecker,
        use_cuda: bool = True,
        rate: float = 50.0,
        viz_update_every: int = 1,
        target_publish_period: float = 0.25,
        target_publish_duration: float = 2.0,
    ) -> None:
        self.checker = checker
        self.use_cuda = bool(use_cuda)
        self.rate = float(rate)
        self.viz_update_every = max(1, int(viz_update_every))
        self.target_publish_period = float(target_publish_period)
        self.target_publish_duration = float(target_publish_duration)

    def run_episode(
        self,
        target: dict,
        episode_id: int,
        max_steps: int,
        success_threshold: float,
        max_runtime: float,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        row = _base_episode_row(episode_id, target["target_id"])
        step_rows: list[dict[str, Any]] = []
        q_path: list[np.ndarray] = []
        ee_path: list[np.ndarray] = []
        control_times: list[float] = []
        command_times: list[float] = []
        goal_world = np.asarray(target["goal_ee_position"], dtype=float)
        goal_q = np.asarray(target.get("goal_joint_positions", [math.nan] * 6), dtype=float)

        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        robot = None
        target_pub = None
        executor = None
        spin_thread = None
        mpc = None
        wall_start = time.time()
        spin_running = [True]
        try:
            robot_params, world_params = _load_robot_and_world_params()
            robot_pos, robot_quat_xyzw = _get_robot_pose_world(robot_params)
            tensor_args = _build_tensor_args(use_cuda=self.use_cuda)

            if not rclpy.ok():
                rclpy.init(args=None)
            robot = CleanTallBenchmarkRobotInterface(JOINT_NAMES, control_rate=self.rate)
            target_pub = TargetPosePublisher(
                goal_world,
                node_name=f"static_compare_sage_target_{episode_id}",
                publish_period_sec=self.target_publish_period,
                publish_duration_sec=self.target_publish_duration,
            )
            executor = MultiThreadedExecutor(num_threads=2)
            executor.add_node(robot)
            executor.add_node(target_pub)

            def _spin() -> None:
                while spin_running[0] and rclpy.ok():
                    try:
                        executor.spin_once(timeout_sec=0.1)
                    except RuntimeError as exc:
                        if "Destroyable" in str(exc):
                            break
                        raise

            spin_thread = threading.Thread(target=_spin, daemon=True)
            spin_thread.start()

            start_wait = time.time()
            while not robot.is_connected():
                if time.time() - start_wait > 10.0:
                    raise TimeoutError("Gazebo joint states were not received within 10s")
                time.sleep(0.1)

            count_primitive_obstacles(world_params, include_ground=False)
            spawn_gazebo_obstacles(robot, world_params, model_prefix=f"static_compare_sage_{episode_id}", include_ground=False)

            mpc = _make_clean_task(tensor_args)
            if not isinstance(mpc.controller, SAGE_MPPI):
                raise RuntimeError("clean SAGE runner did not instantiate SAGE_MPPI")
            control_dt = float(mpc.exp_params.get("control_dt", 0.02))
            refinement_cfg = _apply_refinement_overrides(mpc.deployment_refinement_config, _refinement_args())
            refinement = DeploymentRefinementStack(
                mpc=mpc,
                tensor_args=tensor_args,
                refinement_cfg=refinement_cfg,
                reset_timing_fn=_reset_control_process_timing_strict,
            )
            row["deployment_refinement_enabled"] = bool(refinement.enabled)
            row["local_refinement_enabled"] = bool(refinement.local_refinement is not None)
            _configure_default_goal(mpc, robot_pos, robot_quat_xyzw, inv_transform_point)

            target_pub.start()
            published_target = _wait_for_target(robot, goal_world, timeout_sec=5.0)
            if published_target is None:
                row["target_not_available"] = True
                raise TimeoutError("Published /target_pose was not observed by tuned SAGE robot interface")
            goal_ee_quat = np.ravel(mpc.controller.rollout_fn.goal_ee_quat.detach().cpu().numpy())
            current_goal_ee = inv_transform_point(robot_pos, robot_quat_xyzw, published_target)
            current_goal_world = published_target.copy()
            mpc.update_params(goal_ee_pos=current_goal_ee, goal_ee_quat=goal_ee_quat)
            refinement.on_goal_changed(0.0, control_dt)

            current_state = robot.get_state()
            for warm_idx in range(3):
                if current_state is None:
                    time.sleep(0.01)
                    current_state = robot.get_state()
                    continue
                try:
                    mpc.get_command_and_stats(warm_idx * control_dt, current_state, control_dt=control_dt, WAIT=True)
                except Exception:
                    pass
                time.sleep(0.01)

            rollout_fn = mpc.controller.rollout_fn
            collision_sphere_visualizer = CollisionSphereVisualizer(mpc.exp_params["model"]["robot_collision_params"])
            loop_count = 0
            loop_start = time.time()
            min_margin = float("inf")
            collision = False
            success = False
            timeout = False

            while True:
                iter_start = time.time()
                t_step = time.time() - loop_start
                if max_runtime > 0 and t_step >= max_runtime:
                    timeout = True
                    break
                if max_steps > 0 and loop_count >= max_steps:
                    timeout = True
                    break

                state = robot.get_state()
                if state is None:
                    time.sleep(control_dt)
                    continue
                q = np.asarray(state["position"], dtype=float)
                dq = np.asarray(state["velocity"], dtype=float)
                ddq = np.asarray(state["acceleration"], dtype=float)

                new_target = robot.get_target_position()
                if new_target is not None:
                    target_robot = inv_transform_point(robot_pos, robot_quat_xyzw, new_target)
                    if np.linalg.norm(target_robot - current_goal_ee) > 0.005:
                        current_goal_ee = target_robot.copy()
                        current_goal_world = np.asarray(new_target, dtype=float)
                        mpc.update_params(goal_ee_pos=current_goal_ee, goal_ee_quat=goal_ee_quat)
                        refinement.on_goal_changed(t_step, control_dt)

                curr = np.hstack([q, dq, ddq])
                ee_pose = rollout_fn.get_ee_pose(torch.as_tensor(curr, **tensor_args).unsqueeze(0))
                ee_pos_robot = np.ravel(ee_pose["ee_pos_seq"].detach().cpu().numpy())
                ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos_robot)
                ee_error = float(np.linalg.norm(ee_pos_world - goal_world))
                refinement.update_modes(error=ee_error, q=q, dq=dq, t_step=t_step, control_dt=control_dt)

                stats = {}
                try:
                    cmd_t0 = time.time()
                    cmd, stats = mpc.get_command_and_stats(t_step, state, control_dt=control_dt, WAIT=True)
                    command_times.append(time.time() - cmd_t0)
                except Exception:
                    cmd_t0 = time.time()
                    cmd = _recover_command_strict(mpc, t_step, state, control_dt)
                    command_times.append(time.time() - cmd_t0)
                    stats = mpc.get_latest_stats()

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
                target_positions = np.asarray(target_positions, dtype=np.float64).flatten()[: len(JOINT_NAMES)]
                robot.send_position_command(target_positions)
                robot.publish_ee_pose(ee_pos_world)

                if refinement.enabled:
                    refinement.maybe_trigger_recovery(
                        t_step=t_step,
                        ee_pos_world=ee_pos_world,
                        goal_world=current_goal_world,
                        joint_velocity=dq,
                        control_dt=control_dt,
                    )

                if loop_count % self.viz_update_every == 0:
                    link_pos_robot, link_rot_robot = _compute_link_poses_robot_frame(rollout_fn, q, dq, tensor_args)
                    collision_spheres_world = collision_sphere_visualizer.get_world_spheres(
                        link_pos_robot,
                        link_rot_robot,
                        robot_pos,
                        robot_quat_xyzw,
                    )
                    robot.publish_markers(world_params, current_goal_world, ee_pos_world, collision_spheres=collision_spheres_world)
                    top_trajs_world = _get_clean_top_ee_trajs_world(
                        mpc,
                        robot_pos,
                        robot_quat_xyzw,
                        current_ee_pos_world=ee_pos_world,
                        transform_point_fn=transform_point,
                        max_trajs=5,
                    )
                    robot.publish_top_trajectories(top_trajs_world)

                margin = self.checker.minimum_safety_margin(q)
                joint_error = float(np.linalg.norm(q - goal_q)) if goal_q.shape == (6,) and np.all(np.isfinite(goal_q)) else nan()
                min_margin = min(min_margin, margin)
                collision = collision or bool(margin <= self.checker.collision_threshold)
                q_path.append(q.copy())
                ee_path.append(np.asarray(ee_pos_world, dtype=float).copy())
                step_rows.append(
                    {
                        "method_name": "sage_mppi_tuned",
                        "episode_id": episode_id,
                        "target_id": target["target_id"],
                        "scene": "tall",
                        "step": loop_count,
                        "q": q.tolist(),
                        "ee_position": np.asarray(ee_pos_world, dtype=float).tolist(),
                        "ee_error": ee_error,
                        "joint_error": joint_error,
                        "safety_margin": margin,
                        "collision": bool(margin <= self.checker.collision_threshold),
                        "planning_time": command_times[-1] if command_times else nan(),
                        "control_time": time.time() - iter_start,
                        "wall_time": time.time() - wall_start,
                        "skipped_reason": "",
                    }
                )

                loop_count += 1
                success = bool(ee_error < success_threshold and not collision)
                if success:
                    break

                elapsed = time.time() - iter_start
                control_times.append(elapsed)
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if q_path:
                final_q = q_path[-1]
                final_ee = ee_path[-1]
                row["final_ee_error"] = float(np.linalg.norm(final_ee - goal_world))
                if goal_q.shape == (6,) and np.all(np.isfinite(goal_q)):
                    row["final_joint_error"] = float(np.linalg.norm(final_q - goal_q))
                row["minimum_safety_margin"] = float(min_margin)
                row["trajectory_length_joint"] = trajectory_length_joint(q_path)
                row["trajectory_length_ee"] = trajectory_length_ee(ee_path)
                row["smoothness_jerk"] = smoothness_jerk(q_path)
            row.update(
                {
                    "success": bool(success and not timeout and not collision),
                    "failure": not bool(success and not timeout and not collision),
                    "collision": bool(collision),
                    "timeout": bool(timeout),
                    "steps_to_goal": loop_count,
                    "wall_time": time.time() - wall_start,
                    "planning_time": float(np.sum(command_times)) if command_times else nan(),
                    "control_time_mean": float(np.mean(control_times)) if control_times else nan(),
                    "target_publish_count": target_pub.publish_count if target_pub is not None else 0,
                    "target_publish_duration": target_pub.elapsed if target_pub is not None else 0.0,
                }
            )
            return row, step_rows
        except Exception as exc:
            row.update(
                {
                    "failure": True,
                    "success": False,
                    "timeout": False,
                    "wall_time": time.time() - wall_start,
                    "target_publish_count": target_pub.publish_count if target_pub is not None else 0,
                    "target_publish_duration": target_pub.elapsed if target_pub is not None else 0.0,
                    "skipped_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            row["traceback"] = traceback.format_exc()
            return row, step_rows
        finally:
            spin_running[0] = False
            if mpc is not None:
                try:
                    _shutdown_control_process(getattr(mpc, "control_process", None))
                except Exception:
                    pass
                try:
                    mpc.close()
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
            for node in (target_pub, robot):
                if node is not None:
                    try:
                        node.destroy_node()
                    except Exception:
                        pass
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except Exception:
                    pass
