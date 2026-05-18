#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.sim_gazebo.gazebo_obstacle_utils import count_primitive_obstacles, spawn_gazebo_obstacles
from examples.sim_gazebo.reach_static_ur7e import GazeboReacherTask, inv_transform_point, transform_point
from examples.sim_gazebo.reach_static_ur7e_tall import (
    CollisionSphereVisualizer,
    TallGazeboRobotInterface,
    _compute_link_poses_robot_frame,
    _get_sync_command,
    _get_top_ee_trajs_world,
    _recover_command,
    _reset_control_process_timing,
    _shutdown_control_process,
)
from examples.static_compare.utils.io_utils import resolve_repo_path
from examples.static_compare.utils.metrics import (
    nan,
    smoothness_jerk,
    trajectory_length_ee,
    trajectory_length_joint,
)
from examples.static_compare.utils.ros_nodes import TargetPosePublisher
from examples.static_compare.utils.static_collision_checker import StaticTallCollisionChecker


STORM_ENTRYPOINT = "examples/sim_gazebo/reach_static_ur7e_tall.py"
STORM_REFERENCE_SCRIPT = "examples/sim_gazebo/bash/run_all_reach_static_tall.sh"
STORM_CONFIG = "examples/sim_gazebo/config/ur7e_reacher_gazebo_tall.yml"
ROBOT_FILE = "examples/sim_gazebo/config/ur7e_robot_gazebo.yml"
WORLD_FILE = "examples/sim_gazebo/config/collision_world_gazebo_tall.yml"
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _base_episode_row(episode_id: int, target_id: str) -> dict[str, Any]:
    return {
        "method_name": "storm_mppi_tuned",
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
        "controller_class": "MPPI",
        "controller_entrypoint": STORM_ENTRYPOINT,
        "config_path": STORM_CONFIG,
        "tuned_reference_script": STORM_REFERENCE_SCRIPT,
        "target_publish_count": 0,
        "target_publish_duration": 0.0,
        "uses_clean_controller": False,
        "uses_native_margin": False,
        "deployment_refinement_enabled": False,
        "local_refinement_enabled": False,
        "margin_fallback": True,
        "target_not_available": False,
        "backend": "python_internal_tuned_logic",
        "skipped_reason": "",
    }


def _wait_for_target(robot: TallGazeboRobotInterface, goal_world: np.ndarray, timeout_sec: float = 5.0) -> np.ndarray | None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        target = robot.get_target_position()
        if target is not None and np.linalg.norm(np.asarray(target, dtype=float) - goal_world) < 1.0e-4:
            return np.asarray(target, dtype=float)
        time.sleep(0.05)
    return None


class StormTallRunner:
    """Direct Python runner mirroring examples/sim_gazebo/reach_static_ur7e_tall.py."""

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
        self.use_cuda = bool(use_cuda and torch.cuda.is_available())
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
            robot_file = str(resolve_repo_path(ROBOT_FILE))
            task_file = str(resolve_repo_path(STORM_CONFIG))
            world_file = str(resolve_repo_path(WORLD_FILE))
            with open(robot_file, "r", encoding="utf-8") as f:
                robot_params = yaml.safe_load(f)
            with open(world_file, "r", encoding="utf-8") as f:
                world_params = yaml.safe_load(f)
            robot_pose = robot_params.get("sim_params", {}).get("robot_pose", [0, 0, 0, 0, 0, 0, 1])
            robot_pos = np.asarray(robot_pose[:3], dtype=np.float64)
            robot_quat_xyzw = np.asarray(robot_pose[3:], dtype=np.float64)

            if not rclpy.ok():
                rclpy.init(args=None)
            robot = TallGazeboRobotInterface(JOINT_NAMES, control_rate=self.rate)
            target_pub = TargetPosePublisher(
                goal_world,
                node_name=f"static_compare_storm_target_{episode_id}",
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

            n_world_spheres, n_world_cubes = count_primitive_obstacles(world_params, include_ground=False)
            spawn_gazebo_obstacles(robot, world_params, model_prefix=f"static_compare_storm_{episode_id}", include_ground=False)

            tensor_args = {
                "device": torch.device("cuda", 0) if self.use_cuda else torch.device("cpu"),
                "dtype": torch.float32,
            }
            mpc = GazeboReacherTask(task_file, robot_file, world_file, tensor_args)
            mpc.set_position_only_goal_mode()
            control_dt = float(mpc.exp_params.get("control_dt", 0.02))
            default_goal_seed_state = np.array([0.5, -1.2, 1.2, -1.57, -1.57, 0.0, 0, 0, 0, 0, 0, 0])
            mpc.update_params(goal_state=default_goal_seed_state)
            goal_ee_pos_robot = inv_transform_point(robot_pos, robot_quat_xyzw, np.array([0.5, -0.45, 0.4]))
            mpc.update_params(goal_ee_pos=goal_ee_pos_robot)

            target_pub.start()
            published_target = _wait_for_target(robot, goal_world, timeout_sec=5.0)
            if published_target is None:
                row["target_not_available"] = True
                raise TimeoutError("Published /target_pose was not observed by tuned STORM robot interface")
            current_goal_ee = inv_transform_point(robot_pos, robot_quat_xyzw, published_target)
            current_goal_world = published_target.copy()
            mpc.update_params(goal_ee_pos=current_goal_ee)

            current_state = robot.get_state()
            for warm_idx in range(5):
                if current_state is None:
                    time.sleep(0.01)
                    current_state = robot.get_state()
                    continue
                try:
                    _get_sync_command(mpc, warm_idx * control_dt, current_state, control_dt)
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
                cmd = None
                if new_target is not None:
                    target_robot = inv_transform_point(robot_pos, robot_quat_xyzw, new_target)
                    if np.linalg.norm(target_robot - current_goal_ee) > 0.005:
                        current_goal_ee = target_robot.copy()
                        current_goal_world = np.asarray(new_target, dtype=float)
                        mpc.update_params(goal_ee_pos=current_goal_ee)
                        _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                        cmd_t0 = time.time()
                        cmd = _get_sync_command(mpc, t_step, state, control_dt)
                        command_times.append(time.time() - cmd_t0)

                if cmd is None:
                    try:
                        cmd_t0 = time.time()
                        cmd = _get_sync_command(mpc, t_step, state, control_dt)
                        command_times.append(time.time() - cmd_t0)
                    except (IndexError, RuntimeError, ValueError):
                        cmd_t0 = time.time()
                        cmd = _recover_command(mpc, t_step, state, control_dt)
                        command_times.append(time.time() - cmd_t0)

                if cmd is None or "position" not in cmd:
                    time.sleep(control_dt)
                    continue

                target_positions = cmd["position"]
                if isinstance(target_positions, torch.Tensor):
                    target_positions = target_positions.detach().cpu().numpy()
                target_positions = np.asarray(target_positions, dtype=np.float64).flatten()[: len(JOINT_NAMES)]
                robot.send_position_command(target_positions)

                curr = np.hstack([q, dq, ddq])
                ee_pose = rollout_fn.get_ee_pose(torch.as_tensor(curr, **tensor_args).unsqueeze(0))
                ee_pos_robot = np.ravel(ee_pose["ee_pos_seq"].detach().cpu().numpy())
                ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos_robot)
                robot.publish_ee_pose(ee_pos_world)

                if loop_count % self.viz_update_every == 0:
                    link_pos_robot, link_rot_robot = _compute_link_poses_robot_frame(rollout_fn, q, dq, tensor_args)
                    collision_spheres_world = collision_sphere_visualizer.get_world_spheres(
                        link_pos_robot,
                        link_rot_robot,
                        robot_pos,
                        robot_quat_xyzw,
                    )
                    robot.publish_markers(world_params, current_goal_world, ee_pos_world, collision_spheres=collision_spheres_world)
                    top_trajs_world = _get_top_ee_trajs_world(mpc, robot_pos, robot_quat_xyzw, ee_pos_world, max_trajs=5)
                    robot.publish_top_trajectories(top_trajs_world)

                margin = self.checker.minimum_safety_margin(q)
                ee_error = float(np.linalg.norm(ee_pos_world - goal_world))
                joint_error = float(np.linalg.norm(q - goal_q)) if goal_q.shape == (6,) and np.all(np.isfinite(goal_q)) else nan()
                min_margin = min(min_margin, margin)
                collision = collision or bool(margin <= self.checker.collision_threshold)
                q_path.append(q.copy())
                ee_path.append(np.asarray(ee_pos_world, dtype=float).copy())
                step_rows.append(
                    {
                        "method_name": "storm_mppi_tuned",
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
