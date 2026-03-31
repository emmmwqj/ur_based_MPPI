#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone SAGE-MPPI entry for UR7e reach experiments.

This script is intentionally parallel to the baseline sim_gazebo entrypoints
without modifying them. It supports:

- Offline end-to-end validation using the real task -> ControlProcess ->
  SAGE_MPPI -> rollout -> command chain.
- Optional Gazebo/ROS2 execution with the same task assembly.
- Step-level and episode-level CSV logging for later baseline-vs-SAGE runs.
"""

import argparse
import csv
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import torch
import yaml

torch.multiprocessing.set_start_method("spawn", force=True)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from storm_kit.mpc.task.sage_reacher_task import SageReacherTask


def _repo_path(*parts):
    return os.path.join(REPO_ROOT, *parts)


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _timestamp_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _as_float(value, default=None):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        value = value.detach().reshape(-1)[0].item()
    return float(value)


def _as_bool(value, default=None):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        value = value.detach().reshape(-1)[0].item()
    return bool(value)


class CsvExperimentLogger:
    STEP_FIELDS = [
        "episode_id",
        "step_id",
        "t_step",
        "mode",
        "success",
        "failure",
        "final_goal_distance",
        "minimum_safety_margin",
        "safe_elite_fraction",
        "safe_weight_mass",
        "rho_k",
        "z_t",
        "covariance_fallback",
        "margin_fallback",
    ]
    EPISODE_FIELDS = [
        "episode_id",
        "mode",
        "num_steps",
        "success",
        "failure",
        "final_goal_distance",
        "episode_minimum_safety_margin",
        "safe_elite_fraction",
        "safe_weight_mass",
        "rho_k",
        "z_t",
        "covariance_fallback",
        "margin_fallback",
    ]

    def __init__(self, log_dir, run_name):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.step_path = os.path.join(self.log_dir, f"{run_name}_steps.csv")
        self.episode_path = os.path.join(self.log_dir, f"{run_name}_episodes.csv")
        self._init_csv(self.step_path, self.STEP_FIELDS)
        self._init_csv(self.episode_path, self.EPISODE_FIELDS)
        self.episode_min_margin = {}

    def _init_csv(self, path, fieldnames):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log_step(self, episode_id, step_id, t_step, mode, stats):
        margin = _as_float(stats.get("minimum_safety_margin"))
        if margin is not None:
            prev_min = self.episode_min_margin.get(episode_id)
            self.episode_min_margin[episode_id] = (
                margin if prev_min is None else min(prev_min, margin)
            )

        row = {
            "episode_id": episode_id,
            "step_id": step_id,
            "t_step": float(t_step),
            "mode": mode,
            "success": _as_bool(stats.get("success")),
            "failure": _as_bool(stats.get("failure")),
            "final_goal_distance": _as_float(stats.get("final_goal_distance")),
            "minimum_safety_margin": margin,
            "safe_elite_fraction": _as_float(stats.get("safe_elite_fraction"), 0.0),
            "safe_weight_mass": _as_float(stats.get("safe_weight_mass"), 0.0),
            "rho_k": _as_float(stats.get("rho_k"), 0.0),
            "z_t": int(_as_bool(stats.get("z_t"), False)),
            "covariance_fallback": _as_bool(stats.get("covariance_fallback"), False),
            "margin_fallback": _as_bool(stats.get("margin_fallback"), False),
        }
        with open(self.step_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.STEP_FIELDS)
            writer.writerow(row)

    def log_episode(self, episode_id, mode, num_steps, stats):
        row = {
            "episode_id": episode_id,
            "mode": mode,
            "num_steps": int(num_steps),
            "success": _as_bool(stats.get("success")),
            "failure": _as_bool(stats.get("failure")),
            "final_goal_distance": _as_float(stats.get("final_goal_distance")),
            "episode_minimum_safety_margin": self.episode_min_margin.get(episode_id),
            "safe_elite_fraction": _as_float(stats.get("safe_elite_fraction"), 0.0),
            "safe_weight_mass": _as_float(stats.get("safe_weight_mass"), 0.0),
            "rho_k": _as_float(stats.get("rho_k"), 0.0),
            "z_t": int(_as_bool(stats.get("z_t"), False)),
            "covariance_fallback": _as_bool(stats.get("covariance_fallback"), False),
            "margin_fallback": _as_bool(stats.get("margin_fallback"), False),
        }
        with open(self.episode_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.EPISODE_FIELDS)
            writer.writerow(row)


def _default_paths():
    return {
        "task_file": _repo_path("content", "configs", "mpc", "ur7e_reacher_sage.yml"),
        "robot_file": _repo_path("examples", "sim_gazebo", "config", "ur7e_robot_gazebo.yml"),
        "world_file": _repo_path("examples", "sim_gazebo", "config", "collision_world_gazebo.yml"),
    }


def _build_tensor_args(use_cuda):
    cuda_enabled = bool(use_cuda and torch.cuda.is_available())
    if use_cuda and not cuda_enabled:
        print("CUDA 不可用，自动回退到 CPU", flush=True)
    device = torch.device("cuda", 0) if cuda_enabled else torch.device("cpu")
    return {"device": device, "dtype": torch.float32}


def _goal_joint_positions_from_args(args):
    if args.goal is not None:
        return np.asarray(args.goal, dtype=np.float64)
    return np.asarray([0.5, -1.2, 1.2, -1.57, -1.57, 0.0], dtype=np.float64)


def _goal_state_from_positions(goal_positions):
    return np.concatenate([goal_positions, np.zeros_like(goal_positions)])


def _initial_state_dict(task, robot_file):
    robot_cfg = _load_yaml(robot_file)
    init_state = robot_cfg.get("sim_params", {}).get(
        "init_state",
        task.exp_params["model"]["init_state"],
    )
    init_state = np.asarray(init_state, dtype=np.float64)
    n_dofs = task.n_dofs
    return {
        "position": init_state[:n_dofs].copy(),
        "velocity": np.zeros(n_dofs, dtype=np.float64),
        "acceleration": np.zeros(n_dofs, dtype=np.float64),
    }


def _command_to_state(command):
    return {
        "position": np.asarray(command["position"], dtype=np.float64).copy(),
        "velocity": np.asarray(command["velocity"], dtype=np.float64).copy(),
        "acceleration": np.asarray(command["acceleration"], dtype=np.float64).copy(),
    }


def _build_task(args):
    paths = _default_paths()
    tensor_args = _build_tensor_args(args.cuda)
    task = SageReacherTask(
        task_file=args.task_file or paths["task_file"],
        robot_file=args.robot_file or paths["robot_file"],
        world_file=args.world_file or paths["world_file"],
        tensor_args=tensor_args,
    )
    goal_state = _goal_state_from_positions(_goal_joint_positions_from_args(args))
    task.update_params(goal_state=goal_state)
    return task


def _validate_async_controlprocess(args):
    task = _build_task(args)
    try:
        state = _initial_state_dict(task, task.robot_file)
        control_dt = task.exp_params["control_dt"]
        command = task.get_command(0.0, state, control_dt=control_dt, WAIT=False)
        if command is None or "position" not in command:
            raise RuntimeError("Async ControlProcess did not return a position command")
        return {
            "async_controlprocess_ok": True,
            "async_command_position_norm": float(np.linalg.norm(command["position"])),
        }
    finally:
        task.close()


def run_offline_episode(args):
    episode_id = args.episode_id or _timestamp_tag()
    logger = CsvExperimentLogger(args.log_dir, f"{args.run_name}_{episode_id}")

    async_result = None
    if args.validate_async:
        async_result = _validate_async_controlprocess(args)
        print(f"异步 ControlProcess 验证: {async_result}", flush=True)

    task = _build_task(args)
    try:
        state = _initial_state_dict(task, task.robot_file)
        control_dt = task.exp_params["control_dt"]
        t_step = 0.0
        final_stats = {}

        for step_id in range(args.steps):
            command, stats = task.get_command_and_stats(
                t_step=t_step,
                curr_state=state,
                control_dt=control_dt,
                WAIT=True,
            )
            logger.log_step(episode_id, step_id, t_step, "offline", stats)
            final_stats = dict(stats)
            print(
                "[offline] step=%03d t=%.3f goal_dist=%s min_margin=%s rho_k=%.4f z_t=%d"
                % (
                    step_id,
                    t_step,
                    stats.get("final_goal_distance"),
                    stats.get("minimum_safety_margin"),
                    float(stats.get("rho_k", 0.0)),
                    int(stats.get("z_t", 0)),
                ),
                flush=True,
            )
            state = _command_to_state(command)
            t_step += control_dt
            if args.stop_on_success and bool(stats.get("success")):
                break

        logger.log_episode(episode_id, "offline", step_id + 1, final_stats)
        print(f"离线日志写入: {logger.step_path}", flush=True)
        print(f"离线摘要写入: {logger.episode_path}", flush=True)
        return {
            "episode_id": episode_id,
            "async_result": async_result,
            "step_log": logger.step_path,
            "episode_log": logger.episode_path,
            "final_stats": final_stats,
        }
    finally:
        task.close()


def run_gazebo_episode(args):
    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
    except ImportError as exc:
        raise RuntimeError(
            "Gazebo 模式需要 ROS2 Python 环境，请先 source /opt/ros/humble/setup.bash"
        ) from exc

    class GazeboRobotInterface(Node):
        def __init__(self, joint_names, control_rate):
            super().__init__("storm_sage_mpc_reach_static")
            self.joint_names = joint_names
            self.n_dof = len(joint_names)
            self.control_rate = control_rate
            self.current_positions = None
            self.current_velocities = None
            self.prev_velocities = None
            self.prev_time = None
            self.state_received = False
            self.sub_joint_states = self.create_subscription(
                JointState,
                "/joint_states",
                self._joint_state_callback,
                10,
            )
            self.pub_position_cmd = self.create_publisher(
                Float64MultiArray,
                "/forward_position_controller/commands",
                10,
            )

        def _joint_state_callback(self, msg):
            positions = np.zeros(self.n_dof)
            velocities = np.zeros(self.n_dof)
            for i, name in enumerate(self.joint_names):
                if name in msg.name:
                    idx = msg.name.index(name)
                    positions[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        velocities[i] = msg.velocity[idx]
            self.current_positions = positions
            self.current_velocities = velocities
            self.state_received = True

        def wait_for_state(self, timeout=10.0):
            start = time.time()
            while not self.state_received:
                rclpy.spin_once(self, timeout_sec=0.1)
                if time.time() - start > timeout:
                    return False
            return True

        def get_state(self):
            if not self.state_received:
                return None
            current_time = time.time()
            if self.prev_velocities is not None and self.prev_time is not None:
                dt = max(current_time - self.prev_time, 0.001)
                acceleration = (self.current_velocities - self.prev_velocities) / dt
            else:
                acceleration = np.zeros(self.n_dof)
            self.prev_velocities = self.current_velocities.copy()
            self.prev_time = current_time
            return {
                "position": self.current_positions.copy(),
                "velocity": self.current_velocities.copy(),
                "acceleration": acceleration,
            }

        def send_position_command(self, positions):
            msg = Float64MultiArray()
            msg.data = positions.tolist()
            self.pub_position_cmd.publish(msg)

    episode_id = args.episode_id or _timestamp_tag()
    logger = CsvExperimentLogger(args.log_dir, f"{args.run_name}_{episode_id}")
    task = _build_task(args)
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    control_rate = args.rate
    executor = None
    robot = None
    spin_thread = None
    spin_running = True
    try:
        rclpy.init(args=None)
        robot = GazeboRobotInterface(joint_names, control_rate=control_rate)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(robot)

        def _spin():
            while spin_running and rclpy.ok():
                executor.spin_once(timeout_sec=0.1)

        spin_thread = threading.Thread(target=_spin, daemon=True)
        spin_thread.start()

        if not robot.wait_for_state(timeout=10.0):
            raise RuntimeError("未能从 Gazebo 接收到 /joint_states")

        control_dt = task.exp_params["control_dt"]
        t_step = 0.0
        final_stats = {}

        for step_id in range(args.steps):
            current_state = robot.get_state()
            if current_state is None:
                time.sleep(0.01)
                continue

            command, stats = task.get_command_and_stats(
                t_step=t_step,
                curr_state=current_state,
                control_dt=control_dt,
                WAIT=True,
            )
            robot.send_position_command(np.asarray(command["position"]).flatten()[: task.n_dofs])
            logger.log_step(episode_id, step_id, t_step, "gazebo", stats)
            final_stats = dict(stats)

            if step_id % max(args.print_every, 1) == 0:
                print(
                    "[gazebo] step=%03d t=%.3f goal_dist=%s min_margin=%s rho_k=%.4f z_t=%d"
                    % (
                        step_id,
                        t_step,
                        stats.get("final_goal_distance"),
                        stats.get("minimum_safety_margin"),
                        float(stats.get("rho_k", 0.0)),
                        int(stats.get("z_t", 0)),
                    ),
                    flush=True,
                )

            t_step += control_dt
            if args.stop_on_success and bool(stats.get("success")):
                break
            time.sleep(max((1.0 / control_rate) - control_dt, 0.0))

        logger.log_episode(episode_id, "gazebo", step_id + 1, final_stats)
        print(f"Gazebo 日志写入: {logger.step_path}", flush=True)
        print(f"Gazebo 摘要写入: {logger.episode_path}", flush=True)
        return {
            "episode_id": episode_id,
            "step_log": logger.step_path,
            "episode_log": logger.episode_path,
            "final_stats": final_stats,
        }
    finally:
        if task is not None:
            task.close()
        spin_running = False
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)
        if robot is not None:
            robot.destroy_node()
        if executor is not None:
            executor.shutdown(timeout_sec=1.0)
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="UR7e SAGE-MPPI static reach entrypoint")
    parser.add_argument("--mode", choices=["offline", "gazebo"], default="offline")
    parser.add_argument("--task-file", default=None)
    parser.add_argument("--robot-file", default=None)
    parser.add_argument("--world-file", default=None)
    parser.add_argument("--log-dir", default=_repo_path("examples", "sim_gazebo", "logs", "sage"))
    parser.add_argument("--run-name", default="ur7e_sage")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--validate-async", action="store_true", default=False)
    parser.add_argument("--stop-on-success", action="store_true", default=False)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--no-cuda", dest="cuda", action="store_false")
    parser.add_argument("--goal", type=float, nargs=6, default=None)
    args = parser.parse_args()

    if args.mode == "offline":
        result = run_offline_episode(args)
    else:
        result = run_gazebo_episode(args)

    print("运行完成:", result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
