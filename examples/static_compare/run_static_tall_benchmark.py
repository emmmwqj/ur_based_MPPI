#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.static_compare import rrtstar_ompl_adapter
from examples.static_compare.utils.io_utils import ensure_dir, load_json, resolve_repo_path, write_json
from examples.static_compare.utils.metrics import (
    EPISODE_FIELDS,
    STEP_FIELDS,
    nan,
    smoothness_jerk,
    trajectory_length_ee,
    trajectory_length_joint,
    write_csv,
)
from examples.static_compare.utils.static_collision_checker import StaticTallCollisionChecker


STORM_REFERENCE_SCRIPT = "examples/sim_gazebo/bash/run_all_reach_static_tall.sh"
SAGE_REFERENCE_SCRIPT = "examples/SAGE_MPPI/clean_SAGE/run_all_reach_static_tall.sh"
STORM_ENTRYPOINT = "examples/sim_gazebo/reach_static_ur7e_tall.py"
SAGE_ENTRYPOINT = "examples/SAGE_MPPI/clean_SAGE/reach_static_ur7e_tall.py"
STORM_CONFIG = "examples/sim_gazebo/config/ur7e_reacher_gazebo_tall.yml"
SAGE_CONFIG = "examples/SAGE_MPPI/clean_SAGE/config/ur7e_reacher_gazebo_tall_sage_clean.yml"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@dataclass
class RosMonitorResult:
    q_path: list[np.ndarray] = field(default_factory=list)
    ee_path: list[np.ndarray] = field(default_factory=list)
    step_rows: list[dict] = field(default_factory=list)
    success: bool = False
    timeout: bool = False
    collision: bool = False
    skipped_reason: str = ""
    steps_to_goal: int = 0


def _base_episode_row(method_name: str, episode_id: int, target_id: str) -> dict:
    row = {field: nan() for field in EPISODE_FIELDS}
    row.update(
        {
            "method_name": method_name,
            "episode_id": episode_id,
            "target_id": target_id,
            "scene": "tall",
            "success": False,
            "failure": True,
            "collision": False,
            "timeout": False,
            "rrtstar_available": nan(),
            "rrtstar_exact_solution": nan(),
            "rrtstar_approximate_solution": nan(),
            "skipped_reason": "",
        }
    )
    return row


def _terminate_process_group(proc: subprocess.Popen, grace_sec: float = 8.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    deadline = time.time() + grace_sec
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.1)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + 4.0
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_ros_monitor(
    method_name: str,
    episode_id: int,
    target: dict,
    checker: StaticTallCollisionChecker,
    success_threshold: float,
    max_steps: int,
    max_wall_time: float,
) -> RosMonitorResult:
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except Exception as exc:
        return RosMonitorResult(skipped_reason=f"ROS2 Python bindings unavailable to benchmark wrapper: {exc}")

    goal_ee = np.asarray(target["goal_ee_position"], dtype=float)
    goal_q = np.asarray(target["goal_joint_positions"], dtype=float)
    result = RosMonitorResult()

    class MonitorNode(Node):
        def __init__(self) -> None:
            super().__init__(f"static_compare_{method_name}_{episode_id}")
            self.target_pub = self.create_publisher(PoseStamped, "/target_pose", 10)
            self.create_subscription(JointState, "/joint_states", self.joint_cb, 10)
            self.create_subscription(PoseStamped, "/ee_pose", self.ee_cb, 10)
            self.timer = self.create_timer(0.25, self.publish_target)
            self.last_q = None
            self.last_ee = None
            self.step = 0
            self.sample_seq = 0

        def publish_target(self) -> None:
            msg = PoseStamped()
            msg.header.frame_id = "world"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x = float(goal_ee[0])
            msg.pose.position.y = float(goal_ee[1])
            msg.pose.position.z = float(goal_ee[2])
            msg.pose.orientation.w = 1.0
            self.target_pub.publish(msg)

        def joint_cb(self, msg: JointState) -> None:
            name_to_pos = {name: pos for name, pos in zip(msg.name, msg.position)}
            if not all(name in name_to_pos for name in JOINT_NAMES):
                return
            self.last_q = np.asarray([name_to_pos[name] for name in JOINT_NAMES], dtype=float)
            self.step += 1
            self.sample_seq += 1

        def ee_cb(self, msg: PoseStamped) -> None:
            self.last_ee = np.asarray(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                dtype=float,
            )

    started_here = False
    if not rclpy.ok():
        rclpy.init(args=None)
        started_here = True
    node = MonitorNode()
    start = time.time()
    last_sample_seq = -1
    try:
        while time.time() - start <= max_wall_time:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.last_q is None or node.last_ee is None:
                continue
            if node.sample_seq == last_sample_seq:
                continue
            last_sample_seq = node.sample_seq
            q = node.last_q.copy()
            ee = node.last_ee.copy()
            margin = checker.minimum_safety_margin(q)
            ee_error = float(np.linalg.norm(ee - goal_ee))
            joint_error = float(np.linalg.norm(q - goal_q))
            result.q_path.append(q)
            result.ee_path.append(ee)
            result.step_rows.append(
                {
                    "method_name": method_name,
                    "episode_id": episode_id,
                    "target_id": target["target_id"],
                    "scene": "tall",
                    "step": len(result.step_rows),
                    "q": q.tolist(),
                    "ee_position": ee.tolist(),
                    "ee_error": ee_error,
                    "joint_error": joint_error,
                    "safety_margin": margin,
                    "collision": bool(margin <= checker.collision_threshold),
                    "planning_time": nan(),
                    "wall_time": time.time() - start,
                    "skipped_reason": "",
                }
            )
            result.collision = result.collision or bool(margin <= checker.collision_threshold)
            if ee_error < success_threshold and not result.collision:
                result.success = True
                result.steps_to_goal = len(result.step_rows)
                break
            if len(result.step_rows) >= max_steps:
                result.timeout = True
                result.steps_to_goal = len(result.step_rows)
                break
        else:
            result.timeout = True
            result.steps_to_goal = len(result.step_rows)
        if not result.steps_to_goal:
            result.steps_to_goal = len(result.step_rows)
    finally:
        node.destroy_node()
        if started_here and rclpy.ok():
            rclpy.shutdown()
    if not result.q_path:
        result.timeout = True
        result.skipped_reason = "No controller /ee_pose samples observed before monitor timeout"
    return result


def _fill_trajectory_metrics(
    row: dict,
    checker: StaticTallCollisionChecker,
    q_path: list[np.ndarray],
    ee_path: list[np.ndarray],
    goal_q: np.ndarray,
    goal_ee: np.ndarray,
) -> None:
    if q_path:
        final_q = q_path[-1]
        final_ee = ee_path[-1] if ee_path else checker.ee_position(final_q)
        margins = [checker.minimum_safety_margin(q) for q in q_path]
        row["final_joint_error"] = float(np.linalg.norm(final_q - goal_q))
        row["final_ee_error"] = float(np.linalg.norm(final_ee - goal_ee))
        row["minimum_safety_margin"] = float(np.min(margins))
        row["trajectory_length_joint"] = trajectory_length_joint(q_path)
        row["trajectory_length_ee"] = trajectory_length_ee(ee_path if ee_path else [checker.ee_position(q) for q in q_path])
        row["smoothness_jerk"] = smoothness_jerk(q_path)


def _run_tuned_reference_episode(
    method_name: str,
    reference_script: str,
    entrypoint: str,
    config_path: str,
    target: dict,
    episode_id: int,
    output_root: Path,
    checker: StaticTallCollisionChecker,
    success_threshold: float,
    max_steps: int,
    max_wall_time: float,
) -> tuple[dict, list[dict]]:
    row = _base_episode_row(method_name, episode_id, target["target_id"])
    row.update(
        {
            "tuned_reference_script": reference_script,
            "controller_or_entrypoint": entrypoint,
            "config_path": config_path,
            "planning_time": nan(),
        }
    )
    if method_name == "storm_mppi_tuned":
        row.update(
            {
                "controller_class": "MPPI",
                "uses_clean_controller": False,
                "uses_native_margin": False,
                "deployment_refinement_enabled": False,
                "local_refinement_enabled": False,
                "margin_fallback": True,
            }
        )
    else:
        row.update(
            {
                "controller_class": "SAGE_MPPI",
                "uses_clean_controller": True,
                "uses_native_margin": True,
                "deployment_refinement_enabled": True,
                "local_refinement_enabled": True,
                "margin_fallback": True,
            }
        )

    log_dir = ensure_dir(output_root / "reference_logs")
    log_path = log_dir / f"{episode_id:03d}_{method_name}_{target['target_id']}.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    start = time.time()
    proc = None
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                ["bash", str(resolve_repo_path(reference_script))],
                cwd=str(resolve_repo_path(".")),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
            monitor = _run_ros_monitor(
                method_name=method_name,
                episode_id=episode_id,
                target=target,
                checker=checker,
                success_threshold=success_threshold,
                max_steps=max_steps,
                max_wall_time=max_wall_time,
            )
    except Exception as exc:
        row["skipped_reason"] = f"Exception while running tuned reference: {exc}"
        row["failure"] = True
        return row, []
    finally:
        if proc is not None:
            _terminate_process_group(proc)

    goal_q = np.asarray(target["goal_joint_positions"], dtype=float)
    goal_ee = np.asarray(target["goal_ee_position"], dtype=float)
    _fill_trajectory_metrics(row, checker, monitor.q_path, monitor.ee_path, goal_q, goal_ee)
    row.update(
        {
            "success": bool(monitor.success),
            "failure": not bool(monitor.success),
            "collision": bool(monitor.collision),
            "timeout": bool(monitor.timeout),
            "steps_to_goal": monitor.steps_to_goal,
            "wall_time": time.time() - start,
            "control_time_mean": nan(),
            "skipped_reason": monitor.skipped_reason,
        }
    )
    if monitor.skipped_reason:
        row["skipped_reason"] = f"{monitor.skipped_reason}; see {log_path}"
    return row, monitor.step_rows


def _run_rrtstar_episode(
    target: dict,
    episode_id: int,
    checker: StaticTallCollisionChecker,
    params: dict,
    success_threshold: float,
) -> tuple[dict, list[dict]]:
    row = _base_episode_row("rrtstar", episode_id, target["target_id"])
    row.update(
        {
            "controller_class": "OMPL_RRTstar",
            "controller_or_entrypoint": "examples/static_compare/rrtstar_ompl_adapter.py",
            "config_path": "examples/static_compare/config/static_tall_benchmark.yml",
            "tuned_reference_script": "",
            "uses_clean_controller": False,
            "uses_native_margin": False,
            "deployment_refinement_enabled": False,
            "local_refinement_enabled": False,
            "margin_fallback": False,
            "planning_time_limit": params["planning_time_limit"],
            "goal_bias": params["goal_bias"],
            "interpolation_resolution": params["interpolation_resolution"],
            "collision_check_resolution": params["collision_check_resolution"],
        }
    )
    start_q = np.asarray(target["initial_joint_positions"], dtype=float)
    goal_q = np.asarray(target["goal_joint_positions"], dtype=float)
    goal_ee = np.asarray(target["goal_ee_position"], dtype=float)
    wall_start = time.time()
    result = rrtstar_ompl_adapter.plan_joint_space_rrtstar(
        start_q,
        goal_q,
        checker,
        planning_time_limit=float(params["planning_time_limit"]),
        goal_bias=float(params["goal_bias"]),
        interpolation_resolution=float(params["interpolation_resolution"]),
        collision_check_resolution=float(params["collision_check_resolution"]),
    )
    q_path = [np.asarray(q, dtype=float) for q in result.get("path", [])] or [start_q]
    ee_path = [checker.ee_position(q) for q in q_path]
    path_valid = bool(result.get("rrtstar_path_valid", False))
    _fill_trajectory_metrics(row, checker, q_path, ee_path, goal_q, goal_ee)
    final_ee_error = float(row["final_ee_error"]) if not math.isnan(float(row["final_ee_error"])) else math.inf
    exact = bool(result.get("rrtstar_exact_solution", False))
    approx = bool(result.get("rrtstar_approximate_solution", False))
    success = bool(exact and path_valid and final_ee_error < success_threshold)
    row.update(
        {
            "success": success,
            "failure": not success,
            "collision": bool(result.get("path")) and not path_valid,
            "timeout": False,
            "steps_to_goal": max(0, len(q_path) - 1),
            "wall_time": time.time() - wall_start,
            "planning_time": result.get("planning_time", nan()),
            "control_time_mean": nan(),
            "rrtstar_available": result.get("rrtstar_available", False),
            "rrtstar_exact_solution": exact,
            "rrtstar_approximate_solution": approx,
            "minimum_safety_margin": result.get("minimum_safety_margin", row["minimum_safety_margin"]),
            "path_length_joint": result.get("path_length_joint", row["trajectory_length_joint"]),
            "number_of_validity_checks": result.get("number_of_validity_checks", 0),
            "number_of_invalid_states": result.get("number_of_invalid_states", 0),
            "skipped_reason": result.get("skipped_reason", ""),
        }
    )

    step_rows = []
    for step, (q, ee) in enumerate(zip(q_path, ee_path)):
        margin = checker.minimum_safety_margin(q)
        step_rows.append(
            {
                "method_name": "rrtstar",
                "episode_id": episode_id,
                "target_id": target["target_id"],
                "scene": "tall",
                "step": step,
                "q": q.tolist(),
                "ee_position": ee.tolist(),
                "ee_error": float(np.linalg.norm(ee - goal_ee)),
                "joint_error": float(np.linalg.norm(q - goal_q)),
                "safety_margin": margin,
                "collision": bool(margin <= checker.collision_threshold),
                "planning_time": result.get("planning_time", nan()) if step == 0 else 0.0,
                "wall_time": 0.0,
                "skipped_reason": result.get("skipped_reason", ""),
            }
        )
    return row, step_rows


def _normalize_methods(methods: list[str]) -> list[str]:
    if not methods or "all" in methods:
        return ["storm", "sage", "rrtstar"]
    return methods


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tuned-reference static tall reaching benchmark pilot")
    parser.add_argument("--methods", nargs="+", default=["all"], choices=["storm", "sage", "rrtstar", "all"])
    parser.add_argument("--targets-path", default="examples/static_compare/targets/static_tall_targets.json")
    parser.add_argument("--output-root", default="examples/static_compare/results")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-wall-time", type=float, default=90.0)
    parser.add_argument("--rrtstar-time-limit", type=float, default=2.0)
    args = parser.parse_args()

    output_root = ensure_dir(resolve_repo_path(args.output_root))
    targets_payload = load_json(args.targets_path)
    targets = targets_payload["targets"]
    methods = _normalize_methods(args.methods)
    checker = StaticTallCollisionChecker(include_ground=True)

    metadata = {
        "scene": "tall",
        "backend": "tuned_reference_subprocess_wrapper",
        "seed": args.seed,
        "success_threshold": args.success_threshold,
        "max_steps": args.max_steps,
        "max_wall_time": args.max_wall_time,
        "targets_path": str(resolve_repo_path(args.targets_path)),
        "methods_requested": methods,
        "reset_after_each_target": True,
        "target_injection": "/target_pose",
        "call_chains": {
            "storm_mppi_tuned": {
                "reference_script": STORM_REFERENCE_SCRIPT,
                "entrypoint": STORM_ENTRYPOINT,
                "config_path": STORM_CONFIG,
                "conda_env": "storm_py310",
            },
            "sage_mppi_tuned": {
                "reference_script": SAGE_REFERENCE_SCRIPT,
                "entrypoint": SAGE_ENTRYPOINT,
                "config_path": SAGE_CONFIG,
                "conda_env": "whole_control",
                "deployment_refinement_enabled_by_tuned_config": True,
                "local_refinement_enabled_by_tuned_config": True,
            },
        },
        "rrtstar": {
            "available": rrtstar_ompl_adapter.rrtstar_available,
            "skipped_reason": rrtstar_ompl_adapter.skipped_reason,
            "planning_time_limit": args.rrtstar_time_limit,
            "goal_bias": 0.05,
            "interpolation_resolution": 0.05,
            "collision_check_resolution": 0.05,
        },
        "limitations": [
            "The wrapper monitors tuned references through ROS topics and does not modify their controller code.",
            "STORM/SAGE per-step optimize time is not exposed by the reference scripts, so planning_time is NaN.",
            "Safety margin is recomputed by static_collision_checker from observed joint states for uniform logging.",
        ],
    }

    episode_rows: list[dict] = []
    step_rows: list[dict] = []
    episode_id = 0
    for method in methods:
        for target in targets:
            try:
                if method == "storm":
                    row, steps = _run_tuned_reference_episode(
                        "storm_mppi_tuned",
                        STORM_REFERENCE_SCRIPT,
                        STORM_ENTRYPOINT,
                        STORM_CONFIG,
                        target,
                        episode_id,
                        output_root,
                        checker,
                        args.success_threshold,
                        args.max_steps,
                        args.max_wall_time,
                    )
                elif method == "sage":
                    row, steps = _run_tuned_reference_episode(
                        "sage_mppi_tuned",
                        SAGE_REFERENCE_SCRIPT,
                        SAGE_ENTRYPOINT,
                        SAGE_CONFIG,
                        target,
                        episode_id,
                        output_root,
                        checker,
                        args.success_threshold,
                        args.max_steps,
                        args.max_wall_time,
                    )
                elif method == "rrtstar":
                    row, steps = _run_rrtstar_episode(target, episode_id, checker, metadata["rrtstar"], args.success_threshold)
                else:
                    raise ValueError(f"Unknown method: {method}")
            except Exception as exc:
                method_name = {"storm": "storm_mppi_tuned", "sage": "sage_mppi_tuned"}.get(method, method)
                row = _base_episode_row(method_name, episode_id, target.get("target_id", f"episode_{episode_id}"))
                row["skipped_reason"] = f"Exception: {exc}"
                row["failure"] = True
                steps = []
                metadata.setdefault("exceptions", []).append(
                    {"method": method, "target_id": target.get("target_id"), "traceback": traceback.format_exc()}
                )
            episode_rows.append(row)
            step_rows.extend(steps)
            episode_id += 1

    write_csv(output_root / "static_tall_episode_log.csv", episode_rows, EPISODE_FIELDS)
    write_csv(output_root / "static_tall_step_log.csv", step_rows, STEP_FIELDS)
    metadata["num_episode_rows"] = len(episode_rows)
    metadata["num_step_rows"] = len(step_rows)
    write_json(output_root / "metadata.json", metadata)
    print(f"wrote {len(episode_rows)} episode rows and {len(step_rows)} step rows to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
