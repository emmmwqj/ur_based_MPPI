#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.static_compare import rrtstar_ompl_adapter
from examples.static_compare.sage_tall_runner import SAGE_CONFIG, SAGE_ENTRYPOINT, SAGE_REFERENCE_SCRIPT, SageTallRunner
from examples.static_compare.storm_tall_runner import STORM_CONFIG, STORM_ENTRYPOINT, STORM_REFERENCE_SCRIPT, StormTallRunner
from examples.static_compare.utils.gazebo_utils import cleanup_existing_ur_gazebo, start_gazebo, stop_gazebo
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


def _base_rrtstar_row(episode_id: int, target_id: str) -> dict:
    row = {field: nan() for field in EPISODE_FIELDS}
    row.update(
        {
            "method_name": "rrtstar_ompl",
            "episode_id": episode_id,
            "target_id": target_id,
            "difficulty_tag": "",
            "scene": "tall",
            "success": False,
            "failure": True,
            "collision": False,
            "timeout": False,
            "rrtstar_available": rrtstar_ompl_adapter.rrtstar_available,
            "rrtstar_exact_solution": False,
            "rrtstar_approximate_solution": False,
            "controller_class": "OMPL_RRTstar",
            "controller_entrypoint": "examples/static_compare/rrtstar_ompl_adapter.py",
            "config_path": "examples/static_compare/config/static_tall_benchmark.yml",
            "tuned_reference_script": "",
            "target_publish_count": 0,
            "target_publish_duration": 0.0,
            "uses_clean_controller": False,
            "uses_native_margin": False,
            "deployment_refinement_enabled": False,
            "local_refinement_enabled": False,
            "margin_fallback": False,
            "target_not_available": False,
            "backend": "ompl_joint_space_static_checker",
            "skipped_reason": "",
        }
    )
    return row


def _run_rrtstar_episode(
    target: dict,
    episode_id: int,
    checker: StaticTallCollisionChecker,
    params: dict,
    success_threshold: float,
) -> tuple[dict, list[dict]]:
    row = _base_rrtstar_row(episode_id, target["target_id"])
    row.update(
        {
            "planning_time_limit": params["planning_time_limit"],
            "goal_bias": params["goal_bias"],
            "interpolation_resolution": params["interpolation_resolution"],
            "collision_check_resolution": params["collision_check_resolution"],
        }
    )
    if "goal_joint_positions" not in target:
        row.update(
            {
                "target_not_available": True,
                "skipped_reason": "RRT* requires goal_joint_positions for a fair joint-space baseline.",
                "rrtstar_available": rrtstar_ompl_adapter.rrtstar_available,
            }
        )
        return row, []

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
    exact = bool(result.get("rrtstar_exact_solution", False))
    approx = bool(result.get("rrtstar_approximate_solution", False))
    final_ee_error = float(np.linalg.norm(ee_path[-1] - goal_ee)) if ee_path else math.inf
    minimum_margin = (
        float(result.get("minimum_safety_margin"))
        if result.get("minimum_safety_margin") is not None and not math.isnan(float(result.get("minimum_safety_margin")))
        else min(checker.minimum_safety_margin(q) for q in q_path)
    )
    success = bool(exact and path_valid and final_ee_error < success_threshold and minimum_margin > checker.collision_threshold)
    row.update(
        {
            "success": success,
            "failure": not success,
            "collision": bool(result.get("path")) and not path_valid,
            "timeout": False,
            "final_ee_error": final_ee_error,
            "final_joint_error": float(np.linalg.norm(q_path[-1] - goal_q)),
            "minimum_safety_margin": minimum_margin,
            "steps_to_goal": max(0, len(q_path) - 1),
            "wall_time": time.time() - wall_start,
            "planning_time": result.get("planning_time", nan()),
            "control_time_mean": nan(),
            "trajectory_length_joint": trajectory_length_joint(q_path),
            "trajectory_length_ee": trajectory_length_ee(ee_path),
            "smoothness_jerk": smoothness_jerk(q_path),
            "rrtstar_available": result.get("rrtstar_available", False),
            "rrtstar_exact_solution": exact,
            "rrtstar_approximate_solution": approx,
            "path_length_joint": result.get("path_length_joint", trajectory_length_joint(q_path)),
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
                "method_name": "rrtstar_ompl",
                "episode_id": episode_id,
                "target_id": target["target_id"],
                "difficulty_tag": target.get("difficulty_tag", ""),
                "scene": "tall",
                "step": step,
                "q": q.tolist(),
                "ee_position": ee.tolist(),
                "ee_error": float(np.linalg.norm(ee - goal_ee)),
                "joint_error": float(np.linalg.norm(q - goal_q)),
                "safety_margin": margin,
                "collision": bool(margin <= checker.collision_threshold),
                "planning_time": result.get("planning_time", nan()) if step == 0 else 0.0,
                "control_time": nan(),
                "wall_time": 0.0,
                "skipped_reason": result.get("skipped_reason", ""),
            }
        )
    return row, step_rows


def _normalize_methods(methods: list[str]) -> list[str]:
    if not methods or "all" in methods:
        return ["storm_mppi_tuned", "sage_mppi_tuned", "rrtstar_ompl"]
    mapping = {
        "storm": "storm_mppi_tuned",
        "sage": "sage_mppi_tuned",
        "rrtstar": "rrtstar_ompl",
        "storm_mppi_tuned": "storm_mppi_tuned",
        "sage_mppi_tuned": "sage_mppi_tuned",
        "rrtstar_ompl": "rrtstar_ompl",
    }
    return [mapping[m] for m in methods]


def _exception_row(method_name: str, episode_id: int, target_id: str, exc: Exception) -> dict:
    row = {field: nan() for field in EPISODE_FIELDS}
    row.update(
        {
            "method_name": method_name,
            "episode_id": episode_id,
            "target_id": target_id,
            "difficulty_tag": "",
            "scene": "tall",
            "success": False,
            "failure": True,
            "collision": False,
            "timeout": False,
            "skipped_reason": f"{type(exc).__name__}: {exc}",
            "backend": "python_internal_tuned_logic" if method_name != "rrtstar_ompl" else "ompl_joint_space_static_checker",
        }
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run static tall benchmark using internal tuned STORM/SAGE Python logic")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=["storm", "sage", "rrtstar", "storm_mppi_tuned", "sage_mppi_tuned", "rrtstar_ompl", "all"],
    )
    parser.add_argument("--targets-path", default="examples/static_compare/targets/static_tall_targets.json")
    parser.add_argument("--output-root", default="examples/static_compare/results")
    parser.add_argument("--limit-targets", type=int, default=0, help="Debug/pilot helper: use first N targets when >0")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--max-runtime", type=float, default=60.0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--viz-update-every", type=int, default=1)
    parser.add_argument("--target-publish-duration", type=float, default=2.0)
    parser.add_argument("--rrtstar-time-limit", type=float, default=2.0)
    args = parser.parse_args()

    output_root = ensure_dir(resolve_repo_path(args.output_root))
    log_root = ensure_dir(output_root / "runtime_logs")
    targets_payload = load_json(args.targets_path)
    targets = targets_payload["targets"]
    if args.limit_targets > 0:
        targets = targets[: args.limit_targets]
    methods = _normalize_methods(args.methods)
    checker = StaticTallCollisionChecker(include_ground=True)
    use_cuda = not args.no_cuda

    metadata = {
        "scene": "tall",
        "backend": "python_internal_tuned_logic",
        "seed": args.seed,
        "success_threshold": args.success_threshold,
        "max_steps": args.max_steps,
        "max_runtime": args.max_runtime,
        "targets_path": str(resolve_repo_path(args.targets_path)),
        "limit_targets": args.limit_targets,
        "methods_requested": methods,
        "reset_after_each_target": True,
        "target_injection": "/target_pose",
        "target_publish_duration": args.target_publish_duration,
        "gazebo_launch": {
            "entrypoint": "ros2 launch ur_simulation_gazebo ur_sim_control.launch.py",
            "initial_positions_file": "examples/sim_gazebo/config/initial_positions.yaml",
            "launch_rviz": False,
            "gazebo_gui": False,
        },
        "call_chains": {
            "storm_mppi_tuned": {
                "reference_script_read_only": STORM_REFERENCE_SCRIPT,
                "controller_entrypoint_logic_reproduced_from": STORM_ENTRYPOINT,
                "config_path": STORM_CONFIG,
                "world_path": "examples/sim_gazebo/config/collision_world_gazebo_tall.yml",
                "controller_initialized_in_static_compare": True,
            },
            "sage_mppi_tuned": {
                "reference_script_read_only": SAGE_REFERENCE_SCRIPT,
                "controller_entrypoint_logic_reproduced_from": SAGE_ENTRYPOINT,
                "config_path": SAGE_CONFIG,
                "world_path": "examples/sim_gazebo/config/collision_world_gazebo_tall.yml",
                "controller_initialized_in_static_compare": True,
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
            "validity_checker": "examples/static_compare/utils/static_collision_checker.py",
        },
        "metric_definitions": {
            "success": "final_ee_error < success_threshold and collision=False and timeout=False",
            "planning_time": "STORM/SAGE cumulative synchronous command solve wall time; RRT* OMPL solve wall time.",
            "minimum_safety_margin": "StaticTallCollisionChecker geometric minimum margin over executed or planned states.",
        },
        "limitations": [
            "STORM/SAGE safety margin is recomputed from observed joint states by the shared static_collision_checker for unified logging.",
            "RRT* uses paired joint-space goals generated for the target set; STORM/SAGE receive the shared Cartesian goal through /target_pose.",
            "Gazebo is restarted for each STORM/SAGE target to reset to the tuned initial_positions.yaml state.",
        ],
    }

    episode_rows: list[dict] = []
    step_rows: list[dict] = []
    episode_id = 0
    for method in methods:
        for target in targets:
            try:
                if method == "storm_mppi_tuned":
                    cleanup_existing_ur_gazebo()
                    gazebo = None
                    try:
                        gazebo = start_gazebo(log_root / f"{episode_id:03d}_{method}_{target['target_id']}_gazebo.log")
                        runner = StormTallRunner(
                            checker=checker,
                            use_cuda=use_cuda,
                            rate=args.rate,
                            viz_update_every=args.viz_update_every,
                            target_publish_duration=args.target_publish_duration,
                        )
                        row, steps = runner.run_episode(target, episode_id, args.max_steps, args.success_threshold, args.max_runtime)
                    finally:
                        stop_gazebo(gazebo)
                        cleanup_existing_ur_gazebo()
                elif method == "sage_mppi_tuned":
                    cleanup_existing_ur_gazebo()
                    gazebo = None
                    try:
                        gazebo = start_gazebo(log_root / f"{episode_id:03d}_{method}_{target['target_id']}_gazebo.log")
                        runner = SageTallRunner(
                            checker=checker,
                            use_cuda=use_cuda,
                            rate=args.rate,
                            viz_update_every=args.viz_update_every,
                            target_publish_duration=args.target_publish_duration,
                        )
                        row, steps = runner.run_episode(target, episode_id, args.max_steps, args.success_threshold, args.max_runtime)
                    finally:
                        stop_gazebo(gazebo)
                        cleanup_existing_ur_gazebo()
                elif method == "rrtstar_ompl":
                    row, steps = _run_rrtstar_episode(target, episode_id, checker, metadata["rrtstar"], args.success_threshold)
                else:
                    raise ValueError(f"Unknown method: {method}")
            except Exception as exc:
                row = _exception_row(method, episode_id, target.get("target_id", f"episode_{episode_id}"), exc)
                steps = []
                metadata.setdefault("exceptions", []).append(
                    {"method": method, "target_id": target.get("target_id"), "traceback": traceback.format_exc()}
                )
            row["difficulty_tag"] = target.get("difficulty_tag", "")
            for step in steps:
                step["difficulty_tag"] = target.get("difficulty_tag", "")
            episode_rows.append(row)
            step_rows.extend(steps)
            write_csv(output_root / "static_tall_episode_log.csv", episode_rows, EPISODE_FIELDS)
            write_csv(output_root / "static_tall_step_log.csv", step_rows, STEP_FIELDS)
            episode_id += 1

    metadata["num_episode_rows"] = len(episode_rows)
    metadata["num_step_rows"] = len(step_rows)
    write_json(output_root / "metadata.json", metadata)
    write_csv(output_root / "static_tall_episode_log.csv", episode_rows, EPISODE_FIELDS)
    write_csv(output_root / "static_tall_step_log.csv", step_rows, STEP_FIELDS)
    print(f"wrote {len(episode_rows)} episode rows and {len(step_rows)} step rows to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
