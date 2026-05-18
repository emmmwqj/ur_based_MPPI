#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

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
from storm_kit.mpc.control import MPPI
from storm_kit.mpc.control.sage_mppi import SAGE_MPPI
from storm_kit.mpc.rollout.arm_reacher import ArmReacher
from storm_kit.mpc.rollout.sage_arm_reacher import SageArmReacher
from storm_kit.mpc.utils.state_filter import JointStateFilter


STORM_TASK_FILE = "examples/sim_gazebo/config/ur7e_reacher_gazebo_tall.yml"
SAGE_TASK_FILE = "examples/SAGE_MPPI/clean_SAGE/config/ur7e_reacher_gazebo_tall_sage_clean.yml"
ROBOT_FILE = "examples/sim_gazebo/config/ur7e_robot_gazebo.yml"
WORLD_FILE = "examples/sim_gazebo/config/collision_world_gazebo_tall.yml"


@dataclass
class ControllerAdapter:
    method_name: str
    controller: object
    rollout_fn: object
    exp_params: dict
    tensor_args: dict
    controller_class: str
    uses_clean_controller: bool
    uses_native_margin: bool
    deployment_refinement_enabled: bool
    local_refinement_enabled: bool

    def reset_episode(self, q0: Iterable[float]) -> None:
        q0 = np.asarray(q0, dtype=float)
        zero = np.zeros_like(q0)
        self.state_filter = JointStateFilter(
            raw_joint_state={"position": q0.copy(), "velocity": zero.copy(), "acceleration": zero.copy()},
            filter_coeff=self.exp_params["state_filter_coeff"],
            dt=float(self.exp_params["control_dt"]),
        )
        if hasattr(self.controller, "reset"):
            self.controller.reset()

    @property
    def control_dt(self) -> float:
        return float(self.exp_params["control_dt"])

    @property
    def n_dofs(self) -> int:
        return int(self.rollout_fn.dynamics_model.n_dofs)

    def update_goal(self, goal_q: Iterable[float], goal_ee_position: Iterable[float]) -> None:
        goal_q = np.asarray(goal_q, dtype=float)
        zero = np.zeros_like(goal_q)
        goal_state = np.concatenate([goal_q, zero])
        self.rollout_fn.update_params(goal_state=goal_state)
        goal_quat = getattr(self.rollout_fn, "goal_ee_quat", None)
        if goal_quat is not None:
            goal_quat_np = np.ravel(goal_quat.detach().cpu().numpy())
            self.rollout_fn.update_params(goal_ee_pos=np.asarray(goal_ee_position, dtype=float), goal_ee_quat=goal_quat_np)
        else:
            self.rollout_fn.update_params(goal_ee_pos=np.asarray(goal_ee_position, dtype=float))

    def optimize_once(self, state: dict, t_step: float, shift_steps: int) -> tuple[np.ndarray, dict, float]:
        filtered = self.state_filter.filter_joint_state(copy.deepcopy(state))
        state_vec = np.concatenate([filtered["position"], filtered["velocity"], filtered["acceleration"], [t_step]])
        state_tensor = torch.as_tensor(state_vec, **self.tensor_args).unsqueeze(0)
        start = time.time()
        action_seq, _value, info = self.controller.optimize(state_tensor, shift_steps=shift_steps)
        opt_time = time.time() - start
        if isinstance(action_seq, torch.Tensor):
            action_seq_np = action_seq.detach().cpu().numpy()
        else:
            action_seq_np = np.asarray(action_seq)

        action = action_seq_np[0]
        if self.method_name == "storm_mppi":
            mode = str(self.exp_params.get("mppi", {}).get("execution_mode", "best_sample")).lower()
            best_traj = getattr(self.controller, "best_traj", None)
            if mode == "best_sample" and best_traj is not None:
                action = np.asarray(best_traj.detach().cpu().numpy())[0]

        qdd = np.asarray(
            self.rollout_fn.dynamics_model.integrate_action_step(
                torch.as_tensor(action, **self.tensor_args),
                self.control_dt,
            ).detach().cpu().numpy(),
            dtype=float,
        )
        cmd = self.state_filter.integrate_acc(qdd, dt=self.control_dt)
        return np.asarray(cmd["position"], dtype=float).copy(), dict(info or {}), float(opt_time)


def _load_yaml(path: str) -> dict:
    with open(resolve_repo_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_tensor_args(no_cuda: bool) -> dict:
    use_cuda = (not no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")
    return {"device": device, "dtype": torch.float32}


def _set_position_only_goal_mode(rollout_fn) -> None:
    goal_cost = getattr(rollout_fn, "goal_cost", None)
    if goal_cost is None:
        return
    weight = getattr(goal_cost, "weight", None)
    try:
        if isinstance(weight, torch.Tensor):
            weight[0] = 0.0
        elif isinstance(weight, (list, tuple)):
            goal_cost.weight = [0.0, float(weight[1])]
    except Exception:
        pass


def _build_controller(
    method_name: str,
    task_file: str,
    seed: int,
    tensor_args: dict,
) -> ControllerAdapter:
    exp_params = _load_yaml(task_file)
    world_params = _load_yaml(WORLD_FILE)
    exp_params["robot_params"] = exp_params["model"]
    exp_params["mppi"]["sample_params"]["seed"] = int(seed)

    if method_name == "storm_mppi":
        rollout_fn = ArmReacher(exp_params=exp_params, tensor_args=tensor_args, world_params=world_params)
        _set_position_only_goal_mode(rollout_fn)
        controller_cls = MPPI
        mppi_params = dict(exp_params["mppi"])
        mppi_params.pop("execution_mode", None)
        uses_clean_controller = False
        uses_native_margin = False
        deployment_refinement_enabled = False
        local_refinement_enabled = False
    elif method_name == "sage_mppi_core":
        rollout_fn = SageArmReacher(exp_params=exp_params, tensor_args=tensor_args, world_params=world_params)
        controller_cls = SAGE_MPPI
        mppi_params = dict(exp_params["mppi"])
        execution_mode = str(mppi_params.get("execution_mode", "mean")).lower()
        mppi_params.pop("execution_mode", None)
        mppi_params.update(dict(exp_params.get("sage_controller_core", {})))
        mppi_params["execute_best"] = execution_mode == "best_sample"
        uses_clean_controller = True
        uses_native_margin = True
        deployment_refinement_enabled = False
        local_refinement_enabled = False
    else:
        raise ValueError(f"Unsupported controller method: {method_name}")

    dynamics_model = rollout_fn.dynamics_model
    mppi_params["d_action"] = dynamics_model.d_action
    mppi_params["action_lows"] = -exp_params["model"]["max_acc"] * torch.ones(dynamics_model.d_action, **tensor_args)
    mppi_params["action_highs"] = exp_params["model"]["max_acc"] * torch.ones(dynamics_model.d_action, **tensor_args)
    init_q = torch.tensor(exp_params["model"]["init_state"], **tensor_args)
    init_action = torch.zeros((mppi_params["horizon"], dynamics_model.d_action), **tensor_args)
    init_action[:, :] += init_q
    if exp_params["control_space"] == "acc":
        mppi_params["init_mean"] = init_action * 0.0
    elif exp_params["control_space"] == "pos":
        mppi_params["init_mean"] = init_action
    else:
        raise ValueError(f"Unsupported control_space={exp_params['control_space']}")
    mppi_params["rollout_fn"] = rollout_fn
    mppi_params["tensor_args"] = tensor_args

    controller = controller_cls(**mppi_params)
    return ControllerAdapter(
        method_name=method_name,
        controller=controller,
        rollout_fn=rollout_fn,
        exp_params=exp_params,
        tensor_args=tensor_args,
        controller_class=type(controller).__name__,
        uses_clean_controller=uses_clean_controller,
        uses_native_margin=uses_native_margin,
        deployment_refinement_enabled=deployment_refinement_enabled,
        local_refinement_enabled=local_refinement_enabled,
    )


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
            "rrtstar_path_found": nan(),
            "skipped_reason": "",
        }
    )
    return row


def _episode_metrics(
    checker: StaticTallCollisionChecker,
    q_path: list[np.ndarray],
    goal_q: np.ndarray,
    goal_ee: np.ndarray,
    success_threshold: float,
    collision: bool,
    timeout: bool,
) -> dict:
    final_q = np.asarray(q_path[-1], dtype=float)
    ee_path = [checker.ee_position(q) for q in q_path]
    final_ee_error = float(np.linalg.norm(ee_path[-1] - goal_ee))
    final_joint_error = float(np.linalg.norm(final_q - goal_q))
    margins = [checker.minimum_safety_margin(q) for q in q_path]
    success = bool(final_ee_error < success_threshold and not collision and not timeout)
    return {
        "success": success,
        "failure": not success,
        "final_ee_error": final_ee_error,
        "final_joint_error": final_joint_error,
        "minimum_safety_margin": float(np.min(margins)),
        "trajectory_length_joint": trajectory_length_joint(q_path),
        "trajectory_length_ee": trajectory_length_ee(ee_path),
        "smoothness_jerk": smoothness_jerk(q_path),
    }


def _run_controller_episode(
    adapter: ControllerAdapter,
    target: dict,
    episode_id: int,
    checker: StaticTallCollisionChecker,
    max_steps: int,
    success_threshold: float,
) -> tuple[dict, list[dict]]:
    target_id = str(target["target_id"])
    row = _base_episode_row(adapter.method_name, episode_id, target_id)
    row.update(
        {
            "controller_class": adapter.controller_class,
            "uses_clean_controller": adapter.uses_clean_controller,
            "uses_native_margin": adapter.uses_native_margin,
            "deployment_refinement_enabled": adapter.deployment_refinement_enabled,
            "local_refinement_enabled": adapter.local_refinement_enabled,
            "margin_fallback": False,
            "rrtstar_available": nan(),
            "rrtstar_path_found": nan(),
        }
    )
    step_rows: list[dict] = []
    q = np.asarray(target["initial_joint_positions"], dtype=float)
    goal_q = np.asarray(target["goal_joint_positions"], dtype=float)
    goal_ee = np.asarray(target["goal_ee_position"], dtype=float)
    adapter.reset_episode(q)
    adapter.update_goal(goal_q, goal_ee)

    dq = np.zeros_like(q)
    ddq = np.zeros_like(q)
    q_path = [q.copy()]
    planning_times: list[float] = []
    collision = False
    timeout = False
    steps_to_goal = max_steps
    wall_start = time.time()

    for step in range(max_steps):
        t_step = step * adapter.control_dt
        iter_start = time.time()
        state = {"position": q.copy(), "velocity": dq.copy(), "acceleration": ddq.copy()}
        q_cmd, _info, opt_time = adapter.optimize_once(state, t_step=t_step, shift_steps=0 if step == 0 else 1)
        q_next = np.clip(q_cmd, checker.joint_lower, checker.joint_upper)
        dq_next = (q_next - q) / max(adapter.control_dt, 1.0e-6)
        ddq_next = (dq_next - dq) / max(adapter.control_dt, 1.0e-6)
        q, dq, ddq = q_next, dq_next, ddq_next
        q_path.append(q.copy())
        planning_times.append(opt_time)

        ee = checker.ee_position(q)
        ee_error = float(np.linalg.norm(ee - goal_ee))
        joint_error = float(np.linalg.norm(q - goal_q))
        margin = checker.minimum_safety_margin(q)
        collision = bool(margin <= checker.collision_threshold)
        step_rows.append(
            {
                "method_name": adapter.method_name,
                "episode_id": episode_id,
                "target_id": target_id,
                "scene": "tall",
                "step": step,
                "q": q.tolist(),
                "ee_position": ee.tolist(),
                "ee_error": ee_error,
                "joint_error": joint_error,
                "safety_margin": margin,
                "collision": collision,
                "planning_time": opt_time,
                "wall_time": time.time() - iter_start,
                "skipped_reason": "",
            }
        )
        if collision:
            steps_to_goal = step + 1
            break
        if ee_error < success_threshold:
            steps_to_goal = step + 1
            break
    else:
        timeout = True

    metric_row = _episode_metrics(checker, q_path, goal_q, goal_ee, success_threshold, collision, timeout)
    row.update(metric_row)
    row.update(
        {
            "collision": collision,
            "timeout": timeout,
            "steps_to_goal": steps_to_goal,
            "wall_time": time.time() - wall_start,
            "planning_time": float(np.sum(planning_times)) if planning_times else 0.0,
            "control_time_mean": float(np.mean(planning_times)) if planning_times else nan(),
        }
    )
    return row, step_rows


def _run_rrtstar_episode(
    target: dict,
    episode_id: int,
    checker: StaticTallCollisionChecker,
    params: dict,
    success_threshold: float,
) -> tuple[dict, list[dict]]:
    target_id = str(target["target_id"])
    row = _base_episode_row("rrtstar", episode_id, target_id)
    row.update(
        {
            "controller_class": "OMPL_RRTstar",
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
    goal_q = np.asarray(target["goal_joint_positions"], dtype=float)
    goal_ee = np.asarray(target["goal_ee_position"], dtype=float)
    start_q = np.asarray(target["initial_joint_positions"], dtype=float)
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
    q_path = [np.asarray(q, dtype=float) for q in result.get("path", [])]
    if not q_path:
        q_path = [start_q]

    path_valid = bool(result.get("rrtstar_path_valid", False))
    final_metrics = _episode_metrics(
        checker,
        q_path,
        goal_q,
        goal_ee,
        success_threshold,
        collision=bool(result.get("path")) and not path_valid,
        timeout=False,
    )
    path_found = bool(result.get("rrtstar_path_found", False))
    row.update(final_metrics)
    row.update(
        {
            "success": bool(path_found and final_metrics["success"]),
            "failure": not bool(path_found and final_metrics["success"]),
            "collision": bool(result.get("path")) and not path_valid,
            "timeout": False,
            "steps_to_goal": max(0, len(q_path) - 1),
            "wall_time": time.time() - wall_start,
            "planning_time": result.get("planning_time", nan()),
            "control_time_mean": nan(),
            "rrtstar_available": result.get("rrtstar_available", False),
            "rrtstar_path_found": path_found,
            "minimum_safety_margin": result.get("minimum_safety_margin", final_metrics["minimum_safety_margin"]),
            "path_length_joint": result.get("path_length_joint", final_metrics["trajectory_length_joint"]),
            "number_of_validity_checks": result.get("number_of_validity_checks", 0),
            "number_of_invalid_states": result.get("number_of_invalid_states", 0),
            "skipped_reason": result.get("skipped_reason", ""),
        }
    )
    if not result.get("rrtstar_available", False):
        row["success"] = False
        row["failure"] = True

    step_rows = []
    for step, q in enumerate(q_path):
        ee = checker.ee_position(q)
        margin = checker.minimum_safety_margin(q)
        step_rows.append(
            {
                "method_name": "rrtstar",
                "episode_id": episode_id,
                "target_id": target_id,
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
    parser = argparse.ArgumentParser(description="Run static tall reaching benchmark pilot")
    parser.add_argument("--methods", nargs="+", default=["all"], choices=["storm", "sage", "rrtstar", "all"])
    parser.add_argument("--targets-path", default="examples/static_compare/targets/static_tall_targets.json")
    parser.add_argument("--output-root", default="examples/static_compare/results")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--success-threshold", type=float, default=0.02)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rrtstar-time-limit", type=float, default=2.0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_root = ensure_dir(resolve_repo_path(args.output_root))
    targets_payload = load_json(args.targets_path)
    targets = targets_payload["targets"]
    methods = _normalize_methods(args.methods)
    tensor_args = _build_tensor_args(args.no_cuda)
    checker = StaticTallCollisionChecker(include_ground=True, tensor_args={"device": torch.device("cpu"), "dtype": torch.float32})

    metadata = {
        "scene": "tall",
        "backend": "offline_controller_pilot",
        "headless": bool(args.headless),
        "seed": args.seed,
        "targets_path": str(resolve_repo_path(args.targets_path)),
        "output_root": str(output_root),
        "success_threshold": args.success_threshold,
        "max_steps": args.max_steps,
        "methods_requested": methods,
        "device": str(tensor_args["device"]),
        "metric_definitions": {
            "success": "final_ee_error < success_threshold and collision=False and timeout=False",
            "final_ee_error": "||p_ee(q_final) - p_goal||_2",
            "final_joint_error": "||q_final - q_goal||_2",
            "minimum_safety_margin": "minimum static_collision_checker geometric margin over discrete trajectory/path states",
            "trajectory_length_joint": "sum ||q[t+1] - q[t]||_2",
            "trajectory_length_ee": "sum ||p_ee(q[t+1]) - p_ee(q[t])||_2",
            "smoothness_jerk": "mean norm of third finite difference in joint positions; NaN if fewer than 4 states",
            "planning_time": "RRT*: OMPL solve wall time; STORM/SAGE: cumulative direct optimize wall time in offline pilot",
        },
        "paths": {
            "storm_task_file": str(resolve_repo_path(STORM_TASK_FILE)),
            "sage_task_file": str(resolve_repo_path(SAGE_TASK_FILE)),
            "robot_file": str(resolve_repo_path(ROBOT_FILE)),
            "world_file": str(resolve_repo_path(WORLD_FILE)),
        },
        "sage_overrides": {
            "uses_clean_controller": True,
            "deployment_refinement_enabled": False,
            "local_refinement_enabled": False,
        },
        "collision_checker": {
            "implementation": "examples/static_compare/utils/static_collision_checker.py",
            "uses_gazebo": False,
            "include_ground": True,
            "warnings": checker.warnings,
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
            "Pilot uses an offline controller loop to validate infrastructure; it is not a final Gazebo benchmark result.",
            "STORM/SAGE logged safety margins are computed by static_collision_checker for schema consistency.",
            "SAGE deployment and local refinement stacks are not instantiated.",
        ],
    }

    adapters: dict[str, ControllerAdapter] = {}
    if "storm" in methods:
        adapters["storm"] = _build_controller("storm_mppi", STORM_TASK_FILE, args.seed, tensor_args)
    if "sage" in methods:
        adapters["sage"] = _build_controller("sage_mppi_core", SAGE_TASK_FILE, args.seed, tensor_args)
        if adapters["sage"].controller_class != "SAGE_MPPI":
            raise RuntimeError(f"SAGE clean controller mismatch: {adapters['sage'].controller_class}")

    episode_rows: list[dict] = []
    step_rows: list[dict] = []
    episode_id = 0
    for method in methods:
        for target in targets:
            try:
                if method == "storm":
                    row, steps = _run_controller_episode(adapters["storm"], target, episode_id, checker, args.max_steps, args.success_threshold)
                elif method == "sage":
                    row, steps = _run_controller_episode(adapters["sage"], target, episode_id, checker, args.max_steps, args.success_threshold)
                elif method == "rrtstar":
                    row, steps = _run_rrtstar_episode(target, episode_id, checker, metadata["rrtstar"], args.success_threshold)
                else:
                    raise ValueError(f"Unknown method {method}")
            except Exception as exc:
                target_id = str(target.get("target_id", f"episode_{episode_id}"))
                method_name = {"storm": "storm_mppi", "sage": "sage_mppi_core"}.get(method, method)
                row = _base_episode_row(method_name, episode_id, target_id)
                row["skipped_reason"] = f"Exception: {exc}"
                row["wall_time"] = 0.0
                row["planning_time"] = 0.0
                row["control_time_mean"] = nan()
                steps = []
                metadata.setdefault("exceptions", []).append(
                    {"method": method, "target_id": target_id, "traceback": traceback.format_exc()}
                )
            episode_rows.append(row)
            step_rows.extend(steps)
            episode_id += 1

    episode_csv = output_root / "static_tall_episode_log.csv"
    step_csv = output_root / "static_tall_step_log.csv"
    metadata_path = output_root / "metadata.json"
    write_csv(episode_csv, episode_rows, EPISODE_FIELDS)
    write_csv(step_csv, step_rows, STEP_FIELDS)
    metadata["num_episode_rows"] = len(episode_rows)
    metadata["num_step_rows"] = len(step_rows)
    metadata["episode_log"] = str(episode_csv)
    metadata["step_log"] = str(step_csv)
    write_json(metadata_path, metadata)
    print(f"wrote episode log to {episode_csv}")
    print(f"wrote step log to {step_csv}")
    print(f"wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
