#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
from typing import Dict, Iterable, Optional

import numpy as np
import torch


STEP_FIELDS = [
    "controller_name",
    "episode_id",
    "step_id",
    "seed",
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
    "controller_name",
    "episode_id",
    "seed",
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


def _reduce_scalar(value):
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _reduce_scalar(value[-1])
    return value


def to_float_or_nan(value):
    value = _reduce_scalar(value)
    if value is None:
        return math.nan
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return math.nan
        value = value.detach().reshape(-1)[0].item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def to_bool_or_none(value):
    value = _reduce_scalar(value)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().reshape(-1)[0].item()
    return bool(value)


def _task_state_to_tensor(task, current_state):
    state_tensor = task._state_to_tensor(current_state)
    dynamics_model = task.controller.rollout_fn.dynamics_model
    if state_tensor.shape[0] < dynamics_model.d_state:
        padded_state = torch.zeros(dynamics_model.d_state, **task.controller.tensor_args)
        padded_state[: state_tensor.shape[0]] = state_tensor.to(**task.controller.tensor_args)
        state_tensor = padded_state
    else:
        state_tensor = state_tensor.to(**task.controller.tensor_args)
    return state_tensor.unsqueeze(0)


def compute_goal_distance(task, current_state):
    rollout_fn = task.controller.rollout_fn
    state_tensor = _task_state_to_tensor(task, current_state)

    if (
        hasattr(rollout_fn, "get_ee_pose")
        and getattr(rollout_fn, "goal_ee_pos", None) is not None
    ):
        ee_state = rollout_fn.get_ee_pose(
            state_tensor[:, : rollout_fn.dynamics_model.d_state]
        )
        ee_pos = ee_state["ee_pos_seq"]
        goal_pos = rollout_fn.goal_ee_pos.to(**task.controller.tensor_args)
        return float(torch.norm(ee_pos - goal_pos, dim=-1).mean().item())

    if getattr(rollout_fn, "goal_state", None) is not None:
        n_dofs = rollout_fn.dynamics_model.n_dofs
        goal_q = rollout_fn.goal_state[:, :n_dofs].to(**task.controller.tensor_args)
        q = state_tensor[:, :n_dofs]
        return float(torch.norm(q - goal_q, dim=-1).mean().item())

    return math.nan


def _compute_safety_margin_from_state_dict(rollout_fn, state_dict):
    margins = []

    if (
        hasattr(rollout_fn, "primitive_collision_cost")
        and "link_pos_seq" in state_dict
        and "link_rot_seq" in state_dict
    ):
        p_cost = rollout_fn.primitive_collision_cost
        batch_size, horizon, n_links = state_dict["link_pos_seq"].shape[:3]
        if p_cost.batch_size != batch_size:
            p_cost.batch_size = batch_size
            p_cost.robot_world_coll.build_batch_features(
                batch_size * horizon,
                clone_pose=True,
                clone_points=True,
            )
        link_pos_batch = state_dict["link_pos_seq"].view(batch_size * horizon, n_links, 3)
        link_rot_batch = state_dict["link_rot_seq"].view(batch_size * horizon, n_links, 3, 3)
        raw_signed_dist = p_cost.robot_world_coll.check_robot_sphere_collisions(
            link_pos_batch,
            link_rot_batch,
        ).view(batch_size, horizon, n_links)
        primitive_margin = -(raw_signed_dist + p_cost.distance_threshold)
        margins.append(primitive_margin.amin(dim=(1, 2)))

    if hasattr(rollout_fn, "robot_self_collision_cost") and "state_seq" in state_dict:
        self_cost = rollout_fn.robot_self_collision_cost
        n_dofs = rollout_fn.dynamics_model.n_dofs
        batch_size, horizon = state_dict["state_seq"].shape[:2]
        q_seq = state_dict["state_seq"][:, :, :n_dofs]
        q_flat = q_seq.reshape(batch_size * horizon, n_dofs)
        raw_signed_dist = self_cost.coll.check_self_collisions_nn(q_flat).view(
            batch_size,
            horizon,
        )
        self_margin = -(raw_signed_dist + self_cost.distance_threshold)
        margins.append(self_margin.amin(dim=1))

    if not margins:
        return None
    return torch.stack(margins, dim=0).amin(dim=0)


def compute_minimum_safety_margin(task, current_state):
    controller = task.controller
    if hasattr(controller, "get_latest_stats"):
        latest_stats = controller.get_latest_stats()
        value = to_float_or_nan(latest_stats.get("minimum_safety_margin"))
        if not math.isnan(value):
            return value

    trajectories = getattr(controller, "trajectories", None)
    if trajectories is None or "actions" not in trajectories:
        return math.nan

    rollout_fn = controller.rollout_fn
    if "state_dict" in trajectories:
        state_dict = trajectories["state_dict"]
    else:
        state_tensor = _task_state_to_tensor(task, current_state)
        actions = trajectories["actions"]
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, **controller.tensor_args)
        else:
            actions = actions.to(**controller.tensor_args)
        state_dict = rollout_fn.dynamics_model.rollout_open_loop(state_tensor, actions)

    margin_seq = _compute_safety_margin_from_state_dict(rollout_fn, state_dict)
    if margin_seq is None:
        return math.nan
    return float(margin_seq.amin().item())


def extract_raw_info(task) -> Dict[str, object]:
    control_process = getattr(task, "control_process", None)
    command = getattr(control_process, "command", None)
    if isinstance(command, list) and len(command) > 2 and isinstance(command[2], dict):
        return command[2]
    if hasattr(task.controller, "get_latest_stats"):
        return {"stats": task.controller.get_latest_stats()}
    return {}


def normalize_step_record(
    controller_name,
    task,
    current_state,
    episode_id,
    step_id,
    seed,
    raw_info=None,
    success_threshold=None,
):
    raw_info = {} if raw_info is None else raw_info
    raw_stats = raw_info.get("stats", raw_info)

    goal_distance = to_float_or_nan(
        raw_stats.get("final_goal_distance", raw_stats.get("goal_dist"))
    )
    if math.isnan(goal_distance):
        goal_distance = compute_goal_distance(task, current_state)

    minimum_safety_margin = to_float_or_nan(raw_stats.get("minimum_safety_margin"))
    if math.isnan(minimum_safety_margin):
        minimum_safety_margin = compute_minimum_safety_margin(task, current_state)

    if success_threshold is not None and not math.isnan(goal_distance):
        success = goal_distance <= float(success_threshold)
        failure = not success
    else:
        success = to_bool_or_none(raw_stats.get("success"))
        failure = to_bool_or_none(raw_stats.get("failure"))

    return {
        "controller_name": controller_name,
        "episode_id": episode_id,
        "step_id": step_id,
        "seed": seed,
        "success": success,
        "failure": failure,
        "final_goal_distance": goal_distance,
        "minimum_safety_margin": minimum_safety_margin,
        "safe_elite_fraction": to_float_or_nan(raw_stats.get("safe_elite_fraction")),
        "safe_weight_mass": to_float_or_nan(raw_stats.get("safe_weight_mass")),
        "rho_k": to_float_or_nan(raw_stats.get("rho_k")),
        "z_t": to_float_or_nan(raw_stats.get("z_t")),
        "covariance_fallback": to_bool_or_none(raw_stats.get("covariance_fallback")),
        "margin_fallback": to_bool_or_none(raw_stats.get("margin_fallback")),
    }


def summarize_episode(step_records: Iterable[Dict[str, object]]):
    step_records = list(step_records)
    if not step_records:
        raise ValueError("Cannot summarize empty episode records")

    last = dict(step_records[-1])
    valid_margins = [
        record["minimum_safety_margin"]
        for record in step_records
        if not math.isnan(to_float_or_nan(record["minimum_safety_margin"]))
    ]
    last["minimum_safety_margin"] = min(valid_margins) if valid_margins else math.nan
    last.pop("step_id", None)
    return last


class CsvExperimentLogger:
    def __init__(self, output_dir):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.step_csv = os.path.join(self.output_dir, "step_metrics.csv")
        self.episode_csv = os.path.join(self.output_dir, "episode_metrics.csv")
        self._init_csv(self.step_csv, STEP_FIELDS)
        self._init_csv(self.episode_csv, EPISODE_FIELDS)

    def _init_csv(self, path, fieldnames):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log_step(self, record):
        with open(self.step_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=STEP_FIELDS)
            writer.writerow(record)

    def log_episode(self, record):
        with open(self.episode_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
            writer.writerow(record)
