from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def nan() -> float:
    return float("nan")


EPISODE_FIELDS = [
    "method_name",
    "episode_id",
    "target_id",
    "scene",
    "success",
    "failure",
    "collision",
    "timeout",
    "final_ee_error",
    "final_joint_error",
    "minimum_safety_margin",
    "steps_to_goal",
    "wall_time",
    "planning_time",
    "control_time_mean",
    "trajectory_length_joint",
    "trajectory_length_ee",
    "smoothness_jerk",
    "rrtstar_available",
    "rrtstar_path_found",
    "controller_class",
    "uses_clean_controller",
    "uses_native_margin",
    "deployment_refinement_enabled",
    "local_refinement_enabled",
    "margin_fallback",
    "skipped_reason",
    "path_length_joint",
    "planning_time_limit",
    "goal_bias",
    "interpolation_resolution",
    "collision_check_resolution",
    "number_of_validity_checks",
    "number_of_invalid_states",
]


STEP_FIELDS = [
    "method_name",
    "episode_id",
    "target_id",
    "scene",
    "step",
    "q",
    "ee_position",
    "ee_error",
    "joint_error",
    "safety_margin",
    "collision",
    "planning_time",
    "wall_time",
    "skipped_reason",
]


def safe_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def bool_to_csv(value) -> str:
    if value is None:
        return ""
    return "true" if bool(value) else "false"


def vector_to_jsonish(values: Iterable[float]) -> str:
    arr = np.asarray(list(values), dtype=float)
    return "[" + ",".join(f"{x:.8g}" for x in arr.tolist()) + "]"


def trajectory_length_joint(q_path: Iterable[Iterable[float]]) -> float:
    q = np.asarray(list(q_path), dtype=float)
    if q.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


def trajectory_length_ee(ee_path: Iterable[Iterable[float]]) -> float:
    ee = np.asarray(list(ee_path), dtype=float)
    if ee.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum())


def smoothness_jerk(q_path: Iterable[Iterable[float]]) -> float:
    q = np.asarray(list(q_path), dtype=float)
    if q.shape[0] < 4:
        return float("nan")
    jerk = np.diff(q, n=3, axis=0)
    return float(np.linalg.norm(jerk, axis=1).mean())


def csv_value(value):
    if isinstance(value, bool):
        return bool_to_csv(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return vector_to_jsonish(value)
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if value is None:
        return ""
    return value


def write_csv(path: str | Path, rows: list[dict], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})
