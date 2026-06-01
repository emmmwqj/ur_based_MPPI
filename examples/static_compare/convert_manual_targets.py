#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.static_compare.utils.io_utils import resolve_repo_path, write_json
from examples.static_compare.utils.static_collision_checker import StaticTallCollisionChecker, _signed_distance_to_box


INITIAL_POSITIONS_FILE = "examples/sim_gazebo/config/initial_positions.yaml"
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _load_initial_joint_positions() -> list[float]:
    with open(resolve_repo_path(INITIAL_POSITIONS_FILE), "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return [float(params[name]) for name in JOINT_NAMES]


def _parse_targets(path: str | Path) -> list[list[float]]:
    text = resolve_repo_path(path).read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path), mode="exec")
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "targets":
                    values = ast.literal_eval(node.value)
                    arr = np.asarray(values, dtype=float)
                    if arr.ndim != 2 or arr.shape[1] != 3:
                        raise ValueError(f"targets must be an Nx3 list; got shape={arr.shape}")
                    return arr.tolist()
    raise ValueError(f"No `targets = [...]` assignment found in {path}")


def _point_obstacle_margin(checker: StaticTallCollisionChecker, point: np.ndarray) -> float:
    margins: list[float] = []
    for obstacle in checker.obstacles:
        if obstacle["type"] == "sphere":
            margins.append(float(np.linalg.norm(point - obstacle["position"])) - obstacle["radius"])
        elif obstacle["type"] == "box":
            local = obstacle["rotation"].T @ (point - obstacle["position"])
            margins.append(_signed_distance_to_box(local, obstacle["half_extents"]))
    return float(min(margins)) if margins else float("inf")


def _difficulty_from_margin(point: np.ndarray, margin: float) -> str:
    if margin < 0.0:
        return "inside_obstacle"
    if margin < 0.03:
        return "near_obstacle"
    if point[2] < 0.22 or point[0] < 0.18:
        return "hard_reach"
    return "manual_grid"


def convert_targets(input_path: str | Path, profile: str = "manual_v1") -> dict:
    checker = StaticTallCollisionChecker(include_ground=True)
    initial_q = _load_initial_joint_positions()
    initial_ee = checker.ee_position(initial_q)
    targets = []
    for idx, xyz in enumerate(_parse_targets(input_path)):
        goal = np.asarray(xyz, dtype=float)
        point_margin = _point_obstacle_margin(checker, goal)
        initial_goal_ee_distance = float(np.linalg.norm(goal - initial_ee))
        difficulty = _difficulty_from_margin(goal, point_margin)
        targets.append(
            {
                "target_id": f"manual_static_tall_{idx:03d}",
                "scene": "tall",
                "initial_joint_positions": initial_q,
                "goal_ee_position": goal.round(8).tolist(),
                "difficulty_tag": difficulty,
                "initial_goal_ee_distance": initial_goal_ee_distance,
                "goal_point_obstacle_margin": point_margin,
                "notes": (
                    "Manual Cartesian target from examples/static_compare/targets/setup_targets.txt. "
                    "STORM/SAGE receive this target through /target_pose. No joint-space goal is supplied; "
                    "RRT* is not run as a fair joint-space baseline for this target set."
                ),
            }
        )
    return {
        "schema_version": 5,
        "scene": "tall",
        "profile": profile,
        "target_count": len(targets),
        "source": str(resolve_repo_path(input_path)),
        "initial_positions_source": INITIAL_POSITIONS_FILE,
        "success_threshold": 0.01,
        "max_runtime_sec": 20.0,
        "reset_policy": "Gazebo is restarted for each episode using initial_positions.yaml; this directly resets the arm.",
        "target_interface": "/target_pose Cartesian position in world frame",
        "rrtstar_note": "Manual target set has Cartesian goals only; RRT* requires goal_joint_positions and is skipped unless paired joint goals are supplied.",
        "targets": targets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert manual static tall Cartesian targets into benchmark JSON")
    parser.add_argument("--input", default="examples/static_compare/targets/setup_targets.txt")
    parser.add_argument("--output", default="examples/static_compare/targets/static_tall_targets_manual.json")
    parser.add_argument("--profile", default="manual_v1")
    args = parser.parse_args()
    payload = convert_targets(args.input, profile=args.profile)
    write_json(args.output, payload)
    invalid = [t for t in payload["targets"] if t["goal_point_obstacle_margin"] < 0.0]
    print(
        f"wrote {payload['target_count']} manual targets to {args.output}; "
        f"inside_obstacle={len(invalid)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
