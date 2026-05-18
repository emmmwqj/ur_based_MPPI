#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.static_compare.utils.io_utils import resolve_repo_path, write_json
from examples.static_compare.utils.static_collision_checker import StaticTallCollisionChecker


INITIAL_POSITIONS_FILE = "examples/sim_gazebo/config/initial_positions.yaml"
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _load_tuned_initial_joint_positions() -> np.ndarray:
    with open(resolve_repo_path(INITIAL_POSITIONS_FILE), "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return np.asarray([params[name] for name in JOINT_NAMES], dtype=float)


def _sample_goal(rng: np.random.Generator, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    practical_lower = np.maximum(lower, np.array([-2.4, -2.45, -1.8, -2.8, -2.6, -2.8]))
    practical_upper = np.minimum(upper, np.array([2.4, -0.45, 2.4, -0.35, -0.45, 2.8]))
    return rng.uniform(practical_lower, practical_upper)


def generate_targets(num_targets: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    checker = StaticTallCollisionChecker(include_ground=True)
    q0 = _load_tuned_initial_joint_positions()
    q0_valid = checker.check_state(q0)
    if not q0_valid.valid:
        raise RuntimeError(f"Tuned initial state is invalid under static checker: margin={q0_valid.minimum_safety_margin}")

    targets = []
    seen_goal_ee: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(2000, num_targets * 1000)
    while len(targets) < num_targets and attempts < max_attempts:
        attempts += 1
        qg = _sample_goal(rng, checker.joint_lower, checker.joint_upper)
        q_dist = float(np.linalg.norm(qg - q0))
        if q_dist < 1.0:
            continue
        goal_valid = checker.check_state(qg)
        if not goal_valid.valid:
            continue
        goal_ee = checker.ee_position(qg)
        if not (0.18 <= goal_ee[0] <= 0.82 and -0.62 <= goal_ee[1] <= 0.62 and 0.18 <= goal_ee[2] <= 0.86):
            continue
        if seen_goal_ee and min(float(np.linalg.norm(goal_ee - p)) for p in seen_goal_ee) < 0.08:
            continue

        motion = checker.check_motion(q0, qg, resolution=0.10)
        difficulty = "pilot_tall_near_wall" if motion.minimum_safety_margin < 0.12 else "pilot_tall_open"
        targets.append(
            {
                "target_id": f"tuned_tall_pilot_{len(targets):02d}",
                "scene": "tall",
                "initial_joint_positions": q0.round(8).tolist(),
                "goal_joint_positions": qg.round(8).tolist(),
                "goal_ee_position": goal_ee.round(8).tolist(),
                "difficulty_tag": difficulty,
                "notes": (
                    "Common target for tuned STORM/SAGE references. The reference scripts accept "
                    "Cartesian goals through /target_pose; RRT* uses the paired joint-space goal. "
                    "Initial state is exactly examples/sim_gazebo/config/initial_positions.yaml."
                ),
                "generation": {
                    "seed": seed,
                    "attempt_index": attempts,
                    "initial_margin": float(q0_valid.minimum_safety_margin),
                    "goal_margin": float(goal_valid.minimum_safety_margin),
                    "straight_line_min_margin": float(motion.minimum_safety_margin),
                    "joint_distance": q_dist,
                },
            }
        )
        seen_goal_ee.append(goal_ee)

    if len(targets) < num_targets:
        raise RuntimeError(f"Generated only {len(targets)} targets after {attempts} attempts")

    return {
        "schema_version": 2,
        "scene": "tall",
        "seed": seed,
        "target_count": len(targets),
        "initial_positions_source": INITIAL_POSITIONS_FILE,
        "tuned_reference_default_goals": {
            "storm_mppi_tuned": [0.5, -0.45, 0.4],
            "sage_mppi_tuned": [0.4, -0.5, 0.4],
            "note": "Defaults differ; static_compare injects this shared target list through /target_pose.",
        },
        "targets": targets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate common targets for tuned static tall references")
    parser.add_argument("--num-targets", type=int, default=3)
    parser.add_argument("--output", default="examples/static_compare/targets/static_tall_targets.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not (3 <= args.num_targets <= 5):
        raise ValueError("--num-targets must be between 3 and 5 for this tuned-reference pilot")
    payload = generate_targets(args.num_targets, args.seed)
    write_json(args.output, payload)
    print(f"wrote {payload['target_count']} tuned static tall targets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
