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


WORKSPACE_BOUNDS = {
    "x": [0.18, 0.82],
    "y": [-0.62, 0.62],
    "z": [0.18, 0.86],
}

FORMAL_DIFFICULTY_TAGS = [
    "easy",
    "near_obstacle",
    "around_tall_obstacle",
    "far_reach",
]
FORMAL_V2_DIFFICULTY_TAGS = [
    "easy",
    "near_obstacle",
    "around_tall_obstacle",
    "hard_reach",
]
FORMAL_MIN_GOAL_MARGIN = 0.015
FORMAL_V2_MIN_GOAL_MARGIN = 0.012
FORMAL_V3_MIN_GOAL_MARGIN = 0.012
FORMAL_V3_MIN_INITIAL_GOAL_EE_DISTANCE = 0.16


def _sample_goal(rng: np.random.Generator, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    practical_lower = np.maximum(lower, np.array([-2.5, -2.45, -1.85, -2.85, -2.65, -2.85]))
    practical_upper = np.minimum(upper, np.array([2.5, -0.40, 2.45, -0.30, -0.40, 2.85]))
    return rng.uniform(practical_lower, practical_upper)


def _workspace_ok(goal_ee: np.ndarray) -> bool:
    return bool(
        WORKSPACE_BOUNDS["x"][0] <= goal_ee[0] <= WORKSPACE_BOUNDS["x"][1]
        and WORKSPACE_BOUNDS["y"][0] <= goal_ee[1] <= WORKSPACE_BOUNDS["y"][1]
        and WORKSPACE_BOUNDS["z"][0] <= goal_ee[2] <= WORKSPACE_BOUNDS["z"][1]
    )


def _formal_difficulty(
    q0: np.ndarray,
    qg: np.ndarray,
    goal_ee: np.ndarray,
    goal_margin: float,
    straight_margin: float,
) -> str:
    joint_dist = float(np.linalg.norm(qg - q0))
    ee_dist = float(np.linalg.norm(goal_ee - np.array([0.48397353, 0.13337938, 0.72136778])))
    if joint_dist > 3.2 or ee_dist > 0.45:
        return "far_reach"
    if goal_margin < 0.04 or straight_margin < 0.03:
        return "around_tall_obstacle"
    if goal_margin < 0.075 or straight_margin < 0.06:
        return "near_obstacle"
    return "easy"


def _formal_v2_difficulty(
    q0: np.ndarray,
    qg: np.ndarray,
    goal_ee: np.ndarray,
    goal_margin: float,
    straight_margin: float,
) -> str:
    joint_dist = float(np.linalg.norm(qg - q0))
    ee_dist = float(np.linalg.norm(goal_ee - np.array([0.48397353, 0.13337938, 0.72136778])))
    if (
        straight_margin < 0.018
        or (goal_margin < 0.035 and joint_dist > 1.8)
        or (goal_margin < 0.055 and (joint_dist > 3.0 or ee_dist > 0.43))
    ):
        return "hard_reach"
    if goal_margin < 0.055 or straight_margin < 0.04:
        return "around_tall_obstacle"
    if goal_margin < 0.085 or straight_margin < 0.075:
        return "near_obstacle"
    return "easy"


def _formal_v3_difficulty(
    q0: np.ndarray,
    qg: np.ndarray,
    initial_ee: np.ndarray,
    goal_ee: np.ndarray,
    goal_margin: float,
    straight_margin: float,
) -> str:
    joint_dist = float(np.linalg.norm(qg - q0))
    ee_dist = float(np.linalg.norm(goal_ee - initial_ee))
    if (
        straight_margin < 0.018
        or (goal_margin < 0.035 and joint_dist > 1.8)
        or (goal_margin < 0.055 and (joint_dist > 3.0 or ee_dist > 0.40))
    ):
        return "hard_reach"
    if goal_margin < 0.055 or straight_margin < 0.04:
        return "around_tall_obstacle"
    if goal_margin < 0.085 or straight_margin < 0.075:
        return "near_obstacle"
    return "easy"


def _target_quota(num_targets: int, profile: str) -> dict[str, int]:
    if profile == "pilot":
        return {"pilot_tall_near_wall": num_targets}
    if profile in {"formal_v2", "formal_v3"}:
        return {
            "easy": 10,
            "near_obstacle": 15,
            "around_tall_obstacle": 20,
            "hard_reach": 15,
        }
    tags = FORMAL_DIFFICULTY_TAGS
    base = num_targets // len(tags)
    quota = {tag: base for tag in tags}
    for tag in tags[: num_targets - base * len(tags)]:
        quota[tag] += 1
    return quota


def generate_targets(num_targets: int, seed: int, profile: str = "pilot") -> dict:
    rng = np.random.default_rng(seed)
    checker = StaticTallCollisionChecker(include_ground=True)
    q0 = _load_tuned_initial_joint_positions()
    q0_ee = checker.ee_position(q0)
    q0_valid = checker.check_state(q0)
    if not q0_valid.valid:
        raise RuntimeError(f"Tuned initial state is invalid under static checker: margin={q0_valid.minimum_safety_margin}")

    targets = []
    seen_goal_ee: list[np.ndarray] = []
    quota = _target_quota(num_targets, profile)
    counts = {key: 0 for key in quota}
    attempts = 0
    max_attempts = max(20000, num_targets * 3000)
    while len(targets) < num_targets and attempts < max_attempts:
        attempts += 1
        qg = _sample_goal(rng, checker.joint_lower, checker.joint_upper)
        q_dist = float(np.linalg.norm(qg - q0))
        if q_dist < 1.0:
            continue
        goal_valid = checker.check_state(qg)
        if not goal_valid.valid:
            continue
        if profile == "formal_v3":
            min_goal_margin = FORMAL_V3_MIN_GOAL_MARGIN
        elif profile == "formal_v2":
            min_goal_margin = FORMAL_V2_MIN_GOAL_MARGIN
        else:
            min_goal_margin = FORMAL_MIN_GOAL_MARGIN
        if profile in {"formal", "formal_v2", "formal_v3"} and goal_valid.minimum_safety_margin < min_goal_margin:
            continue
        goal_ee = checker.ee_position(qg)
        initial_goal_ee_distance = float(np.linalg.norm(goal_ee - q0_ee))
        if profile == "formal_v3" and initial_goal_ee_distance < FORMAL_V3_MIN_INITIAL_GOAL_EE_DISTANCE:
            continue
        if not _workspace_ok(goal_ee):
            continue
        min_ee_separation = 0.045 if profile in {"formal_v2", "formal_v3"} else 0.08
        if seen_goal_ee and min(float(np.linalg.norm(goal_ee - p)) for p in seen_goal_ee) < min_ee_separation:
            continue

        motion = checker.check_motion(q0, qg, resolution=0.10)
        if profile == "formal_v3":
            difficulty = _formal_v3_difficulty(
                q0,
                qg,
                q0_ee,
                goal_ee,
                goal_valid.minimum_safety_margin,
                motion.minimum_safety_margin,
            )
            if counts.get(difficulty, 0) >= quota.get(difficulty, 0):
                continue
        elif profile == "formal_v2":
            difficulty = _formal_v2_difficulty(q0, qg, goal_ee, goal_valid.minimum_safety_margin, motion.minimum_safety_margin)
            if counts.get(difficulty, 0) >= quota.get(difficulty, 0):
                continue
        elif profile == "formal":
            difficulty = _formal_difficulty(q0, qg, goal_ee, goal_valid.minimum_safety_margin, motion.minimum_safety_margin)
            if counts.get(difficulty, 0) >= quota.get(difficulty, 0):
                continue
        else:
            difficulty = "pilot_tall_near_wall" if motion.minimum_safety_margin < 0.12 else "pilot_tall_open"
        targets.append(
            {
                "target_id": f"static_tall_{profile}_{len(targets):03d}",
                "scene": "tall",
                "initial_joint_positions": q0.round(8).tolist(),
                "goal_joint_positions": qg.round(8).tolist(),
                "goal_ee_position": goal_ee.round(8).tolist(),
                "difficulty_tag": difficulty,
                "initial_goal_ee_distance": initial_goal_ee_distance,
                "goal_margin": float(goal_valid.minimum_safety_margin),
                "straight_line_min_margin": float(motion.minimum_safety_margin),
                "notes": (
                    "Common target for tuned STORM/SAGE references. The reference scripts accept "
                    "Cartesian goals through /target_pose; RRT* uses the paired joint-space goal. "
                    "Initial state is exactly examples/sim_gazebo/config/initial_positions.yaml. "
                    f"Profile={profile}; difficulty_tag={difficulty}. "
                    "V3 filters out initial EE positions that are already close to the goal."
                ),
                "generation": {
                    "seed": seed,
                    "profile": profile,
                    "attempt_index": attempts,
                    "initial_margin": float(q0_valid.minimum_safety_margin),
                    "goal_margin": float(goal_valid.minimum_safety_margin),
                    "straight_line_min_margin": float(motion.minimum_safety_margin),
                    "initial_goal_ee_distance": initial_goal_ee_distance,
                    "joint_distance": q_dist,
                    "workspace_bounds": WORKSPACE_BOUNDS,
                    "formal_min_goal_margin": min_goal_margin if profile in {"formal", "formal_v2", "formal_v3"} else None,
                    "min_initial_goal_ee_distance": (
                        FORMAL_V3_MIN_INITIAL_GOAL_EE_DISTANCE if profile == "formal_v3" else None
                    ),
                    "min_ee_separation": min_ee_separation,
                },
            }
        )
        seen_goal_ee.append(goal_ee)
        counts[difficulty] = counts.get(difficulty, 0) + 1

    if len(targets) < num_targets:
        raise RuntimeError(
            f"Generated only {len(targets)} targets after {attempts} attempts; "
            f"counts={counts}, quota={quota}"
        )

    return {
        "schema_version": 3,
        "scene": "tall",
        "seed": seed,
        "profile": profile,
        "target_count": len(targets),
        "initial_positions_source": INITIAL_POSITIONS_FILE,
        "workspace_bounds": WORKSPACE_BOUNDS,
        "min_initial_goal_ee_distance": FORMAL_V3_MIN_INITIAL_GOAL_EE_DISTANCE if profile == "formal_v3" else None,
        "difficulty_counts": counts,
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
    parser.add_argument("--profile", choices=["pilot", "formal", "formal_v2", "formal_v3"], default="pilot")
    args = parser.parse_args()
    if args.profile == "pilot" and not (3 <= args.num_targets <= 5):
        raise ValueError("--num-targets must be between 3 and 5 for this tuned-reference pilot")
    if args.profile == "formal" and args.num_targets not in {20, 30}:
        raise ValueError("--num-targets must be 20 or 30 for formal static tall preparation")
    if args.profile == "formal_v2" and args.num_targets != 60:
        raise ValueError("--num-targets must be 60 for formal_v2 static tall preparation")
    if args.profile == "formal_v3" and not (55 <= args.num_targets <= 65):
        raise ValueError("--num-targets must be around 60 for formal_v3 static tall preparation")
    payload = generate_targets(args.num_targets, args.seed, profile=args.profile)
    write_json(args.output, payload)
    print(
        f"wrote {payload['target_count']} {args.profile} tuned static tall targets to {args.output}; "
        f"difficulty_counts={payload.get('difficulty_counts')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
