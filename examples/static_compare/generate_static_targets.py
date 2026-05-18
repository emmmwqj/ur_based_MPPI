#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.static_compare.utils.io_utils import write_json
from examples.static_compare.utils.static_collision_checker import StaticTallCollisionChecker


def _difficulty_tag(line_min_margin: float, joint_distance: float) -> str:
    if line_min_margin < 0.08 or joint_distance > 2.0:
        return "pilot_tall_near_wall"
    if line_min_margin < 0.18:
        return "pilot_tall_medium"
    return "pilot_tall_open"


def _sample_joint(rng: np.random.Generator, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    # Keep the pilot samples in a practical UR7e reaching envelope while still
    # respecting true joint limits from the URDF/config.
    practical_lower = np.maximum(lower, np.array([-2.8, -2.5, -2.3, -3.0, -2.7, -3.1]))
    practical_upper = np.minimum(upper, np.array([2.8, -0.45, 2.5, -0.35, -0.45, 3.1]))
    return rng.uniform(practical_lower, practical_upper)


def _candidate_ok(checker: StaticTallCollisionChecker, q: np.ndarray) -> tuple[bool, np.ndarray, float]:
    validity = checker.check_state(q)
    if not validity.valid:
        return False, np.zeros(3), validity.minimum_safety_margin
    ee = checker.ee_position(q)
    workspace_ok = (
        0.15 <= ee[0] <= 0.85
        and -0.65 <= ee[1] <= 0.65
        and 0.15 <= ee[2] <= 0.85
    )
    return bool(workspace_ok), ee, validity.minimum_safety_margin


def generate_targets(num_targets: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    checker = StaticTallCollisionChecker(include_ground=True)
    lower, upper = checker.joint_lower, checker.joint_upper
    targets: list[dict] = []
    used_goal_positions: list[np.ndarray] = []

    attempts = 0
    max_attempts = max(2500, num_targets * 1200)
    while len(targets) < num_targets and attempts < max_attempts:
        attempts += 1
        q0 = _sample_joint(rng, lower, upper)
        qg = _sample_joint(rng, lower, upper)
        q_dist = float(np.linalg.norm(qg - q0))
        if q_dist < 1.05:
            continue

        ok0, ee0, margin0 = _candidate_ok(checker, q0)
        okg, eeg, marging = _candidate_ok(checker, qg)
        if not (ok0 and okg):
            continue
        if used_goal_positions and min(float(np.linalg.norm(eeg - prev)) for prev in used_goal_positions) < 0.06:
            continue

        motion = checker.check_motion(q0, qg, resolution=0.10)
        line_margin = motion.minimum_safety_margin
        if line_margin > 0.42 and len(targets) < max(1, num_targets // 2):
            continue

        target_id = f"tall_pilot_{len(targets):02d}"
        notes = (
            "Generated from joint-space initial/goal samples; endpoints pass "
            "static_collision_checker. Goal EE is FK(goal_joint_positions). "
            "Straight-line joint motion margin is logged for difficulty only; "
            "the task is not sequential retargeting."
        )
        targets.append(
            {
                "target_id": target_id,
                "scene": "tall",
                "initial_joint_positions": q0.round(8).tolist(),
                "goal_joint_positions": qg.round(8).tolist(),
                "goal_ee_position": eeg.round(8).tolist(),
                "difficulty_tag": _difficulty_tag(line_margin, q_dist),
                "notes": notes,
                "generation": {
                    "seed": seed,
                    "attempt_index": attempts,
                    "initial_ee_position": ee0.round(8).tolist(),
                    "initial_margin": float(margin0),
                    "goal_margin": float(marging),
                    "straight_line_min_margin": float(line_margin),
                    "joint_distance": q_dist,
                },
            }
        )
        used_goal_positions.append(eeg)

    if len(targets) < num_targets:
        raise RuntimeError(
            f"Generated only {len(targets)} valid targets after {attempts} attempts; "
            "relax sampling filters or inspect the collision checker."
        )
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate static tall-scene joint-space targets")
    parser.add_argument("--num-targets", type=int, default=5)
    parser.add_argument("--output", default="examples/static_compare/targets/static_tall_targets.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not (5 <= args.num_targets <= 10):
        raise ValueError("--num-targets must be between 5 and 10 for this pilot")

    targets = generate_targets(args.num_targets, args.seed)
    payload = {
        "schema_version": 1,
        "scene": "tall",
        "seed": args.seed,
        "target_count": len(targets),
        "targets": targets,
    }
    write_json(args.output, payload)
    print(f"wrote {len(targets)} static tall targets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
