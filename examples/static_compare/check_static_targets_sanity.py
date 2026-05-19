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

from examples.static_compare.utils.io_utils import ensure_dir, load_json, resolve_repo_path, write_json
from examples.static_compare.utils.static_collision_checker import StaticTallCollisionChecker


DEFAULT_WORKSPACE_BOUNDS = {
    "x": [0.18, 0.82],
    "y": [-0.62, 0.62],
    "z": [0.18, 0.86],
}
DIFFICULTY_TAGS = {"easy", "near_obstacle", "around_tall_obstacle", "far_reach", "hard_reach"}


def _workspace_ok(goal_ee: np.ndarray, bounds: dict) -> bool:
    return bool(
        bounds["x"][0] <= goal_ee[0] <= bounds["x"][1]
        and bounds["y"][0] <= goal_ee[1] <= bounds["y"][1]
        and bounds["z"][0] <= goal_ee[2] <= bounds["z"][1]
    )


def _validate_target(target: dict, checker: StaticTallCollisionChecker, bounds: dict, min_goal_margin: float) -> dict:
    target_id = str(target.get("target_id", ""))
    reasons: list[str] = []
    warnings: list[str] = []
    try:
        q0 = np.asarray(target["initial_joint_positions"], dtype=float)
        qg = np.asarray(target["goal_joint_positions"], dtype=float)
        goal_ee = np.asarray(target["goal_ee_position"], dtype=float)
    except Exception as exc:
        return {
            "target_id": target_id,
            "valid": False,
            "reasons": [f"missing or malformed required fields: {exc}"],
            "warnings": [],
        }

    if target.get("scene") != "tall":
        reasons.append("scene is not tall")
    if q0.shape != (6,) or qg.shape != (6,) or goal_ee.shape != (3,):
        reasons.append("invalid vector shape")
    if target.get("difficulty_tag") not in DIFFICULTY_TAGS:
        reasons.append(f"difficulty_tag must be one of {sorted(DIFFICULTY_TAGS)}")
    if not checker.within_joint_limits(q0):
        reasons.append("initial_joint_positions outside joint limits")
    if not checker.within_joint_limits(qg):
        reasons.append("goal_joint_positions outside joint limits")
    if not _workspace_ok(goal_ee, bounds):
        reasons.append("goal_ee_position outside formal workspace bounds")

    init_valid = checker.check_state(q0)
    goal_valid = checker.check_state(qg)
    if not init_valid.valid:
        reasons.append(f"initial state invalid; margin={init_valid.minimum_safety_margin:.6g}")
    if not goal_valid.valid:
        reasons.append(f"goal joint state invalid; margin={goal_valid.minimum_safety_margin:.6g}")
    if goal_valid.minimum_safety_margin < min_goal_margin:
        reasons.append(f"goal is too close to/in obstacle; margin={goal_valid.minimum_safety_margin:.6g}")

    fk_goal = checker.ee_position(qg)
    fk_error = float(np.linalg.norm(fk_goal - goal_ee))
    if fk_error > 2.0e-4:
        reasons.append(f"goal_ee_position does not match FK(goal_joint_positions); error={fk_error:.6g}")

    motion = checker.check_motion(q0, qg, resolution=0.10)
    if not motion.valid:
        warnings.append(
            "straight-line joint interpolation is invalid; this is allowed for hard/around-obstacle targets "
            f"but recorded. straight_line_min_margin={motion.minimum_safety_margin:.6g}"
        )
    elif motion.minimum_safety_margin < 0.02:
        warnings.append(
            "straight-line joint interpolation is very close to obstacles; retained as a hard case. "
            f"straight_line_min_margin={motion.minimum_safety_margin:.6g}"
        )

    return {
        "target_id": target_id,
        "valid": len(reasons) == 0,
        "difficulty_tag": target.get("difficulty_tag"),
        "goal_ee_position": goal_ee.round(8).tolist() if goal_ee.shape == (3,) else [],
        "goal_margin": float(goal_valid.minimum_safety_margin),
        "initial_margin": float(init_valid.minimum_safety_margin),
        "straight_line_min_margin": float(motion.minimum_safety_margin),
        "fk_error": fk_error,
        "reasons": reasons,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity-check static tall formal target set")
    parser.add_argument("--targets-path", default="examples/static_compare/targets/static_tall_targets_formal.json")
    parser.add_argument("--output", default="examples/static_compare/targets/target_sanity_report.json")
    parser.add_argument("--valid-output", default="", help="Optional path for a filtered valid target set")
    parser.add_argument("--min-goal-margin", type=float, default=0.015)
    args = parser.parse_args()

    payload = load_json(args.targets_path)
    bounds = payload.get("workspace_bounds", DEFAULT_WORKSPACE_BOUNDS)
    checker = StaticTallCollisionChecker(include_ground=True)
    checked = [_validate_target(target, checker, bounds, args.min_goal_margin) for target in payload.get("targets", [])]
    valid_ids = {entry["target_id"] for entry in checked if entry["valid"]}
    invalid = [entry for entry in checked if not entry["valid"]]
    warnings = [entry for entry in checked if entry.get("warnings")]
    difficulty_counts = {}
    for target in payload.get("targets", []):
        if target.get("target_id") in valid_ids:
            difficulty_counts[target.get("difficulty_tag", "")] = difficulty_counts.get(target.get("difficulty_tag", ""), 0) + 1

    report = {
        "status": "pass" if not invalid else "fail",
        "targets_path": str(resolve_repo_path(args.targets_path)),
        "total_targets": len(payload.get("targets", [])),
        "valid_count": len(valid_ids),
        "invalid_count": len(invalid),
        "warning_count": len(warnings),
        "difficulty_counts_valid": difficulty_counts,
        "tuned_target_interface": {
            "storm_mppi_tuned": "/target_pose PoseStamped position only",
            "sage_mppi_tuned": "/target_pose PoseStamped position only",
            "rrtstar_ompl": "initial_joint_positions + goal_joint_positions",
        },
        "workspace_bounds": bounds,
        "checks": checked,
    }
    write_json(args.output, report)

    if args.valid_output:
        filtered = dict(payload)
        filtered["targets"] = [target for target in payload.get("targets", []) if target.get("target_id") in valid_ids]
        filtered["target_count"] = len(filtered["targets"])
        filtered["target_sanity_report"] = str(resolve_repo_path(args.output))
        write_json(args.valid_output, filtered)

    print(
        f"target sanity status={report['status']} valid={report['valid_count']} "
        f"invalid={report['invalid_count']} warnings={report['warning_count']}"
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
