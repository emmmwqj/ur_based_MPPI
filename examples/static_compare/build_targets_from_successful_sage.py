#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.static_compare.utils.io_utils import load_json, resolve_repo_path, write_json
from examples.static_compare.utils.static_collision_checker import StaticTallCollisionChecker


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_vector(value: str) -> list[float]:
    parsed = ast.literal_eval(value)
    arr = np.asarray(parsed, dtype=float)
    if arr.shape != (6,):
        raise ValueError(f"Expected 6D joint vector, got shape={arr.shape}")
    return arr.tolist()


def _last_sage_step_by_target(step_log_path: str | Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    with open(resolve_repo_path(step_log_path), "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method_name") != "sage_mppi_tuned":
                continue
            target_id = row["target_id"]
            step = int(row["step"])
            if target_id not in latest or step > int(latest[target_id]["step"]):
                latest[target_id] = row
    return latest


def build_targets(
    manual_targets_path: str | Path,
    episode_log_path: str | Path,
    step_log_path: str | Path,
) -> dict:
    manual_payload = load_json(manual_targets_path)
    manual_by_id = {target["target_id"]: target for target in manual_payload["targets"]}
    last_steps = _last_sage_step_by_target(step_log_path)
    checker = StaticTallCollisionChecker(include_ground=True)

    targets = []
    with open(resolve_repo_path(episode_log_path), "r", encoding="utf-8", newline="") as f:
        for episode in csv.DictReader(f):
            if episode.get("method_name") != "sage_mppi_tuned":
                continue
            if not (_parse_bool(episode.get("success", "")) and not _parse_bool(episode.get("collision", ""))):
                continue
            target_id = episode["target_id"]
            source = manual_by_id[target_id]
            if target_id not in last_steps:
                raise RuntimeError(f"Missing final SAGE step for {target_id}")
            q_goal = _parse_vector(last_steps[target_id]["q"])
            goal_valid = checker.check_state(q_goal)
            goal_ee = np.asarray(source["goal_ee_position"], dtype=float)
            fk_goal = checker.ee_position(q_goal)
            fk_error_to_manual_goal = float(np.linalg.norm(fk_goal - goal_ee))
            if fk_error_to_manual_goal >= 0.01:
                raise RuntimeError(
                    f"SAGE final q for {target_id} is not within 0.01m of manual goal: {fk_error_to_manual_goal}"
                )
            if not goal_valid.valid:
                raise RuntimeError(f"SAGE final q for {target_id} is invalid: margin={goal_valid.minimum_safety_margin}")
            copied = dict(source)
            copied["target_id"] = f"sage44_{target_id}"
            copied["source_target_id"] = target_id
            copied["goal_joint_positions"] = [round(float(x), 8) for x in q_goal]
            copied["rrtstar_goal_source"] = "final joint state from prior SAGE success/no-collision episode"
            copied["rrtstar_goal_fk_error_to_manual_goal"] = fk_error_to_manual_goal
            copied["goal_state_margin"] = float(goal_valid.minimum_safety_margin)
            copied["notes"] = (
                copied.get("notes", "")
                + " This filtered target set keeps only prior SAGE success/no-collision targets; "
                + "goal_joint_positions is the final SAGE joint state from that prior episode for RRT* comparison."
            )
            targets.append(copied)

    return {
        "schema_version": 6,
        "scene": "tall",
        "profile": "manual_sage_success_no_collision_44",
        "target_count": len(targets),
        "source_manual_targets": str(resolve_repo_path(manual_targets_path)),
        "source_episode_log": str(resolve_repo_path(episode_log_path)),
        "source_step_log": str(resolve_repo_path(step_log_path)),
        "success_threshold": 0.01,
        "max_runtime_sec": 20.0,
        "rrtstar_goal_policy": "Use final SAGE joint state from previous success/no-collision record.",
        "targets": targets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a 44-target benchmark set from successful SAGE manual targets")
    parser.add_argument("--manual-targets", default="examples/static_compare/targets/static_tall_targets_manual.json")
    parser.add_argument("--episode-log", default="examples/static_compare/results/manual_static_tall_v1/static_tall_episode_log.csv")
    parser.add_argument("--step-log", default="examples/static_compare/results/manual_static_tall_v1/static_tall_step_log.csv")
    parser.add_argument("--output", default="examples/static_compare/targets/static_tall_targets_sage44.json")
    args = parser.parse_args()
    payload = build_targets(args.manual_targets, args.episode_log, args.step_log)
    write_json(args.output, payload)
    print(f"wrote {payload['target_count']} targets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
