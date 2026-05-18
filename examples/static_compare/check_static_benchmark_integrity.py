#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.static_compare import rrtstar_ompl_adapter
from examples.static_compare.utils.io_utils import ensure_dir, load_json, resolve_repo_path, write_json
from examples.static_compare.utils.metrics import EPISODE_FIELDS
from examples.static_compare.utils.static_collision_checker import StaticTallCollisionChecker


CONFIG_PATH = "examples/static_compare/config/static_tall_benchmark.yml"


def _record(report: dict, name: str, status: str, detail: str = "") -> None:
    report["checks"].append({"name": name, "status": status, "detail": detail})
    if status == "fail":
        report["failed"] += 1
    elif status == "warn":
        report["warnings"] += 1
    else:
        report["passed"] += 1


def _load_config() -> dict:
    with open(resolve_repo_path(CONFIG_PATH), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_no_dynamic_paths(report: dict, config: dict) -> None:
    bad = []
    for key, value in config.get("paths", {}).items():
        text = str(value)
        if "sim_dynamic" in text or "sage_sim_dynamic" in text:
            bad.append(f"{key}={text}")
    if bad:
        _record(report, "no_dynamic_scene_paths", "fail", "; ".join(bad))
    else:
        _record(report, "no_dynamic_scene_paths", "pass", "No sim_dynamic or sage_sim_dynamic paths referenced.")


def _check_target_set(report: dict, targets_payload: dict, checker: StaticTallCollisionChecker) -> None:
    targets = targets_payload.get("targets", [])
    _record(report, "scene_is_tall", "pass" if targets_payload.get("scene") == "tall" else "fail", str(targets_payload.get("scene")))
    _record(report, "target_count_pilot_size", "pass" if 3 <= len(targets) <= 5 else "fail", f"count={len(targets)}")

    seen_ids = set()
    seen_goals: list[np.ndarray] = []
    for target in targets:
        target_id = str(target.get("target_id", ""))
        status = "pass"
        details = []
        if target_id in seen_ids:
            status = "fail"
            details.append("duplicate target_id")
        seen_ids.add(target_id)
        if target.get("scene") != "tall":
            status = "fail"
            details.append("scene is not tall")

        try:
            q0 = np.asarray(target["initial_joint_positions"], dtype=float)
            qg = np.asarray(target["goal_joint_positions"], dtype=float)
            peg = np.asarray(target["goal_ee_position"], dtype=float)
        except Exception as exc:
            _record(report, f"target_{target_id}_required_fields", "fail", str(exc))
            continue

        if q0.shape != (6,) or qg.shape != (6,) or peg.shape != (3,):
            status = "fail"
            details.append("invalid vector shapes")
        if not checker.within_joint_limits(q0) or not checker.within_joint_limits(qg):
            status = "fail"
            details.append("joint limits violated")

        fk_goal = checker.ee_position(qg)
        if float(np.linalg.norm(fk_goal - peg)) > 2.0e-4:
            status = "fail"
            details.append(f"goal_ee_position does not match FK; error={np.linalg.norm(fk_goal - peg):.6g}")

        init_valid = checker.check_state(q0)
        goal_valid = checker.check_state(qg)
        if not init_valid.valid or not goal_valid.valid:
            status = "warn"
            details.append(
                "endpoint validity warning: "
                f"initial_margin={init_valid.minimum_safety_margin:.5f}, "
                f"goal_margin={goal_valid.minimum_safety_margin:.5f}"
            )

        if seen_goals and min(float(np.linalg.norm(qg - prev)) for prev in seen_goals) < 0.05:
            status = "fail"
            details.append("goal joint positions are not sufficiently distinct")
        seen_goals.append(qg)
        _record(report, f"target_{target_id}_validity", status, "; ".join(details) if details else "ok")


def _check_method_config(report: dict, config: dict) -> None:
    paths = config.get("paths", {})
    overrides = config.get("benchmark_overrides", {})

    storm_ref = str(overrides.get("storm_tuned_reference_script", ""))
    sage_ref = str(overrides.get("sage_tuned_reference_script", ""))
    _record(
        report,
        "storm_uses_tuned_reference",
        "pass" if storm_ref == "examples/sim_gazebo/bash/run_all_reach_static_tall.sh" else "fail",
        storm_ref,
    )
    _record(
        report,
        "sage_uses_tuned_reference",
        "pass" if sage_ref == "examples/SAGE_MPPI/clean_SAGE/run_all_reach_static_tall.sh" else "fail",
        sage_ref,
    )

    sage_entry = str(paths.get("sage_clean_controller_entry", ""))
    if "clean_SAGE" in sage_entry and bool(overrides.get("sage_uses_clean_controller")):
        _record(report, "sage_clean_core_path", "pass", sage_entry)
    else:
        _record(report, "sage_clean_core_path", "fail", sage_entry)

    dep = bool(overrides.get("sage_deployment_refinement_enabled"))
    local = bool(overrides.get("sage_local_refinement_enabled"))
    _record(report, "sage_deployment_refinement_matches_tuned", "pass" if dep else "warn", f"enabled={dep}")
    _record(report, "sage_local_refinement_matches_tuned", "pass" if local else "warn", f"enabled={local}")
    _record(report, "reset_after_each_target", "pass" if bool(overrides.get("reset_after_each_target")) else "fail", str(overrides.get("reset_after_each_target")))

    checker_name = str(overrides.get("rrtstar_internal_validity_checker", ""))
    _record(
        report,
        "rrtstar_uses_static_collision_checker",
        "pass" if checker_name == "static_collision_checker" else "fail",
        checker_name,
    )

    if rrtstar_ompl_adapter.rrtstar_available:
        _record(report, "rrtstar_ompl_import", "pass", "OMPL Python binding imported.")
    else:
        _record(report, "rrtstar_ompl_import", "warn", rrtstar_ompl_adapter.skipped_reason)


def _check_result_schema_if_present(report: dict, output_root: Path) -> None:
    episode_path = output_root / "static_tall_episode_log.csv"
    if not episode_path.exists():
        _record(report, "result_schema_consistency", "warn", "Episode log not present yet; runner will be checked after pilot.")
        return
    with open(episode_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
    if fields == EPISODE_FIELDS:
        _record(report, "result_schema_consistency", "pass", "Episode CSV fields match unified schema.")
    else:
        _record(report, "result_schema_consistency", "fail", f"fields={fields}")


def _write_text_report(path: Path, report: dict) -> None:
    lines = [
        "Static tall benchmark integrity report",
        f"status: {report['status']}",
        f"passed: {report['passed']}",
        f"warnings: {report['warnings']}",
        f"failed: {report['failed']}",
        "",
    ]
    for check in report["checks"]:
        lines.append(f"[{check['status']}] {check['name']}: {check.get('detail', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check static tall benchmark integrity")
    parser.add_argument("--targets-path", default="examples/static_compare/targets/static_tall_targets.json")
    parser.add_argument("--output-root", default="examples/static_compare/results")
    args = parser.parse_args()

    output_root = ensure_dir(resolve_repo_path(args.output_root))
    report = {"status": "unknown", "passed": 0, "warnings": 0, "failed": 0, "checks": []}

    config = _load_config()
    targets_payload = load_json(args.targets_path)
    checker = StaticTallCollisionChecker(include_ground=True)

    _check_no_dynamic_paths(report, config)
    _check_target_set(report, targets_payload, checker)
    _check_method_config(report, config)
    _check_result_schema_if_present(report, output_root)

    report["collision_checker_warnings"] = checker.warnings
    report["status"] = "pass" if report["failed"] == 0 else "fail"

    json_path = output_root / "integrity_report.json"
    txt_path = output_root / "integrity_report.txt"
    write_json(json_path, report)
    _write_text_report(txt_path, report)
    print(f"integrity status={report['status']} failed={report['failed']} warnings={report['warnings']}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
