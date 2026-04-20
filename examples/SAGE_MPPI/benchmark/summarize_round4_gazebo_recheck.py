#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os
from collections import defaultdict


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _as_float(value):
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _as_bool(value):
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def _find_scene_from_path(path):
    for part in os.path.abspath(path).split(os.sep):
        if part.startswith("scene="):
            return part.split("=", 1)[1]
    return "unknown"


def _steps_to_success(step_rows):
    first_success = {}
    for row in step_rows:
        episode_id = row.get("episode_id")
        if episode_id in first_success:
            continue
        if _as_bool(row.get("success")):
            step_id = _as_float(row.get("step_id"))
            if not math.isnan(step_id):
                first_success[episode_id] = step_id + 1.0
    return first_success


def _mean(values):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def _comparison_flag(baseline_value, sage_value, larger_is_better):
    if math.isnan(baseline_value) or math.isnan(sage_value):
        return ""
    return sage_value >= baseline_value if larger_is_better else sage_value <= baseline_value


def summarize(input_root, headless_summary_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(input_root, "round4_gazebo_recheck_metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        import json

        with open(metadata_path) as f:
            metadata = json.load(f)

    grouped = defaultdict(list)
    for dirpath, _, filenames in os.walk(input_root):
        if "episode_metrics.csv" not in filenames:
            continue
        scene = _find_scene_from_path(dirpath)
        episode_rows = _read_csv(os.path.join(dirpath, "episode_metrics.csv"))
        step_rows = _read_csv(os.path.join(dirpath, "step_metrics.csv"))
        step_lookup = _steps_to_success(step_rows)
        for row in episode_rows:
            merged = dict(row)
            merged["scene"] = scene
            merged["steps_to_success"] = step_lookup.get(row["episode_id"], math.nan)
            grouped[(scene, row["controller_name"])].append(merged)

    summary_fields = [
        "scene",
        "controller_name",
        "num_episodes",
        "num_success",
        "num_failure",
        "success_rate",
        "mean_steps_to_success",
        "mean_minimum_safety_margin",
        "mean_final_goal_distance",
    ]
    summary_rows = []
    for (scene, controller_name), rows in sorted(grouped.items()):
        success_flags = [_as_bool(row["success"]) for row in rows]
        failure_flags = [_as_bool(row["failure"]) for row in rows]
        success_steps = [
            _as_float(row["steps_to_success"])
            for row, success in zip(rows, success_flags)
            if success is True
        ]
        summary_rows.append(
            {
                "scene": scene,
                "controller_name": controller_name,
                "num_episodes": len(rows),
                "num_success": sum(1 for x in success_flags if x is True),
                "num_failure": sum(1 for x in failure_flags if x is True),
                "success_rate": _mean([1.0 if x is True else 0.0 for x in success_flags if x is not None]),
                "mean_steps_to_success": _mean(success_steps),
                "mean_minimum_safety_margin": _mean([_as_float(row["minimum_safety_margin"]) for row in rows]),
                "mean_final_goal_distance": _mean([_as_float(row["final_goal_distance"]) for row in rows]),
            }
        )

    summary_csv = os.path.join(output_dir, "gazebo_recheck_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    headless_rows = _read_csv(headless_summary_csv)
    headless = {(row["scene"], row["controller_name"]): row for row in headless_rows}
    gazebo = {(row["scene"], row["controller_name"]): row for row in summary_rows}

    compare_fields = [
        "scene",
        "metric",
        "headless_baseline",
        "headless_sage",
        "gazebo_baseline",
        "gazebo_sage",
        "headless_sage_better",
        "gazebo_sage_better",
        "trend_consistent",
    ]
    compare_rows = []
    metric_specs = [
        ("success_rate", True),
        ("mean_steps_to_success", False),
        ("mean_minimum_safety_margin", True),
        ("mean_final_goal_distance", False),
    ]
    for scene in ("obstacle_hard", "narrow_hard"):
        for metric_name, larger_is_better in metric_specs:
            hb = _as_float(headless[(scene, "baseline")][metric_name])
            hs = _as_float(headless[(scene, "sage")][metric_name])
            gb = _as_float(gazebo[(scene, "baseline")][metric_name])
            gs = _as_float(gazebo[(scene, "sage")][metric_name])
            headless_better = _comparison_flag(hb, hs, larger_is_better)
            gazebo_better = _comparison_flag(gb, gs, larger_is_better)
            trend_consistent = (
                ""
                if headless_better == "" or gazebo_better == ""
                else headless_better == gazebo_better
            )
            compare_rows.append(
                {
                    "scene": scene,
                    "metric": metric_name,
                    "headless_baseline": hb,
                    "headless_sage": hs,
                    "gazebo_baseline": gb,
                    "gazebo_sage": gs,
                    "headless_sage_better": headless_better,
                    "gazebo_sage_better": gazebo_better,
                    "trend_consistent": trend_consistent,
                }
            )

    compare_csv = os.path.join(output_dir, "gazebo_recheck_vs_headless.csv")
    with open(compare_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=compare_fields)
        writer.writeheader()
        for row in compare_rows:
            writer.writerow(row)

    md_path = os.path.join(output_dir, "gazebo_recheck_notes.md")
    with open(md_path, "w") as f:
        f.write("# Round4 Gazebo Recheck Notes\n\n")
        f.write("Small-scale Gazebo recheck on round4 hard scenes using the same pair dataset subset as baseline and SAGE.\n\n")
        if metadata.get("physical_obstacles_spawned") is not None:
            f.write("## Physical obstacle spawn status\n")
            for scene, status in metadata.get("physical_obstacles_spawned", {}).items():
                f.write(f"- {scene}: physical_obstacles_spawned={status}\n")
            f.write("\n")
        f.write("## Trend consistency against headless round4\n")
        for row in compare_rows:
            f.write(
                f"- {row['scene']} / {row['metric']}: "
                f"headless_sage_better={row['headless_sage_better']}, "
                f"gazebo_sage_better={row['gazebo_sage_better']}, "
                f"trend_consistent={row['trend_consistent']}.\n"
            )

    print(summary_csv)
    print(compare_csv)
    print(md_path)
    return summary_csv, compare_csv, md_path


def main():
    parser = argparse.ArgumentParser(description="Summarize round4 Gazebo recheck results")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--headless-summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summarize(args.input_root, args.headless_summary_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
