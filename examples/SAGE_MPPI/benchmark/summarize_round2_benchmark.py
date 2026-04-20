#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os
from collections import defaultdict
from statistics import mean, pstdev


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


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _collect_runs(input_root):
    runs = []
    for dirpath, _, filenames in os.walk(input_root):
        if "episode_metrics.csv" not in filenames:
            continue
        runs.append(
            {
                "scene": _find_scene_from_path(dirpath),
                "episode_csv": os.path.join(dirpath, "episode_metrics.csv"),
                "step_csv": os.path.join(dirpath, "step_metrics.csv"),
            }
        )
    return sorted(runs, key=lambda item: (item["scene"], item["episode_csv"]))


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


def _valid(values):
    return [value for value in values if not math.isnan(value)]


def _mean(values):
    values = _valid(values)
    if not values:
        return math.nan
    return float(mean(values))


def _std(values):
    values = _valid(values)
    if not values:
        return math.nan
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))


def _rate(values):
    mapped = []
    for value in values:
        parsed = _as_bool(value)
        if parsed is None:
            continue
        mapped.append(1.0 if parsed else 0.0)
    if not mapped:
        return math.nan
    return float(mean(mapped))


def summarize(input_root, output_dir):
    runs = _collect_runs(input_root)
    os.makedirs(output_dir, exist_ok=True)

    merged_rows = []
    grouped = defaultdict(list)

    for run in runs:
        episode_rows = _read_csv(run["episode_csv"])
        step_lookup = _steps_to_success(_read_csv(run["step_csv"])) if os.path.exists(run["step_csv"]) else {}
        for row in episode_rows:
            merged = dict(row)
            merged["scene"] = run["scene"]
            merged["steps_to_success"] = step_lookup.get(row.get("episode_id"), math.nan)
            merged_rows.append(merged)
            grouped[(run["scene"], row.get("controller_name", "unknown"))].append(merged)

    merged_fields = [
        "scene",
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
        "steps_to_success",
    ]
    merged_csv = os.path.join(output_dir, "episode_metrics_merged.csv")
    with open(merged_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({field: row.get(field, "") for field in merged_fields})

    summary_fields = [
        "scene",
        "controller_name",
        "num_episodes",
        "num_success",
        "num_failure",
        "success_rate",
        "mean_final_goal_distance",
        "std_final_goal_distance",
        "mean_minimum_safety_margin",
        "std_minimum_safety_margin",
        "mean_steps_to_success",
        "mean_safe_elite_fraction",
        "mean_safe_weight_mass",
        "mean_rho_k",
        "mean_z_t",
        "covariance_fallback_rate",
    ]
    summary_rows = []

    for (scene, controller_name), rows in sorted(grouped.items()):
        success_flags = [_as_bool(row.get("success")) for row in rows]
        failure_flags = [_as_bool(row.get("failure")) for row in rows]
        num_success = sum(1 for value in success_flags if value is True)
        num_failure = sum(1 for value in failure_flags if value is True)
        success_only_steps = [
            _as_float(row.get("steps_to_success"))
            for row, success in zip(rows, success_flags)
            if success is True
        ]

        summary_rows.append(
            {
                "scene": scene,
                "controller_name": controller_name,
                "num_episodes": len(rows),
                "num_success": num_success,
                "num_failure": num_failure,
                "success_rate": float(num_success) / float(len(rows)) if rows else math.nan,
                "mean_final_goal_distance": _mean([_as_float(row.get("final_goal_distance")) for row in rows]),
                "std_final_goal_distance": _std([_as_float(row.get("final_goal_distance")) for row in rows]),
                "mean_minimum_safety_margin": _mean([_as_float(row.get("minimum_safety_margin")) for row in rows]),
                "std_minimum_safety_margin": _std([_as_float(row.get("minimum_safety_margin")) for row in rows]),
                "mean_steps_to_success": _mean(success_only_steps),
                "mean_safe_elite_fraction": _mean([_as_float(row.get("safe_elite_fraction")) for row in rows]),
                "mean_safe_weight_mass": _mean([_as_float(row.get("safe_weight_mass")) for row in rows]),
                "mean_rho_k": _mean([_as_float(row.get("rho_k")) for row in rows]),
                "mean_z_t": _mean([_as_float(row.get("z_t")) for row in rows]),
                "covariance_fallback_rate": _rate([row.get("covariance_fallback") for row in rows]),
            }
        )

    summary_csv = os.path.join(output_dir, "summary_by_scene_controller.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(merged_csv)
    print(summary_csv)
    return merged_csv, summary_csv


def main():
    parser = argparse.ArgumentParser(description="Summarize round2 benchmark experiment CSV files")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summarize(args.input_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
