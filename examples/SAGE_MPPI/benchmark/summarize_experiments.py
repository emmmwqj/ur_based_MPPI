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
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def _find_scene_from_path(path):
    parts = os.path.abspath(path).split(os.sep)
    for part in parts:
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
        episode_csv = os.path.join(dirpath, "episode_metrics.csv")
        step_csv = os.path.join(dirpath, "step_metrics.csv")
        scene = _find_scene_from_path(dirpath)
        runs.append(
            {
                "scene": scene,
                "episode_csv": episode_csv,
                "step_csv": step_csv if os.path.exists(step_csv) else None,
            }
        )
    return sorted(runs, key=lambda item: (item["scene"], item["episode_csv"]))


def _collect_steps_to_success(step_rows):
    by_episode = {}
    for row in step_rows:
        episode_id = row.get("episode_id")
        if episode_id not in by_episode:
            by_episode[episode_id] = math.nan
        success = _as_bool(row.get("success"))
        if success and math.isnan(by_episode[episode_id]):
            step_id = _as_float(row.get("step_id"))
            if not math.isnan(step_id):
                by_episode[episode_id] = step_id + 1.0
    return by_episode


def _safe_mean(values):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return math.nan
    return float(mean(values))


def _safe_std(values):
    values = [v for v in values if not math.isnan(v)]
    if len(values) < 2:
        return 0.0 if values else math.nan
    return float(pstdev(values))


def _safe_rate_from_bool(values):
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
        step_to_success = {}
        if run["step_csv"] is not None:
            step_to_success = _collect_steps_to_success(_read_csv(run["step_csv"]))

        for row in episode_rows:
            merged = dict(row)
            merged["scene"] = run["scene"]
            merged["steps_to_success"] = step_to_success.get(row.get("episode_id"), math.nan)
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
        "success_rate",
        "mean_final_goal_distance",
        "std_final_goal_distance",
        "mean_minimum_safety_margin",
        "std_minimum_safety_margin",
        "mean_steps_to_success",
        "mean_failure_rate",
        "mean_safe_elite_fraction",
        "mean_safe_weight_mass",
        "mean_rho_k",
        "mean_z_t",
        "covariance_fallback_rate",
    ]
    summary_rows = []

    for (scene, controller_name), rows in sorted(grouped.items()):
        final_goal_distance = [_as_float(row.get("final_goal_distance")) for row in rows]
        minimum_safety_margin = [_as_float(row.get("minimum_safety_margin")) for row in rows]
        steps_to_success = [_as_float(row.get("steps_to_success")) for row in rows]
        safe_elite_fraction = [_as_float(row.get("safe_elite_fraction")) for row in rows]
        safe_weight_mass = [_as_float(row.get("safe_weight_mass")) for row in rows]
        rho_k = [_as_float(row.get("rho_k")) for row in rows]
        z_t = [_as_float(row.get("z_t")) for row in rows]

        summary_rows.append(
            {
                "scene": scene,
                "controller_name": controller_name,
                "num_episodes": len(rows),
                "success_rate": _safe_rate_from_bool([row.get("success") for row in rows]),
                "mean_final_goal_distance": _safe_mean(final_goal_distance),
                "std_final_goal_distance": _safe_std(final_goal_distance),
                "mean_minimum_safety_margin": _safe_mean(minimum_safety_margin),
                "std_minimum_safety_margin": _safe_std(minimum_safety_margin),
                "mean_steps_to_success": _safe_mean(steps_to_success),
                "mean_failure_rate": _safe_rate_from_bool([row.get("failure") for row in rows]),
                "mean_safe_elite_fraction": _safe_mean(safe_elite_fraction),
                "mean_safe_weight_mass": _safe_mean(safe_weight_mass),
                "mean_rho_k": _safe_mean(rho_k),
                "mean_z_t": _safe_mean(z_t),
                "covariance_fallback_rate": _safe_rate_from_bool(
                    [row.get("covariance_fallback") for row in rows]
                ),
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
    parser = argparse.ArgumentParser(description="Summarize batch-run experiment CSV files")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summarize(args.input_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
