#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_pairs(path):
    import json

    with open(path) as f:
        return json.load(f)


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


def _pair_id_from_episode_id(episode_id):
    if episode_id.endswith("_baseline"):
        return episode_id[: -len("_baseline")]
    if episode_id.endswith("_sage"):
        return episode_id[: -len("_sage")]
    return episode_id


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
    arr = np.asarray(
        [_as_float(v) if not isinstance(v, (float, int)) else float(v) for v in values],
        dtype=np.float64,
    )
    return arr[np.isfinite(arr)]


def _mean(values):
    values = _valid(values)
    if values.size == 0:
        return math.nan
    return float(np.mean(values))


def _std(values):
    values = _valid(values)
    if values.size == 0:
        return math.nan
    if values.size == 1:
        return 0.0
    return float(np.std(values))


def _paired_wilcoxon(baseline_values, sage_values):
    baseline = np.asarray(baseline_values, dtype=np.float64)
    sage = np.asarray(sage_values, dtype=np.float64)
    diffs = sage - baseline
    valid_mask = np.isfinite(diffs)
    baseline = baseline[valid_mask]
    sage = sage[valid_mask]
    diffs = diffs[valid_mask]
    if diffs.size == 0:
        return math.nan, math.nan, math.nan, 0
    if np.allclose(diffs, 0.0):
        return 1.0, float(np.mean(diffs)), float(np.median(diffs)), int(diffs.size)
    _, p_value = wilcoxon(baseline, sage, zero_method="wilcox", alternative="two-sided", mode="auto")
    return float(p_value), float(np.mean(diffs)), float(np.median(diffs)), int(diffs.size)


def analyze(input_root, pairs_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pairs = _load_pairs(pairs_path)
    runs = _collect_runs(input_root)

    pair_steps = {
        (scene_name, pair["pair_id"]): int(pair.get("max_steps", pairs["scene_meta"][scene_name]["max_steps"]))
        for scene_name, scene_pairs in pairs["scenes"].items()
        for pair in scene_pairs
    }

    merged_rows = []
    grouped = defaultdict(list)
    paired_by_scene = defaultdict(dict)

    for run in runs:
        episode_rows = _read_csv(run["episode_csv"])
        step_lookup = _steps_to_success(_read_csv(run["step_csv"])) if os.path.exists(run["step_csv"]) else {}
        for row in episode_rows:
            controller_name = row.get("controller_name", "unknown")
            pair_id = _pair_id_from_episode_id(row.get("episode_id", ""))
            max_steps = pair_steps.get((run["scene"], pair_id))
            steps_to_success = step_lookup.get(row.get("episode_id"), math.nan)
            merged = dict(row)
            merged["scene"] = run["scene"]
            merged["pair_id"] = pair_id
            merged["base_scene"] = pairs["scene_meta"][run["scene"]]["base_scene"]
            merged["difficulty"] = pairs["scene_meta"][run["scene"]]["difficulty"]
            merged["max_steps"] = max_steps
            merged["steps_to_success"] = steps_to_success
            merged["steps_to_success_filled"] = (
                steps_to_success
                if not math.isnan(_as_float(steps_to_success))
                else (float(max_steps) + 1.0 if max_steps is not None else math.nan)
            )
            merged_rows.append(merged)
            grouped[(run["scene"], controller_name)].append(merged)
            paired_by_scene[run["scene"]][(pair_id, controller_name)] = merged

    merged_fields = [
        "scene",
        "base_scene",
        "difficulty",
        "controller_name",
        "pair_id",
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
        "steps_to_success_filled",
        "max_steps",
    ]
    merged_csv = os.path.join(output_dir, "episode_metrics_merged.csv")
    with open(merged_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({field: row.get(field, "") for field in merged_fields})

    summary_fields = [
        "scene",
        "base_scene",
        "difficulty",
        "controller_name",
        "num_episodes",
        "num_success",
        "num_failure",
        "success_rate",
        "mean_steps_to_success",
        "std_steps_to_success",
        "mean_minimum_safety_margin",
        "std_minimum_safety_margin",
        "mean_final_goal_distance",
        "std_final_goal_distance",
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
        success_steps = [
            _as_float(row.get("steps_to_success"))
            for row, success in zip(rows, success_flags)
            if success is True
        ]
        summary_rows.append(
            {
                "scene": scene,
                "base_scene": pairs["scene_meta"][scene]["base_scene"],
                "difficulty": pairs["scene_meta"][scene]["difficulty"],
                "controller_name": controller_name,
                "num_episodes": len(rows),
                "num_success": sum(1 for x in success_flags if x is True),
                "num_failure": sum(1 for x in failure_flags if x is True),
                "success_rate": _mean([1.0 if x is True else 0.0 for x in success_flags if x is not None]),
                "mean_steps_to_success": _mean(success_steps),
                "std_steps_to_success": _std(success_steps),
                "mean_minimum_safety_margin": _mean([row.get("minimum_safety_margin") for row in rows]),
                "std_minimum_safety_margin": _std([row.get("minimum_safety_margin") for row in rows]),
                "mean_final_goal_distance": _mean([row.get("final_goal_distance") for row in rows]),
                "std_final_goal_distance": _std([row.get("final_goal_distance") for row in rows]),
                "mean_safe_elite_fraction": _mean([row.get("safe_elite_fraction") for row in rows]),
                "mean_safe_weight_mass": _mean([row.get("safe_weight_mass") for row in rows]),
                "mean_rho_k": _mean([row.get("rho_k") for row in rows]),
                "mean_z_t": _mean([row.get("z_t") for row in rows]),
                "covariance_fallback_rate": _mean(
                    [1.0 if _as_bool(row.get("covariance_fallback")) else 0.0 for row in rows]
                ),
            }
        )

    summary_csv = os.path.join(output_dir, "summary_by_scene_controller.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    paired_fields = [
        "scene",
        "base_scene",
        "difficulty",
        "metric",
        "test_name",
        "metric_definition",
        "num_pairs",
        "baseline_value",
        "sage_value",
        "paired_mean_difference_sage_minus_baseline",
        "paired_median_difference_sage_minus_baseline",
        "p_value",
        "n_both_success",
        "n_both_failure",
        "n_baseline_success_sage_failure",
        "n_baseline_failure_sage_success",
    ]
    paired_rows = []

    for scene in sorted(pairs["scenes"].keys()):
        scene_pairs = pairs["scenes"][scene]
        baseline_rows = []
        sage_rows = []
        for pair in scene_pairs:
            pair_id = pair["pair_id"]
            baseline_row = paired_by_scene[scene].get((pair_id, "baseline"))
            sage_row = paired_by_scene[scene].get((pair_id, "sage"))
            if baseline_row is None or sage_row is None:
                continue
            baseline_rows.append(baseline_row)
            sage_rows.append(sage_row)

        if not baseline_rows or not sage_rows:
            continue

        baseline_success = np.asarray([1 if _as_bool(row["success"]) else 0 for row in baseline_rows], dtype=np.int32)
        sage_success = np.asarray([1 if _as_bool(row["success"]) else 0 for row in sage_rows], dtype=np.int32)
        n_both_success = int(np.sum((baseline_success == 1) & (sage_success == 1)))
        n_both_failure = int(np.sum((baseline_success == 0) & (sage_success == 0)))
        n_baseline_success_sage_failure = int(np.sum((baseline_success == 1) & (sage_success == 0)))
        n_baseline_failure_sage_success = int(np.sum((baseline_success == 0) & (sage_success == 1)))
        discordant = n_baseline_success_sage_failure + n_baseline_failure_sage_success
        if discordant == 0:
            success_p = 1.0
        else:
            success_p = float(
                binomtest(
                    min(n_baseline_success_sage_failure, n_baseline_failure_sage_success),
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )

        paired_rows.append(
            {
                "scene": scene,
                "base_scene": pairs["scene_meta"][scene]["base_scene"],
                "difficulty": pairs["scene_meta"][scene]["difficulty"],
                "metric": "success_rate",
                "test_name": "Exact McNemar via binomial test",
                "metric_definition": "Paired success/failure on identical initial-goal pairs",
                "num_pairs": len(baseline_rows),
                "baseline_value": float(np.mean(baseline_success)),
                "sage_value": float(np.mean(sage_success)),
                "paired_mean_difference_sage_minus_baseline": float(np.mean(sage_success - baseline_success)),
                "paired_median_difference_sage_minus_baseline": float(np.median(sage_success - baseline_success)),
                "p_value": success_p,
                "n_both_success": n_both_success,
                "n_both_failure": n_both_failure,
                "n_baseline_success_sage_failure": n_baseline_success_sage_failure,
                "n_baseline_failure_sage_success": n_baseline_failure_sage_success,
            }
        )

        metric_specs = [
            (
                "steps_to_success",
                "Wilcoxon signed-rank",
                "Paired steps_to_success with failures mapped to max_steps + 1",
                [row["steps_to_success_filled"] for row in baseline_rows],
                [row["steps_to_success_filled"] for row in sage_rows],
            ),
            (
                "minimum_safety_margin",
                "Wilcoxon signed-rank",
                "Paired minimum safety margin over each episode",
                [row["minimum_safety_margin"] for row in baseline_rows],
                [row["minimum_safety_margin"] for row in sage_rows],
            ),
            (
                "final_goal_distance",
                "Wilcoxon signed-rank",
                "Paired final end-effector goal distance at episode termination",
                [row["final_goal_distance"] for row in baseline_rows],
                [row["final_goal_distance"] for row in sage_rows],
            ),
        ]

        for metric_name, test_name, metric_definition, baseline_values, sage_values in metric_specs:
            baseline_arr = np.asarray([_as_float(v) for v in baseline_values], dtype=np.float64)
            sage_arr = np.asarray([_as_float(v) for v in sage_values], dtype=np.float64)
            p_value, mean_diff, median_diff, n_pairs = _paired_wilcoxon(baseline_arr, sage_arr)
            paired_rows.append(
                {
                    "scene": scene,
                    "base_scene": pairs["scene_meta"][scene]["base_scene"],
                    "difficulty": pairs["scene_meta"][scene]["difficulty"],
                    "metric": metric_name,
                    "test_name": test_name,
                    "metric_definition": metric_definition,
                    "num_pairs": n_pairs,
                    "baseline_value": float(np.nanmean(baseline_arr)),
                    "sage_value": float(np.nanmean(sage_arr)),
                    "paired_mean_difference_sage_minus_baseline": mean_diff,
                    "paired_median_difference_sage_minus_baseline": median_diff,
                    "p_value": p_value,
                    "n_both_success": n_both_success,
                    "n_both_failure": n_both_failure,
                    "n_baseline_success_sage_failure": n_baseline_success_sage_failure,
                    "n_baseline_failure_sage_success": n_baseline_failure_sage_success,
                }
            )

    paired_csv = os.path.join(output_dir, "paired_statistics_by_scene.csv")
    with open(paired_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=paired_fields)
        writer.writeheader()
        for row in paired_rows:
            writer.writerow(row)

    print(merged_csv)
    print(summary_csv)
    print(paired_csv)
    return merged_csv, summary_csv, paired_csv


def main():
    parser = argparse.ArgumentParser(description="Analyze harder round4 benchmark with paired statistics")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--pairs-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    analyze(args.input_root, args.pairs_path, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
