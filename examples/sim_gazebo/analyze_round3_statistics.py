#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu


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


def _valid(values):
    values = [_as_float(v) if not isinstance(v, (int, float)) else float(v) for v in values]
    return np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)


def _mean_ci(values, rng, num_bootstrap=2000):
    values = _valid(values)
    if values.size == 0:
        return math.nan, math.nan, math.nan
    mean_value = float(np.mean(values))
    if values.size == 1:
        return mean_value, mean_value, mean_value
    samples = rng.choice(values, size=(num_bootstrap, values.size), replace=True)
    boot_means = np.mean(samples, axis=1)
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return mean_value, float(lo), float(hi)


def _wilson_ci(num_success, num_total, z=1.96):
    if num_total <= 0:
        return math.nan, math.nan, math.nan
    p_hat = float(num_success) / float(num_total)
    denom = 1.0 + (z * z) / num_total
    center = (p_hat + (z * z) / (2.0 * num_total)) / denom
    half = (z / denom) * math.sqrt(
        (p_hat * (1.0 - p_hat) / num_total) + (z * z) / (4.0 * num_total * num_total)
    )
    return p_hat, max(0.0, center - half), min(1.0, center + half)


def _scene_groups(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["scene"], row["controller_name"]), []).append(row)
    return grouped


def analyze(merged_csv, output_dir, bootstrap_seed=20260403):
    rows = _read_csv(merged_csv)
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(bootstrap_seed)
    grouped = _scene_groups(rows)

    ci_rows = []
    for (scene, controller_name), group_rows in sorted(grouped.items()):
        success_values = [_as_bool(r["success"]) for r in group_rows]
        num_total = len(group_rows)
        num_success = sum(1 for value in success_values if value is True)
        success_rate, success_ci_lo, success_ci_hi = _wilson_ci(num_success, num_total)

        goal_mean, goal_ci_lo, goal_ci_hi = _mean_ci(
            [r["final_goal_distance"] for r in group_rows],
            rng,
        )
        margin_mean, margin_ci_lo, margin_ci_hi = _mean_ci(
            [r["minimum_safety_margin"] for r in group_rows],
            rng,
        )
        steps_mean, steps_ci_lo, steps_ci_hi = _mean_ci(
            [r["steps_to_success"] for r in group_rows if _as_bool(r["success"]) is True],
            rng,
        )

        ci_rows.append(
            {
                "scene": scene,
                "controller_name": controller_name,
                "num_episodes": num_total,
                "num_success": num_success,
                "num_failure": num_total - num_success,
                "success_rate": success_rate,
                "success_rate_ci_low": success_ci_lo,
                "success_rate_ci_high": success_ci_hi,
                "mean_final_goal_distance": goal_mean,
                "final_goal_distance_ci_low": goal_ci_lo,
                "final_goal_distance_ci_high": goal_ci_hi,
                "mean_minimum_safety_margin": margin_mean,
                "minimum_safety_margin_ci_low": margin_ci_lo,
                "minimum_safety_margin_ci_high": margin_ci_hi,
                "mean_steps_to_success": steps_mean,
                "steps_to_success_ci_low": steps_ci_lo,
                "steps_to_success_ci_high": steps_ci_hi,
            }
        )

    ci_csv = os.path.join(output_dir, "scene_controller_ci_summary.csv")
    ci_fields = [
        "scene",
        "controller_name",
        "num_episodes",
        "num_success",
        "num_failure",
        "success_rate",
        "success_rate_ci_low",
        "success_rate_ci_high",
        "mean_final_goal_distance",
        "final_goal_distance_ci_low",
        "final_goal_distance_ci_high",
        "mean_minimum_safety_margin",
        "minimum_safety_margin_ci_low",
        "minimum_safety_margin_ci_high",
        "mean_steps_to_success",
        "steps_to_success_ci_low",
        "steps_to_success_ci_high",
    ]
    with open(ci_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ci_fields)
        writer.writeheader()
        for row in ci_rows:
            writer.writerow(row)

    stat_rows = []
    scenes = sorted({row["scene"] for row in rows})
    for scene in scenes:
        baseline_rows = grouped.get((scene, "baseline"), [])
        sage_rows = grouped.get((scene, "sage"), [])
        if not baseline_rows or not sage_rows:
            continue

        baseline_success = sum(1 for row in baseline_rows if _as_bool(row["success"]) is True)
        sage_success = sum(1 for row in sage_rows if _as_bool(row["success"]) is True)
        baseline_failure = len(baseline_rows) - baseline_success
        sage_failure = len(sage_rows) - sage_success
        _, success_p = fisher_exact(
            [[baseline_success, baseline_failure], [sage_success, sage_failure]],
            alternative="two-sided",
        )

        metric_specs = [
            ("steps_to_success", "Mann-Whitney U", [r["steps_to_success"] for r in baseline_rows if _as_bool(r["success"]) is True], [r["steps_to_success"] for r in sage_rows if _as_bool(r["success"]) is True]),
            ("minimum_safety_margin", "Mann-Whitney U", [r["minimum_safety_margin"] for r in baseline_rows], [r["minimum_safety_margin"] for r in sage_rows]),
            ("final_goal_distance", "Mann-Whitney U", [r["final_goal_distance"] for r in baseline_rows], [r["final_goal_distance"] for r in sage_rows]),
        ]

        stat_rows.append(
            {
                "scene": scene,
                "metric": "success_rate",
                "test_name": "Fisher exact",
                "baseline_value": float(baseline_success) / float(len(baseline_rows)),
                "sage_value": float(sage_success) / float(len(sage_rows)),
                "delta_sage_minus_baseline": (float(sage_success) / float(len(sage_rows))) - (float(baseline_success) / float(len(baseline_rows))),
                "p_value": float(success_p),
            }
        )

        for metric_name, test_name, baseline_values, sage_values in metric_specs:
            baseline_arr = _valid(baseline_values)
            sage_arr = _valid(sage_values)
            if baseline_arr.size == 0 or sage_arr.size == 0:
                p_value = math.nan
                baseline_mean = math.nan
                sage_mean = math.nan
            else:
                if np.allclose(baseline_arr, baseline_arr[0]) and np.allclose(sage_arr, sage_arr[0]) and np.isclose(baseline_arr[0], sage_arr[0]):
                    p_value = 1.0
                else:
                    _, p_value = mannwhitneyu(baseline_arr, sage_arr, alternative="two-sided")
                baseline_mean = float(np.mean(baseline_arr))
                sage_mean = float(np.mean(sage_arr))

            stat_rows.append(
                {
                    "scene": scene,
                    "metric": metric_name,
                    "test_name": test_name,
                    "baseline_value": baseline_mean,
                    "sage_value": sage_mean,
                    "delta_sage_minus_baseline": sage_mean - baseline_mean if not math.isnan(baseline_mean) and not math.isnan(sage_mean) else math.nan,
                    "p_value": float(p_value) if not math.isnan(p_value) else math.nan,
                }
            )

    stat_csv = os.path.join(output_dir, "scene_pairwise_statistics.csv")
    stat_fields = [
        "scene",
        "metric",
        "test_name",
        "baseline_value",
        "sage_value",
        "delta_sage_minus_baseline",
        "p_value",
    ]
    with open(stat_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stat_fields)
        writer.writeheader()
        for row in stat_rows:
            writer.writerow(row)

    print(ci_csv)
    print(stat_csv)
    return ci_csv, stat_csv


def main():
    parser = argparse.ArgumentParser(description="Analyze round3 benchmark with CI and significance tests")
    parser.add_argument("--merged-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260403)
    args = parser.parse_args()
    analyze(args.merged_csv, args.output_dir, bootstrap_seed=args.bootstrap_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
