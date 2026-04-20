#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCENE_ORDER = ["easy", "obstacle", "narrow"]
CONTROLLER_ORDER = ["baseline", "sage"]
COLORS = {"baseline": "#6b7280", "sage": "#c2410c"}


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


def _fmt_mean_ci(mean_value, ci_low, ci_high, scale=1.0, digits=3):
    if any(math.isnan(x) for x in (mean_value, ci_low, ci_high)):
        return "nan"
    mean_scaled = mean_value * scale
    lo_err = mean_scaled - (ci_low * scale)
    hi_err = (ci_high * scale) - mean_scaled
    return f"{mean_scaled:.{digits}f} [{mean_scaled - lo_err:.{digits}f}, {mean_scaled + hi_err:.{digits}f}]"


def _group_ci(rows):
    grouped = {}
    for row in rows:
        grouped[(row["scene"], row["controller_name"])] = row
    return grouped


def _group_stats(rows):
    grouped = {}
    for row in rows:
        grouped[(row["scene"], row["metric"])] = row
    return grouped


def _plot_metric(ci_rows, output_dir, metric_key, ci_low_key, ci_high_key, ylabel, filename):
    ci_lookup = _group_ci(ci_rows)
    x = np.arange(len(SCENE_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for idx, controller_name in enumerate(CONTROLLER_ORDER):
        values = []
        lower = []
        upper = []
        for scene in SCENE_ORDER:
            row = ci_lookup[(scene, controller_name)]
            mean_value = _as_float(row[metric_key])
            ci_low = _as_float(row[ci_low_key])
            ci_high = _as_float(row[ci_high_key])
            values.append(mean_value)
            lower.append(mean_value - ci_low if not any(math.isnan(v) for v in (mean_value, ci_low)) else 0.0)
            upper.append(ci_high - mean_value if not any(math.isnan(v) for v in (mean_value, ci_high)) else 0.0)
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[controller_name],
            label=controller_name,
            yerr=np.asarray([lower, upper]),
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SCENE_ORDER)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    png_path = os.path.join(output_dir, f"{filename}.png")
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def make_tables_and_plots(ci_csv, stat_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ci_rows = _read_csv(ci_csv)
    stat_rows = _read_csv(stat_csv)
    ci_lookup = _group_ci(ci_rows)
    stat_lookup = _group_stats(stat_rows)

    table_rows = []
    for scene in SCENE_ORDER:
        for controller_name in CONTROLLER_ORDER:
            ci_row = ci_lookup[(scene, controller_name)]
            table_rows.append(
                {
                    "scene": scene,
                    "controller_name": controller_name,
                    "success_rate": _fmt_mean_ci(
                        _as_float(ci_row["success_rate"]),
                        _as_float(ci_row["success_rate_ci_low"]),
                        _as_float(ci_row["success_rate_ci_high"]),
                        scale=100.0,
                        digits=1,
                    ),
                    "mean_steps_to_success": _fmt_mean_ci(
                        _as_float(ci_row["mean_steps_to_success"]),
                        _as_float(ci_row["steps_to_success_ci_low"]),
                        _as_float(ci_row["steps_to_success_ci_high"]),
                        digits=2,
                    ),
                    "mean_minimum_safety_margin": _fmt_mean_ci(
                        _as_float(ci_row["mean_minimum_safety_margin"]),
                        _as_float(ci_row["minimum_safety_margin_ci_low"]),
                        _as_float(ci_row["minimum_safety_margin_ci_high"]),
                        digits=4,
                    ),
                    "mean_final_goal_distance": _fmt_mean_ci(
                        _as_float(ci_row["mean_final_goal_distance"]),
                        _as_float(ci_row["final_goal_distance_ci_low"]),
                        _as_float(ci_row["final_goal_distance_ci_high"]),
                        digits=4,
                    ),
                    "p_success_rate": stat_lookup[(scene, "success_rate")]["p_value"],
                    "p_steps_to_success": stat_lookup[(scene, "steps_to_success")]["p_value"],
                    "p_minimum_safety_margin": stat_lookup[(scene, "minimum_safety_margin")]["p_value"],
                    "p_final_goal_distance": stat_lookup[(scene, "final_goal_distance")]["p_value"],
                }
            )

    table_csv = os.path.join(output_dir, "paper_main_table.csv")
    table_fields = [
        "scene",
        "controller_name",
        "success_rate",
        "mean_steps_to_success",
        "mean_minimum_safety_margin",
        "mean_final_goal_distance",
        "p_success_rate",
        "p_steps_to_success",
        "p_minimum_safety_margin",
        "p_final_goal_distance",
    ]
    with open(table_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table_fields)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    generated = [table_csv]
    generated.extend(
        _plot_metric(
            ci_rows,
            output_dir,
            metric_key="success_rate",
            ci_low_key="success_rate_ci_low",
            ci_high_key="success_rate_ci_high",
            ylabel="Success Rate",
            filename="success_rate_by_scene",
        )
    )
    generated.extend(
        _plot_metric(
            ci_rows,
            output_dir,
            metric_key="mean_steps_to_success",
            ci_low_key="steps_to_success_ci_low",
            ci_high_key="steps_to_success_ci_high",
            ylabel="Steps To Success",
            filename="steps_to_success_by_scene",
        )
    )
    generated.extend(
        _plot_metric(
            ci_rows,
            output_dir,
            metric_key="mean_minimum_safety_margin",
            ci_low_key="minimum_safety_margin_ci_low",
            ci_high_key="minimum_safety_margin_ci_high",
            ylabel="Minimum Safety Margin",
            filename="minimum_safety_margin_by_scene",
        )
    )
    for path in generated:
        print(path)
    return generated


def main():
    parser = argparse.ArgumentParser(description="Create round3 paper-ready tables and plots")
    parser.add_argument("--ci-csv", required=True)
    parser.add_argument("--stat-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    make_tables_and_plots(args.ci_csv, args.stat_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
