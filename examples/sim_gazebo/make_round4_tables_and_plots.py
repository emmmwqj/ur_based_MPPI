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


SCENE_ORDER = ["obstacle_medium", "obstacle_hard", "narrow_medium", "narrow_hard"]
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


def _group_summary(rows):
    grouped = {}
    for row in rows:
        grouped[(row["scene"], row["controller_name"])] = row
    return grouped


def _group_stats(rows):
    grouped = {}
    for row in rows:
        grouped[(row["scene"], row["metric"])] = row
    return grouped


def _plot_metric(summary_rows, output_dir, metric_key, ylabel, filename):
    summary_lookup = _group_summary(summary_rows)
    x = np.arange(len(SCENE_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.2, 4.4))

    for idx, controller_name in enumerate(CONTROLLER_ORDER):
        values = []
        for scene in SCENE_ORDER:
            values.append(_as_float(summary_lookup[(scene, controller_name)][metric_key]))
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[controller_name],
            label=controller_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SCENE_ORDER, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for idx, scene in enumerate(SCENE_ORDER):
        if scene.endswith("_hard"):
            ax.axvspan(idx - 0.5, idx + 0.5, color="#f5f5f4", zorder=0)
    fig.tight_layout()
    png_path = os.path.join(output_dir, f"{filename}.png")
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def make_tables_and_plots(summary_csv, paired_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = _read_csv(summary_csv)
    paired_rows = _read_csv(paired_csv)
    summary_lookup = _group_summary(summary_rows)
    paired_lookup = _group_stats(paired_rows)

    table_fields = [
        "scene",
        "base_scene",
        "difficulty",
        "controller_name",
        "num_episodes",
        "success_rate",
        "mean_steps_to_success",
        "mean_minimum_safety_margin",
        "mean_final_goal_distance",
        "paired_p_success_rate",
        "paired_p_steps_to_success",
        "paired_p_minimum_safety_margin",
        "paired_p_final_goal_distance",
        "paired_delta_success_rate",
        "paired_delta_steps_to_success",
        "paired_delta_minimum_safety_margin",
        "paired_delta_final_goal_distance",
    ]

    table_rows = []
    for scene in SCENE_ORDER:
        for controller_name in CONTROLLER_ORDER:
            summary_row = summary_lookup[(scene, controller_name)]
            table_rows.append(
                {
                    "scene": scene,
                    "base_scene": summary_row["base_scene"],
                    "difficulty": summary_row["difficulty"],
                    "controller_name": controller_name,
                    "num_episodes": summary_row["num_episodes"],
                    "success_rate": summary_row["success_rate"],
                    "mean_steps_to_success": summary_row["mean_steps_to_success"],
                    "mean_minimum_safety_margin": summary_row["mean_minimum_safety_margin"],
                    "mean_final_goal_distance": summary_row["mean_final_goal_distance"],
                    "paired_p_success_rate": paired_lookup[(scene, "success_rate")]["p_value"],
                    "paired_p_steps_to_success": paired_lookup[(scene, "steps_to_success")]["p_value"],
                    "paired_p_minimum_safety_margin": paired_lookup[(scene, "minimum_safety_margin")]["p_value"],
                    "paired_p_final_goal_distance": paired_lookup[(scene, "final_goal_distance")]["p_value"],
                    "paired_delta_success_rate": paired_lookup[(scene, "success_rate")]["paired_mean_difference_sage_minus_baseline"],
                    "paired_delta_steps_to_success": paired_lookup[(scene, "steps_to_success")]["paired_mean_difference_sage_minus_baseline"],
                    "paired_delta_minimum_safety_margin": paired_lookup[(scene, "minimum_safety_margin")]["paired_mean_difference_sage_minus_baseline"],
                    "paired_delta_final_goal_distance": paired_lookup[(scene, "final_goal_distance")]["paired_mean_difference_sage_minus_baseline"],
                }
            )

    table_csv = os.path.join(output_dir, "paper_round4_harder_table.csv")
    with open(table_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table_fields)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    generated = [table_csv]
    generated.extend(
        _plot_metric(
            summary_rows,
            output_dir,
            metric_key="success_rate",
            ylabel="Success Rate",
            filename="round4_success_rate_by_scene",
        )
    )
    generated.extend(
        _plot_metric(
            summary_rows,
            output_dir,
            metric_key="mean_steps_to_success",
            ylabel="Mean Steps To Success",
            filename="round4_steps_to_success_by_scene",
        )
    )
    generated.extend(
        _plot_metric(
            summary_rows,
            output_dir,
            metric_key="mean_minimum_safety_margin",
            ylabel="Mean Minimum Safety Margin",
            filename="round4_minimum_safety_margin_by_scene",
        )
    )
    for path in generated:
        print(path)
    return generated


def main():
    parser = argparse.ArgumentParser(description="Create round4 harder-scene tables and plots")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--paired-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    make_tables_and_plots(args.summary_csv, args.paired_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
