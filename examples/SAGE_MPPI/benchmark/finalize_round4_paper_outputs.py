#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import shutil


SCENE_ORDER = ["obstacle_medium", "obstacle_hard", "narrow_medium", "narrow_hard"]
METRIC_ORDER = [
    "success_rate",
    "steps_to_success",
    "minimum_safety_margin",
    "final_goal_distance",
]


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _summary_lookup(rows):
    return {(row["scene"], row["controller_name"]): row for row in rows}


def _paired_lookup(rows):
    return {(row["scene"], row["metric"]): row for row in rows}


def finalize(summary_csv, paired_csv, paper_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = _read_csv(summary_csv)
    paired_rows = _read_csv(paired_csv)
    summary = _summary_lookup(summary_rows)
    paired = _paired_lookup(paired_rows)

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
        "paired_test_success_rate",
        "paired_p_success_rate",
        "paired_test_steps_to_success",
        "paired_p_steps_to_success",
        "paired_test_minimum_safety_margin",
        "paired_p_minimum_safety_margin",
        "paired_test_final_goal_distance",
        "paired_p_final_goal_distance",
        "paired_delta_success_rate",
        "paired_delta_steps_to_success",
        "paired_delta_minimum_safety_margin",
        "paired_delta_final_goal_distance",
    ]

    table_rows = []
    for scene in SCENE_ORDER:
        for controller_name in ("baseline", "sage"):
            srow = summary[(scene, controller_name)]
            table_rows.append(
                {
                    "scene": scene,
                    "base_scene": srow["base_scene"],
                    "difficulty": srow["difficulty"],
                    "controller_name": controller_name,
                    "num_episodes": srow["num_episodes"],
                    "success_rate": srow["success_rate"],
                    "mean_steps_to_success": srow["mean_steps_to_success"],
                    "mean_minimum_safety_margin": srow["mean_minimum_safety_margin"],
                    "mean_final_goal_distance": srow["mean_final_goal_distance"],
                    "paired_test_success_rate": paired[(scene, "success_rate")]["test_name"],
                    "paired_p_success_rate": paired[(scene, "success_rate")]["p_value"],
                    "paired_test_steps_to_success": paired[(scene, "steps_to_success")]["test_name"],
                    "paired_p_steps_to_success": paired[(scene, "steps_to_success")]["p_value"],
                    "paired_test_minimum_safety_margin": paired[(scene, "minimum_safety_margin")]["test_name"],
                    "paired_p_minimum_safety_margin": paired[(scene, "minimum_safety_margin")]["p_value"],
                    "paired_test_final_goal_distance": paired[(scene, "final_goal_distance")]["test_name"],
                    "paired_p_final_goal_distance": paired[(scene, "final_goal_distance")]["p_value"],
                    "paired_delta_success_rate": paired[(scene, "success_rate")]["paired_mean_difference_sage_minus_baseline"],
                    "paired_delta_steps_to_success": paired[(scene, "steps_to_success")]["paired_mean_difference_sage_minus_baseline"],
                    "paired_delta_minimum_safety_margin": paired[(scene, "minimum_safety_margin")]["paired_mean_difference_sage_minus_baseline"],
                    "paired_delta_final_goal_distance": paired[(scene, "final_goal_distance")]["paired_mean_difference_sage_minus_baseline"],
                }
            )

    final_table_csv = os.path.join(output_dir, "round4_paper_main_table_final.csv")
    with open(final_table_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table_fields)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    notes_path = os.path.join(output_dir, "round4_paper_statistics_notes.md")
    with open(notes_path, "w") as f:
        f.write("# Round4 Harder Benchmark Statistical Notes\n\n")
        f.write("## Statistical tests\n")
        f.write("- Success/failure: Exact McNemar via binomial test on paired outcomes.\n")
        f.write("- Steps to success: Wilcoxon signed-rank on paired episode values; failures are mapped to `max_steps + 1`.\n")
        f.write("- Minimum safety margin: Wilcoxon signed-rank on paired episode minima.\n")
        f.write("- Final goal distance: Wilcoxon signed-rank on paired terminal distances.\n\n")
        f.write("## Scene-wise summary\n")
        for scene in SCENE_ORDER:
            success = paired[(scene, "success_rate")]
            steps = paired[(scene, "steps_to_success")]
            margin = paired[(scene, "minimum_safety_margin")]
            goal = paired[(scene, "final_goal_distance")]
            f.write(f"### {scene}\n")
            f.write(
                f"- Success rate: baseline={success['baseline_value']}, sage={success['sage_value']}, "
                f"paired mean diff={success['paired_mean_difference_sage_minus_baseline']}, p={success['p_value']}.\n"
            )
            f.write(
                f"- Steps to success: baseline={steps['baseline_value']}, sage={steps['sage_value']}, "
                f"paired mean diff={steps['paired_mean_difference_sage_minus_baseline']}, p={steps['p_value']}.\n"
            )
            f.write(
                f"- Minimum safety margin: baseline={margin['baseline_value']}, sage={margin['sage_value']}, "
                f"paired mean diff={margin['paired_mean_difference_sage_minus_baseline']}, p={margin['p_value']}.\n"
            )
            f.write(
                f"- Final goal distance: baseline={goal['baseline_value']}, sage={goal['sage_value']}, "
                f"paired mean diff={goal['paired_mean_difference_sage_minus_baseline']}, p={goal['p_value']}.\n\n"
            )

    copied = [final_table_csv, notes_path]
    rename_map = {
        "round4_success_rate_by_scene.png": "round4_paper_figure_success_rate.png",
        "round4_success_rate_by_scene.pdf": "round4_paper_figure_success_rate.pdf",
        "round4_steps_to_success_by_scene.png": "round4_paper_figure_steps_to_success.png",
        "round4_steps_to_success_by_scene.pdf": "round4_paper_figure_steps_to_success.pdf",
        "round4_minimum_safety_margin_by_scene.png": "round4_paper_figure_minimum_safety_margin.png",
        "round4_minimum_safety_margin_by_scene.pdf": "round4_paper_figure_minimum_safety_margin.pdf",
    }
    for src_name, dst_name in rename_map.items():
        src = os.path.join(paper_dir, src_name)
        dst = os.path.join(output_dir, dst_name)
        shutil.copy2(src, dst)
        copied.append(dst)

    for path in copied:
        print(path)
    return copied


def main():
    parser = argparse.ArgumentParser(description="Finalize round4 paper outputs")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--paired-csv", required=True)
    parser.add_argument("--paper-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    finalize(args.summary_csv, args.paired_csv, args.paper_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
