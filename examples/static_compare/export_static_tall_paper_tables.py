#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.static_compare.summarize_static_benchmark import (
    PAIRED_OUTCOME_FIELDS,
    PAIRED_SUMMARY_FIELDS,
    PAIRED_TARGET_FIELDS,
    SUMMARY_FIELDS,
    _summarize_group,
    _write_csv,
    _write_summary_csv,
    paired_comparison,
    summarize,
    summarize_by_difficulty,
)
from examples.static_compare.utils.io_utils import ensure_dir, load_json, resolve_repo_path, write_json


PAPER_SUMMARY_FIELDS = [
    "method_name",
    "difficulty_tag",
    "num_episodes",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "mean_final_ee_error",
    "median_final_ee_error",
    "mean_wall_time",
    "median_wall_time",
    "iqr_wall_time",
    "mean_steps_to_goal",
    "median_steps_to_goal",
    "mean_minimum_safety_margin",
    "median_minimum_safety_margin",
    "mean_trajectory_length_ee",
    "median_trajectory_length_ee",
    "mean_trajectory_length_joint",
    "median_trajectory_length_joint",
    "mean_smoothness_jerk",
    "median_smoothness_jerk",
    "rrtstar_exact_rate",
    "rrtstar_approximate_rate",
]

RRT_SWEEP_FIELDS = [
    "planning_time_limit",
    "num_episodes",
    "success_rate",
    "rrtstar_exact_rate",
    "rrtstar_approximate_rate",
    "mean_final_ee_error",
    "median_final_ee_error",
    "mean_wall_time",
    "median_wall_time",
    "mean_planning_time",
    "mean_minimum_safety_margin",
    "median_minimum_safety_margin",
    "mean_trajectory_length_joint",
    "median_trajectory_length_joint",
    "mean_trajectory_length_ee",
    "median_trajectory_length_ee",
]


def _read_episode_log(path: Path, benchmark_seed: str = "") -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if not row.get("benchmark_seed") and benchmark_seed:
            row["benchmark_seed"] = benchmark_seed
    return rows


def _seed_from_run_dir(run_dir: Path) -> str:
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        try:
            return str(load_json(metadata_path).get("seed", ""))
        except Exception:
            pass
    match = re.search(r"seed_(\d+)", run_dir.name)
    return match.group(1) if match else ""


def _load_storm_sage_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for log_path in sorted(results_root.glob("storm_sage_seed_*/static_tall_episode_log.csv")):
        rows.extend(_read_episode_log(log_path, _seed_from_run_dir(log_path.parent)))
    return rows


def _load_rrt_rows(results_root: Path, limit_label: str) -> list[dict]:
    log_path = results_root / f"rrtstar_sweep_{limit_label}" / "static_tall_episode_log.csv"
    if not log_path.exists():
        return []
    return _read_episode_log(log_path)


def _paper_row(row: dict) -> dict:
    return {field: row.get(field, "") for field in PAPER_SUMMARY_FIELDS}


def _write_paper_summary(path: Path, rows: list[dict]) -> None:
    _write_csv(path, [_paper_row(row) for row in rows], PAPER_SUMMARY_FIELDS)


def _rrt_sweep_row(rows: list[dict], limit_label: str) -> dict:
    summary = _summarize_group("rrtstar_ompl", rows, group="overall")
    return {
        "planning_time_limit": limit_label.rstrip("s"),
        "num_episodes": summary["num_episodes"],
        "success_rate": summary["success_rate"],
        "rrtstar_exact_rate": summary["rrtstar_exact_rate"],
        "rrtstar_approximate_rate": summary["rrtstar_approximate_rate"],
        "mean_final_ee_error": summary["mean_final_ee_error"],
        "median_final_ee_error": summary["median_final_ee_error"],
        "mean_wall_time": summary["mean_wall_time"],
        "median_wall_time": summary["median_wall_time"],
        "mean_planning_time": summary["mean_planning_time"],
        "mean_minimum_safety_margin": summary["mean_minimum_safety_margin"],
        "median_minimum_safety_margin": summary["median_minimum_safety_margin"],
        "mean_trajectory_length_joint": summary["mean_trajectory_length_joint"],
        "median_trajectory_length_joint": summary["median_trajectory_length_joint"],
        "mean_trajectory_length_ee": summary["mean_trajectory_length_ee"],
        "median_trajectory_length_ee": summary["median_trajectory_length_ee"],
    }


def _write_minimal_latex(path: Path, rows: list[dict], caption: str) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Episodes & Success & Timeout & Median wall (s) \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row.get("difficulty_tag") != "all":
            continue
        lines.append(
            f"{row['method_name']} & {row['num_episodes']} & "
            f"{float(row['success_rate']):.3f} & {float(row['timeout_rate']):.3f} & "
            f"{float(row['median_wall_time']):.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export paper-facing static tall benchmark v3 tables")
    parser.add_argument("--results-root", default="examples/static_compare/results/formal_static_tall_v3")
    parser.add_argument("--targets-path", default="examples/static_compare/targets/static_tall_targets_formal_v3.json")
    args = parser.parse_args()

    results_root = ensure_dir(resolve_repo_path(args.results_root))
    targets_path = resolve_repo_path(args.targets_path)
    storm_sage_rows = _load_storm_sage_rows(results_root)
    if not storm_sage_rows:
        raise FileNotFoundError(f"No storm_sage_seed_* episode logs found under {results_root}")

    combined_path = results_root / "static_tall_storm_sage_multiseed_episode_log.csv"
    all_fields = sorted({field for row in storm_sage_rows for field in row.keys()})
    _write_csv(combined_path, storm_sage_rows, all_fields)

    storm_sage_summary = summarize(storm_sage_rows)
    storm_sage_by_difficulty = summarize_by_difficulty(storm_sage_rows)
    paired_targets, paired_summary, paired_outcome = paired_comparison(storm_sage_rows)

    rrt_rows_by_limit = {limit: _load_rrt_rows(results_root, limit) for limit in ["5s", "10s", "20s"]}
    rrt5_summary = []
    if rrt_rows_by_limit["5s"]:
        rrt5 = _summarize_group("rrtstar_ompl", rrt_rows_by_limit["5s"], group="overall")
        rrt5["method_name"] = "rrtstar_ompl_5s"
        rrt5_summary.append(rrt5)
    overall_rows = storm_sage_summary + rrt5_summary
    per_difficulty_rows = storm_sage_by_difficulty
    if rrt_rows_by_limit["5s"]:
        for row in summarize_by_difficulty(rrt_rows_by_limit["5s"]):
            row["method_name"] = "rrtstar_ompl_5s"
            per_difficulty_rows.append(row)

    rrt_sweep_rows = [
        _rrt_sweep_row(rows, limit)
        for limit, rows in rrt_rows_by_limit.items()
        if rows
    ]

    _write_summary_csv(results_root / "summary.csv", overall_rows)
    write_json(
        results_root / "summary.json",
        {
            "summary": overall_rows,
            "summary_by_difficulty": per_difficulty_rows,
            "targets_path": str(targets_path),
        },
    )
    _write_summary_csv(results_root / "summary_by_difficulty.csv", per_difficulty_rows)
    write_json(results_root / "summary_by_difficulty.json", {"summary_by_difficulty": per_difficulty_rows})

    _write_paper_summary(results_root / "static_tall_overall_table.csv", overall_rows)
    _write_paper_summary(results_root / "static_tall_per_difficulty_table.csv", per_difficulty_rows)
    _write_csv(results_root / "static_tall_paired_table.csv", paired_summary, PAIRED_SUMMARY_FIELDS)
    _write_csv(results_root / "static_tall_paired_outcome_table.csv", paired_outcome, PAIRED_OUTCOME_FIELDS)
    _write_csv(results_root / "static_tall_paired_by_target_table.csv", paired_targets, PAIRED_TARGET_FIELDS)
    _write_csv(results_root / "static_tall_rrtstar_sweep_table.csv", rrt_sweep_rows, RRT_SWEEP_FIELDS)
    write_json(
        results_root / "static_tall_paper_tables.json",
        {
            "overall": overall_rows,
            "per_difficulty": per_difficulty_rows,
            "paired": paired_summary,
            "paired_outcome": paired_outcome,
            "rrtstar_sweep": rrt_sweep_rows,
        },
    )
    _write_minimal_latex(results_root / "static_tall_overall_table.tex", overall_rows, "Static tall benchmark overall results.")
    print(f"wrote paper tables to {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
