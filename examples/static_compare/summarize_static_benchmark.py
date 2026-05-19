#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.static_compare.utils.io_utils import ensure_dir, load_json, resolve_repo_path, write_json


SUMMARY_FIELDS = [
    "group",
    "difficulty_tag",
    "method_name",
    "num_episodes",
    "status",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "mean_final_ee_error",
    "std_final_ee_error",
    "mean_final_joint_error",
    "mean_minimum_safety_margin",
    "std_minimum_safety_margin",
    "mean_steps_to_goal",
    "mean_wall_time",
    "mean_planning_time",
    "mean_trajectory_length_joint",
    "mean_trajectory_length_ee",
    "mean_smoothness_jerk",
    "rrtstar_exact_rate",
    "rrtstar_approximate_rate",
    "skipped_reason",
]

PAIRED_TARGET_FIELDS = [
    "target_id",
    "difficulty_tag",
    "storm_success",
    "sage_success",
    "wall_time_storm",
    "wall_time_sage",
    "faster_winner",
    "final_ee_error_storm",
    "final_ee_error_sage",
    "final_error_winner",
    "minimum_safety_margin_storm",
    "minimum_safety_margin_sage",
    "minimum_margin_winner",
    "trajectory_length_ee_storm",
    "trajectory_length_ee_sage",
    "trajectory_length_ee_winner",
    "trajectory_length_joint_storm",
    "trajectory_length_joint_sage",
    "trajectory_length_joint_winner",
]

PAIRED_SUMMARY_FIELDS = [
    "difficulty_tag",
    "metric",
    "num_pairs",
    "sage_wins",
    "ties",
    "storm_wins",
    "sage_win_rate",
    "tie_rate",
    "storm_win_rate",
]


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_float(value: str) -> float:
    try:
        parsed = float(value)
        return parsed
    except Exception:
        return math.nan


def _mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(statistics.mean(vals)) if vals else math.nan


def _std(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(statistics.pstdev(vals)) if len(vals) > 1 else math.nan


def _rate(rows: list[dict], field: str) -> float:
    if not rows:
        return math.nan
    return sum(1 for row in rows if _parse_bool(row.get(field, ""))) / float(len(rows))


def _target_difficulty_map(targets_path: str | None) -> dict[str, str]:
    if not targets_path:
        return {}
    payload = load_json(targets_path)
    return {str(target["target_id"]): str(target.get("difficulty_tag", "")) for target in payload.get("targets", [])}


def _with_difficulty(episode_rows: list[dict], difficulty_by_target: dict[str, str]) -> list[dict]:
    rows = []
    for row in episode_rows:
        copied = dict(row)
        copied["difficulty_tag"] = copied.get("difficulty_tag") or difficulty_by_target.get(copied.get("target_id", ""), "")
        rows.append(copied)
    return rows


def _summarize_group(method: str, rows: list[dict], group: str, difficulty_tag: str = "all") -> dict:
    skipped_reasons = sorted({row.get("skipped_reason", "") for row in rows if row.get("skipped_reason", "")})
    all_rrt_unavailable = (
        method in {"rrtstar", "rrtstar_ompl"}
        and rows
        and all(str(row.get("rrtstar_available", "")).strip().lower() == "false" for row in rows)
    )
    status = "unavailable" if all_rrt_unavailable else ("skipped" if skipped_reasons and not any(_parse_bool(r.get("success", "")) for r in rows) else "ok")
    return {
        "group": group,
        "difficulty_tag": difficulty_tag,
        "method_name": method,
        "num_episodes": len(rows),
        "status": status,
        "success_rate": _rate(rows, "success"),
        "collision_rate": _rate(rows, "collision"),
        "timeout_rate": _rate(rows, "timeout"),
        "mean_final_ee_error": _mean([_parse_float(r.get("final_ee_error", "")) for r in rows]),
        "std_final_ee_error": _std([_parse_float(r.get("final_ee_error", "")) for r in rows]),
        "mean_final_joint_error": _mean([_parse_float(r.get("final_joint_error", "")) for r in rows]),
        "mean_minimum_safety_margin": _mean([_parse_float(r.get("minimum_safety_margin", "")) for r in rows]),
        "std_minimum_safety_margin": _std([_parse_float(r.get("minimum_safety_margin", "")) for r in rows]),
        "mean_steps_to_goal": _mean([_parse_float(r.get("steps_to_goal", "")) for r in rows]),
        "mean_wall_time": _mean([_parse_float(r.get("wall_time", "")) for r in rows]),
        "mean_planning_time": _mean([_parse_float(r.get("planning_time", "")) for r in rows]),
        "mean_trajectory_length_joint": _mean([_parse_float(r.get("trajectory_length_joint", "")) for r in rows]),
        "mean_trajectory_length_ee": _mean([_parse_float(r.get("trajectory_length_ee", "")) for r in rows]),
        "mean_smoothness_jerk": _mean([_parse_float(r.get("smoothness_jerk", "")) for r in rows]),
        "rrtstar_exact_rate": _rate(rows, "rrtstar_exact_solution") if method in {"rrtstar", "rrtstar_ompl"} else math.nan,
        "rrtstar_approximate_rate": _rate(rows, "rrtstar_approximate_solution") if method in {"rrtstar", "rrtstar_ompl"} else math.nan,
        "skipped_reason": " | ".join(skipped_reasons),
    }


def summarize(episode_rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in episode_rows:
        groups[row["method_name"]].append(row)

    summary_rows = []
    for method, rows in sorted(groups.items()):
        summary_rows.append(_summarize_group(method, rows, group="overall"))
    return summary_rows


def summarize_by_difficulty(episode_rows: list[dict]) -> list[dict]:
    nested: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in episode_rows:
        nested[(row["method_name"], row.get("difficulty_tag", "") or "unknown")].append(row)
    return [
        _summarize_group(method, rows, group="difficulty", difficulty_tag=difficulty)
        for (method, difficulty), rows in sorted(nested.items())
    ]


def _winner(smaller_is_better: bool, storm_value: float, sage_value: float, tolerance: float) -> str:
    if math.isnan(storm_value) or math.isnan(sage_value):
        return "tie"
    delta = sage_value - storm_value
    if abs(delta) <= tolerance:
        return "tie"
    if smaller_is_better:
        return "sage" if delta < 0 else "storm"
    return "sage" if delta > 0 else "storm"


def paired_comparison(episode_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_method_target = {(row["method_name"], row["target_id"]): row for row in episode_rows}
    target_ids = sorted(
        {row["target_id"] for row in episode_rows if row["method_name"] == "storm_mppi_tuned"}
        & {row["target_id"] for row in episode_rows if row["method_name"] == "sage_mppi_tuned"}
    )
    target_rows = []
    for target_id in target_ids:
        storm = by_method_target[("storm_mppi_tuned", target_id)]
        sage = by_method_target[("sage_mppi_tuned", target_id)]
        row = {
            "target_id": target_id,
            "difficulty_tag": storm.get("difficulty_tag") or sage.get("difficulty_tag") or "unknown",
            "storm_success": storm.get("success", ""),
            "sage_success": sage.get("success", ""),
            "wall_time_storm": _parse_float(storm.get("wall_time", "")),
            "wall_time_sage": _parse_float(sage.get("wall_time", "")),
            "final_ee_error_storm": _parse_float(storm.get("final_ee_error", "")),
            "final_ee_error_sage": _parse_float(sage.get("final_ee_error", "")),
            "minimum_safety_margin_storm": _parse_float(storm.get("minimum_safety_margin", "")),
            "minimum_safety_margin_sage": _parse_float(sage.get("minimum_safety_margin", "")),
            "trajectory_length_ee_storm": _parse_float(storm.get("trajectory_length_ee", "")),
            "trajectory_length_ee_sage": _parse_float(sage.get("trajectory_length_ee", "")),
            "trajectory_length_joint_storm": _parse_float(storm.get("trajectory_length_joint", "")),
            "trajectory_length_joint_sage": _parse_float(sage.get("trajectory_length_joint", "")),
        }
        row["faster_winner"] = _winner(True, row["wall_time_storm"], row["wall_time_sage"], 0.05)
        row["final_error_winner"] = _winner(True, row["final_ee_error_storm"], row["final_ee_error_sage"], 1.0e-4)
        row["minimum_margin_winner"] = _winner(False, row["minimum_safety_margin_storm"], row["minimum_safety_margin_sage"], 1.0e-4)
        row["trajectory_length_ee_winner"] = _winner(True, row["trajectory_length_ee_storm"], row["trajectory_length_ee_sage"], 1.0e-4)
        row["trajectory_length_joint_winner"] = _winner(True, row["trajectory_length_joint_storm"], row["trajectory_length_joint_sage"], 1.0e-4)
        target_rows.append(row)

    summary_rows = []
    metrics = [
        ("faster_winner", "wall_time"),
        ("final_error_winner", "final_ee_error"),
        ("minimum_margin_winner", "minimum_safety_margin"),
        ("trajectory_length_ee_winner", "trajectory_length_ee"),
        ("trajectory_length_joint_winner", "trajectory_length_joint"),
    ]
    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    for row in target_rows:
        by_difficulty[row["difficulty_tag"]].append(row)
        by_difficulty["all"].append(row)
    for difficulty, rows in sorted(by_difficulty.items()):
        for winner_field, metric in metrics:
            n = len(rows)
            sage_wins = sum(1 for row in rows if row[winner_field] == "sage")
            ties = sum(1 for row in rows if row[winner_field] == "tie")
            storm_wins = sum(1 for row in rows if row[winner_field] == "storm")
            summary_rows.append(
                {
                    "difficulty_tag": difficulty,
                    "metric": metric,
                    "num_pairs": n,
                    "sage_wins": sage_wins,
                    "ties": ties,
                    "storm_wins": storm_wins,
                    "sage_win_rate": sage_wins / n if n else math.nan,
                    "tie_rate": ties / n if n else math.nan,
                    "storm_win_rate": storm_wins / n if n else math.nan,
                }
            )
    return target_rows, summary_rows


def _csv_value(value):
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in SUMMARY_FIELDS})


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _write_summary_txt(path: Path, rows: list[dict]) -> None:
    lines = ["Static tall benchmark summary", ""]
    for row in rows:
        lines.append(
            "{method}: episodes={n}, status={status}, success_rate={succ:.3g}, "
            "collision_rate={coll:.3g}, timeout_rate={timeout:.3g}, "
            "mean_final_ee_error={ee:.5g}, mean_min_margin={margin:.5g}".format(
                method=row["method_name"],
                n=row["num_episodes"],
                status=row["status"],
                succ=row["success_rate"],
                coll=row["collision_rate"],
                timeout=row["timeout_rate"],
                ee=row["mean_final_ee_error"],
                margin=row["mean_minimum_safety_margin"],
            )
        )
        if row.get("skipped_reason"):
            lines.append(f"  skipped_reason: {row['skipped_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize static tall benchmark results")
    parser.add_argument("--results-root", default="examples/static_compare/results")
    parser.add_argument("--targets-path", default="")
    args = parser.parse_args()

    results_root = ensure_dir(resolve_repo_path(args.results_root))
    episode_path = results_root / "static_tall_episode_log.csv"
    if not episode_path.exists():
        raise FileNotFoundError(f"Missing episode log: {episode_path}")

    with open(episode_path, "r", encoding="utf-8", newline="") as f:
        episode_rows = _with_difficulty(list(csv.DictReader(f)), _target_difficulty_map(args.targets_path or None))
    summary_rows = summarize(episode_rows)
    difficulty_rows = summarize_by_difficulty(episode_rows)
    paired_target_rows, paired_summary_rows = paired_comparison(episode_rows)

    csv_path = results_root / "summary.csv"
    json_path = results_root / "summary.json"
    txt_path = results_root / "summary.txt"
    _write_summary_csv(csv_path, summary_rows)
    write_json(json_path, {"summary": summary_rows, "summary_by_difficulty": difficulty_rows})
    _write_summary_txt(txt_path, summary_rows)
    _write_summary_csv(results_root / "summary_by_difficulty.csv", difficulty_rows)
    write_json(results_root / "summary_by_difficulty.json", {"summary_by_difficulty": difficulty_rows})
    _write_csv(results_root / "paired_storm_sage_by_target.csv", paired_target_rows, PAIRED_TARGET_FIELDS)
    _write_csv(results_root / "paired_storm_sage_summary.csv", paired_summary_rows, PAIRED_SUMMARY_FIELDS)
    write_json(
        results_root / "paired_storm_sage.json",
        {"by_target": paired_target_rows, "summary": paired_summary_rows},
    )
    print(f"wrote summary to {csv_path}, {json_path}, {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
