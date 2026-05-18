#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.static_compare.utils.io_utils import ensure_dir, resolve_repo_path, write_json


SUMMARY_FIELDS = [
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


def summarize(episode_rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in episode_rows:
        groups[row["method_name"]].append(row)

    summary_rows = []
    for method, rows in sorted(groups.items()):
        skipped_reasons = sorted({row.get("skipped_reason", "") for row in rows if row.get("skipped_reason", "")})
        all_rrt_unavailable = (
            method in {"rrtstar", "rrtstar_ompl"}
            and rows
            and all(str(row.get("rrtstar_available", "")).strip().lower() == "false" for row in rows)
        )
        status = "unavailable" if all_rrt_unavailable else ("skipped" if skipped_reasons and not any(_parse_bool(r.get("success", "")) for r in rows) else "ok")
        summary_rows.append(
            {
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
        )
    return summary_rows


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
    args = parser.parse_args()

    results_root = ensure_dir(resolve_repo_path(args.results_root))
    episode_path = results_root / "static_tall_episode_log.csv"
    if not episode_path.exists():
        raise FileNotFoundError(f"Missing episode log: {episode_path}")

    with open(episode_path, "r", encoding="utf-8", newline="") as f:
        episode_rows = list(csv.DictReader(f))
    summary_rows = summarize(episode_rows)

    csv_path = results_root / "summary.csv"
    json_path = results_root / "summary.json"
    txt_path = results_root / "summary.txt"
    _write_summary_csv(csv_path, summary_rows)
    write_json(json_path, {"summary": summary_rows})
    _write_summary_txt(txt_path, summary_rows)
    print(f"wrote summary to {csv_path}, {json_path}, {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
