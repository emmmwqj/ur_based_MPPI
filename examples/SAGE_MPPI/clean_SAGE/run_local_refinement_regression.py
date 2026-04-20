#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
STORM_ROOT = SCRIPT_DIR.parents[2]
OFFICIAL_TASK_FILE = SCRIPT_DIR / "config" / "ur7e_reacher_gazebo_tall_sage_clean.yml"
INITIAL_POSITIONS_FILE = STORM_ROOT / "examples" / "sim_gazebo" / "config" / "initial_positions.yaml"
RUN_CONTROLLER_SCRIPT = SCRIPT_DIR / "run_reach_static_tall.sh"
DEFAULT_OUTPUT_ROOT = Path("/tmp/sage_local_refinement_regression")

FIXED_TARGETS = [
    {"name": "t1_easy", "x": 0.5, "y": -0.45, "z": 0.40},
    {"name": "t2_mid_low", "x": 0.5, "y": -0.10, "z": 0.42},
    {"name": "t3_center", "x": 0.5, "y": 0.00, "z": 0.45},
    {"name": "t4_upper_edge", "x": 0.5, "y": 0.28, "z": 0.55},
    {"name": "t5_lower_edge", "x": 0.5, "y": -0.22, "z": 0.55},
]
FIXED_SEEDS = [0, 1, 2]


def _bash(command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-lc", command],
        check=False,
        capture_output=True,
        text=True,
    )


def _topics_ready() -> bool:
    proc = _bash("source /opt/ros/humble/setup.bash >/dev/null 2>&1; ros2 topic list 2>/dev/null")
    topics = proc.stdout.splitlines()
    return "/joint_states" in topics and "/forward_position_controller/commands" in topics


def _kill_matching(regex: str) -> None:
    _bash(f"pkill -f {json.dumps(regex)} >/dev/null 2>&1 || true")


def _cleanup_existing() -> None:
    _kill_matching("ros2 launch ur_simulation_gazebo ur_sim_control.launch.py")
    _kill_matching("gzserver .*libgazebo_ros_init.so .*libgazebo_ros_factory.so .*libgazebo_ros_force_system.so")
    _kill_matching("/tmp/sage_local_refinement_regression/.*/run_reach_static_tall.sh --no-rviz --max-steps")
    time.sleep(2.0)


def _wait_for_topics(timeout_s: float = 180.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _topics_ready():
            return True
        time.sleep(1.0)
    return False


def _wait_log_contains(log_file: Path, needle: str, timeout_s: float = 60.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if log_file.exists():
            text = log_file.read_text(errors="ignore")
            if needle in text:
                return True
        time.sleep(0.5)
    return False


def _publish_target(target: dict) -> None:
    msg = (
        "{header: {frame_id: 'world'}, "
        f"pose: {{position: {{x: {target['x']}, y: {target['y']}, z: {target['z']}}}, "
        "orientation: {w: 1.0}}}"
    )
    _bash(
        "source /opt/ros/humble/setup.bash >/dev/null 2>&1; "
        f"ros2 topic pub /target_pose geometry_msgs/PoseStamped {json.dumps(msg)} -1 >/dev/null 2>&1"
    )


def _spawn_gazebo(gazebo_log: Path) -> subprocess.Popen:
    command = f"""
source /opt/ros/humble/setup.bash
if [ -f ~/ur_arm/ros_ur_driver/install/setup.bash ]; then source ~/ur_arm/ros_ur_driver/install/setup.bash; fi
if [ -f ~/ur_arm/gazebo_ur_sim/install/setup.bash ]; then source ~/ur_arm/gazebo_ur_sim/install/setup.bash; fi
ros2 launch ur_simulation_gazebo ur_sim_control.launch.py \
  ur_type:=ur7e \
  initial_joint_controller:=forward_position_controller \
  initial_positions_file:='{INITIAL_POSITIONS_FILE}' \
  launch_rviz:=false \
  gazebo_gui:=false
"""
    log_handle = gazebo_log.open("w")
    proc = subprocess.Popen(
        ["setsid", "bash", "-lc", command],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._sage_log_handle = log_handle  # type: ignore[attr-defined]
    return proc


def _spawn_controller(task_file: Path, run_log: Path, max_steps: int) -> subprocess.Popen:
    command = f"cd {json.dumps(str(SCRIPT_DIR))} && SAGE_TASK_FILE={json.dumps(str(task_file))} ./run_reach_static_tall.sh --no-rviz --max-steps {max_steps}"
    log_handle = run_log.open("w")
    proc = subprocess.Popen(
        ["setsid", "bash", "-lc", command],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._sage_log_handle = log_handle  # type: ignore[attr-defined]
    return proc


def _stop_process_group(proc: subprocess.Popen, grace_s: float = 5.0) -> None:
    try:
        if proc.poll() is not None:
            return
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        deadline = time.time() + grace_s
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(1.0)
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        handle = getattr(proc, "_sage_log_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass


def _make_seed_config(output_dir: Path, seed: int) -> Path:
    task_cfg = yaml.safe_load(OFFICIAL_TASK_FILE.read_text())
    task_cfg["control_dt"] = 0.05
    task_cfg.setdefault("mppi", {}).setdefault("sample_params", {})["seed"] = int(seed)
    out_file = output_dir / "task.yml"
    out_file.write_text(yaml.safe_dump(task_cfg, sort_keys=False))
    return out_file


def _parse_run_log(log_file: Path) -> dict:
    text = log_file.read_text(errors="ignore")
    rows = []
    for line in text.splitlines():
        if "ee_error=" not in line:
            continue

        def get_num(key: str):
            m = re.search(rf"{re.escape(key)}=([0-9.]+)", line)
            return float(m.group(1)) if m else None

        def get_bool(key: str):
            m = re.search(rf"{re.escape(key)}=(True|False)", line)
            return (m.group(1) == "True") if m else None

        rows.append(
            {
                "ee_error": get_num("ee_error"),
                "lr_active": get_bool("lr_active"),
            }
        )

    ee_vals = [r["ee_error"] for r in rows if r["ee_error"] is not None]
    final_error = ee_vals[-1] if ee_vals else math.nan
    min_error = min(ee_vals) if ee_vals else math.nan
    rebound = bool(ee_vals and final_error > min_error + 0.01)
    lr_active_count = sum(1 for r in rows if r["lr_active"])
    return {
        "final_ee_error": final_error,
        "min_ee_error": min_error,
        "final_lt_2cm": bool(final_error < 0.02) if not math.isnan(final_error) else False,
        "final_lt_5mm": bool(final_error < 0.005) if not math.isnan(final_error) else False,
        "rebound": rebound,
        "local_refinement_active_log_count": lr_active_count,
    }


def _run_one_case(output_root: Path, seed: int, target: dict, max_steps: int) -> dict:
    case_dir = output_root / f"seed{seed}_{target['name']}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    task_file = _make_seed_config(case_dir, seed)
    gazebo_log = case_dir / "gazebo.log"
    run_log = case_dir / "run.log"

    _cleanup_existing()
    gazebo_proc = _spawn_gazebo(gazebo_log)
    try:
        if not _wait_for_topics():
            raise RuntimeError("Gazebo topics did not become ready in time")

        ctrl_proc = _spawn_controller(task_file, run_log, max_steps=max_steps)
        try:
            _wait_log_contains(run_log, "说明: /target_pose", timeout_s=60.0)
            time.sleep(1.0)
            _publish_target(target)
            ctrl_proc.wait()
        finally:
            _stop_process_group(ctrl_proc)
    finally:
        _stop_process_group(gazebo_proc)

    case_metrics = _parse_run_log(run_log)
    case_metrics.update({"seed": seed, "target": target["name"]})
    return case_metrics


def _aggregate(results: list[dict]) -> dict:
    finals = [r["final_ee_error"] for r in results]
    mins = [r["min_ee_error"] for r in results]
    worst = max(results, key=lambda r: r["final_ee_error"])
    return {
        "num_runs": len(results),
        "mean_final_ee_error": sum(finals) / len(finals),
        "mean_min_ee_error": sum(mins) / len(mins),
        "final_lt_2cm_rate": sum(1 for r in results if r["final_lt_2cm"]) / len(results),
        "final_lt_5mm_rate": sum(1 for r in results if r["final_lt_5mm"]) / len(results),
        "worst_case": worst,
        "rebound_count": sum(1 for r in results if r["rebound"]),
        "local_refinement_active_runs": sum(
            1 for r in results if r["local_refinement_active_log_count"] > 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed local-refinement regression for clean SAGE tall scene.")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"official_config={OFFICIAL_TASK_FILE}")
    print(f"targets={FIXED_TARGETS}")
    print(f"seeds={FIXED_SEEDS}")

    results = []
    for seed in FIXED_SEEDS:
        for target in FIXED_TARGETS:
            print(f"RUN seed={seed} target={target['name']}", flush=True)
            results.append(_run_one_case(output_root, seed, target, max_steps=args.max_steps))

    summary = _aggregate(results)
    payload = {
        "official_config": str(OFFICIAL_TASK_FILE),
        "targets": FIXED_TARGETS,
        "seeds": FIXED_SEEDS,
        "results": results,
        "summary": summary,
    }
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2))

    print("")
    print("Regression summary")
    print(f"mean_final_ee_error={summary['mean_final_ee_error']:.6f}")
    print(f"mean_min_ee_error={summary['mean_min_ee_error']:.6f}")
    print(f"<2cm_success_rate={summary['final_lt_2cm_rate']:.3f}")
    print(f"<5mm_success_rate={summary['final_lt_5mm_rate']:.3f}")
    worst = summary["worst_case"]
    print(
        "worst_case="
        f"seed{worst['seed']}_{worst['target']}:final={worst['final_ee_error']:.6f},"
        f"min={worst['min_ee_error']:.6f}"
    )
    print(f"summary_file={output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
