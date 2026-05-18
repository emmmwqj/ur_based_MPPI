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
import time
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
BASH_DIR = SCRIPT_DIR / "bash"
OFFICIAL_TASK_FILE = SCRIPT_DIR / "config" / "ur7e_reacher_gazebo_tall.yml"
INITIAL_POSITIONS_FILE = SCRIPT_DIR / "config" / "initial_positions.yaml"
WORLD_FILE = SCRIPT_DIR / "config" / "collision_world_gazebo_tall.yml"
RVIZ_CONFIG_FILE = SCRIPT_DIR / "config" / "reach_static_tall_validation.rviz"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "regression_outputs" / "sequential_retargeting"
DEFAULT_SEEDS = [0, 1, 2]

TARGET_SEQUENCE = [
    {"name": "p0_default", "x": 0.4, "y": -0.5, "z": 0.4, "publish": False},
    {"name": "p1", "x": 0.4, "y": -0.5, "z": 0.4, "publish": True},
    {"name": "p2", "x": 0.33, "y": 0.65, "z": 0.3, "publish": True},
    {"name": "p3", "x": 0.54, "y": 0.0, "z": 0.5, "publish": True},
    {"name": "p4", "x": 0.33, "y": 0.65, "z": 0.3, "publish": True},
    {"name": "p5", "x": 0.36, "y": -0.57, "z": 0.43, "publish": True},
    {"name": "p6", "x": 0.4, "y": 0.1, "z": 0.1, "publish": True},
]

LOG_RE = re.compile(
    r"\[\s*(?P<step>\d+)\]\s+t=(?P<t>[0-9.]+)s\s+\|\s+q=\[[^\]]*\]\s+\|\s+ee_error=(?P<ee>[0-9.]+)\s+\|\s+opt_dt=(?P<opt>[0-9.]+)s"
)


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
    _kill_matching("/home/wqj/storm/examples/sim_gazebo/run_reach_static_tall.sh .*--max-steps")
    _kill_matching("/home/wqj/storm/examples/sim_gazebo/bash/run_sequential_retargeting_validation.sh")
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
        f"timeout 5s ros2 topic pub /target_pose geometry_msgs/PoseStamped {json.dumps(msg)} -1 >/dev/null 2>&1"
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
    proc._storm_log_handle = log_handle  # type: ignore[attr-defined]
    return proc


def _make_seed_config(output_dir: Path, seed: int) -> Path:
    task_cfg = yaml.safe_load(OFFICIAL_TASK_FILE.read_text())
    task_cfg.setdefault("mppi", {}).setdefault("sample_params", {})["seed"] = int(seed)
    out_file = output_dir / "task.yml"
    out_file.write_text(yaml.safe_dump(task_cfg, sort_keys=False))
    return out_file


def _spawn_controller(task_file: Path, run_log: Path, max_steps: int, launch_rviz: bool) -> subprocess.Popen:
    rviz_flag = "" if launch_rviz else " --no-rviz"
    command = (
        f"cd {json.dumps(str(SCRIPT_DIR))} && "
        f"STORM_TASK_FILE={json.dumps(str(task_file))} "
        f"STORM_DEFAULT_GOAL_WORLD={json.dumps(json.dumps([0.4, -0.5, 0.4]))} "
        f"STORM_RVIZ_CONFIG={json.dumps(str(RVIZ_CONFIG_FILE))} "
        f"./run_reach_static_tall.sh{rviz_flag} --max-steps {max_steps}"
    )
    log_handle = run_log.open("w")
    proc = subprocess.Popen(
        ["setsid", "bash", "-lc", command],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._storm_log_handle = log_handle  # type: ignore[attr-defined]
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
        handle = getattr(proc, "_storm_log_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass


def _load_world() -> dict:
    return yaml.safe_load(WORLD_FILE.read_text())


def _point_inside_cube(point: dict, cube: dict) -> bool:
    dims = cube["dims"]
    pose = cube["pose"]
    half = [d * 0.5 for d in dims]
    return (
        pose[0] - half[0] <= point["x"] <= pose[0] + half[0]
        and pose[1] - half[1] <= point["y"] <= pose[1] + half[1]
        and pose[2] - half[2] <= point["z"] <= pose[2] + half[2]
    )


def _point_inside_sphere(point: dict, sphere: dict) -> bool:
    px, py, pz = point["x"], point["y"], point["z"]
    sx, sy, sz = sphere["position"]
    return (px - sx) ** 2 + (py - sy) ** 2 + (pz - sz) ** 2 <= sphere["radius"] ** 2


def _validate_targets(sequence: list[dict]) -> list[str]:
    world = _load_world()["world_model"]["coll_objs"]
    issues = []
    for target in sequence:
        if target["z"] <= 0.0:
            issues.append(f"{target['name']}: z={target['z']:.3f} touches or enters ground")
            continue
        bad = False
        for sphere in world.get("sphere", {}).values():
            if _point_inside_sphere(target, sphere):
                issues.append(f"{target['name']}: inside sphere obstacle")
                bad = True
                break
        if bad:
            continue
        for cube_name, cube in world.get("cube", {}).items():
            if _point_inside_cube(target, cube):
                issues.append(f"{target['name']}: inside cube obstacle {cube_name}")
                bad = True
                break
    return issues


class LogMonitor:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self._offset = 0
        self.events: list[dict] = []

    def poll(self) -> None:
        if not self.log_file.exists():
            return
        with self.log_file.open("r", errors="ignore") as f:
            f.seek(self._offset)
            chunk = f.read()
            self._offset = f.tell()
        if not chunk:
            return
        for line in chunk.splitlines():
            m = LOG_RE.search(line)
            if not m:
                continue
            self.events.append(
                {
                    "step": int(m.group("step")),
                    "t": float(m.group("t")),
                    "ee_error": float(m.group("ee")),
                    "opt_dt": float(m.group("opt")),
                }
            )

    def wait_for_first_event(self, timeout_s: float = 60.0) -> dict:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self.poll()
            if self.events:
                return self.events[-1]
            time.sleep(0.2)
        raise RuntimeError("controller log did not produce ee_error events in time")

    def latest(self) -> dict | None:
        self.poll()
        return self.events[-1] if self.events else None


def _wait_for_segment(
    monitor: LogMonitor,
    ctrl_proc: subprocess.Popen,
    target: dict,
    publish: bool,
    timeout_s: float,
    stable_hits_required: int,
) -> dict:
    before = monitor.latest()
    if before is None:
        before = monitor.wait_for_first_event()

    switch_t = before["t"]
    pre_switch_error = before["ee_error"]
    if publish:
        _publish_target(target)

    start_index = len(monitor.events)
    deadline = time.time() + timeout_s
    stable_hits = 0
    peak_error = pre_switch_error
    min_error = math.inf
    final_error = math.nan
    t_lt_2cm = None
    t_lt_5mm = None
    stalled = False
    stall_reason = None
    controller_exited = False
    sim_timeout_reached = False
    stall_window_s = 5.0
    stall_error_band = 0.01
    stall_error_floor = 0.05

    while time.time() < deadline:
        monitor.poll()
        if len(monitor.events) <= start_index:
            if ctrl_proc.poll() is not None:
                controller_exited = True
                stall_reason = "controller_exited"
                break
            time.sleep(0.1)
            continue

        for event in monitor.events[start_index:]:
            start_index += 1
            ee = event["ee_error"]
            peak_error = max(peak_error, ee)
            min_error = min(min_error, ee)
            final_error = ee
            if t_lt_2cm is None and ee < 0.02:
                t_lt_2cm = max(0.0, event["t"] - switch_t)
            if t_lt_5mm is None and ee < 0.005:
                t_lt_5mm = max(0.0, event["t"] - switch_t)

            if ee < 0.005:
                stable_hits += 1
            else:
                stable_hits = 0

            if (event["t"] - switch_t) >= timeout_s:
                stalled = True
                stall_reason = "segment_timeout"
                sim_timeout_reached = True
                break

        if stable_hits >= stable_hits_required:
            break
        if sim_timeout_reached:
            break

        if ctrl_proc.poll() is not None:
            controller_exited = True
            stall_reason = "controller_exited"
            break

        latest = monitor.events[-1] if monitor.events else None
        if latest is not None:
            recent = [e for e in monitor.events if e["t"] >= latest["t"] - stall_window_s]
            if (
                len(recent) >= 4
                and latest["ee_error"] > stall_error_floor
                and (max(e["ee_error"] for e in recent) - min(e["ee_error"] for e in recent)) < stall_error_band
            ):
                stalled = True
                stall_reason = "no_progress"
                break
        time.sleep(0.1)

    rebound = bool(not math.isnan(final_error) and final_error > min_error + 0.01)
    stable_stop = bool(not math.isnan(final_error) and final_error < 0.005 and stable_hits >= stable_hits_required)
    if min_error is math.inf:
        min_error = math.nan

    return {
        "target": target["name"],
        "switch_t": switch_t,
        "pre_switch_ee_error": pre_switch_error,
        "peak_ee_error": peak_error,
        "time_to_lt_2cm": t_lt_2cm,
        "time_to_lt_5mm": t_lt_5mm,
        "final_ee_error": final_error,
        "min_ee_error": min_error,
        "rebound": rebound,
        "stable_stop": stable_stop,
        "stalled": stalled,
        "controller_exited": controller_exited,
        "stall_reason": stall_reason,
    }


def _run_sequence(
    output_root: Path,
    seed: int,
    max_steps: int,
    segment_timeout_s: float,
    stable_hits_required: int,
    launch_rviz: bool,
) -> dict:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    task_file = _make_seed_config(output_root, seed)
    issues = _validate_targets(TARGET_SEQUENCE)
    if issues:
        return {"target_validation_issues": issues}

    gazebo_log = output_root / "gazebo.log"
    run_log = output_root / "run.log"

    _cleanup_existing()
    gazebo_proc = _spawn_gazebo(gazebo_log)
    try:
        if not _wait_for_topics():
            raise RuntimeError("Gazebo topics did not become ready in time")

        ctrl_proc = _spawn_controller(task_file, run_log, max_steps=max_steps, launch_rviz=launch_rviz)
        try:
            _wait_log_contains(run_log, "发布 PoseStamped 到 /target_pose", timeout_s=60.0)
            monitor = LogMonitor(run_log)
            monitor.wait_for_first_event(timeout_s=60.0)

            segments = []
            for target in TARGET_SEQUENCE:
                segments.append(
                        _wait_for_segment(
                            monitor=monitor,
                            ctrl_proc=ctrl_proc,
                            target=target,
                            publish=target["publish"],
                            timeout_s=segment_timeout_s,
                            stable_hits_required=stable_hits_required,
                        )
                    )
                if segments[-1].get("controller_exited"):
                    break

            if segments and segments[-1].get("controller_exited"):
                completed_targets = {seg["target"] for seg in segments}
                for target in TARGET_SEQUENCE:
                    if target["name"] in completed_targets:
                        continue
                    segments.append(
                        {
                            "target": target["name"],
                            "switch_t": math.nan,
                            "pre_switch_ee_error": math.nan,
                            "peak_ee_error": math.nan,
                            "time_to_lt_2cm": None,
                            "time_to_lt_5mm": None,
                            "final_ee_error": math.inf,
                            "min_ee_error": math.nan,
                            "rebound": False,
                            "stable_stop": False,
                            "stalled": True,
                            "controller_exited": True,
                            "stall_reason": "controller_exited",
                        }
                    )
        finally:
            _stop_process_group(ctrl_proc)
    finally:
        _stop_process_group(gazebo_proc)

    worst = max(
        segments,
        key=lambda x: x["final_ee_error"]
        if isinstance(x["final_ee_error"], (int, float)) and not math.isnan(x["final_ee_error"])
        else -1.0,
    )
    return {
        "seed": seed,
        "target_sequence": TARGET_SEQUENCE,
        "segments": segments,
        "worst_segment": worst,
    }


def _aggregate_runs(results: list[dict]) -> dict:
    segments_by_name: dict[str, list[dict]] = {}
    all_segments: list[dict] = []
    for run in results:
        for seg in run["segments"]:
            seg_with_seed = dict(seg)
            seg_with_seed["seed"] = run["seed"]
            all_segments.append(seg_with_seed)
            segments_by_name.setdefault(seg["target"], []).append(seg_with_seed)

    aggregated_segments = []
    for name, segs in segments_by_name.items():
        finals = [s["final_ee_error"] for s in segs]
        aggregated_segments.append(
            {
                "target": name,
                "num_runs": len(segs),
                "mean_final_ee_error": sum(finals) / len(finals),
                "lt_2cm_rate": sum(1 for s in segs if s["final_ee_error"] < 0.02) / len(segs),
                "lt_5mm_rate": sum(1 for s in segs if s["final_ee_error"] < 0.005) / len(segs),
                "rebound_count": sum(1 for s in segs if s["rebound"]),
                "stalled_count": sum(1 for s in segs if s.get("stalled")),
                "worst_final_ee_error": max(finals),
            }
        )

    worst = max(all_segments, key=lambda s: s["final_ee_error"])
    return {
        "num_runs": len(results),
        "num_segments": len(all_segments),
        "overall_lt_2cm_rate": sum(1 for s in all_segments if s["final_ee_error"] < 0.02) / len(all_segments),
        "overall_lt_5mm_rate": sum(1 for s in all_segments if s["final_ee_error"] < 0.005) / len(all_segments),
        "rebound_count": sum(1 for s in all_segments if s["rebound"]),
        "stalled_count": sum(1 for s in all_segments if s.get("stalled")),
        "worst_segment": worst,
        "segments": aggregated_segments,
    }


def _format_ratio(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _render_report(summary: dict) -> str:
    lines = []
    lines.append("# Baseline STORM MPPI Sequential Retargeting Report")
    lines.append("")
    lines.append(f"- Config: `{OFFICIAL_TASK_FILE}`")
    lines.append(f"- Seeds: `{summary.get('seeds', [])}`")
    lines.append("")
    lines.append("## Target Sequence")
    lines.append("")
    for target in summary.get("target_sequence", []):
        lines.append(f"- {target['name']}: ({target['x']}, {target['y']}, {target['z']})")
    lines.append("")
    agg = summary.get("aggregate", {})
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Runs: `{agg.get('num_runs', 0)}`")
    lines.append(f"- Segments: `{agg.get('num_segments', 0)}`")
    lines.append(f"- <2cm success rate: `{_format_ratio(agg.get('overall_lt_2cm_rate', 0.0))}`")
    lines.append(f"- <5mm success rate: `{_format_ratio(agg.get('overall_lt_5mm_rate', 0.0))}`")
    lines.append(f"- Rebound count: `{agg.get('rebound_count', 0)}`")
    lines.append(f"- Stalled count: `{agg.get('stalled_count', 0)}`")
    lines.append("")
    lines.append("## Per-Target Summary")
    lines.append("")
    lines.append("| Target | Mean Final EE Error | <2cm | <5mm | Rebound Count | Stalled Count | Worst Final |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for seg in agg.get("segments", []):
        lines.append(
            f"| {seg['target']} | {seg['mean_final_ee_error']:.6f} | "
            f"{_format_ratio(seg['lt_2cm_rate'])} | {_format_ratio(seg['lt_5mm_rate'])} | "
            f"{seg['rebound_count']} | {seg['stalled_count']} | {seg['worst_final_ee_error']:.6f} |"
        )
    lines.append("")
    worst = agg.get("worst_segment", {})
    if worst:
        lines.append("## Worst Segment")
        lines.append("")
        lines.append(f"- Seed: `{worst.get('seed')}`")
        lines.append(f"- Target: `{worst.get('target')}`")
        lines.append(f"- Final EE error: `{worst.get('final_ee_error', float('nan')):.6f}`")
        lines.append(f"- Peak EE error: `{worst.get('peak_ee_error', float('nan')):.6f}`")
        lines.append(f"- <2cm time: `{worst.get('time_to_lt_2cm')}`")
        lines.append(f"- <5mm time: `{worst.get('time_to_lt_5mm')}`")
        lines.append(f"- Rebound: `{worst.get('rebound')}`")
        lines.append(f"- Stalled: `{worst.get('stalled')}`")
        lines.append(f"- Stall reason: `{worst.get('stall_reason')}`")
    lines.append("")
    lines.append("## Detailed Per-Run Segments")
    lines.append("")
    for run in summary.get("runs", []):
        seed = run.get("seed")
        lines.append(f"### Seed {seed}")
        lines.append("")
        lines.append("| Target | Peak EE Error | Min EE Error | Final EE Error | <2cm Time | <5mm Time | Rebound | Stalled | Stall Reason |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---|---|")
        for seg in run.get("segments", []):
            t2 = seg.get("time_to_lt_2cm")
            t5 = seg.get("time_to_lt_5mm")
            lines.append(
                f"| {seg['target']} | "
                f"{seg.get('peak_ee_error', float('nan')):.6f} | "
                f"{seg.get('min_ee_error', float('nan')):.6f} | "
                f"{seg.get('final_ee_error', float('nan')):.6f} | "
                f"{'-' if t2 is None else f'{t2:.2f}s'} | "
                f"{'-' if t5 is None else f'{t5:.2f}s'} | "
                f"{seg.get('rebound')} | "
                f"{seg.get('stalled') or seg.get('controller_exited')} | "
                f"{seg.get('stall_reason') or '-'} |"
            )
        lines.append("")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate sequential retargeting for baseline STORM MPPI tall scene.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--segment-timeout-s", type=float, default=35.0)
    parser.add_argument("--stable-hits", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--with-rviz", dest="with_rviz", action="store_true", default=True)
    parser.add_argument("--no-rviz", dest="with_rviz", action="store_false")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    runs = []
    for seed in args.seeds:
        run_dir = output_root / f"seed{seed}"
        result = _run_sequence(
            output_root=run_dir,
            seed=int(seed),
            max_steps=args.max_steps,
            segment_timeout_s=args.segment_timeout_s,
            stable_hits_required=args.stable_hits,
            launch_rviz=bool(args.with_rviz),
        )
        if "target_validation_issues" in result:
            summary_file = output_root / "summary.json"
            summary_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 1
        runs.append(result)

    summary = {
        "target_sequence": TARGET_SEQUENCE,
        "seeds": [int(s) for s in args.seeds],
        "runs": runs,
        "aggregate": _aggregate_runs(runs),
    }

    summary_file = output_root / "summary.json"
    report_file = output_root / "report.md"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    report_file.write_text(_render_report(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
