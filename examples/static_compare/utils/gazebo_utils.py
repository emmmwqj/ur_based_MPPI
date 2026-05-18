from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from .io_utils import ensure_dir, resolve_repo_path


DEFAULT_INITIAL_POSITIONS_FILE = "examples/sim_gazebo/config/initial_positions.yaml"


def _ros_env_command(command: str) -> str:
    return f"""
source /opt/ros/humble/setup.bash
if [ -f ~/ur_arm/ros_ur_driver/install/setup.bash ]; then
    source ~/ur_arm/ros_ur_driver/install/setup.bash
fi
if [ -f ~/ur_arm/gazebo_ur_sim/install/setup.bash ]; then
    source ~/ur_arm/gazebo_ur_sim/install/setup.bash
fi
{command}
"""


def _run_bash(command: str, timeout: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-lc", _ros_env_command(command)],
        cwd=str(resolve_repo_path(".")),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def topics_ready() -> bool:
    try:
        proc = _run_bash("ros2 topic list", timeout=3.0)
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    topics = set(proc.stdout.splitlines())
    return "/joint_states" in topics and "/forward_position_controller/commands" in topics


def _get_pgid(pid: int) -> int | None:
    try:
        proc = subprocess.run(
            ["ps", "-o", "pgid=", str(pid)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode != 0:
            return None
        text = proc.stdout.strip()
        return int(text) if text else None
    except Exception:
        return None


def kill_process_group(pgid: int | None, label: str = "process", grace_sec: float = 8.0) -> None:
    if pgid is None:
        return
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return
    except PermissionError:
        return

    for sig, wait_sec in ((signal.SIGINT, grace_sec), (signal.SIGTERM, 4.0), (signal.SIGKILL, 0.5)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.time() + wait_sec
        while time.time() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)


def terminate_process(proc: subprocess.Popen | None, label: str = "process") -> None:
    if proc is None or proc.poll() is not None:
        return
    kill_process_group(proc.pid, label=label)


def _pgrep(pattern: str) -> list[int]:
    proc = subprocess.run(
        ["pgrep", "-f", pattern],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return []
    pids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                pids.append(int(line))
            except ValueError:
                pass
    return pids


def cleanup_existing_ur_gazebo() -> None:
    patterns = [
        "ros2 launch ur_simulation_gazebo ur_sim_control.launch.py",
        "gzserver .*libgazebo_ros_init.so .*libgazebo_ros_factory.so .*libgazebo_ros_force_system.so",
    ]
    found = False
    for pattern in patterns:
        for pid in _pgrep(pattern):
            pgid = _get_pgid(pid)
            if pgid is not None:
                found = True
                kill_process_group(pgid, label=f"stale gazebo {pattern}")
    if found:
        deadline = time.time() + 10.0
        while time.time() < deadline and topics_ready():
            time.sleep(0.5)


def start_gazebo(
    log_path: str | Path,
    initial_positions_file: str | Path = DEFAULT_INITIAL_POSITIONS_FILE,
    wait_timeout_sec: float = 180.0,
) -> subprocess.Popen:
    initial_positions = resolve_repo_path(initial_positions_file)
    log_path = resolve_repo_path(log_path)
    ensure_dir(log_path.parent)
    log_file = open(log_path, "w", encoding="utf-8")
    command = _ros_env_command(
        "ros2 launch ur_simulation_gazebo ur_sim_control.launch.py "
        "ur_type:=ur7e "
        "initial_joint_controller:=forward_position_controller "
        f"initial_positions_file:='{initial_positions}' "
        "launch_rviz:=false "
        "gazebo_gui:=false"
    )
    proc = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=str(resolve_repo_path(".")),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    proc._static_compare_log_file = log_file  # type: ignore[attr-defined]

    deadline = time.time() + wait_timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Gazebo exited before topics became ready; see {log_path}")
        if topics_ready():
            return proc
        time.sleep(1.0)

    terminate_process(proc, label="gazebo")
    raise TimeoutError(f"Gazebo topics did not become ready within {wait_timeout_sec:.1f}s; see {log_path}")


def stop_gazebo(proc: subprocess.Popen | None) -> None:
    try:
        terminate_process(proc, label="gazebo")
    finally:
        log_file = getattr(proc, "_static_compare_log_file", None)
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass
