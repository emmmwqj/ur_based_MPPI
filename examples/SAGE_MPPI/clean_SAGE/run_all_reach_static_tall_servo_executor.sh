#!/bin/bash
# One-shot clean SAGE tall-scene launcher using the JointServoExecutor output layer.
#
# This keeps the same Gazebo, ros2_control, and controller configuration as
# run_all_reach_static_tall.sh. The planner is unchanged; only the final joint
# command output path is routed through the new execution layer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_GAZEBO_CONFIG_DIR="$(cd "${SCRIPT_DIR}/../../sim_gazebo/config" && pwd)"
INITIAL_POSITIONS_FILE="${SIM_GAZEBO_CONFIG_DIR}/initial_positions.yaml"

GAZEBO_PGID=""
CONTROLLER_PGID=""
SHUTTING_DOWN=false

log() {
    echo "[run_all_reach_static_tall_servo_executor] $*" >&2
}

topics_ready() {
    local topics
    if ! topics="$(ros2 topic list 2>/dev/null)"; then
        return 1
    fi

    if printf '%s\n' "${topics}" | grep -qx '/joint_states' && \
        printf '%s\n' "${topics}" | grep -qx '/forward_position_controller/commands'; then
        return 0
    fi

    return 1
}

get_pgid_for_pid() {
    local pid="$1"
    ps -o pgid= "${pid}" 2>/dev/null | tr -d ' ' || true
}

kill_process_group() {
    local pgid="$1"
    local label="$2"
    if [[ -z "${pgid}" ]]; then
        return
    fi

    if ! kill -0 "-${pgid}" 2>/dev/null; then
        return
    fi

    log "stopping ${label} (pgid=${pgid})"
    kill -INT "-${pgid}" 2>/dev/null || true

    local deadline=$((SECONDS + 8))
    while kill -0 "-${pgid}" 2>/dev/null; do
        if (( SECONDS >= deadline )); then
            break
        fi
        sleep 0.2
    done

    if kill -0 "-${pgid}" 2>/dev/null; then
        log "${label} did not exit on SIGINT, sending SIGTERM"
        kill -TERM "-${pgid}" 2>/dev/null || true
        deadline=$((SECONDS + 5))
        while kill -0 "-${pgid}" 2>/dev/null; do
            if (( SECONDS >= deadline )); then
                break
            fi
            sleep 0.2
        done
    fi

    if kill -0 "-${pgid}" 2>/dev/null; then
        log "${label} still alive, sending SIGKILL"
        kill -KILL "-${pgid}" 2>/dev/null || true
    fi
}

cleanup_existing_ur_gazebo() {
    local pids pgid
    local found=false

    pids="$(pgrep -f 'ros2 launch ur_simulation_gazebo ur_sim_control.launch.py' || true)"
    if [[ -n "${pids}" ]]; then
        found=true
        while read -r pid; do
            [[ -z "${pid}" ]] && continue
            pgid="$(get_pgid_for_pid "${pid}")"
            if [[ -n "${pgid}" ]]; then
                kill_process_group "${pgid}" "stale ur_sim_control launch"
            fi
        done <<< "${pids}"
    fi

    pids="$(pgrep -f 'gzserver .*libgazebo_ros_init.so .*libgazebo_ros_factory.so .*libgazebo_ros_force_system.so' || true)"
    if [[ -n "${pids}" ]]; then
        found=true
        while read -r pid; do
            [[ -z "${pid}" ]] && continue
            pgid="$(get_pgid_for_pid "${pid}")"
            if [[ -n "${pgid}" ]]; then
                kill_process_group "${pgid}" "stale gzserver"
            fi
        done <<< "${pids}"
    fi

    if [[ "${found}" == "true" ]]; then
        log "waiting for stale Gazebo topics to disappear"
        for _ in $(seq 1 20); do
            if ! topics_ready; then
                break
            fi
            sleep 0.5
        done
    fi
}

cleanup() {
    if [[ "${SHUTTING_DOWN}" == "true" ]]; then
        return
    fi
    SHUTTING_DOWN=true

    kill_process_group "${CONTROLLER_PGID}" "controller"
    kill_process_group "${GAZEBO_PGID}" "gazebo"
}

on_interrupt() {
    log "received interrupt, cleaning up"
    cleanup
    exit 130
}

trap on_interrupt INT TERM
trap cleanup EXIT

if ! command -v setsid >/dev/null 2>&1; then
    echo "error: setsid is required to manage process groups" >&2
    exit 1
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
    source /opt/ros/humble/setup.bash
fi

echo "============================================================"
echo "UR7e clean SAGE-MPPI Tall Scene - Servo Executor Launcher"
echo "============================================================"
echo "Gazebo RViz: disabled"
echo "Gazebo GUI: disabled"
echo "Controller RViz: enabled unless --no-rviz is passed"
echo "Execution layer default: servo"
echo ""

log "checking for stale UR Gazebo/controller processes"
cleanup_existing_ur_gazebo

log "starting Gazebo"
setsid bash -lc "
source /opt/ros/humble/setup.bash
if [ -f ~/ur_arm/ros_ur_driver/install/setup.bash ]; then
    source ~/ur_arm/ros_ur_driver/install/setup.bash
fi
if [ -f ~/ur_arm/gazebo_ur_sim/install/setup.bash ]; then
    source ~/ur_arm/gazebo_ur_sim/install/setup.bash
fi
ros2 launch ur_simulation_gazebo ur_sim_control.launch.py \
    ur_type:=ur7e \
    initial_joint_controller:=forward_position_controller \
    initial_positions_file:='${INITIAL_POSITIONS_FILE}' \
    launch_rviz:=false \
    gazebo_gui:=false
" &
GAZEBO_PGID=$!

log "waiting for Gazebo topics"
READY=false
for _ in $(seq 1 180); do
    if topics_ready; then
        READY=true
        break
    fi
    sleep 1
done

if [[ "${READY}" != "true" ]]; then
    log "Gazebo did not become ready in time"
    exit 1
fi

log "Gazebo is ready, starting clean SAGE controller with JointServoExecutor"
setsid bash -lc 'cd "$1" && shift && ./run_reach_static_tall_servo_executor.sh "$@"' _ "${SCRIPT_DIR}" "$@" &
CONTROLLER_PGID=$!

set +e
wait "${CONTROLLER_PGID}"
CONTROLLER_STATUS=$?
set -e
cleanup
exit "${CONTROLLER_STATUS}"
