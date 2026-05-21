#!/bin/bash
# One-shot launcher for the sim_gazebo tall-scene UR7e experiment.
# Default behavior:
# - Gazebo starts headless (no Gazebo GUI, no Gazebo-side RViz)
# - Controller starts with its own RViz
# - Controller uses CUDA by default

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INITIAL_POSITIONS_FILE="${SIM_DIR}/config/initial_positions.yaml"

GAZEBO_PGID=""
CONTROLLER_PGID=""
SHUTTING_DOWN=false

GAZEBO_PATTERN="ros2 launch ur_simulation_gazebo ur_sim_control.launch.py ur_type:=ur7e initial_joint_controller:=forward_position_controller initial_positions_file:=${INITIAL_POSITIONS_FILE} launch_rviz:=false gazebo_gui:=false"
GAZESERVER_PATTERN="gzserver /opt/ros/humble/share/gazebo_ros/worlds/empty.world"
ROBOT_STATE_PUBLISHER_PATTERN="/opt/ros/humble/lib/robot_state_publisher/robot_state_publisher"
SPAWN_ENTITY_PATTERN="/opt/ros/humble/lib/gazebo_ros/spawn_entity.py -entity ur -topic robot_description"
SPAWNER_FPC_PATTERN="/opt/ros/humble/lib/controller_manager/spawner forward_position_controller"
SPAWNER_JSB_PATTERN="/opt/ros/humble/lib/controller_manager/spawner joint_state_broadcaster"
CONTROLLER_SCRIPT_PATTERN="${SIM_DIR}.*run_reach_static_tall.sh"
CONTROLLER_NODE_PATTERN="python3 reach_static_ur7e_tall.py"
RVIZ_PATTERN="rviz2 -d ${SIM_DIR}/config/reach_static_tall_validation.rviz"

log() {
    echo "[run_all_reach_static_tall] $*" >&2
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

controller_manager_ready() {
    local services
    if ! services="$(ros2 service list 2>/dev/null)"; then
        return 1
    fi

    printf '%s\n' "${services}" | grep -qx '/controller_manager/list_controllers'
}

wait_for_stale_ros_graph_to_clear() {
    local deadline=$((SECONDS + 10))
    while (( SECONDS < deadline )); do
        if ! topics_ready && ! controller_manager_ready; then
            return
        fi
        sleep 0.5
    done

    log "stale ROS graph entries are still visible after cleanup; continuing"
}

kill_process_group() {
    local pgid="$1"
    local label="$2"
    if [[ -z "$pgid" ]]; then
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

kill_matching_groups() {
    local pattern="$1"
    local label="$2"
    local pids
    pids=$(pgrep -f -- "$pattern" || true)
    for pid in $pids; do
        if [[ "$pid" == "$$" ]]; then
            continue
        fi

        local pgid
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
        if [[ -n "$pgid" ]]; then
            kill_process_group "$pgid" "${label}"
        fi
    done
}

cleanup_stale_processes() {
    kill_matching_groups "$CONTROLLER_SCRIPT_PATTERN" "controller-script-stale"
    kill_matching_groups "$CONTROLLER_NODE_PATTERN" "controller-node-stale"
    kill_matching_groups "$GAZEBO_PATTERN" "gazebo-stale"
    kill_matching_groups "$GAZESERVER_PATTERN" "gzserver-stale"
    kill_matching_groups "$ROBOT_STATE_PUBLISHER_PATTERN" "robot-state-publisher-stale"
    kill_matching_groups "$SPAWN_ENTITY_PATTERN" "spawn-entity-stale"
    kill_matching_groups "$SPAWNER_FPC_PATTERN" "spawner-fpc-stale"
    kill_matching_groups "$SPAWNER_JSB_PATTERN" "spawner-jsb-stale"

    local rviz_pids
    rviz_pids=$(pgrep -f -- "$RVIZ_PATTERN" || true)
    for pid in $rviz_pids; do
        if [[ "$pid" == "$$" ]]; then
            continue
        fi
        kill -INT "$pid" 2>/dev/null || true
        sleep 0.1
        kill -TERM "$pid" 2>/dev/null || true
    done
}

cleanup() {
    if [[ "${SHUTTING_DOWN}" == "true" ]]; then
        return
    fi
    SHUTTING_DOWN=true

    kill_process_group "${CONTROLLER_PGID}" "controller"
    kill_process_group "${GAZEBO_PGID}" "gazebo"
    cleanup_stale_processes
}

on_interrupt() {
    log "received interrupt, cleaning up"
    cleanup
    exit 130
}

trap on_interrupt INT TERM
trap cleanup EXIT

if ! command -v setsid >/dev/null 2>&1; then
    echo "错误: 当前环境缺少 setsid，无法可靠管理进程组" >&2
    exit 1
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
    source /opt/ros/humble/setup.bash
fi

echo "============================================================"
echo "UR7e STORM MPC Tall Scene - Unified Launcher"
echo "============================================================"
echo "Gazebo RViz: disabled"
echo "Gazebo GUI: disabled"
echo "Controller RViz: enabled"
echo "Controller CUDA: enabled (default)"
echo ""

log "cleaning up stale sim_gazebo tall resources before launch"
cleanup_stale_processes
wait_for_stale_ros_graph_to_clear

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

log "Gazebo is ready, starting STORM controller"
setsid bash -lc 'cd "$1" && shift && ./run_reach_static_tall.sh "$@"' _ "${SIM_DIR}" "$@" &
CONTROLLER_PGID=$!

set +e
wait "${CONTROLLER_PGID}"
CONTROLLER_STATUS=$?
set -e
cleanup
exit "${CONTROLLER_STATUS}"
