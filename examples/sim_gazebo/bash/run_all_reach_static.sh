#!/bin/bash
# One-shot launcher for the default sim_gazebo UR7e reach-static experiment.
# Default behavior matches the original two-script workflow:
# - Gazebo GUI enabled
# - Gazebo-side RViz disabled
# - Controller-side RViz enabled
# - CUDA enabled unless --no-cuda is provided

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INITIAL_POSITIONS_FILE="${SIM_DIR}/config/initial_positions.yaml"
CONTROLLER_RVIZ_CONFIG="${SIM_DIR}/config/reach_static.rviz"

GAZEBO_PGID=""
CONTROLLER_PGID=""
SHUTTING_DOWN=false

GAZEBO_PATTERN="ros2 launch ur_simulation_gazebo ur_sim_control.launch.py.*ur_type:=ur7e.*initial_joint_controller:=forward_position_controller.*initial_positions_file:=${INITIAL_POSITIONS_FILE}"
GAZESERVER_PATTERN="gzserver /opt/ros/humble/share/gazebo_ros/worlds/empty.world"
ROBOT_STATE_PUBLISHER_PATTERN="/opt/ros/humble/lib/robot_state_publisher/robot_state_publisher"
SPAWN_ENTITY_PATTERN="/opt/ros/humble/lib/gazebo_ros/spawn_entity.py -entity ur -topic robot_description"
SPAWNER_FPC_PATTERN="/opt/ros/humble/lib/controller_manager/spawner forward_position_controller"
SPAWNER_JSB_PATTERN="/opt/ros/humble/lib/controller_manager/spawner joint_state_broadcaster"
CONTROLLER_NODE_PATTERN="${SIM_DIR}.*python3 reach_static_ur7e.py"
RVIZ_PATTERN="rviz2 -d ${CONTROLLER_RVIZ_CONFIG}"

LAUNCH_CONTROLLER_RVIZ=true
GAZEBO_GUI=true
CONTROLLER_ARGS=()

log() {
    echo "[run_all_reach_static] $*" >&2
}

usage() {
    cat <<EOF
Usage: ./run_all_reach_static.sh [--no-rviz] [--no-gui] [--no-cuda] [--rate Hz]

Starts Gazebo and the default STORM MPPI reach-static controller in one shell.
EOF
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

quote_args() {
    local quoted=""
    local arg
    for arg in "$@"; do
        quoted+=$(printf '%q ' "${arg}")
    done
    printf '%s' "${quoted}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-rviz)
            LAUNCH_CONTROLLER_RVIZ=false
            shift
            ;;
        --no-gui)
            GAZEBO_GUI=false
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            CONTROLLER_ARGS+=("$1")
            shift
            ;;
    esac
done

trap on_interrupt INT TERM
trap cleanup EXIT

if ! command -v setsid >/dev/null 2>&1; then
    echo "错误: 当前环境缺少 setsid，无法可靠管理进程组" >&2
    exit 1
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
    source /opt/ros/humble/setup.bash
fi

CONTROLLER_ARGS_QUOTED="$(quote_args "${CONTROLLER_ARGS[@]}")"

echo "============================================================"
echo "UR7e STORM MPC Reach Static - Unified Launcher"
echo "============================================================"
echo "Gazebo GUI: ${GAZEBO_GUI}"
echo "Gazebo RViz: false"
echo "Controller RViz: ${LAUNCH_CONTROLLER_RVIZ}"
echo "Controller args: ${CONTROLLER_ARGS[*]:-(none)}"
echo ""

log "cleaning up stale sim_gazebo resources before launch"
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
    gazebo_gui:='${GAZEBO_GUI}'
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
setsid bash -lc "
source /opt/ros/humble/setup.bash
eval \"\$(conda shell.bash hook)\"
conda activate storm_py310
cd '${SIM_DIR}'

RVIZ_PID=''
cleanup_controller() {
    if [[ -n \"\${RVIZ_PID}\" ]]; then
        kill \"\${RVIZ_PID}\" 2>/dev/null || true
    fi
}
trap cleanup_controller EXIT

if ${LAUNCH_CONTROLLER_RVIZ}; then
    (
        sleep 3
        echo '[RViz] 启动可视化...'
        ros2 run rviz2 rviz2 -d '${CONTROLLER_RVIZ_CONFIG}' 2>/dev/null
    ) &
    RVIZ_PID=\$!
fi

python3 reach_static_ur7e.py ${CONTROLLER_ARGS_QUOTED}
" &
CONTROLLER_PGID=$!

set +e
wait "${CONTROLLER_PGID}"
CONTROLLER_STATUS=$?
set -e
cleanup
exit "${CONTROLLER_STATUS}"
