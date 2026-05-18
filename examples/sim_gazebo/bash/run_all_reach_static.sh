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

wait "${CONTROLLER_PGID}"
