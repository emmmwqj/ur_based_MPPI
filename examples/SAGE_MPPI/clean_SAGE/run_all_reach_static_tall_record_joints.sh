#!/bin/bash
# One-shot clean SAGE tall-scene launcher with joint smoothness recording.
#
# This preserves the same Gazebo/controller flow as run_all_reach_static_tall.sh
# and additionally records:
#   - /joint_states
#   - /forward_position_controller/commands

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_GAZEBO_CONFIG_DIR="$(cd "${SCRIPT_DIR}/../../sim_gazebo/config" && pwd)"
INITIAL_POSITIONS_FILE="${SIM_GAZEBO_CONFIG_DIR}/initial_positions.yaml"

GAZEBO_PGID=""
CONTROLLER_PGID=""
RECORDER_PGID=""
SHUTTING_DOWN=false

RECORD_ROOT="${SCRIPT_DIR}/smoothness_records"
RECORD_DIR=""
RECORDER_ARGS=()
CONTROLLER_ARGS=()

usage() {
    cat <<EOF
Usage:
  ./run_all_reach_static_tall_record_joints.sh [recording options] [controller options]

Recording options:
  --record-dir DIR       Exact output directory for this run.
  --record-root DIR      Parent directory for timestamped output.
  --no-plots            Write CSV/JSON/Markdown only.
  --recorder-min-dt DT  Minimum sample interval kept for finite differences.
  --joint-names LIST    Comma-separated joint names for recorder order.

Common controller options passed through:
  --no-rviz
  --max-steps N
  --rate HZ
  --no-cuda
  --disable-deployment-refinement

Outputs:
  joint_states.csv
  commands.csv
  smoothness_summary.json
  smoothness_report.md
  *.png plots unless --no-plots is used
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --record-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --record-dir requires a value" >&2
                exit 2
            fi
            RECORD_DIR="$2"
            shift 2
            ;;
        --record-root)
            if [[ $# -lt 2 ]]; then
                echo "error: --record-root requires a value" >&2
                exit 2
            fi
            RECORD_ROOT="$2"
            shift 2
            ;;
        --no-plots)
            RECORDER_ARGS+=("--no-plots")
            shift
            ;;
        --recorder-min-dt)
            if [[ $# -lt 2 ]]; then
                echo "error: --recorder-min-dt requires a value" >&2
                exit 2
            fi
            RECORDER_ARGS+=("--min-dt" "$2")
            shift 2
            ;;
        --joint-names)
            if [[ $# -lt 2 ]]; then
                echo "error: --joint-names requires a value" >&2
                exit 2
            fi
            RECORDER_ARGS+=("--joint-names" "$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            CONTROLLER_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${RECORD_DIR}" ]]; then
    RECORD_DIR="${RECORD_ROOT}/$(date +%Y%m%d_%H%M%S)"
fi

log() {
    echo "[run_all_reach_static_tall_record_joints] $*" >&2
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
    kill_process_group "${RECORDER_PGID}" "joint smoothness recorder"
    kill_process_group "${GAZEBO_PGID}" "gazebo"
}

on_interrupt() {
    log "received interrupt, cleaning up"
    cleanup
    log "recording output: ${RECORD_DIR}"
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

mkdir -p "${RECORD_DIR}"

echo "============================================================"
echo "UR7e clean SAGE-MPPI Tall Scene - Joint Smoothness Recording"
echo "============================================================"
echo "Gazebo RViz: disabled"
echo "Gazebo GUI: disabled"
echo "Controller RViz: enabled unless --no-rviz is passed"
echo "Controller env: whole_control"
echo "Record dir: ${RECORD_DIR}"
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

log "Gazebo is ready, starting joint smoothness recorder"
setsid bash -lc '
source /opt/ros/humble/setup.bash
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${STORM_CONDA_ENV:-whole_control}" || true
fi
cd "$1"
shift
python3 record_joint_smoothness.py "$@"
' _ "${SCRIPT_DIR}" --output-dir "${RECORD_DIR}" "${RECORDER_ARGS[@]}" &
RECORDER_PGID=$!

sleep 1

log "starting clean SAGE controller"
setsid bash -lc 'cd "$1" && shift && ./run_reach_static_tall.sh "$@"' _ "${SCRIPT_DIR}" "${CONTROLLER_ARGS[@]}" &
CONTROLLER_PGID=$!

set +e
wait "${CONTROLLER_PGID}"
CONTROLLER_STATUS=$?
set -e
cleanup
log "recording output: ${RECORD_DIR}"
exit "${CONTROLLER_STATUS}"
