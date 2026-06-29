#!/bin/bash
# UR7e clean SAGE tall controller with switchable JointServoExecutor output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CONFIG_FILE="${SCRIPT_DIR}/config/reach_static_clean.rviz"
CONDA_ENV="${STORM_CONDA_ENV:-whole_control}"

echo "============================================================"
echo "UR7e SAGE CLEAN MPC Tall Scene - Servo Executor Controller"
echo "============================================================"

if [[ -z "${ROS_DISTRO:-}" ]]; then
    echo "source ROS2 environment"
    source /opt/ros/humble/setup.bash
fi

echo "ROS_DISTRO: ${ROS_DISTRO}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    echo "error: failed to activate ${CONDA_ENV}" >&2
    exit 1
fi

echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: $(python --version)"
echo ""

LAUNCH_RVIZ=true
PASSTHROUGH_ARGS=()
RVIZ_PID=""
for arg in "$@"; do
    if [[ "${arg}" == "--no-rviz" ]]; then
        LAUNCH_RVIZ=false
    else
        PASSTHROUGH_ARGS+=("${arg}")
    fi
done

cleanup() {
    if $LAUNCH_RVIZ && [[ -n "${RVIZ_PID}" ]] && kill -0 "${RVIZ_PID}" 2>/dev/null; then
        echo ""
        echo "stopping RViz..."
        kill "${RVIZ_PID}" 2>/dev/null || true
        wait "${RVIZ_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

cd "${SCRIPT_DIR}"

if $LAUNCH_RVIZ; then
    (
        sleep 3
        echo "[RViz] starting visualization..."
        ros2 run rviz2 rviz2 -d "${RVIZ_CONFIG_FILE}" 2>/dev/null
    ) &
    RVIZ_PID=$!
fi

python3 reach_static_ur7e_tall_servo_executor.py "${PASSTHROUGH_ARGS[@]}"
