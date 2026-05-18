#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CONFIG_FILE="${SCRIPT_DIR}/config/reach_dynamic_ball_sage.rviz"

echo "=============================================="
echo "UR7e SAGE Predictive Dynamic Primitive Ball Simple Baseline"
echo "=============================================="

source /opt/ros/humble/setup.bash
if [ -f ~/ur_arm/ros_ur_driver/install/setup.bash ]; then
    source ~/ur_arm/ros_ur_driver/install/setup.bash
fi
if [ -f ~/ur_arm/gazebo_ur_sim/install/setup.bash ]; then
    source ~/ur_arm/gazebo_ur_sim/install/setup.bash
fi

eval "$(conda shell.bash hook)"
conda activate whole_control

LAUNCH_RVIZ=true
PY_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-rviz" ]]; then
        LAUNCH_RVIZ=false
        continue
    fi
    PY_ARGS+=("$arg")
done

RVIZ_PID=""
cleanup() {
    if $LAUNCH_RVIZ && [[ -n "$RVIZ_PID" ]] && kill -0 "$RVIZ_PID" 2>/dev/null; then
        kill "$RVIZ_PID" 2>/dev/null || true
        wait "$RVIZ_PID" 2>/dev/null || true
    fi
}
on_interrupt() {
    cleanup
    exit 130
}
trap on_interrupt INT TERM
trap cleanup EXIT

if $LAUNCH_RVIZ; then
    (
        sleep 3
        ros2 run rviz2 rviz2 -d "$RVIZ_CONFIG_FILE" 2>/dev/null
    ) &
    RVIZ_PID=$!
fi

cd "$SCRIPT_DIR"
python3 scripts/sage_reach_dynamic_ball_predict_simple.py "${PY_ARGS[@]}"
