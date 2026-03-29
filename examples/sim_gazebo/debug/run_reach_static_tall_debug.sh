#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

if [ -z "${ROS_DISTRO:-}" ]; then
    source /opt/ros/humble/setup.bash
fi

eval "$(conda shell.bash hook)"
conda activate storm_py310

echo "==============================================" | tee "$LOG_FILE"
echo "UR7e STORM MPC Reach Static - Gazebo 高墙场景 DEBUG" | tee -a "$LOG_FILE"
echo "==============================================" | tee -a "$LOG_FILE"
echo "日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "调试数据目录: $SCRIPT_DIR/captures" | tee -a "$LOG_FILE"

LAUNCH_RVIZ=true
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-rviz" ]]; then
        LAUNCH_RVIZ=false
    else
        ARGS+=("$arg")
    fi
done

RVIZ_PID=""
cleanup() {
    if $LAUNCH_RVIZ && [ -n "$RVIZ_PID" ] && kill -0 "$RVIZ_PID" 2>/dev/null; then
        kill "$RVIZ_PID" 2>/dev/null || true
        wait "$RVIZ_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if $LAUNCH_RVIZ; then
    (
        sleep 3
        ros2 run rviz2 rviz2 -d "$PROJECT_DIR/config/reach_static.rviz" 2>/dev/null
    ) &
    RVIZ_PID=$!
fi

cd "$PROJECT_DIR"
python3 "$SCRIPT_DIR/reach_static_ur7e_tall_debug.py" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
