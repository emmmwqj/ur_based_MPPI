#!/bin/bash
set -e

echo "============================================================"
echo "启动 Whole Gazebo Tall ESDF Diffusion MPC 控制器"
echo "============================================================"

source /opt/ros/humble/setup.bash
eval "$(conda shell.bash hook)"
conda activate storm_py310

export PYTHONPATH=~/storm:$PYTHONPATH

if ! ros2 topic list | grep -q "/joint_states"; then
    echo "未检测到 /joint_states，请先运行 ./run_gazebo.sh"
    exit 1
fi

if ! ros2 topic list | grep -q "/forward_position_controller/commands"; then
    echo "未检测到 /forward_position_controller/commands，请确认 Gazebo 已正确启动"
    exit 1
fi

LAUNCH_RVIZ=true
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-rviz" ]]; then
        LAUNCH_RVIZ=false
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RVIZ_PID=""
cleanup() {
    if [[ -n "$RVIZ_PID" ]] && kill -0 "$RVIZ_PID" 2>/dev/null; then
        kill "$RVIZ_PID" 2>/dev/null || true
        wait "$RVIZ_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if $LAUNCH_RVIZ; then
    (
        sleep 3
        echo "[RViz] 启动 whole_sim_gazebo diffusion 可视化..."
        exec "$SCRIPT_DIR/run_rviz.sh"
    ) &
    RVIZ_PID=$!
fi

python3 ur7e_mpc_whole_gazebo_diffusion.py "${PASSTHROUGH_ARGS[@]}"
