#!/bin/bash
# UR7e STORM MPC Reach Static - Gazebo 高墙场景启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "UR7e STORM MPC Reach Static - Gazebo 高墙场景"
echo "=============================================="

if [ -z "$ROS_DISTRO" ]; then
    echo "正在 source ROS2 环境..."
    source /opt/ros/humble/setup.bash
fi

echo "ROS_DISTRO: $ROS_DISTRO"

eval "$(conda shell.bash hook)"
conda activate storm_py310

if [[ "$CONDA_DEFAULT_ENV" != "storm_py310" ]]; then
    echo "错误: 无法激活 storm_py310 环境"
    exit 1
fi

echo "Conda 环境: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"
echo ""

LAUNCH_RVIZ=true
RVIZ_PID=""
for arg in "$@"; do
    if [[ "$arg" == "--no-rviz" ]]; then
        LAUNCH_RVIZ=false
    fi
done

cleanup() {
    if $LAUNCH_RVIZ && [ -n "$RVIZ_PID" ] && kill -0 "$RVIZ_PID" 2>/dev/null; then
        echo ""
        echo "关闭 RViz..."
        kill "$RVIZ_PID" 2>/dev/null || true
        wait "$RVIZ_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "启动 Reach Static 高墙场景控制器..."
echo ""
cd "$SCRIPT_DIR"

if $LAUNCH_RVIZ; then
    (
        sleep 3
        echo "[RViz] 启动可视化..."
        ros2 run rviz2 rviz2 -d "$SCRIPT_DIR/config/reach_static.rviz" 2>/dev/null
    ) &
    RVIZ_PID=$!
fi

python3 reach_static_ur7e_tall.py "$@"
