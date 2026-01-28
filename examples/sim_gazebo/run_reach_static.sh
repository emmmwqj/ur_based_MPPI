#!/bin/bash
# UR7e STORM MPC Reach Static - Gazebo 启动脚本
#
# 用法: ./run_reach_static.sh [--no-cuda] [--rate 50]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "UR7e STORM MPC Reach Static - Gazebo"
echo "=============================================="

# 检查 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    echo "正在 source ROS2 环境..."
    source /opt/ros/humble/setup.bash
fi

echo "ROS_DISTRO: $ROS_DISTRO"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate storm_py310

if [[ "$CONDA_DEFAULT_ENV" != "storm_py310" ]]; then
    echo "错误: 无法激活 storm_py310 环境"
    exit 1
fi

echo "Conda 环境: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"
echo ""

# 检查是否需要启动 RViz
LAUNCH_RVIZ=true
RVIZ_PID=""
for arg in "$@"; do
    if [[ "$arg" == "--no-rviz" ]]; then
        LAUNCH_RVIZ=false
    fi
done

# 运行控制器
echo "启动 Reach Static 控制器..."
echo ""
cd "$SCRIPT_DIR"

# 启动 RViz (后台，延迟启动以避免与 MPC 初始化竞争)
if $LAUNCH_RVIZ; then
    (
        sleep 3  # 等待 MPC 初始化完成后再启动 RViz
        echo "[RViz] 启动可视化..."
        ros2 run rviz2 rviz2 -d "$SCRIPT_DIR/config/reach_static.rviz" 2>/dev/null
    ) &
    RVIZ_PID=$!
fi

python3 reach_static_ur7e.py "$@"

# 清理 RViz
if $LAUNCH_RVIZ && [ -n "$RVIZ_PID" ]; then
    echo ""
    echo "关闭 RViz..."
    kill $RVIZ_PID 2>/dev/null || true
fi
