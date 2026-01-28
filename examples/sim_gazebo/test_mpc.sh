#!/bin/bash
# ============================================================================
# UR7e STORM MPC 控制器启动脚本
# ============================================================================
#
# 前提条件:
#   1. 已启动 Gazebo 仿真 (run_gazebo.sh)
#   2. 已激活 STORM conda 环境
#
# ============================================================================

set -e

echo "============================================================"
echo "启动 UR7e STORM MPC 控制器"
echo "============================================================"

# 检查 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: 未检测到 conda 环境"
    echo "建议: conda activate storm_py310"
fi

# Source ROS2 环境
source /opt/ros/humble/setup.bash
echo "✓ Sourced ROS2 Humble"

# 添加 STORM 到 Python 路径
export PYTHONPATH=~/storm:$PYTHONPATH
echo "✓ Added STORM to PYTHONPATH"

# 检查 Gazebo 是否运行
echo ""
echo "检查 Gazebo 仿真状态..."
if ros2 topic list | grep -q "/joint_states"; then
    echo "✓ 检测到 /joint_states 话题"
else
    echo "✗ 未检测到 /joint_states 话题"
    echo "请先运行 ./run_gazebo.sh 启动 Gazebo 仿真"
    exit 1
fi

if ros2 topic list | grep -q "/forward_position_controller/commands"; then
    echo "✓ 检测到 ForwardPositionController"
else
    echo "✗ 未检测到 ForwardPositionController"
    echo "请确保 Gazebo 启动时使用了 initial_joint_controller:=forward_position_controller"
    exit 1
fi

echo ""
echo "启动 MPC 控制器..."
echo ""

# 进入脚本目录
cd "$(dirname "$0")"

# 运行 MPC 控制器
python3 ur7e_mpc_gazebo.py "$@"
