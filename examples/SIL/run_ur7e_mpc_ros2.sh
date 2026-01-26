#!/bin/bash
# UR7e MPC ROS2 控制器启动脚本
# 
# 需要在 storm_py310 conda 环境中运行 (Python 3.10 + ROS2)

echo "=============================================="
echo "UR7e MPC ROS2 控制器"
echo "=============================================="

# 动态初始化 Conda 环境
eval "$(conda shell.bash hook)"

# 激活 conda 环境
conda activate storm_py310

# 检查 conda 环境
if [[ "$CONDA_DEFAULT_ENV" != "storm_py310" ]]; then
    echo "错误: 无法激活 storm_py310 环境"
    echo "请手动激活: conda activate storm_py310"
    exit 1
fi

# 设置 ROS2 环境
source /opt/ros/humble/setup.bash

echo ""
echo "环境信息:"
echo "  Python: $(python3 --version)"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "  ROS_DISTRO: $ROS_DISTRO"
echo ""

# 检查是否能找到 Isaac Sim 发布的话题
echo "检查 ROS2 话题..."
timeout 3 ros2 topic list 2>/dev/null | grep -E "joint" || echo "  (未发现关节话题，确保 Isaac Sim 正在运行)"
echo ""

# 运行控制器
echo "启动 MPC 控制器..."
echo ""
cd ~/storm/examples/SIL
python3 ur7e_mpc_ros2.py "$@"
