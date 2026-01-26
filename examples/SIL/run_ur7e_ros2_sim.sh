#!/bin/bash
# UR7e ROS2 仿真启动脚本
# 
# 重要: 使用 Isaac Sim 内置 ROS2 库，不要 source 系统 ROS2!
# Isaac Sim (Python 3.11) 通过 DDS 与系统 ROS2 (Python 3.10) 通信

ISAAC_SIM_PATH="/home/wqj/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64"
ROS2_BRIDGE_LIB="${ISAAC_SIM_PATH}/exts/isaacsim.ros2.bridge/humble/lib"

echo "=============================================="
echo "UR7e Isaac Sim + ROS2 仿真 (OmniGraph)"
echo "=============================================="
echo ""

# 重要: 不要 source 系统 ROS2!
# Isaac Sim 使用内置 ROS2 库 (Python 3.11)
# 外部 ROS 节点使用系统 ROS2 (Python 3.10)
# DDS 中间件处理通信

echo "设置 Isaac Sim 内置 ROS2 环境变量..."

# 设置 ROS2 环境变量 (使用 Isaac Sim 内置库)
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="${ROS2_BRIDGE_LIB}:${LD_LIBRARY_PATH}"

# FastDDS 配置
export FASTRTPS_DEFAULT_PROFILES_FILE=""
export ROS_LOCALHOST_ONLY=0

echo "  ROS_DISTRO=$ROS_DISTRO"
echo "  RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "  ROS2 库路径: $ROS2_BRIDGE_LIB"
echo ""

# 动态初始化 conda (用于非交互式 shell)
eval "$(conda shell.bash hook)"

# 激活 conda 环境
conda activate env_isaaclab

if [[ "$CONDA_DEFAULT_ENV" != "env_isaaclab" ]]; then
    echo "错误: 无法激活 env_isaaclab 环境"
    exit 1
fi

echo "Conda 环境: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"
echo ""

# 运行脚本 (默认使用新的 OmniGraph 版本)
SCRIPT="${1:-ur7e_ros2_sim.py}"
shift 2>/dev/null || true

echo "运行脚本: ~/storm/examples/SIL/${SCRIPT}"
echo ""

cd ~/storm/examples/SIL
python ${SCRIPT} "$@"
